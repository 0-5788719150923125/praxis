"""BaseDecoder's layer-stack construction (praxis/decoders/base.py): which
layout each router gets, and what that layout builds."""

import pytest
import torch

from praxis import PraxisConfig, registry
from praxis.containers import LossContainer
from praxis.decoders.base import _router_layout, _wants_expert_bank
from praxis.routers.bank import ExpertBank
from praxis.routers.smear import SMEAR


def _decoder(**overrides):
    config = PraxisConfig(
        **{
            "hidden_size": 64,
            "num_heads": 4,
            "depth": 6,
            "decoder_type": "sequential",
            "block_type": "recurrent",
            **overrides,
        }
    )
    return registry.lookup("decoders", config.decoder_type)(config), config


def test_layout_is_shared_so_the_decoder_builds_one_block():
    for key in ("smear", "vear", "distance"):
        assert _router_layout(key) == "shared"
        assert not _wants_expert_bank(key)


def test_shared_layout_reuses_one_smear_layer_at_every_position():
    """num_layers positions, one LocalLayer object, and num_experts deviations
    per target - not num_experts blocks."""
    decoder, config = _decoder(router_type="smear", num_experts=4, num_layers=3)
    assert len(decoder.locals) == config.num_layers
    assert all(layer is decoder.locals[0] for layer in decoder.locals)
    router = decoder.locals[0].router
    assert isinstance(router, SMEAR)
    assert router.num_experts == config.num_experts
    assert router.targets, "no merge targets discovered"

    # The recurrent block holds its own ExpertBank, which already routes itself
    # and must not be routed a second time.
    assert not any(
        isinstance(decoder.locals[0].block.get_submodule(t.name), ExpertBank)
        for t in router.targets
        if t.name
    )

    hidden_states = torch.randn(2, 10, config.hidden_size)
    output, _, _, losses = decoder(hidden_states, losses=LossContainer())
    assert output.shape == hidden_states.shape
    assert isinstance(losses, LossContainer)


def test_prismatic_experts_cycle_alibi_and_rope():
    """The decoder builds each Prismatic expert under a different encoding and
    puts the config's own encoding back afterwards."""
    decoder, config = _decoder(
        router_type="prismatic",
        block_type="transformer",
        attention_type="causal",
        num_experts=4,
        num_layers=2,
        encoding="nope",
    )
    experts = decoder.locals[0].router.experts
    assert [e.attn.pos_type for e in experts] == ["alibi", "rope", "alibi", "rope"]
    assert all(layer is decoder.locals[0] for layer in decoder.locals)
    assert config.encoding == "nope"
