"""PEER gated experts (praxis/dense/peer.py, DENSE_REGISTRY["peer_glu"]).

``test_dense.py`` already covers forward-pass shape for every registry entry.
What needs its own pins here are the invariants that make ``peer_glu`` a fair
comparison against ``peer`` rather than a disguised size increase.
"""

import pytest
import torch

from praxis import PraxisConfig
from praxis.dense import DENSE_REGISTRY
from praxis.dense.glu import GatedLinearMLP
from praxis.dense.peer import (
    BANK_WIDTH_MULTIPLE,
    ROWS_PER_EXPERT,
    ROWS_PER_GLU_EXPERT,
    ParameterEfficientExpertRetrieval,
)

WIDTHS = [64, 128, 256, 512]


def make_config(hidden_size=256, num_heads=16, activation="gelu", dropout=0.0):
    config = PraxisConfig()
    config.hidden_size = hidden_size
    config.num_heads = num_heads
    config.activation = activation
    config.dropout = dropout
    return config


def params(module):
    return sum(p.numel() for p in module.parameters())


def test_registry_exposes_the_gated_variant():
    assert "peer_glu" in DENSE_REGISTRY
    module = DENSE_REGISTRY["peer_glu"](make_config())
    assert isinstance(module, ParameterEfficientExpertRetrieval)
    assert module.glu is True
    assert module.gate is not None
    # And the ungated entry is untouched.
    assert DENSE_REGISTRY["peer"](make_config()).gate is None


@pytest.mark.parametrize("hidden_size", WIDTHS)
def test_gated_experts_are_capacity_matched(hidden_size):
    """The third bank row per expert comes out of expert COUNT, not the
    parameter budget - otherwise `peer_glu` vs `peer` would measure 1.5x more
    parameters rather than the architecture change."""
    config = make_config(hidden_size=hidden_size)
    plain = ParameterEfficientExpertRetrieval(config, glu=False)
    gated = ParameterEfficientExpertRetrieval(config, glu=True)

    assert gated.num_experts < plain.num_experts  # breadth traded away
    assert gated.rows_per_expert == ROWS_PER_GLU_EXPERT
    assert plain.rows_per_expert == ROWS_PER_EXPERT
    # Rounding the bank to a perfect square is the only slack.
    ratio = params(gated) / params(plain)
    assert 0.9 < ratio < 1.1, (hidden_size, ratio, params(gated), params(plain))


@pytest.mark.parametrize("hidden_size", WIDTHS)
def test_bank_holds_its_ratio_to_the_dense_ffn(hidden_size):
    """The budgeting invariant the module documents, now under either expert
    form: the bank tracks the dense FFN it replaces at every width."""
    config = make_config(hidden_size=hidden_size)
    dense = params(GatedLinearMLP(config))
    for glu in (False, True):
        peer = ParameterEfficientExpertRetrieval(config, glu=glu)
        assert peer.num_experts == peer.num_keys**2  # product-key square
        budget = BANK_WIDTH_MULTIPLE * hidden_size * ROWS_PER_EXPERT
        assert abs(peer.num_experts * peer.rows_per_expert - budget) / budget < 0.25
        assert params(peer) > dense  # a retrieval bank is the bigger object


def test_gated_forward_differs_from_ungated():
    """Sanity that the gate actually participates: zeroing it must change the
    output, and the gated path must not silently reduce to the plain one."""
    torch.manual_seed(0)
    config = make_config(hidden_size=64, num_heads=4)
    module = ParameterEfficientExpertRetrieval(config, glu=True).eval()
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        before = module(x, current_depth=0).clone()
        module.gate.weight.zero_()
        after = module(x, current_depth=0)
    assert not torch.allclose(before, after)


def test_gradients_reach_every_bank():
    config = make_config(hidden_size=64, num_heads=4, dropout=0.1)
    module = ParameterEfficientExpertRetrieval(config, glu=True)
    x = torch.randn(3, 12, 64, requires_grad=True)
    module(x, current_depth=0).sum().backward()
    for name in ("down", "gate", "up", "keys"):
        tensor = getattr(module, name)
        weight = tensor if name == "keys" else tensor.weight
        assert weight.grad is not None, name
        assert float(weight.grad.abs().sum()) > 0.0, name
    assert float(x.grad.abs().sum()) > 0.0


def test_repr_names_the_expert_form():
    """The blueprint tab renders __repr__, so the variant has to be visible
    there - two runs whose configs differ only in ffn_type would otherwise be
    indistinguishable in the architecture view."""
    plain = ParameterEfficientExpertRetrieval(make_config(), glu=False)
    gated = ParameterEfficientExpertRetrieval(make_config(), glu=True)
    assert "expert=rank1" in plain.extra_repr()
    assert "expert=glu" in gated.extra_repr()


def test_explicit_expert_count_overrides_the_budget():
    """Registry profiles may pin the bank; the rows factor must not fight it."""
    module = ParameterEfficientExpertRetrieval(make_config(), num_experts=256, glu=True)
    assert module.num_experts == 256
    assert module.num_keys == 16


def test_sparse_gated_banks_stay_sparse():
    """Sparse gradients come from the embedding lookup, so the gate bank has to
    carry the same flag or the sparse path densifies through it."""
    module = ParameterEfficientExpertRetrieval(
        make_config(hidden_size=64, num_heads=4), glu=True, sparse=True
    )
    assert module.down.sparse and module.up.sparse and module.gate.sparse
    assert module._gathers()  # sparse always gathers
    out = module(torch.randn(2, 8, 64), current_depth=0)
    assert out.shape == (2, 8, 64)
