"""Sweep over the decoders registry.

Every (decoder_type, block_type) pair runs once. The other axes are assigned
round-robin over the pairs, so every controller, residual, hidden size and
layer count is hit without building the full product. Routers are not an
axis here: the causality sweep in tests/test_modeling.py builds every router.
"""

import itertools

import pytest
import torch

from praxis import PraxisConfig, registry
from praxis.containers import LossContainer

DECODERS = list(registry.namespace("decoders"))
BLOCKS = list(registry.namespace("blocks"))
CONTROLLERS = list(registry.namespace("controllers"))
RESIDUALS = list(registry.namespace("residuals"))
HIDDEN_SIZES = [64, 128]
# Shuffling controllers draw depth layers without replacement: keep >= depth.
NUM_LAYERS = [3, 5]


def _cases():
    cases = []
    for i, (decoder, block) in enumerate(itertools.product(DECODERS, BLOCKS)):
        params = dict(
            decoder_type=decoder,
            block_type=block,
            controller_type=CONTROLLERS[i % len(CONTROLLERS)],
            residual_type=RESIDUALS[i % len(RESIDUALS)],
            hidden_size=HIDDEN_SIZES[i % len(HIDDEN_SIZES)],
            num_layers=NUM_LAYERS[(i // 2) % len(NUM_LAYERS)],
        )
        cases.append(pytest.param(params, id="-".join(str(v) for v in params.values())))
    return cases


CASES = _cases()


def test_the_round_robin_covers_every_axis_value():
    seen = [c.values[0] for c in CASES]
    for key, values in (
        ("controller_type", CONTROLLERS),
        ("residual_type", RESIDUALS),
        ("hidden_size", HIDDEN_SIZES),
        ("num_layers", NUM_LAYERS),
    ):
        assert {p[key] for p in seen} == set(values), key


@pytest.mark.parametrize("params", CASES)
def test_forward_pass(params):
    extra = {"num_heads": 2}
    if params["block_type"] == "mru":
        # mru's state head (hidden_size / num_heads) must be a perfect square.
        hidden = params["hidden_size"]
        extra = {"num_heads": hidden // 64, "embed_size": hidden}
    config = PraxisConfig(depth=3, num_experts=params["num_layers"], **params, **extra)
    decoder = registry.lookup("decoders", config.decoder_type)(config)
    inputs = torch.randn(4, 16, config.hidden_size)
    block_ids = torch.full((4, 16), 100, dtype=torch.long)

    hidden_states, _, _, _ = decoder(
        hidden_states=inputs,
        attention_mask=None,
        past_key_values=None,
        current_state=None,
        block_ids=block_ids,
        losses=LossContainer(),
    )
    assert hidden_states.shape == inputs.shape
    assert len(decoder.locals) == config.num_layers
