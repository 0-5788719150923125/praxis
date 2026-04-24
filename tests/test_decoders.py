import itertools
import os
import random
from typing import List

import pytest
import torch

from praxis import (
    BLOCK_REGISTRY,
    CONTROLLER_REGISTRY,
    DECODER_REGISTRY,
    RESIDUAL_REGISTRY,
    PraxisConfig,
)
from praxis.containers import LossContainer
from praxis.routers import ROUTER_REGISTRY

TEST_PARAMS = {
    "debug": [True, False],
    "hidden_size": [64, 128],
    "num_heads": [2],
    "mod": list(ROUTER_REGISTRY.keys()),
    "depth": [3],
    "num_experts": [3, 5],
    "decoder_type": list(DECODER_REGISTRY.keys()),
    "block_type": list(BLOCK_REGISTRY.keys()),
    "controller_type": list(CONTROLLER_REGISTRY.keys()),
    "residual_type": list(RESIDUAL_REGISTRY.keys()),
}
PARAM_KEYS = list(TEST_PARAMS.keys())

# Full Cartesian product is ~34k cases; sample a stratified subset so every
# (decoder_type, block_type) pair still gets coverage but the suite finishes
# in seconds. Override with PRAXIS_DECODER_FULL=1 to run the full grid.
SAMPLES_PER_PAIR = int(os.environ.get("PRAXIS_DECODER_SAMPLES", "2"))
SAMPLE_SEED = 0xDEC0


def get_decoder_configs() -> List[PraxisConfig]:
    """Generate valid configurations."""
    param_value_lists = [TEST_PARAMS[key] for key in PARAM_KEYS]
    configs = []
    for combo in itertools.product(*param_value_lists):
        if combo[PARAM_KEYS.index("block_type")] != "mru":
            params = dict(zip(PARAM_KEYS, combo))
            # Set num_layers to match num_experts for these tests
            params["num_layers"] = params["num_experts"]
            configs.append(PraxisConfig(**params))
    return configs


def _sampled_decoder_configs() -> List[PraxisConfig]:
    """Stratified sample: SAMPLES_PER_PAIR configs per (decoder_type, block_type).

    Set PRAXIS_DECODER_FULL=1 to fall back to the full Cartesian product.
    """
    configs = get_decoder_configs()
    if os.environ.get("PRAXIS_DECODER_FULL"):
        return configs

    buckets: dict = {}
    for cfg in configs:
        buckets.setdefault((cfg.decoder_type, cfg.block_type), []).append(cfg)

    rng = random.Random(SAMPLE_SEED)
    sampled = []
    for key in sorted(buckets):
        pool = buckets[key]
        rng.shuffle(pool)
        sampled.extend(pool[:SAMPLES_PER_PAIR])
    return sampled


@pytest.fixture(params=_sampled_decoder_configs())
def module_setup(request):
    config = request.param
    decoder = DECODER_REGISTRY.get(config.decoder_type)(config)
    return decoder, config.hidden_size, config.num_experts


def test_forward_pass(module_setup):
    """Test forward pass with valid parameter combinations."""
    decoder, hidden_size, num_experts = module_setup
    batch_size = 4
    seq_len = 16

    # Create input tensor
    inputs = torch.randn(batch_size, seq_len, hidden_size)
    block_ids = torch.full(size=(batch_size, seq_len), fill_value=100, dtype=torch.long)

    # Run forward pass
    hidden_states, past_key_values, current_state, aux_loss = decoder(
        hidden_states=inputs,
        attention_mask=None,
        past_key_values=None,
        current_state=None,
        block_ids=block_ids,
        losses=LossContainer(),
    )

    # Verify output shape
    assert hidden_states.shape == inputs.shape
    # Verify correct number of layers/experts
    assert num_experts == len(decoder.locals)
