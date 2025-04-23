import itertools
from typing import List

import pytest
import torch

from praxis import PraxisConfig
from praxis.modules.decoder import PraxisDecoder
from praxis.modules.router import MOD_LAYOUT

TEST_PARAMS = {
    "hidden_size": [64, 128],
    "num_heads": [2],
    "shuffle": [True, False],
    "graph": [True, False],
    "pathfinder": [True, False],
    "mod": list(MOD_LAYOUT.keys()),
    "depth": [3],
    "num_experts": [3],
}
PARAM_KEYS = list(TEST_PARAMS.keys())


def get_decoder_configs() -> List[PraxisConfig]:
    """Generate valid configurations."""
    param_value_lists = [TEST_PARAMS[key] for key in PARAM_KEYS]
    return [
        PraxisConfig(**dict(zip(PARAM_KEYS, combo)))
        for combo in itertools.product(*param_value_lists)
    ]


@pytest.fixture(params=get_decoder_configs())
def module_setup(request):
    config = request.param
    decoder = PraxisDecoder(config)
    return decoder, config.hidden_size, config.depth


def test_forward_pass(module_setup):
    """Test forward pass with valid parameter combinations."""
    decoder, hidden_size, depth = module_setup
    batch_size = 4
    seq_len = 16

    # Create input tensor
    inputs = torch.randn(batch_size, seq_len, hidden_size)
    block_ids = torch.full(size=(batch_size, seq_len), fill_value=100, dtype=torch.long)

    # Run forward pass
    hidden_states, past_key_values, current_state, aux_loss = decoder(
        inputs=inputs,
        current_state=None,
        attention_mask=None,
        past_key_values=None,
        block_ids=block_ids,
    )

    # Verify output shape
    assert hidden_states.shape == inputs.shape
    # Verify correct number of layers/experts
    assert depth == len(decoder.stack.locals)
