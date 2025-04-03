import itertools
from enum import Enum
from typing import List

import pytest
import torch

from praxis import PraxisConfig
from praxis.blocks import BLOCK_REGISTRY

MODULE_CLASSES = list(BLOCK_REGISTRY.keys())
MODULE_CLASSES.remove("mru")

# Define test parameters in a more structured way
TEST_PARAMS = {
    "hidden_sizes": [64, 128, 256],
    "num_heads": [1, 2, 3],
}


def get_block_configs() -> List[PraxisConfig]:
    """Generate valid attention configurations using itertools.product."""
    return [
        PraxisConfig(hidden_size=hidden_size, num_heads=num_heads)
        for hidden_size, num_heads in itertools.product(
            TEST_PARAMS["hidden_sizes"], TEST_PARAMS["num_heads"]
        )
    ]


@pytest.fixture(params=list(itertools.product(MODULE_CLASSES, get_block_configs())))
def module_setup(request, config):
    """
    Parametrized fixture that provides module and its configuration.

    Args:
        request: pytest request object containing the parameter tuple
        config: the base config fixture from conftest.py

    Returns:
        tuple: (module instance, config)
    """
    module_class, block_config = request.param

    setattr(config, "hidden_size", block_config.hidden_size)
    setattr(config, "num_heads", block_config.num_heads)

    module = BLOCK_REGISTRY.get(module_class)(config)
    return module, block_config


def test_forward_pass(module_setup):
    """Test forward pass with valid parameter combinations."""
    module, block_config = module_setup
    batch_size = 32
    seq_len = 16

    # Create input tensor
    x = torch.randn(batch_size, seq_len, block_config.hidden_size)

    # Run forward pass
    output, layer_kv, prev_state, aux_loss = module(
        x, attention_mask=None, router_weights=None, current_state=None, current_depth=0
    )

    # Verify output shape
    assert output.shape == (batch_size, seq_len, block_config.hidden_size)
