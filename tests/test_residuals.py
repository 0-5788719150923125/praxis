import itertools
from enum import Enum
from typing import List

import pytest
import torch
from torch import nn

from praxis import PraxisConfig
from praxis.modules.residual import HyperConnection, ResidualConnection

MODULE_CLASSES = [HyperConnection, ResidualConnection]


# Define test parameters in a more structured way
TEST_PARAMS = {
    "hidden_sizes": [64, 128, 256],
    "num_heads": [1, 2, 3],
}


def get_residual_configs() -> List[PraxisConfig]:
    """Generate valid attention configurations using itertools.product."""
    return [
        PraxisConfig(hidden_size=hidden_size, num_heads=num_heads)
        for hidden_size, num_heads in itertools.product(
            TEST_PARAMS["hidden_sizes"], TEST_PARAMS["num_heads"]
        )
    ]


@pytest.fixture(params=list(itertools.product(MODULE_CLASSES, get_residual_configs())))
def module_setup(request, config):
    """
    Parametrized fixture that provides module and its configuration.

    Args:
        request: pytest request object containing the parameter tuple
        config: the base config fixture from conftest.py

    Returns:
        tuple: (module instance, config)
    """
    module_class, res_config = request.param

    setattr(config, "hidden_size", res_config.hidden_size)
    setattr(config, "num_heads", res_config.num_heads)

    module = module_class(config.hidden_size)
    return module, res_config


def test_forward_pass(module_setup):
    """Test forward pass with valid parameter combinations."""
    module, res_config = module_setup
    batch_size = 32
    seq_len = 16

    # Create input tensor
    x = torch.randn(batch_size, seq_len, res_config.hidden_size)

    # Run forward pass
    residual, beta = module.connect_width(x)
    y = nn.Identity()(module.format_state(residual))
    merged = module.connect_depth(residual, y, beta)
    final = module.format_state(merged)

    # Verify output shape
    assert final.shape == (batch_size, seq_len, res_config.hidden_size)
