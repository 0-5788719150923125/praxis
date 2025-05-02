from itertools import product

import pytest
import torch

from praxis.dense import DENSE_REGISTRY

# Define test parameters
MODULE_CLASSES = list(DENSE_REGISTRY.values())
HIDDEN_SIZES = [64, 256]

# Create parameter combinations
MODULE_PARAMS = list(product(MODULE_CLASSES, HIDDEN_SIZES))


@pytest.fixture(params=MODULE_PARAMS)
def module_setup(request, config):
    """
    Parametrized fixture that provides both module and its configuration.

    Args:
        request: pytest request object containing the parameter tuple
        config: the base config fixture from conftest.py

    Returns:
        tuple: (module instance, hidden_size)
    """
    module_class, hidden_size = request.param
    # Use the update method from our existing config
    setattr(config, "hidden_size", hidden_size)
    module = module_class(config)
    return module, hidden_size


def test_forward_pass(module_setup):
    """Test using parametrized module and dimensions."""
    module, hidden_size = module_setup
    batch_size = 32
    seq_len = 16
    x = torch.randn(batch_size, seq_len, hidden_size)
    output = module(x)
    assert output.shape == (batch_size, seq_len, hidden_size)
