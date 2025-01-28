from itertools import product

import pytest
import torch

from praxis.activations import ACT2CLS, ACT2FN
from praxis.modules.dense import PraxisGLU, PraxisMLP, PraxisPoly, PraxisScatter
from praxis.modules.kan import PraxisKAN

# Define test parameters
MODULE_CLASSES = [PraxisGLU, PraxisMLP, PraxisPoly, PraxisKAN, PraxisScatter]
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


# def test_large_module(large_config):
#     """Test using the large config fixture."""
#     module = PraxisGLU(large_config)
#     batch_size = 16
#     x = torch.randn(batch_size, large_config.hidden_size)
#     output = module(x)
#     assert output.shape == (batch_size, large_config.hidden_size)


# @pytest.mark.parametrize("hidden_size", [32, 64, 128, 256])
# def test_custom_config(hidden_size, config):
#     """Test using modified base config."""
#     custom_config = config.update(hidden_size=hidden_size)
#     module = PraxisGLU(custom_config)
#     batch_size = 8
#     x = torch.randn(batch_size, custom_config.hidden_size)
#     output = module(x)
#     assert output.shape == (batch_size, custom_config.hidden_size)


# from dataclasses import dataclass

# import pytest
# import torch
# import torch.nn as nn

# from praxis.activations import ACT2CLS, ACT2FN
# from praxis.modules.dense import PraxisGLU, PraxisMLP, PraxisPoly

# # List of all module classes to test
# MODULE_CLASSES = [PraxisGLU, PraxisMLP, PraxisPoly]


# # Mock AutoConfig for testing
# @dataclass
# class MockAutoConfig:
#     hidden_size: int = 64
#     activation: str = "gelu"
#     dropout: float = 0.1


# @pytest.fixture
# def config():
#     """Fixture to provide a consistent config object for tests."""
#     return MockAutoConfig()


# # Parametrize the module fixture
# @pytest.fixture(params=MODULE_CLASSES)
# def module(request, config):
#     """
#     Parametrized fixture that provides instances of all module classes.

#     Args:
#         request: pytest request object containing the parameter
#         config: the config fixture

#     Returns:
#         An instance of the current module class being tested
#     """
#     module_class = request.param
#     return module_class(config)


# @pytest.mark.parametrize(
#     "batch_size,seq_length",
#     [(32, 50), (1, 10), (128, 512)],  # Standard case  # Minimum case  # Large case
# )
# def test_forward_pass_dimensions(module, batch_size, seq_length):
#     """
#     Test that the forward pass produces expected output dimensions.

#     Tests both 3D input (batch_size, seq_length, hidden_size) and
#     2D input (batch_size, hidden_size) cases.
#     """
#     hidden_size = 64

#     # Test with 3D input
#     x = torch.randn(batch_size, seq_length, hidden_size)
#     output = module(x)
#     assert output.shape == (batch_size, seq_length, hidden_size)

#     # Test with 2D input
#     x = torch.randn(batch_size, hidden_size)
#     output = module(x)
#     assert output.shape == (batch_size, hidden_size)


# def test_dropout_behavior(module):
#     """Test that dropout behaves differently in train vs eval mode."""
#     x = torch.ones(32, 64)

#     # Test in training mode
#     module.train()
#     out1 = module(x)
#     out2 = module(x)
#     assert not torch.allclose(out1, out2)  # Outputs should differ due to dropout

#     # Test in eval mode
#     module.eval()
#     out1 = module(x)
#     out2 = module(x)
#     assert torch.allclose(out1, out2)  # Outputs should be identical


# def test_initialization(module, config):
#     """Test that the module initializes with correct attributes."""
#     # Common checks for all modules
#     assert isinstance(module, nn.Module)
#     assert hasattr(module, "forward")

#     # Test output shape
#     x = torch.randn(1, config.hidden_size)
#     output = module(x)
#     assert output.shape == (1, config.hidden_size)


# @pytest.mark.parametrize("activation", ["gelu", "relu", "silu"])
# def test_activation_functions(config, activation):
#     """Test that modules work with different activation functions."""
#     config.activation = activation

#     for module_class in MODULE_CLASSES:
#         module = module_class(config)
#         x = torch.randn(1, config.hidden_size)
#         output = module(x)
#         assert output.shape == (1, config.hidden_size)


# # Custom marker for slow tests
# @pytest.mark.slow
# @pytest.mark.parametrize("hidden_size", [32, 64, 128, 256])
# def test_different_hidden_sizes(hidden_size, config):
#     """Test modules with different hidden sizes (marked as slow)."""
#     config.hidden_size = hidden_size

#     for module_class in MODULE_CLASSES:
#         module = module_class(config)
#         x = torch.randn(1, hidden_size)
#         output = module(x)
#         assert output.shape == (1, hidden_size)
