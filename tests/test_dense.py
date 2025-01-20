# tests/test_dense.py
from itertools import product

import pytest
import torch

from praxis.activations import ACT2CLS, ACT2FN
from praxis.modules.dense import PraxisGLU, PraxisMLP, PraxisPoly

# Define test parameters
MODEL_CLASSES = [PraxisGLU, PraxisMLP, PraxisPoly]
HIDDEN_SIZES = [32, 64, 128, 256]

# Create parameter combinations
MODEL_PARAMS = list(product(MODEL_CLASSES, HIDDEN_SIZES))


@pytest.fixture(params=MODEL_PARAMS)
def model_setup(request, config):
    """
    Parametrized fixture that provides both model and its configuration.

    Args:
        request: pytest request object containing the parameter tuple
        config: the base config fixture from conftest.py

    Returns:
        tuple: (model instance, hidden_size)
    """
    model_class, hidden_size = request.param
    # Use the update method from our existing config
    updated_config = config.update(hidden_size=hidden_size)
    model = model_class(updated_config)
    return model, hidden_size


def test_forward_pass_dimensions(model_setup):
    """Test using parametrized model and dimensions."""
    model, hidden_size = model_setup
    batch_size = 32
    x = torch.randn(batch_size, hidden_size)
    output = model(x)
    assert output.shape == (batch_size, hidden_size)


# def test_large_model(large_config):
#     """Test using the large config fixture."""
#     model = PraxisGLU(large_config)
#     batch_size = 16
#     x = torch.randn(batch_size, large_config.hidden_size)
#     output = model(x)
#     assert output.shape == (batch_size, large_config.hidden_size)


# @pytest.mark.parametrize("hidden_size", [32, 64, 128, 256])
# def test_custom_config(hidden_size, config):
#     """Test using modified base config."""
#     custom_config = config.update(hidden_size=hidden_size)
#     model = PraxisGLU(custom_config)
#     batch_size = 8
#     x = torch.randn(batch_size, custom_config.hidden_size)
#     output = model(x)
#     assert output.shape == (batch_size, custom_config.hidden_size)


# from dataclasses import dataclass

# import pytest
# import torch
# import torch.nn as nn

# from praxis.activations import ACT2CLS, ACT2FN
# from praxis.modules.dense import PraxisGLU, PraxisMLP, PraxisPoly

# # List of all model classes to test
# MODEL_CLASSES = [PraxisGLU, PraxisMLP, PraxisPoly]


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


# # Parametrize the model fixture
# @pytest.fixture(params=MODEL_CLASSES)
# def model(request, config):
#     """
#     Parametrized fixture that provides instances of all model classes.

#     Args:
#         request: pytest request object containing the parameter
#         config: the config fixture

#     Returns:
#         An instance of the current model class being tested
#     """
#     model_class = request.param
#     return model_class(config)


# @pytest.mark.parametrize(
#     "batch_size,seq_length",
#     [(32, 50), (1, 10), (128, 512)],  # Standard case  # Minimum case  # Large case
# )
# def test_forward_pass_dimensions(model, batch_size, seq_length):
#     """
#     Test that the forward pass produces expected output dimensions.

#     Tests both 3D input (batch_size, seq_length, hidden_size) and
#     2D input (batch_size, hidden_size) cases.
#     """
#     hidden_size = 64

#     # Test with 3D input
#     x = torch.randn(batch_size, seq_length, hidden_size)
#     output = model(x)
#     assert output.shape == (batch_size, seq_length, hidden_size)

#     # Test with 2D input
#     x = torch.randn(batch_size, hidden_size)
#     output = model(x)
#     assert output.shape == (batch_size, hidden_size)


# def test_dropout_behavior(model):
#     """Test that dropout behaves differently in train vs eval mode."""
#     x = torch.ones(32, 64)

#     # Test in training mode
#     model.train()
#     out1 = model(x)
#     out2 = model(x)
#     assert not torch.allclose(out1, out2)  # Outputs should differ due to dropout

#     # Test in eval mode
#     model.eval()
#     out1 = model(x)
#     out2 = model(x)
#     assert torch.allclose(out1, out2)  # Outputs should be identical


# def test_initialization(model, config):
#     """Test that the model initializes with correct attributes."""
#     # Common checks for all models
#     assert isinstance(model, nn.Module)
#     assert hasattr(model, "forward")

#     # Test output shape
#     x = torch.randn(1, config.hidden_size)
#     output = model(x)
#     assert output.shape == (1, config.hidden_size)


# @pytest.mark.parametrize("activation", ["gelu", "relu", "silu"])
# def test_activation_functions(config, activation):
#     """Test that models work with different activation functions."""
#     config.activation = activation

#     for model_class in MODEL_CLASSES:
#         model = model_class(config)
#         x = torch.randn(1, config.hidden_size)
#         output = model(x)
#         assert output.shape == (1, config.hidden_size)


# # Custom marker for slow tests
# @pytest.mark.slow
# @pytest.mark.parametrize("hidden_size", [32, 64, 128, 256])
# def test_different_hidden_sizes(hidden_size, config):
#     """Test models with different hidden sizes (marked as slow)."""
#     config.hidden_size = hidden_size

#     for model_class in MODEL_CLASSES:
#         model = model_class(config)
#         x = torch.randn(1, hidden_size)
#         output = model(x)
#         assert output.shape == (1, hidden_size)
