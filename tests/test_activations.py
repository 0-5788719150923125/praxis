import itertools

import pytest
import torch

from praxis.activations import ACTIVATION_MAP

MODULE_CLASSES = list(ACTIVATION_MAP.values())


@pytest.fixture(params=MODULE_CLASSES)
def function(request):  # Renamed to match test parameter
    module_class = request.param
    try:
        # Instantiate the module if it's a class
        if isinstance(module_class, type):
            return module_class()
        return module_class
    except Exception as e:
        pytest.skip(f"Failed to initialize module: {str(e)}")


def test_is_plottable_on_the_dashboard(function):
    """Every activation must be samplable by /api/activation_curves.

    That endpoint swallows sampling failures, so an activation it cannot probe
    just disappears from the Activation Forward / Derivative charts with no
    error anywhere - it reads as "not in the model" rather than as a bug.
    """
    import torch

    from praxis.web.routes.dynamics import _activation_classes, _sample_activation

    if not isinstance(function, _activation_classes()):
        pytest.skip("not matched as an activation by the walker")

    function(torch.randn(2, 4, 111))  # materialize lazy params at width 111
    sample = _sample_activation(
        function, -5.0, 5.0, 32, torch.device("cpu"), torch.float32
    )
    assert sample is not None, f"{type(function).__name__} cannot be sampled"
    assert len(sample["forward"]) == 32
    assert len(sample["backward"]) == 32


def test_forward_pass(function):
    """Test forward pass with valid parameter combinations."""
    batch_size = 32
    seq_len = 16
    hidden_size = 64

    # Create input tensor
    x = torch.randn(batch_size, seq_len, hidden_size)

    try:
        # Run forward pass
        output = function(x)

        # Verify output shape
        assert output.shape == (batch_size, seq_len, hidden_size)

        # Additional checks for valid output
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains infinite values"

    except Exception as e:
        pytest.fail(f"Forward pass failed: {str(e)}")
