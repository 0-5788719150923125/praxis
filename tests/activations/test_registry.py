"""Sweeps over every ``activations`` entry and every ``activation_types`` mixture."""

import pytest
import torch

from praxis import registry
from praxis.activations import build_activation
from praxis.web.routes.dynamics import _activation_classes, _sample_activation

# Every concrete activation, plus every combination type over a small bank. Two
# parameter-free values, so a mixture costs the same to exercise as a plain one.
TEST_VALUES = ["gelu", "tanh"]
SPECS = [(name, name) for name in sorted(registry.namespace("activations"))] + [
    (
        f"type:{name}",
        {"type": name, "values": TEST_VALUES[: 1 if name == "single" else 2]},
    )
    for name in registry.namespace("activation_types")
]


@pytest.fixture(params=[s for _, s in SPECS], ids=[i for i, _ in SPECS])
def function(request):
    """One activation, built through the same resolver the model uses."""
    return build_activation(request.param)


def test_forward_pass(function):
    x = torch.randn(32, 16, 64)
    output = function(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_is_plottable_on_the_dashboard(function):
    """Every activation must be samplable by /api/activation_curves.

    That endpoint swallows sampling failures, so an activation it cannot probe
    just disappears from the Activation Forward / Derivative charts with no
    error anywhere - it reads as "not in the model" rather than as a bug.
    """
    assert isinstance(
        function, _activation_classes()
    ), f"the dashboard's model walker does not see {type(function).__name__}"

    function(torch.randn(2, 4, 111))  # materialize lazy params at width 111
    sample = _sample_activation(
        function, -5.0, 5.0, 32, torch.device("cpu"), torch.float32
    )
    assert sample is not None, f"{type(function).__name__} cannot be sampled"
    assert len(sample["forward"]) == 32
    assert len(sample["backward"]) == 32


def test_no_zero_dim_parameters(function):
    """The schedule_free wrapper swaps parameters with
    ``x.view(torch.uint8).bitwise_xor_(y.view(torch.uint8))``, which raises
    "self.dim() cannot be 0 to view Float as Byte" on a 0-dim tensor. This
    reproduces that view rather than only checking the shape."""
    function(torch.randn(2, 4, 16))  # materialize lazy params
    for name, param in function.named_parameters():
        assert param.dim() > 0, f"{name} is 0-dim"
        param.detach().clone().view(torch.uint8)
