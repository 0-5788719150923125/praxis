"""Tests for the Titans NeuralMemory core and surfacings (praxis.memory)."""

import pytest
import torch


def test_spline_dense_shapes_and_knot_gradients():
    """The spline dense variant maps dim -> dim, stays finite on extreme
    inputs (compact support: far-out values ride the base path), and its knot
    positions/widths receive gradient - they must be learnable for the
    test-time re-knotting thesis to apply."""
    from praxis.dense.spline import SplineNetwork

    class Cfg:
        hidden_size = 32
        activation = "gelu"

    torch.manual_seed(0)
    net = SplineNetwork(Cfg(), num_knots=6)
    x = torch.randn(2, 16, 32, requires_grad=True)
    y = net(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert net.knots.grad is not None and torch.isfinite(net.knots.grad).all()
    assert net.log_widths.grad is not None

    assert torch.isfinite(net(torch.ones(1, 1, 32) * 1000)).all()
