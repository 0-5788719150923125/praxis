"""Tests for praxis/normalization/rms_norm.py."""

import torch

from praxis import registry


def test_rms_norm_behavior():
    """Test RMSNorm basic functionality."""
    hidden_size = 64
    rms_norm = registry.lookup("normalization", "rms_norm")(hidden_size)

    x = torch.randn(32, 16, hidden_size)
    output = rms_norm(x)

    assert output.shape == x.shape

    # Check that RMS is normalized
    rms = torch.sqrt(torch.mean(output**2, dim=-1))
    expected_rms = torch.ones_like(rms)

    assert torch.allclose(rms, expected_rms, atol=1e-4)
