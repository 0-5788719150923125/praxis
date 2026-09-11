import pytest
import torch

from praxis import registry
from praxis.normalization import LayerNorm


def test_layer_norm_behavior():
    """Test LayerNorm basic functionality."""
    hidden_size = 64
    layer_norm = registry.lookup("normalization", "layer_norm")(hidden_size)

    x = torch.randn(32, 16, hidden_size)
    output = layer_norm(x)

    assert output.shape == x.shape

    # Check that output is normalized (mean ≈ 0, std ≈ 1)
    mean = output.mean(dim=-1)
    std = output.std(dim=-1)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-2)
