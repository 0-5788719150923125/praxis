import pytest
import torch

from praxis import registry
from praxis.normalization import NoNorm


def test_no_normalization_identity():
    """Test that NoNorm returns input unchanged."""
    normalized_shape = 64
    norm = NoNorm(normalized_shape)

    x = torch.randn(10, 20, 64)
    output = norm(x)

    assert torch.equal(x, output), "NoNorm should return input unchanged"


def test_no_normalization_parameters():
    """Test that NoNorm stores parameters correctly."""
    normalized_shape = 128
    eps = 1e-6
    norm = NoNorm(normalized_shape, eps=eps)

    assert norm.normalized_shape == normalized_shape
    assert norm.eps == eps


def test_no_normalization_all_modes():
    """Test that NoNorm always returns input unchanged regardless of mode."""
    hidden_size = 64
    x = torch.randn(10, 20, hidden_size)

    none_norm = registry.lookup("normalization", "none")(hidden_size)

    # All modes should return input unchanged
    assert torch.equal(none_norm(x, mode="pre"), x)
    assert torch.equal(none_norm(x, mode="post"), x)
    assert torch.equal(none_norm(x, mode="both"), x)
    assert torch.equal(none_norm(x, mode="none"), x)
    assert torch.equal(none_norm(x, mode="direct"), x)
