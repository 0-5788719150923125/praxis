"""Tests for praxis/normalization/base.py."""

import pytest
import torch

from praxis.normalization import NoNorm


@pytest.mark.parametrize("mode", ["pre", "post", "direct", "none"])
def test_no_normalization_is_the_identity_in_every_mode(mode):
    norm = NoNorm(128, eps=1e-6)
    assert norm.normalized_shape == 128 and norm.eps == 1e-6
    assert not list(norm.parameters())
    x = torch.randn(10, 20, 128)
    assert torch.equal(norm(x, mode=mode), x)
