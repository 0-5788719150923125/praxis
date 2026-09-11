"""Sweeps over every ``normalization`` entry."""

import pytest
import torch

from praxis import registry

KEYS = sorted(registry.namespace("normalization"))

# Which of a sublayer's two positions each entry fills: (pre, post).
POSITIONS = {
    "none": (False, False),
    "layer_norm": (True, False),
    "rms_norm": (True, False),
    "post_rms_norm": (False, True),
    "sandwich": (True, True),
    "sandwich_tied": (True, True),
    "hero": (True, True),
    "hero_inverted": (True, True),
}


@pytest.mark.parametrize("key", KEYS)
def test_forward_pass(key):
    norm = registry.lookup("normalization", key)(64, eps=1e-5)
    x = torch.randn(32, 16, 64)
    output = norm(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("key", KEYS)
def test_each_mode_normalizes_exactly_the_positions_the_entry_fills(key):
    assert key in POSITIONS, f"declare which positions {key!r} fills in POSITIONS"
    pre, post = POSITIONS[key]
    norm = registry.lookup("normalization", key)(64)
    x = torch.randn(10, 20, 64)

    expected = {"pre": pre, "post": post, "direct": pre or post, "none": False}
    for mode, normalizes in expected.items():
        assert torch.equal(norm(x, mode=mode), x) is not normalizes, (
            f"{key} mode={mode} should {'' if normalizes else 'not '}normalize"
        )
