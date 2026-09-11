"""HelicalWidth: an arch-shaped per-depth width profile whose active window
precesses around the inner channels."""

import torch

from praxis import registry
from praxis.width.helical import width_fraction


def test_profile_is_an_arch():
    """Inflate early, decay through the tail: ends sit at the floor, the crest
    near the front rises well above them."""
    prof = registry.lookup("width", "helical")().profile(6)
    assert abs(prof[0] - 0.25) < 1e-6 and abs(prof[-1] - 0.25) < 1e-6
    assert max(prof) > 0.9
    assert prof.index(max(prof)) < len(prof) // 2  # crest is in the front half


def test_deflation_matches_profile(glu, active_channels):
    policy = registry.lookup("width", "helical")()
    block, x = glu(hidden=8, inner=12), torch.randn(2, 3, 8)
    prof = policy.profile(6)
    for d in range(6):
        expected = max(1, min(12, round(prof[d] * 12)))
        assert len(active_channels(block, policy, d, 6, x)) == expected


def test_helix_window_precesses_with_depth(glu, active_channels):
    """The active set at successive depths is rotated, not identical (coverage).
    ``helical_steady`` holds the width constant, so only the start moves."""
    policy = registry.lookup("width", "helical_steady")()
    block, x = glu(hidden=8, inner=12), torch.randn(2, 3, 8)
    assert active_channels(block, policy, 0, 8, x) != active_channels(
        block, policy, 1, 8, x
    )


def test_width_fraction_single_depth():
    assert width_fraction(0, 1, 0.25, 0.3) == 1.0  # degenerate stack = full width
