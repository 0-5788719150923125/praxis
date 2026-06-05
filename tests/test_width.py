"""Mixture-of-widths: the helical deflation policy and its profile."""

import torch
import torch.nn as nn

from praxis.width import WIDTH_REGISTRY
from praxis.width.helical import width_fraction


class _Block(nn.Module):
    """A GLU-shaped stand-in: ``down`` is the inner projection width policies mask."""

    def __init__(self, hidden=8, inner=12):
        super().__init__()
        self.up = nn.Linear(hidden, 2 * inner)
        self.down = nn.Linear(inner, hidden)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(a * b)


def _active_channels(block, policy, depth, max_depth, x):
    """Count inner channels that survive the policy's mask at this depth."""
    seen = {}
    with policy.scope([block], current_depth=depth, max_depth=max_depth):
        # Register AFTER the policy hook so we observe the masked input.
        handle = block.down.register_forward_pre_hook(
            lambda m, a: seen.setdefault("x", a[0].detach().clone())
        )
        block(x)
        handle.remove()
    return int((seen["x"].abs().sum(dim=(0, 1)) > 0).sum().item())


def test_registry_keys():
    assert "none" in WIDTH_REGISTRY and "helical" in WIDTH_REGISTRY


def test_full_width_is_noop_and_has_no_profile():
    policy = WIDTH_REGISTRY["none"]()
    assert policy.profile(8) is None
    block, x = _Block(), torch.randn(2, 3, 8)
    assert _active_channels(block, policy, 0, 8, x) == 12  # nothing masked


def test_profile_is_an_arch():
    """Inflate early, decay through the tail: ends sit at the floor, the crest
    near the front rises well above them."""
    prof = WIDTH_REGISTRY["helical"]().profile(6)
    assert abs(prof[0] - 0.25) < 1e-6 and abs(prof[-1] - 0.25) < 1e-6
    assert max(prof) > 0.9
    assert prof.index(max(prof)) < len(prof) // 2  # crest is in the front half


def test_deflation_matches_profile():
    policy = WIDTH_REGISTRY["helical"]()
    block, x = _Block(), torch.randn(2, 3, 8)
    prof = policy.profile(6)
    for d in range(6):
        expected = max(1, min(12, round(prof[d] * 12)))
        assert _active_channels(block, policy, d, 6, x) == expected


def test_hooks_are_removed_on_exit():
    policy = WIDTH_REGISTRY["helical"]()
    block, x = _Block(), torch.randn(2, 3, 8)
    with policy.scope([block], current_depth=3, max_depth=6):
        pass
    assert _active_channels(block, WIDTH_REGISTRY["none"](), 0, 6, x) == 12


def test_helix_window_precesses_with_depth():
    """The active set at successive depths is rotated, not identical (coverage)."""
    policy = WIDTH_REGISTRY["helical_steady"]()  # constant width, so only the start moves
    block, x = _Block(), torch.randn(2, 3, 8)

    def active_set(depth):
        seen = {}
        with policy.scope([block], current_depth=depth, max_depth=8):
            handle = block.down.register_forward_pre_hook(
                lambda m, a: seen.setdefault("x", a[0].detach().clone())
            )
            block(x)
            handle.remove()
        return set((seen["x"].abs().sum(dim=(0, 1)) > 0).nonzero().flatten().tolist())

    assert active_set(0) != active_set(1)


def test_width_fraction_single_depth():
    assert width_fraction(0, 1, 0.25, 0.3) == 1.0  # degenerate stack = full width
