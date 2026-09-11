"""Mixture-of-widths: the helical deflation policy and its profile."""

import torch
import torch.nn as nn

from praxis import registry


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


def test_full_width_is_noop_and_has_no_profile():
    policy = registry.lookup("width", "none")()
    assert policy.profile(8) is None
    block, x = _Block(), torch.randn(2, 3, 8)
    assert _active_channels(block, policy, 0, 8, x) == 12  # nothing masked
