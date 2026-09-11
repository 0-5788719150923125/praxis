"""A GLU-shaped stand-in for the dense experts width policies patch."""

import pytest
import torch
import torch.nn as nn


class GLU(nn.Module):
    """up -> chunk(2) -> a*act(b) -> down, like GatedLinearMLP/ArcGLU. ``down``'s
    input is the inner width a policy masks or slices."""

    def __init__(self, hidden=16, inner=24, act=None):
        super().__init__()
        self.up = nn.Linear(hidden, 2 * inner)
        self.act = act if act is not None else nn.Tanh()
        self.down = nn.Linear(inner, hidden)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(a * self.act(b))


@pytest.fixture
def glu():
    torch.manual_seed(0)
    return GLU


@pytest.fixture
def active_channels():
    """Which inner channels reach ``down`` under the policy at this depth."""

    def count(block, policy, depth, max_depth, x):
        seen = {}
        with policy.scope([block], current_depth=depth, max_depth=max_depth):
            # Register AFTER the policy hook so we observe the masked input.
            handle = block.down.register_forward_pre_hook(
                lambda m, a: seen.setdefault("x", a[0].detach().clone())
            )
            block(x)
            handle.remove()
        return set((seen["x"].abs().sum(dim=(0, 1)) > 0).nonzero().flatten().tolist())

    return count
