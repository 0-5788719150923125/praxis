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


def test_registry_keys():
    assert "none" in registry.namespace("width") and "helical" in registry.namespace(
        "width"
    )


def test_hooks_are_removed_on_exit():
    policy = registry.lookup("width", "helical")()
    block, x = _Block(), torch.randn(2, 3, 8)
    with policy.scope([block], current_depth=3, max_depth=6):
        pass
    assert _active_channels(block, registry.lookup("width", "none")(), 0, 6, x) == 12


# ─── Sparse (true-slice) variant ─────────────────────────────────────────────


class _GLU(nn.Module):
    """up -> chunk(2) -> a*act(b) -> down, like GatedLinearMLP/ArcGLU."""

    def __init__(self, hidden=16, inner=24):
        super().__init__()
        self.up = nn.Linear(hidden, 2 * inner)
        self.down = nn.Linear(inner, hidden)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(a * torch.tanh(b))


def test_sparse_preserves_output_shape():
    pol = registry.lookup("width", "helical_sparse")()
    blk, x = _GLU(), torch.randn(2, 4, 16)
    for d in range(6):
        with pol.scope([blk], current_depth=d, max_depth=6):
            out = blk(x)
        assert out.shape == (2, 4, 16)


def test_sparse_restores_full_forward_on_exit():
    pol = registry.lookup("width", "helical_sparse")()
    blk, x = _GLU(inner=24), torch.randn(2, 4, 16)
    with pol.scope([blk], current_depth=5, max_depth=6):
        pass
    assert blk.up(x).shape[-1] == 48  # back to full 2*inner
