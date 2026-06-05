"""Helical width deflation.

At each recurrent step we keep only a contiguous block of ``rank`` inner channels
of every dense block; the block's start precesses with depth by a golden-ratio
(Weyl) stride, so the active window winds around the full width like a helix and
the union over depth covers everything while any single step is cheap.

The active fraction follows a skewed raised-sine arch over depth: it inflates
from the floor at the first step up to a peak near the front of the stack, then
decays back toward the floor through the tail - features grow early and thin out
late, biased to run narrow (and so to exit) as depth grows.

Phase 0 masks the inactive channels with a forward pre-hook on each block's
``down`` projection (full matmul, zeroed input). This proves the
capacity-fluctuation dynamics and the arch; the bucketed slice that actually
shrinks the matmul is Phase 1.
"""

import math
from contextlib import contextmanager

import torch
import torch.nn as nn

# Weyl stride: precess the active window without resonance, so successive depths
# tile the full width rather than revisiting the same channels.
GOLDEN = 0.6180339887498949


def is_compiling() -> bool:
    """True while torch.compile / Dynamo is tracing. Width policies mutate module
    forwards and weights at runtime, which Dynamo cannot trace, so they no-op
    under compile (the model runs at full width). Use ``--no-compile`` to keep
    width active - it is an eager-mode optimization."""
    check = getattr(torch.compiler, "is_compiling", None)
    return bool(check()) if check is not None else False


def width_fraction(depth: int, max_depth: int, floor: float, peak: float) -> float:
    """Active fraction of inner channels at ``depth``: a raised-sine arch from
    ``floor`` up to 1.0 (peaking at the ``peak`` fraction of the stack) and back
    down toward ``floor``."""
    if max_depth <= 1:
        return 1.0
    u = depth / (max_depth - 1)  # 0 .. 1 over the depth stack
    # Skew the symmetric arch so its peak lands at `peak` instead of the middle.
    if 0.0 < peak < 1.0 and u > 0.0:
        u = u ** (math.log(0.5) / math.log(peak))
    arch = math.sin(math.pi * u)  # 0 -> 1 -> 0
    return floor + (1.0 - floor) * arch


def _dense_down_projections(experts):
    """The inner ('down') projection of every GLU/MLP in these experts. Masking
    its input deflates the dense rank while the residual stream stays full."""
    for expert in experts:
        if not isinstance(expert, nn.Module):
            continue
        for module in expert.modules():
            down = getattr(module, "down", None)
            if isinstance(down, nn.Linear):
                yield down


def _helix_pre_hook(rank: int, start: int, width: int):
    """Pre-hook that zeros all but a depth-rotated block of ``rank`` channels of
    the projection's input. The mask is built on the input's device/dtype so it
    never up-casts a half-precision activation."""

    def hook(module, args):
        x = args[0]
        idx = (start + torch.arange(rank, device=x.device)) % width
        mask = torch.zeros(width, device=x.device, dtype=x.dtype)
        mask[idx] = 1.0
        return (x * mask, *args[1:])

    return hook


class HelicalWidth(nn.Module):
    """Deflate each dense block to a helically-precessing low-rank slice per
    recurrent step, on a raised-sine arch over depth (inflate early, decay the
    tail). ``floor`` is the minimum width fraction; ``peak`` is the depth
    fraction where width crests. An ``nn.Module`` (no parameters of its own) so
    it appears on the architecture blueprint."""

    def __init__(self, floor: float = 0.25, peak: float = 0.3):
        super().__init__()
        self.floor = floor
        self.peak = peak

    def extra_repr(self) -> str:
        return f"floor={self.floor}, peak={self.peak}"

    def fraction(self, depth, max_depth):
        """Active width fraction at a single depth (for realized-usage metrics)."""
        return width_fraction(depth, max_depth, self.floor, self.peak)

    def profile(self, max_depth):
        """The active fraction at every depth - the arch the dashboard plots."""
        return [
            width_fraction(d, max_depth, self.floor, self.peak)
            for d in range(max_depth)
        ]

    @contextmanager
    def scope(self, experts, current_depth, max_depth):
        """Patch the experts' inner rank for one step, restoring on exit. Pure
        and scoped - hooks are removed in the finally."""
        if is_compiling():  # Dynamo can't trace runtime hooks; run full width
            yield
            return
        frac = width_fraction(current_depth, max_depth, self.floor, self.peak)
        handles = []
        for down in _dense_down_projections(experts):
            width = down.in_features
            rank = max(1, min(width, round(frac * width)))
            start = round(width * GOLDEN * current_depth) % width
            handles.append(
                down.register_forward_pre_hook(_helix_pre_hook(rank, start, width))
            )
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
