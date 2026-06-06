"""Mixture-of-widths: per-depth deflation of the inner rank of recurrent layers.

A width policy live-patches the experts the decoder is about to run so they
compute at a lower rank for one step, restoring them on exit. The residual
stream stays full width - only each block's *inner* dense rank deflates - so a
deep recurrent stack becomes a population of narrow, low-rank voters whose
consensus over depth recovers a full-rank computation. See
``next/mixture_of_widths.md`` for the framing.

Like every Praxis subsystem this is registry-driven: ``--width-type`` selects a
named profile from ``WIDTH_REGISTRY`` (bare classes for the base shapes, partial
presets for tuned variants, exactly as the routers do). A policy exposes two
methods::

    with policy.apply(experts, current_depth, max_depth):  # scoped rank patch
        run_one_step(experts)

    policy.profile(max_depth)  # per-depth active-width arch, or None (for the dash)
"""

from functools import partial

from praxis.width.base import FullWidth
from praxis.width.helical import HelicalWidth
from praxis.width.sparse import HelicalSparseWidth

WIDTH_REGISTRY = {
    "none": FullWidth,
    # Mask variants (full matmul, zeroed channels): prove the dynamics.
    "helical": HelicalWidth,  # inflate early (peak 0.3), floor 0.25
    "helical_late": partial(HelicalWidth, peak=0.6),  # crest mid-stack
    "helical_steady": partial(HelicalWidth, floor=0.5, peak=0.5),  # gentle breathing
    "helical_tight": partial(
        HelicalWidth, floor=0.1, peak=0.25
    ),  # aggressive deflation
    # Sparse variants (sliced weights, smaller matmul): real FLOP reduction.
    "helical_sparse": HelicalSparseWidth,
    "helical_sparse_tight": partial(HelicalSparseWidth, floor=0.1, peak=0.25),
}
