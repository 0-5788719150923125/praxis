"""Mixture-of-widths: per-depth deflation of the inner rank of recurrent layers.

A width policy live-patches the experts the decoder is about to run so they
compute at a lower rank for one step, restoring them on exit. The residual
stream stays full width - only each block's *inner* dense rank deflates - so a
deep recurrent stack becomes a population of narrow, low-rank voters whose
consensus over depth recovers a full-rank computation. See
``next/mixture_of_widths.md`` for the framing.

Like every Praxis subsystem this is registry-driven: ``--width-type`` selects a
named profile from the ``width`` registry (bare classes for the base shapes, partial
presets for tuned variants, exactly as the routers do). A policy exposes two
methods::

    with policy.apply(experts, current_depth, max_depth):  # scoped rank patch
        run_one_step(experts)

    policy.profile(max_depth)  # per-depth active-width arch, or None (for the dash)
"""

from functools import partial

from praxis import registry
from praxis.registry import Entry
from praxis.width.base import FullWidth
from praxis.width.helical import HelicalWidth
from praxis.width.sparse import HelicalSparseWidth

registry.declare(
    "width",
    title="Mixture-of-widths",
    doc=(
        (
            "Per-depth deflation of each block's inner rank over the recurrent loop (a "
            "helically-precessing low-rank slice), turning deep recurrence into a "
            "population of narrow voters. The ``helical`` variants mask channels under a "
            "full matmul, which proves the dynamics; the ``helical_sparse`` variants slice "
            "the weights so the matmul shrinks, a real FLOP reduction."
        )
    ),
    entries={
        "none": FullWidth,
        "helical": Entry(
            HelicalWidth,
            (
                "Mask variant on the default arch: inflates early (peak 0.3) with a "
                "width floor of 0.25."
            ),
        ),
        "helical_late": Entry(
            partial(HelicalWidth, peak=0.6),
            "``helical`` with the width crest moved mid-stack (peak 0.6).",
        ),
        "helical_steady": Entry(
            partial(HelicalWidth, floor=0.5, peak=0.5),
            (
                "``helical`` with floor and peak both 0.5: a gentle breathing that "
                "never drops below half width."
            ),
        ),
        "helical_tight": Entry(
            partial(HelicalWidth, floor=0.1, peak=0.25),
            (
                "``helical`` with aggressive deflation: floor 0.1, crest at 0.25 of "
                "depth."
            ),
        ),
        "helical_sparse": Entry(
            HelicalSparseWidth,
            (
                "Sparse variant on ``helical``'s default arch: sliced weights and a "
                "smaller matmul, so the FLOP saving is real."
            ),
        ),
        "helical_sparse_tight": Entry(
            partial(HelicalSparseWidth, floor=0.1, peak=0.25),
            (
                "``helical_tight``'s schedule (floor 0.1, peak 0.25) on the sparse, "
                "sliced-weight implementation."
            ),
        ),
    },
)
