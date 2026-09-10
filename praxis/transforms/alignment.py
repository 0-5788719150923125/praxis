"""Shape requests a ghost transform makes of modules that size themselves.

Ghosting runs AFTER the model is assembled, so any module that DERIVES a
dimension rather than reading it off the config has already fixed its shape by
the time the transform walks the tree. The expansion needs ``d`` to divide both
axes of the tensor it rewrites, so a derived axis one step off a lattice takes
the whole tensor out of reach.

PEER is the case in point. Its banks are ``[num_experts * num_sets,
hidden_size]`` with ``num_experts = num_keys ** 2`` and ``num_keys =
round(sqrt(budget))``. At hidden_size 272 that rounds to 27 keys, giving
729 = 3^6 rows against a hidden width of 2^4 * 17. gcd(729, 272) = 1, so no
d > 1 divides both axes and the largest tensor group in the decoder is
unreachable for want of one key.

THE MODULE ASKS; THE TRANSFORM DOES NOT PUSH. A module states which of its
dimensions is derived and what axis it produces. The alternative - a constructor
flag plus a second registry profile per arm to pass it - writes a ghost
implementation detail into the signature of every auto-sizing module and doubles
the registry each time one appears.

WHAT A REQUEST DOES NOT BUY. Only profiles that set ``request_alignment`` are
answered, and only the broad ones do. A profile naming a single site
(``ghost_conv_complex``) must leave every other site's arithmetic exactly as its
baseline had it, or the run stops being one change off its control - a bank
quietly shrinking 729 -> 676 in an experiment that never meant to touch PEER is
the kind of confound found in a loss curve instead.

Requests are advisory: ``align_axis`` returns the unaligned value when no lattice
near it satisfies ``d``, and the transform reports the tensor as ``indivisible``
in the ``[GHOST]`` block.
"""

from __future__ import annotations

from typing import Callable, Optional

__all__ = ["align_axis"]


def align_axis(
    value: float,
    d: int,
    extent: Optional[Callable[[int], int]] = None,
    minimum: int = 1,
) -> int:
    """Nearest integer to ``value`` whose resulting axis length is divisible by ``d``.

    Args:
        value: the unrounded derived size, e.g. ``sqrt(budgeted_rows)``. Kept as a
            float on purpose: the tie-break measures distance from the TRUE
            derived value, not from its rounding, so 26.93 aligns down to 26
            rather than up to 28 from a base of 27.
        d: block count the transform will need. ``d <= 1`` means no request, and
            the value is simply rounded.
        extent: maps a candidate integer to the axis length a ghost expansion
            would have to divide. ``lambda k: k ** 2 * sets`` for a product-key
            bank whose rows are ``num_keys ** 2 * num_sets``. Identity by default.
        minimum: floor on the returned integer, which the caller owns.

    Returns ``round(value)`` (floored at ``minimum``) when nothing within ``d``
    steps qualifies, leaving the tensor to be reported as indivisible rather than
    silently moving the model a long way to accommodate a transform.
    """
    base = max(minimum, round(value))
    if d <= 1:
        return base
    measure = extent if extent is not None else (lambda k: k)
    for step in range(0, base + d):
        candidates = [
            c
            for c in (base - step, base + step)
            if c >= minimum and measure(c) % d == 0
        ]
        if candidates:
            # Distance from the true derived value, then the smaller size, so an
            # exact tie spends fewer parameters.
            return min(candidates, key=lambda c: (abs(c - value), c))
    return base
