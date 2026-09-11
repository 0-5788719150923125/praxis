"""Tests for praxis/transforms/alignment.py: sizing an axis so an algebra divides it."""

import math
from types import SimpleNamespace

import pytest

from praxis.transforms import aligned_size
from praxis.transforms.alignment import align_axis
from praxis.transforms.ghost import pick_algebra


@pytest.mark.parametrize(
    "hidden, expected", [(272, 26), (284, 28), (512, 36), (1024, 52)]
)
def test_alignment_generalizes_across_widths(hidden, expected):
    """Derived, not tuned: the same sqrt of the same budget on a different lattice.
    These are the widths the module's own comment claims, asserted rather than
    trusted."""
    root = math.sqrt(4 * hidden * 2 / 3)
    assert align_axis(root, 2, lambda k: k**2, minimum=2) == expected


def test_alignment_measures_from_the_true_derived_value():
    """26.93 rounds to 27, and both 26 and 28 satisfy d = 2 - but 26 is nearer the
    value the budget actually produced. Rounding first and then stepping off the
    rounded base would pick 28 half the time."""
    assert align_axis(26.93, 2, lambda k: k**2, minimum=2) == 26
    assert align_axis(27.4, 2, lambda k: k**2, minimum=2) == 28
    # No request is an ordinary round, and the floor still holds.
    assert align_axis(26.93, 1) == 27
    assert align_axis(1.2, 2, minimum=4) == 4


def test_alignment_is_advisory_not_a_forced_march():
    """A request that cannot be met comes back unaligned rather than dragging the
    model somewhere far away, and the transform then reports the tensor as
    indivisible in the [GHOST] block. A missed request is a log line."""
    # An extent that is odd at every candidate: nothing satisfies d = 2.
    assert align_axis(27.0, 2, lambda k: 2 * k + 1, minimum=2) == 27
    # And a config carrying no `transform_type` at all is simply not asking.
    assert aligned_size(SimpleNamespace(), 27.0) == 27


def test_alignment_cannot_grant_the_other_axis():
    """Granting the row axis is not the same as being ghost-eligible: `d` has to
    divide the hidden axis too, and that one belongs to the config. cyclic3 would
    be satisfied by 27 keys (729 = 3^6) and still refused, because 272 = 2^4 * 17.
    Named so the request is not mistaken for a guarantee."""
    assert align_axis(26.93, 3, lambda k: k**2, minimum=2) == 27
    assert pick_algebra((729, 272), "cyclic3") is None
