"""How a run's several loss terms fold into one scalar.

A strategy receives every registered objective for the step. Those that need to
tell the terms apart also receive their NAMES and the trunk activation the
classifier classified, because the two questions a fold can be wrong about - which term is
this, and how hard is it pulling - are answerable from nothing else.
"""

from functools import partial

from praxis import registry
from praxis.registry import Entry
from praxis.strategies.anchor_capped import AnchorCapped
from praxis.strategies.naive import NaiveSummation
from praxis.strategies.real_time import RealTime
from praxis.strategies.uncertainty_weighted import UncertaintyWeighted

registry.declare(
    "strategies",
    title="Training strategies",
    doc=(
        (
            "How a run's loss terms (the main loss, regularizers, encoder and auxiliary "
            "losses) fold into one scalar to backpropagate: ``naive`` sums them "
            "unweighted; the others reweight each term, by its own magnitude, by learned "
            "uncertainty, or (``capped``) by shrinking any term whose gradient at the "
            "trunk out-pulls the main loss. A strategy only runs when there is more than "
            "one term."
        )
    ),
    entries={
        "naive": NaiveSummation,
        "real_time": Entry(
            RealTime,
            (
                "Divides each term by its own detached value, so every term equals 1 "
                "and the gradient proportions follow the latest loss values."
            ),
        ),
        "weighted": UncertaintyWeighted,
        "weighted_clamped": Entry(
            partial(UncertaintyWeighted, clamped=True),
            (
                "weighted with each learned uncertainty clamped to [0.1, 10], so the "
                "1/sigma^2 factor cannot explode when sigma gets small."
            ),
        ),
        "capped": Entry(
            AnchorCapped,
            (
                "Gradient-space rather than value-space: scales back any term that "
                "pulls on the trunk harder than the task objective does, and leaves "
                "everything else exactly as the plain sum has it."
            ),
        ),
    },
)
