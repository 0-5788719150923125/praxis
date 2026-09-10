"""How a run's several loss terms fold into one scalar.

A strategy receives every registered objective for the step. Those that need to
tell the terms apart also receive their NAMES and the trunk activation the head
classified, because the two questions a fold can be wrong about - which term is
this, and how hard is it pulling - are answerable from nothing else.
"""

from functools import partial

from praxis.strategies.anchor_capped import AnchorCapped
from praxis.strategies.naive import NaiveSummation
from praxis.strategies.real_time import RealTime
from praxis.strategies.uncertainty_weighted import UncertaintyWeighted

STRATEGIES_REGISTRY = dict(
    naive=NaiveSummation,
    real_time=RealTime,
    weighted=UncertaintyWeighted,
    weighted_clamped=partial(UncertaintyWeighted, clamped=True),
    # Gradient-space rather than value-space: scale back any term that pulls on
    # the trunk harder than the task objective does, and leave everything else
    # exactly as the plain sum had it.
    capped=AnchorCapped,
)
