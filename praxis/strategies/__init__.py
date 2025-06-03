from functools import partial

from praxis.strategies.naive import NaiveSummation
from praxis.strategies.real_time import RealTime
from praxis.strategies.uncertainty_weighted import UncertaintyWeighted

STRATEGIES_REGISTRY = dict(
    naive=NaiveSummation,
    real_time=RealTime,
    weighted=UncertaintyWeighted,
    weighted_clamped=partial(UncertaintyWeighted, clamped=True),
)
