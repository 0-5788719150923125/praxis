from praxis.strategies.naive import NaiveSummation
from praxis.strategies.uncertainty_weighted import UncertaintyWeighted

STRATEGIES_REGISTRY = dict(naive=NaiveSummation, weighted=UncertaintyWeighted)
