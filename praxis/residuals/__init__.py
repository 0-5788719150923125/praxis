from praxis.residuals.base import ResidualConnection
from praxis.residuals.hyper import HyperConnection

RESIDUAL_REGISTRY = dict(standard=ResidualConnection, hyper=HyperConnection)
