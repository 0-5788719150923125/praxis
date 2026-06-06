from praxis.residuals.base import ResidualConnection
from praxis.residuals.hyper import HyperConnection
from praxis.residuals.rezero import ReZeroConnection

RESIDUAL_REGISTRY = dict(
    standard=ResidualConnection,
    hyper=HyperConnection,
    rezero=ReZeroConnection,
)
