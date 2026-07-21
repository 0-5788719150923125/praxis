from praxis.residuals.base import ResidualConnection
from praxis.residuals.hyper import HyperConnection
from praxis.residuals.rezero import ReZeroConnection
from praxis.residuals.smear import SmearResidual

RESIDUAL_REGISTRY = dict(
    standard=ResidualConnection,
    hyper=HyperConnection,
    rezero=ReZeroConnection,
    # Per-depth soft mix of standard + rezero (praxis/residuals/smear.py);
    # hyper joins once its stream-state is unified with the base contract.
    smear=SmearResidual,
)
