from praxis import registry
from praxis.registry import Entry
from praxis.residuals.base import ResidualConnection
from praxis.residuals.hyper import HyperConnection
from praxis.residuals.rezero import ReZeroConnection
from praxis.residuals.smear import SmearResidual

registry.declare(
    "residuals",
    title="Residual connections",
    doc=(
        (
            "Standard residuals vs. hyper-connections, and the per-depth gains and "
            "mixtures between them."
        )
    ),
    entries={
        "standard": ResidualConnection,
        "hyper": HyperConnection,
        "rezero": ReZeroConnection,
        "smear": Entry(
            SmearResidual,
            (
                "Per-depth soft mix of the ``standard`` and ``rezero`` connections. "
                "``hyper`` cannot join the mix, since it widens the residual to "
                "several streams."
            ),
        ),
    },
)
