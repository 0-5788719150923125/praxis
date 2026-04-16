from praxis.exits.base import BaseExit
from praxis.exits.kl import KLDivergenceExit

EXIT_REGISTRY = {
    "none": BaseExit,
    "kl": KLDivergenceExit,
}
