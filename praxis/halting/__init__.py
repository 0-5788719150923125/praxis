from praxis.halting.base import BaseHalting
from praxis.halting.kl import KLDivergenceHalting

HALTING_REGISTRY = {
    "none": BaseHalting,
    "kl": KLDivergenceHalting,
}
