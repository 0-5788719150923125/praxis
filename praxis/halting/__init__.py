from functools import partial

from praxis.halting.base import BaseHalting
from praxis.halting.kl import KLDivergenceHalting

HALTING_REGISTRY = {
    "none": BaseHalting,
    # Randomized training depth centred at half the budget - the recurrent-depth
    # paper's own setting, and the right shape while the budget is small.
    "kl": KLDivergenceHalting,
    # The same halting rule under a training prior whose centre grows with the
    # LOG of the budget rather than half of it. For deep loop counts, where the
    # linear rule flattens into something close to uniform over the range and
    # spends most forwards on inputs that converged after three loops. Keeps the
    # ramp (P(r=1) < P(r=2)) and makes full depth rare rather than routine; see
    # LOOP_PRIORS in praxis/halting/kl.py for the measured curves.
    "kl_log": partial(KLDivergenceHalting, prior="log"),
}
