from functools import partial

from praxis.routers.mixture_of_depths import MixtureOfDepths
from praxis.routers.smear import SMEAR

ROUTER_REGISTRY = dict(
    mixture_of_depths=MixtureOfDepths,
    mixture_of_depths_u=partial(MixtureOfDepths, layout="u"),
    mixture_of_depths_decayed=partial(MixtureOfDepths, layout="decayed"),
    mixture_of_depths_ramped=partial(MixtureOfDepths, layout="ramped"),
    mixture_of_depths_skip_2=partial(MixtureOfDepths, layout="skip_2"),
    smear=SMEAR,
)
