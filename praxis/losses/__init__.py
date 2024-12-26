from functools import partial

import torch.nn as nn

from praxis.losses.cross_entropy import PraxisCrossEntropyLoss
from praxis.losses.focal import FocalLoss
from praxis.losses.mile import MiLeLoss

LOSS_REGISTRY = {
    "cross_entropy": partial(PraxisCrossEntropyLoss, penalty_weight=0),
    "dedup": PraxisCrossEntropyLoss,
    "focal": FocalLoss,
    "mile": MiLeLoss,
}
