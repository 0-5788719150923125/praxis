from functools import partial

import torch.nn as nn

from praxis.losses.cross_entropy import CrossEntropyLoss
from praxis.losses.cut_cross_entropy import CutCrossEntropyLoss
from praxis.losses.focal import FocalLoss
from praxis.losses.mile import MiLeLoss
from praxis.losses.stablemax import StableMaxCrossEntropyLoss

LOSS_REGISTRY = {
    "cross_entropy": partial(CrossEntropyLoss, penalty_weight=0),
    "dedup": partial(CrossEntropyLoss, penalty_weight=0.1),
    "focal": FocalLoss,
    "mile": MiLeLoss,
    "stablemax": StableMaxCrossEntropyLoss,
    "cut_cross_entropy": CutCrossEntropyLoss,
}
