import torch.nn as nn

from praxis.losses.focal import FocalLoss
from praxis.losses.mile import MiLeLoss

LOSS_REGISTRY = {
    "cross_entropy": nn.CrossEntropyLoss,
    "focal": FocalLoss,
    "mile": MiLeLoss,
}
