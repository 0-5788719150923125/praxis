from functools import partial

import torch
import torch.nn as nn

from praxis.losses.cross_entropy import CrossEntropyLoss
from praxis.losses.cut_cross_entropy import CutCrossEntropyLoss
from praxis.losses.focal import FocalLoss
from praxis.losses.mile import MiLeLoss
from praxis.losses.stablemax import StableMaxCrossEntropyLoss


def alpha_vector_factory(
    cls, vocab_size=1024, alpha_start=0.5, alpha_end=1.5, *args, **kwargs
):
    """Factory function that creates a FocalLoss with vector alpha."""
    alpha_vector = torch.linspace(alpha_start, alpha_end, vocab_size)
    return cls(alpha=alpha_vector, *args, **kwargs)


LOSS_REGISTRY = {
    "cross_entropy": partial(CrossEntropyLoss, penalty_weight=0),
    "dedup": partial(CrossEntropyLoss, penalty_weight=0.1),
    "focal": partial(FocalLoss, alpha=1.0, gamma=2.0),
    "focal_alpha": partial(
        alpha_vector_factory, cls=FocalLoss, alpha_start=0.5, alpha_end=1.5, gamma=2.0
    ),
    "mile": MiLeLoss,
    "stablemax": StableMaxCrossEntropyLoss,
    "cut_cross_entropy": CutCrossEntropyLoss,
}


def get_loss_function(name, vocab_size):
    return LOSS_REGISTRY[name](vocab_size=vocab_size)
