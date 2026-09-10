from functools import partial

import torch
import torch.nn as nn

from praxis.losses.batchmode import ModeBaselineCrossEntropyLoss, ModeCrossEntropyLoss
from praxis.losses.contrastive_token import ContrastiveTokenLoss
from praxis.losses.cross_entropy import CrossEntropyLoss
from praxis.losses.focal import FocalLoss
from praxis.losses.halo import HALOLoss
from praxis.losses.layer_wise import compute_layer_wise_loss
from praxis.losses.mile import MiLeLoss
from praxis.losses.objectives import Objectives
from praxis.losses.regression import MeanSquaredErrorLoss
from praxis.losses.regularizers import REGULARIZER_REGISTRY, build_regularizers
from praxis.losses.stablemax import StableMaxCrossEntropyLoss


def alpha_vector_factory(
    cls, vocab_size=1024, alpha_start=0.5, alpha_end=1.5, *args, **kwargs
):
    """Factory function that creates a FocalLoss with vector alpha."""
    alpha_vector = torch.linspace(alpha_start, alpha_end, vocab_size)
    return cls(alpha=alpha_vector, *args, **kwargs)


LOSS_REGISTRY = {
    "cross_entropy": CrossEntropyLoss,
    "dedup": partial(CrossEntropyLoss, penalty_weight=0.1),
    "focal": partial(FocalLoss, alpha=1.0, gamma=2.0),
    "focal_alpha": partial(
        alpha_vector_factory, cls=FocalLoss, alpha_start=0.5, alpha_end=1.5, gamma=2.0
    ),
    # Mode-level criteria over the batch loss distribution: "mode_cross_entropy"
    # is floored mode-as-target (consensus band leads, tail keeps FLOOR * lr);
    # "mode_baseline_cross_entropy" is the deviation-above-mode fallback dual.
    "mode_cross_entropy": ModeCrossEntropyLoss,
    "mode_baseline_cross_entropy": ModeBaselineCrossEntropyLoss,
    "mile": MiLeLoss,
    "stablemax": StableMaxCrossEntropyLoss,
    "contrastive_token": ContrastiveTokenLoss,
    # Loss-owning encoders may reroute this: CALM treats "halo" as its
    # geometric mode (recon stays CE; HALO steers the energy head through the
    # frozen codec), keeping HALO off the centroid-shaping recon path.
    "halo": HALOLoss,
}


def get_loss_function(name, vocab_size):
    return LOSS_REGISTRY[name](vocab_size=vocab_size)


def build_objectives(config, encoder=None) -> Objectives:
    """The model's loss terms: the main criterion plus its regularizers.

    Producers that own a term (MTP, a parallel head's arms) register theirs
    on the returned container as they are built. An encoder that owns the
    loss (CALM) leaves ``main`` unregistered - there is no criterion for the
    model to call.
    """
    objectives = Objectives()
    if not (encoder and getattr(encoder, "handles_loss", False)):
        objectives.register(
            "main", get_loss_function(config.loss_func, config.vocab_size)
        )
    for regularizer in build_regularizers(
        getattr(config, "regularizers", None),
        # `or 0`: a pure-byte tokenizer defines no pad token, and this is only
        # a gather-safe index - padding itself is identified by ignore_index
        # in the labels.
        pad_id=config.pad_token_id or 0,
    ):
        objectives.register(regularizer.name, regularizer)
    return objectives
