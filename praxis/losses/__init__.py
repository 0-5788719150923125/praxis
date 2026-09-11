from functools import partial

import torch
import torch.nn as nn

from praxis import registry
from praxis.losses.batchmode import ModeBaselineCrossEntropyLoss, ModeCrossEntropyLoss
from praxis.losses.contrastive_token import ContrastiveTokenLoss
from praxis.losses.cross_entropy import CrossEntropyLoss
from praxis.losses.focal import FocalLoss
from praxis.losses.halo import HALOLoss
from praxis.losses.layer_wise import compute_layer_wise_loss
from praxis.losses.mile import MiLeLoss
from praxis.losses.objectives import Objectives
from praxis.losses.regression import MeanSquaredErrorLoss
from praxis.losses.regularizers import build_regularizers
from praxis.losses.stablemax import StableMaxCrossEntropyLoss
from praxis.registry import Entry


def alpha_vector_factory(
    cls, vocab_size=1024, alpha_start=0.5, alpha_end=1.5, *args, **kwargs
):
    """Factory function that creates a FocalLoss with vector alpha."""
    alpha_vector = torch.linspace(alpha_start, alpha_end, vocab_size)
    return cls(alpha=alpha_vector, *args, **kwargs)


registry.declare(
    "losses",
    title="Loss functions",
    doc=(
        (
            "Per-token training criteria. Most accept optional ``loss_weights`` for "
            "task-weighted training."
        )
    ),
    entries={
        "cross_entropy": CrossEntropyLoss,
        "dedup": Entry(
            partial(CrossEntropyLoss, penalty_weight=0.1),
            (
                "cross_entropy with its anti-duplication penalty on "
                "(``penalty_weight=0.1``): a token the model is about to predict equal "
                "to one already in the prompt costs 1.1x."
            ),
        ),
        "focal": Entry(
            partial(FocalLoss, alpha=1.0, gamma=2.0),
            (
                "Focal loss with a scalar alpha of 1.0 and gamma 2.0, which "
                "down-weights well-classified tokens so the gradient concentrates on "
                "hard ones."
            ),
        ),
        "focal_alpha": Entry(
            partial(
                alpha_vector_factory,
                cls=FocalLoss,
                alpha_start=0.5,
                alpha_end=1.5,
                gamma=2.0,
            ),
            (
                "Focal loss (gamma 2.0) with a per-token-id alpha rising linearly from "
                "0.5 to 1.5 across the vocabulary."
            ),
        ),
        "mode_cross_entropy": Entry(
            ModeCrossEntropyLoss,
            (
                "Mode-level criterion over the batch's per-token loss distribution: "
                "floored mode-as-target. Each token's CE is weighted by the loss "
                "density at its own loss value, so the consensus band leads and the "
                "tail keeps a floored share (``FLOOR``)."
            ),
        ),
        "mode_baseline_cross_entropy": Entry(
            ModeBaselineCrossEntropyLoss,
            (
                "The fallback dual of mode_cross_entropy: each token is weighted by "
                "its deviation above the modal loss, so the consensus band gets "
                "floor-level gradient only and tokens above it get pressure in "
                "proportion to their deviation."
            ),
        ),
        "mile": Entry(
            MiLeLoss,
            (
                "MiLe (Mitigating the bias of learning difficulties with tokens): "
                "reweights each token's CE by the detached entropy of its predicted "
                "distribution. Costs more VRAM than cross_entropy."
            ),
        ),
        "stablemax": StableMaxCrossEntropyLoss,
        "contrastive_token": ContrastiveTokenLoss,
        "halo": Entry(
            HALOLoss,
            (
                "HALO (Hyperspherical Active Learning Objective) adapted for language "
                "modeling. A loss-owning encoder may reroute it: CALM treats ``halo`` "
                "as its geometric mode, where reconstruction stays CE and HALO steers "
                "the energy generator through the frozen codec, keeping HALO off the "
                "centroid-shaping reconstruction path."
            ),
        ),
    },
)


def get_loss_function(name, vocab_size):
    return registry.lookup("losses", name)(vocab_size=vocab_size)


def build_objectives(config, encoder=None) -> Objectives:
    """The model's loss terms: the main criterion plus its regularizers.

    Producers that own a term (MTP, a parallel classifier's arms) register theirs
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
