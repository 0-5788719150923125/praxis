from functools import partial
from typing import List, Optional

import torch.nn as nn

from praxis import registry
from praxis.losses.activation import ActivationRegularizer
from praxis.losses.contrastive_isotropy import ContrastiveIsotropyLoss
from praxis.losses.dissonance import Dissonance
from praxis.losses.harmonic_kl import HarmonicKLRegularizer
from praxis.losses.ouroboros_budget import OuroborosBudget
from praxis.registry import Entry

registry.declare(
    "regularizers",
    title="Regularizers",
    doc=(
        (
            "Additive representation-shaping losses layered beside the main criterion, "
            "each computed from the final hidden states during training; ``--strategy`` "
            "decides how the terms combine. The model runs every name listed, so a new "
            "option is an entry here rather than a new flag. Listing any names replaces "
            "the default rather than adding to it, and passing the flag with no values "
            "disables all. The ``*_probe`` variants observe only: they log their metric "
            "without contributing to the loss."
        )
    ),
    entries={
        "contrastive_isotropy": ContrastiveIsotropyLoss,
        "isotropy_probe": Entry(
            partial(ContrastiveIsotropyLoss, observe_only=True),
            (
                "contrastive_isotropy as a pure instrument: every metric, zero "
                "gradient. Pair it with a config that drops ``contrastive_isotropy`` "
                "when the question is what the isotropy term does to the "
                "representation, since removing the loss also removes repr_anisotropy, "
                "repr_nematic and repr_dimensions, the only evidence that could answer "
                "it."
            ),
        ),
        "activation": ActivationRegularizer,
        "harmonic_kl": Entry(
            HarmonicKLRegularizer,
            (
                "KL between the output readout and a slow EMA of itself: a trust "
                "region the RL path otherwise lacks, and a direct test of the paper's "
                "constitutive-basis claim."
            ),
        ),
        "ouroboros_budget": Entry(
            OuroborosBudget,
            (
                "Holds the Ouroboros activation's expected step count at the un-looped "
                "baseline's budget, via a Lagrange multiplier rather than a penalty "
                "weight. Pairs with ``activation: ouroboros``; a no-op without it."
            ),
        ),
        "dissonance": Entry(
            Dissonance,
            (
                "Holds the harmonic field's Plomp-Levelt roughness at or above that of "
                "the signal it multiplies, by a dual variable. Needs a harmonic field "
                "under the head; a no-op otherwise."
            ),
        ),
        "dissonance_probe": Entry(
            partial(Dissonance, observe_only=True),
            (
                "dissonance's spectrum readings with zero gradient: what the field "
                "does before deciding to push on it. Pair it with a config that drops "
                "``dissonance`` when the question is what the roughness term does."
            ),
        ),
    },
)

# Resolves config.regularizers when left unset (None). Empty list disables all.
DEFAULT_REGULARIZERS = ["contrastive_isotropy"]


def build_regularizers(names: Optional[List[str]], pad_id: int = 0) -> nn.ModuleList:
    """Build the model's regularizer list from registry names.

    ``None`` means "use the default"; an empty list disables them entirely.
    """
    if names is None:
        names = DEFAULT_REGULARIZERS
    mods = []
    for name in names:
        if name not in registry.namespace("regularizers"):
            raise KeyError(
                f"unknown regularizer '{name}'; "
                f"choices: {sorted(registry.namespace("regularizers"))}"
            )
        mods.append(registry.lookup("regularizers", name)(pad_id=pad_id))
    return nn.ModuleList(mods)
