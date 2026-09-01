from typing import List, Optional

import torch.nn as nn

from praxis.losses.activation import ActivationRegularizer
from praxis.losses.contrastive_isotropy import ContrastiveIsotropyLoss
from praxis.losses.harmonic_kl import HarmonicKLRegularizer
from praxis.losses.ouroboros_budget import OuroborosBudget
from functools import partial

# Additive representation-shaping losses. Add an option here (not a new CLI
# flag) to make it selectable; the model runs every name in config.regularizers.
REGULARIZER_REGISTRY = {
    "contrastive_isotropy": ContrastiveIsotropyLoss,
    # The same class as a pure INSTRUMENT: every metric, zero gradient. Pair it
    # with a config that drops `contrastive_isotropy` when the question is what
    # the isotropy term was doing to the representation - otherwise removing the
    # loss also removes repr_anisotropy / repr_nematic / repr_dimensions, which
    # are the only evidence that could answer it.
    "isotropy_probe": partial(ContrastiveIsotropyLoss, observe_only=True),
    "activation": ActivationRegularizer,
    # KL between the output readout and a slow EMA of itself - the trust region
    # the RL path never had, and a direct test of the paper's constitutive-basis
    # claim. Opt-in; see praxis/losses/harmonic_kl.py and next/rl.md.
    "harmonic_kl": HarmonicKLRegularizer,
    # Holds the Ouroboros activation's expected step count at the un-looped
    # baseline's budget, via a Lagrange multiplier rather than a penalty weight.
    # Pairs with `activation: ouroboros`; a no-op without it. See
    # praxis/activations/ouroboros.py and next/ouroboros.md.
    "ouroboros_budget": OuroborosBudget,
}

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
        if name not in REGULARIZER_REGISTRY:
            raise KeyError(
                f"unknown regularizer '{name}'; "
                f"choices: {sorted(REGULARIZER_REGISTRY)}"
            )
        mods.append(REGULARIZER_REGISTRY[name](pad_id=pad_id))
    return nn.ModuleList(mods)
