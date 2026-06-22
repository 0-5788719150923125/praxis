from typing import List, Optional

import torch.nn as nn

from praxis.losses.activation import ActivationRegularizer
from praxis.losses.contrastive_isotropy import ContrastiveIsotropyLoss

# Additive representation-shaping losses. Add an option here (not a new CLI
# flag) to make it selectable; the model runs every name in config.regularizers.
REGULARIZER_REGISTRY = {
    "contrastive_isotropy": ContrastiveIsotropyLoss,
    "activation": ActivationRegularizer,
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
