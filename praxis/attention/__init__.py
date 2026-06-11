from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn

from praxis.attention.arc import ArcAttention
from praxis.attention.modular import ModularAttention
from praxis.attention.causal import CausalAttention
from praxis.attention.components import VanillaMHA
from praxis.attention.infini import InfiniAttention
from praxis.attention.pk_attention import ProductKeyAttention
from praxis.attention.syntaxes import SyntaxesAttention

# Registry of available attention mechanisms
ATTENTION_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "modular": ModularAttention,
    "vanilla": VanillaMHA,
    "pk": ProductKeyAttention,
    "syntaxes": SyntaxesAttention,
    "causal": CausalAttention,
    "infini": InfiniAttention,
    "arc": ArcAttention,
    # Arc + the dropoff ablation (next/dropoff.md): withhold the causal tip
    # via the "warp" value sink at the first layer of the last recurrent pass
    # (heuristic: depth - num_layers; e.g. 2 layers x 4 loops -> step 6), so
    # the model leans on delayed context for that beat and the remaining
    # layers recorrect.
    "arc_dropoff": partial(ArcAttention, dropoff="warp"),
}
