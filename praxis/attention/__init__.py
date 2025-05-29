from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn

from praxis.attention.base import ModularAttention
from praxis.attention.components import VanillaMHA
from praxis.attention.pk_attention import ProductKeyAttention
from praxis.attention.syntaxis import SyntaxisAttention

# Registry of available attention mechanisms
ATTENTION_REGISTRY: Dict[str, Type[nn.Module]] = {
    "standard": ModularAttention,
    "vanilla": VanillaMHA,
    "pk": ProductKeyAttention,
    "syntaxis": SyntaxisAttention,
}
