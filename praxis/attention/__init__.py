from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn

from praxis.attention.base import ModularAttention
from praxis.attention.components import VanillaMHA
from praxis.attention.flex_attention import FlexAttention
from praxis.attention.pk_attention import ProductKeyAttention
from praxis.attention.syntaxes import SyntaxesAttention

# Registry of available attention mechanisms
ATTENTION_REGISTRY: Dict[str, Type[nn.Module]] = {
    "standard": ModularAttention,
    "vanilla": VanillaMHA,
    "pk": ProductKeyAttention,
    "syntaxes": SyntaxesAttention,
    "flex_attention": FlexAttention,
}
