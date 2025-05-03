from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn

from praxis.attention.base import ModularAttention, ProductKeyAttention, VanillaMHA

# Registry of available attention mechanisms
ATTENTION_REGISTRY: Dict[str, Type[nn.Module]] = {
    "standard": ModularAttention,
    "vanilla": VanillaMHA,
    "pk": ProductKeyAttention,
}
