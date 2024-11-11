from typing import Optional, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.moe.server.layers.custom_experts import register_expert_class
from torch import Tensor
from transformers import AutoConfig

from praxis.modules.attention import PraxisAttention
from praxis.modules.dense import PraxisGLU, PraxisMLP
from praxis.modules.peer import PraxisPEER
from praxis.modules.smear import PraxisSMEAR

EXPERT_REGISTRY = {
    "mlp": PraxisMLP,
    "glu": PraxisGLU,
    "peer": PraxisPEER,
    "smear": PraxisSMEAR,
}

EXPERT_CONFIGS = {
    "peer": {
        "num_experts": 32**2,
        "num_heads": 4,
        "k": 8,
        "key_dims": 90,
        "offset_heads": False,
    },
    "smear": {"num_experts": 3},
    "glu": {},
    "mlp": {},
}

input_shape = lambda batch_size, hidden_dim: torch.empty((batch_size, hidden_dim))


@register_expert_class("hivemind_expert", input_shape)
class PraxisBlock(nn.Module):
    """
    A standard transformer block, with adjustable feedforward "experts".

    When using Hivemind, there are certain limitations:

    1. All inputs to the `forward()` method must be Tensors.
    2. No inputs are allowed to be empty (None) types.
    3. All inputs must be of a constant shape.
    3. All inputs/outputs must be a part of the computation graph (i.e. returning detached aux_loss tensors is invalid).

    Essentially, Hivemind experts have static inputs/outputs - in contrast to the "dynamic" nature of Pytorch.
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.num_dims, eps=config.epsilon)
        self.attn = PraxisAttention(config)
        self.mlp_norm = nn.RMSNorm(config.num_dims, eps=config.epsilon)
        self.mlp = EXPERT_REGISTRY[config.expert["type"]](config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Tensor,
        router_weights: Optional[Tensor] = None,
        token_indices: Optional[Tensor] = None,
    ):
        # this is a super hack because hivemind
        if torch.is_tensor(router_weights) and self._is_zero_tensor(router_weights):
            router_weights = None
        if torch.is_tensor(token_indices) and self._is_zero_tensor(token_indices):
            token_indices = None
        residual = inputs
        normalized = self.attn_norm(inputs)
        outputs = self.attn(normalized, attention_mask, token_indices)
        outputs = self.dropout(outputs)
        outputs = outputs + residual
        residual = outputs
        normalized = self.mlp_norm(outputs)
        outputs = self.mlp(normalized)
        outputs = self.dropout(outputs)
        if torch.is_tensor(router_weights):
            outputs = outputs + router_weights
        outputs = outputs + residual
        return outputs

    def _is_zero_tensor(self, tensor: torch.Tensor, tolerance: float = 1e-10) -> bool:
        """Check if a tensor is filled with zeros (within numerical tolerance)"""
        try:
            if tensor.dtype == torch.int64:
                return torch.all(tensor == 0).item()
            return torch.abs(tensor).max().item() < tolerance
        except Exception as e:
            return True