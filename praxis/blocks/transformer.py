from typing import Optional, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.moe.server.layers.custom_experts import register_expert_class
from torch import Tensor
from transformers import AutoConfig

from praxis.modules.attention import ATTENTION_REGISTRY
from praxis.modules.experts import EXPERT_REGISTRY, get_expert_config

input_shape = lambda batch_size, hidden_dim: torch.empty((batch_size, hidden_dim))


@register_expert_class("hivemind_expert", input_shape)
class PraxisTransformer(nn.Module):
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

    def __init__(self, config: AutoConfig, *args, **kwargs):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.epsilon)
        self.attn = ATTENTION_REGISTRY[config.attention_type](config)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.epsilon)
        self.mlp = EXPERT_REGISTRY[get_expert_config(config.expert)["type"]](config)
        # Force exploration of attention subnetworks
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Tensor,
        router_weights: Optional[Tensor] = None,
        *args,
        **kwargs,
    ):
        residual = inputs
        normalized = self.attn_norm(inputs)
        outputs = self.attn(normalized, attention_mask)
        outputs = outputs + residual
        outputs = self.dropout(outputs)
        residual = outputs
        normalized = self.mlp_norm(outputs)
        outputs = self.mlp(normalized)
        if torch.is_tensor(router_weights):
            # this is a super hack because hivemind
            if not self._is_zero_tensor(router_weights):
                outputs = outputs * router_weights
        outputs = outputs + residual
        return self.dropout(outputs)

    def _is_zero_tensor(self, tensor: torch.Tensor, tolerance: float = 1e-10) -> bool:
        """Check if a tensor is filled with zeros (within numerical tolerance)"""
        try:
            if tensor.dtype == torch.int64:
                return torch.all(tensor == 0).item()
            return torch.abs(tensor).max().item() < tolerance
        except Exception as e:
            return True
