from typing import Optional, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.moe.server.layers.custom_experts import register_expert_class
from torch import Tensor

from praxis.modules.attention import ATTENTION_REGISTRY
from praxis.modules.experts import EXPERT_REGISTRY, get_expert_config
from praxis.modules.residual import HyperConnection

input_shape = lambda batch_size, hidden_dim: torch.empty((batch_size, hidden_dim))


@register_expert_class("hivemind_expert", input_shape)
class PraxisTransformer(nn.Module):
    """
    A standard transformer block, with adjustable feedforward "experts".
    """

    __version__ = "0.1.0"

    def __init__(self, config: "AutoConfig", *args, **kwargs):
        super().__init__()
        self.attn_res = HyperConnection(config.hidden_size) if config.hyper else False
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.epsilon)
        self.attn = ATTENTION_REGISTRY[config.attention_type](config)
        self.ffn_res = HyperConnection(config.hidden_size) if config.hyper else False
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.epsilon)
        self.ffn = EXPERT_REGISTRY[get_expert_config(config.expert)["type"]](config)

    def forward(
        self,
        inputs: Tensor,
        current_state: Tensor,
        attention_mask: Tensor,
        router_weights: Optional[Tensor] = None,
        *args,
        **kwargs,
    ):
        residual = self.attn_res(inputs=inputs) if self.attn_res else inputs
        normalized = self.attn_norm(inputs)
        outputs = self.attn(normalized, attention_mask)
        outputs = (
            self.attn_res(outputs=outputs) if self.attn_res else outputs + residual
        )
        residual = self.ffn_res(inputs=outputs) if self.ffn_res else outputs
        normalized = self.ffn_norm(outputs)
        outputs = self.ffn(normalized)
        if torch.is_tensor(router_weights):
            # this is a super hack because hivemind
            if not self._is_zero_tensor(router_weights):
                outputs = outputs * router_weights
        outputs = self.ffn_res(outputs=outputs) if self.ffn_res else outputs + residual
        return outputs, None, 0

    def _is_zero_tensor(self, tensor: torch.Tensor, tolerance: float = 1e-10) -> bool:
        """Check if a tensor is filled with zeros (within numerical tolerance)"""
        try:
            if tensor.dtype == torch.int64:
                return torch.all(tensor == 0).item()
            return torch.abs(tensor).max().item() < tolerance
        except Exception as e:
            return True
