from typing import Optional, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.moe.server.layers.custom_experts import register_expert_class
from torch import Tensor

from praxis.modules.attention import ATTENTION_REGISTRY
from praxis.modules.experts import EXPERT_REGISTRY, get_expert_config
from praxis.modules.residual import HyperConnection, ResidualConnection

input_shape = lambda batch_size, hidden_dim: torch.empty((batch_size, hidden_dim))


@register_expert_class("hivemind_expert", input_shape)
class PraxisTransformer(nn.Module):
    """
    A standard transformer block, with adjustable feedforward "experts".
    """

    def __init__(self, config: "AutoConfig", *args, **kwargs):
        super().__init__()
        self.attn_res = (
            HyperConnection(config.hidden_size)
            if config.hyper
            else ResidualConnection()
        )
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.epsilon)
        self.attn = ATTENTION_REGISTRY[config.attention_type](config)
        self.ffn_res = (
            HyperConnection(config.hidden_size)
            if config.hyper
            else ResidualConnection()
        )
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.epsilon)
        self.ffn = EXPERT_REGISTRY[get_expert_config(config.expert)["type"]](config)

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: torch.Tensor = None,
        current_state: torch.Tensor = None,
        current_depth: int = 0,
        sequence_ids: torch.Tensor = None,
        router_weights: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        aux_loss = 0
        # =========== Attention Block =============
        residual, beta = self.attn_res.connect_width(inputs)
        attn_input = self.attn_norm(self.attn_res.format_state(residual))
        attn_output, layer_kv, aux_loss = self.attn(
            attn_input, past_key_values, attention_mask, sequence_ids
        )
        attn_merged = self.attn_res.connect_depth(residual, attn_output, beta)

        # =========== FeedForward Block ===========
        residual, beta_ffn = self.ffn_res.connect_width(
            self.ffn_res.format_state(attn_merged)
        )
        ffn_input = self.ffn_norm(self.ffn_res.format_state(residual))
        ffn_output = self.ffn(ffn_input, current_depth)

        if torch.is_tensor(router_weights):
            # this is a super hack because hivemind
            if not self._is_zero_tensor(router_weights):
                ffn_output = ffn_output * router_weights

        # Merge expansions
        final_output = self.ffn_res.connect_depth(residual, ffn_output, beta_ffn)
        return self.ffn_res.format_state(final_output), layer_kv, None, aux_loss

    def _is_zero_tensor(self, tensor: torch.Tensor, tolerance: float = 1e-10) -> bool:
        if tensor.dtype == torch.int64:
            return bool(torch.all(tensor == 0))
        return bool(torch.abs(tensor).max().item() < tolerance)
