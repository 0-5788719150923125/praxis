from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.moe.server.layers.custom_experts import register_expert_class
from transformers.activations import ACT2FN

from ..configuration_praxis import PraxisConfig
from .attention import PraxisAttention

input_shape = lambda batch_size, hid_dim: torch.empty((batch_size, hid_dim))


@register_expert_class("praxis_expert", input_shape)
class PraxisBlock(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.attn = PraxisAttention(config)
        self.norm = nn.RMSNorm(config.n_dim, eps=config.epsilon)
        self.mlp = PraxisGLU(config)
        self.drop = nn.Dropout(config.dropout)

    def forward(
        self,
        inputs,
        attention_mask=None,
        router_weights=None,
    ):
        residual = inputs
        outputs = self.attn(inputs, attention_mask) + residual
        residual = outputs
        outputs = self.mlp(self.norm(outputs))
        outputs = self.drop(outputs)
        if router_weights is not None:
            outputs *= router_weights
        return dict(hidden_states=outputs + residual)


@register_expert_class("praxis_mlp", input_shape)
class PraxisMLP(nn.Sequential):
    def __init__(self, config: PraxisConfig):
        super().__init__(
            OrderedDict(
                [
                    ("up", nn.Linear(config.n_dim, 4 * config.n_dim)),
                    ("act", ACT2FN[config.activation]),
                    ("down", nn.Linear(4 * config.n_dim, config.n_dim)),
                ]
            )
        )


@register_expert_class("praxis_glu", input_shape)
class PraxisGLU(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.up = nn.Linear(config.n_dim, 8 * config.n_dim)
        self.act = ACT2FN[config.activation]
        self.down = nn.Linear(4 * config.n_dim, config.n_dim)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(a * self.act(b))