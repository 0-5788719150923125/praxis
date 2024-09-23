import torch
import torch.nn as nn

from ..configuration_praxis import PraxisConfig
from .attention import PraxisAttention
from .mlp import PraxisMLP


class PraxisBlock(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.n_dim, eps=config.epsilon)
        self.attn = PraxisAttention(config)
        self.mlp_norm = nn.RMSNorm(config.n_dim, eps=config.epsilon)
        self.mlp = PraxisMLP(config)

    def forward(
        self,
        x,
        attention_mask=None,
        router_weights=False,
    ):
        residual = x
        norm = self.attn_norm(x)
        y = self.attn(norm, attention_mask) + residual
        residual = y
        norm = self.mlp_norm(y)
        y = self.mlp(norm)
        if router_weights is not None:
            y *= router_weights
        y += residual
        return dict(hidden_states=x)
