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
        router_weights=None,
    ):
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, attention_mask)
        x = residual + x
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        if router_weights is not None:
            x *= router_weights
        x = residual + x
        return dict(hidden_states=x)
