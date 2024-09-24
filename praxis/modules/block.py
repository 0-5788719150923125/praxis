import torch
import torch.nn as nn

from ..configuration_praxis import PraxisConfig
from .attention import PraxisAttention
from .mlp import PraxisMLP


class PraxisBlock(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.attn = PraxisAttention(config)
        self.norm = nn.RMSNorm(config.n_dim, eps=config.epsilon)
        self.mlp = PraxisMLP(config)

    def forward(
        self,
        x,
        attention_mask=None,
        router_weights=False,
    ):
        residual = x
        y = self.attn(x, attention_mask) + residual
        residual = y
        y = self.mlp(self.norm(y))
        if router_weights is not None:
            y *= router_weights
        y += residual
        return dict(hidden_states=x)
