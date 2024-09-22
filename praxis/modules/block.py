import torch
import torch.nn as nn

from ..configuration_praxis import PraxisConfig
from .attention import PraxisAttention
from .mlp import PraxisMLP


class PraxisBlock(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.n_dim, eps=config.rms_norm_epsilon)
        self.attn = PraxisAttention(config)
        self.mlp_norm = nn.RMSNorm(config.n_dim, eps=config.rms_norm_epsilon)
        self.mlp = PraxisMLP(config)

    def forward(
        self,
        x,
        attention_mask=None,
        weights=None,
    ):
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, attention_mask)
        x = residual + x
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        if weights is not None:
            x *= weights
        x = residual + x
        return dict(hidden_states=x)
