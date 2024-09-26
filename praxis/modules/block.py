import torch
import torch.nn as nn

from ..configuration_praxis import PraxisConfig
from .attention import PraxisAttention
from .dense import PraxisGLU


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
        outputs = self.drop(self.mlp(self.norm(outputs)))
        if router_weights is not None:
            outputs *= router_weights
        outputs += residual
        return dict(hidden_states=outputs)
