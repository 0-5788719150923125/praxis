import torch
import torch.nn as nn

from .attention import PraxisAttention
from .mlp import PraxisMLP


class PraxisBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)
        self.attn = PraxisAttention(config)

        self.n_experts = config.n_experts
        self.k_best = config.k_best

        self.mlp_norm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)
        self.mlp = PraxisMLP(config)

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, attention_mask)
        x = residual + x
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x
        aux_loss = 0
        return x, aux_loss
