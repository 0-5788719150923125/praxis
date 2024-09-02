import torch.nn as nn
from ..layers.attention import ThornsAttention
from ..layers.mlp import ThornsMLP


class ThornsBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.RMSNorm(config.n_embd, eps=config.rms_norm_epsilon)
        self.attn = ThornsAttention(config)
        self.mlp = ThornsMLP(hid_dim=config.n_embd, config=config)

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.norm(x)
        x = self.attn(x, attention_mask)
        x = residual + x
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        x = residual + x
        return x
