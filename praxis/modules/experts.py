from typing import OrderedDict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.moe.server.layers.custom_experts import register_expert_class
from torch import Tensor
from transformers.activations import ACT2FN

from praxis import PraxisConfig
from praxis.modules.attention import PraxisAttention
from praxis.modules.peer import PraxisPEER


# input_shape = lambda batch_size, hid_dim: torch.empty((batch_size, hid_dim))
input_shape = lambda batch_size, hid_dim: (
    torch.empty((batch_size, hid_dim)),
    torch.empty((batch_size)),
    torch.empty((batch_size)),
    torch.empty((batch_size)),
)


@register_expert_class("praxis_block", input_shape)
class PraxisBlock(nn.Module):
    """
    A standard transformer block, with adjustable feedforward "experts".
    """

    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.num_dims, eps=config.epsilon)
        self.attn = PraxisAttention(config)
        self.mlp_norm = nn.RMSNorm(config.num_dims, eps=config.epsilon)
        self.mlp = EXPERT_DICT[config.expert_type](config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        router_weights: Optional[Tensor] = None,
        token_indices: Optional[Tensor] = None,
    ):
        residual = inputs
        normalized = self.attn_norm(inputs)
        outputs = self.attn(normalized, attention_mask, token_indices)
        outputs = self.dropout(outputs)
        outputs = outputs + residual
        residual = outputs
        normalized = self.mlp_norm(outputs)
        outputs = self.mlp(normalized)
        outputs = self.dropout(outputs)
        if torch.is_tensor(router_weights):
            outputs *= router_weights
        aux_loss = 0
        outputs = outputs + residual
        return outputs, aux_loss


class PraxisMLP(nn.Sequential):
    """
    A standard Multi-Layer Perceptron.
    """

    def __init__(self, config: PraxisConfig):
        super().__init__(
            OrderedDict(
                [
                    ("up", nn.Linear(config.num_dims, 4 * config.num_dims)),
                    ("act", ACT2FN[config.activation]),
                    ("dropout", nn.Dropout(config.dropout)),
                    ("down", nn.Linear(4 * config.num_dims, config.num_dims)),
                ]
            )
        )


class PraxisGLU(nn.Module):
    """
    A standard MLP, augmented with a Gated Linear Units.
    """

    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.up = nn.Linear(config.num_dims, 8 * config.num_dims)
        self.act = ACT2FN[config.activation]
        self.dropout = nn.Dropout(config.dropout)
        self.down = nn.Linear(4 * config.num_dims, config.num_dims)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(self.dropout(a * self.act(b)))


EXPERT_DICT = {"mlp": PraxisMLP, "glu": PraxisGLU, "peer": PraxisPEER}
