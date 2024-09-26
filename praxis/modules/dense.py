from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.moe.server.layers.custom_experts import register_expert_class
from transformers.activations import ACT2FN

from ..configuration_praxis import PraxisConfig

input_shape = lambda batch_size, hid_dim: torch.empty((batch_size, hid_dim))


@register_expert_class("praxis_mlp", input_shape)
class PraxisMLP(nn.Sequential):
    def __init__(self, config: PraxisConfig):
        super().__init__(
            OrderedDict(
                [
                    ("in", nn.Linear(config.n_dim, 4 * config.n_dim)),
                    ("act", ACT2FN[config.activation]),
                    ("out", nn.Linear(4 * config.n_dim, config.n_dim)),
                ]
            )
        )


@register_expert_class("praxis_glu", input_shape)
class PraxisGLU(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.inputs = nn.Linear(config.n_dim, 8 * config.n_dim)
        self.activation = ACT2FN[config.activation]
        self.output = nn.Linear(4 * config.n_dim, config.n_dim)

    def forward(self, x):
        x = self.inputs(x)
        x = self.activation(x)
        x, gate = x.chunk(2, dim=-1)
        x = F.silu(gate) * x
        return self.output(x)
