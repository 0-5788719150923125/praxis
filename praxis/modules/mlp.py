from typing import OrderedDict

import torch
import torch.nn as nn
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
