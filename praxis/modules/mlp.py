from typing import OrderedDict

import torch
import torch.nn as nn
from hivemind.moe.server.layers.custom_experts import register_expert_class
from transformers.activations import ACT2FN

input_shape = lambda batch_size, hid_dim: torch.empty((batch_size, hid_dim))


@register_expert_class("praxis", input_shape)
class PraxisMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.network = nn.Sequential(
            OrderedDict(
                [
                    ("in_proj", nn.Linear(config.n_embd, 4 * config.n_embd)),
                    ("act", ACT2FN[config.activation_function]),
                    ("out_proj", nn.Linear(4 * config.n_embd, config.n_embd)),
                ]
            )
        )

    def forward(self, x):
        return self.network(x)
