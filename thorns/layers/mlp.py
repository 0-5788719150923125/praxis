import torch.nn as nn
from transformers.activations import ACT2FN
from typing import OrderedDict
from hivemind.moe.server.layers.custom_experts import register_expert_class


ffn_sample_input = lambda batch_size, hid_dim: torch.empty((batch_size, hid_dim))


@register_expert_class("thorn", ffn_sample_input)
class ThornsMLP(nn.Module):
    def __init__(self, hid_dim, config):
        super().__init__()
        self.network = nn.Sequential(
            OrderedDict(
                [
                    ("in_proj", nn.Linear(hid_dim, 4 * hid_dim)),
                    ("act", ACT2FN[config.activation_function]),
                    ("out_proj", nn.Linear(4 * hid_dim, hid_dim)),
                ]
            )
        )

    def forward(self, x):
        return self.network(x)
