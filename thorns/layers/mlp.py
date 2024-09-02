import torch.nn as nn
from transformers.activations import ACT2FN
from typing import OrderedDict


class ThornsMLP(nn.Module):
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
