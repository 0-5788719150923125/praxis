import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis import PraxisConfig
from praxis.modules.experts import PraxisBlock
from praxis.modules.router import PraxisMixtureOfDepths


class PraxisDecoder(nn.Module):
    """
    A module that wraps the entire decoder block (and all intermediate layers)
    in a single class.
    """

    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.gradient_checkpointing = True
        self.shuffle = config.shuffle
        self.experts = nn.ModuleList(PraxisBlock(config) for _ in range(config.n_layer))
        self.routers = None
        if config.sparse:
            self.routers = nn.ModuleList(
                PraxisMixtureOfDepths(config) for _ in range(config.n_layer // 2)
            )

    def forward(self, inputs: Tensor, attention_mask: Tensor):
        if self.shuffle:
            random.shuffle(self.experts)

        hidden_states = inputs
        aux_losses = []

        for i, expert in enumerate(self.experts):
            router = (
                self.routers[(i - 1) // 2] if self.routers and i % 2 != 0 else None
            )  # select odd layers
            hidden_states, aux_loss = self._custom_forward(
                expert, hidden_states, attention_mask, router
            )
            aux_losses.append(aux_loss)

        return hidden_states, sum(aux_losses)

    def _custom_forward(self, expert, hidden_states, attention_mask, router=None):
        def custom_forward(*inputs):
            return router(expert, *inputs) if router else expert(*inputs)

        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                custom_forward, hidden_states, attention_mask, use_reentrant=False
            )
        else:
            return custom_forward(hidden_states, attention_mask)
