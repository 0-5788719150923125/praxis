import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis import PraxisConfig
from praxis.modules.experts import PraxisExpert


class PraxisDecoder(nn.Module):
    """
    A module that wraps the entire decoder block (and all intermediate layers)
    in a single class.
    """

    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.sparse = config.sparse
        self.shuffle = config.shuffle
        self.checkpoint_indices = self._checkpoint_strategy(
            config.memory_profile, config.num_layers
        )
        self.experts = nn.ModuleList(
            [PraxisExpert(config) for _ in range(config.num_layers)]
        )

    def forward(self, inputs: Tensor, attention_mask: Tensor):
        if self.shuffle:
            random.shuffle(self.experts)

        hidden_states = inputs
        aux_losses = []

        for i, expert in enumerate(self.experts):
            use_router = True if self.sparse and i % 2 != 0 else False
            bit_tensor = torch.tensor([1 if use_router else 0], dtype=torch.bool)
            gradient_checkpointing = True if i in self.checkpoint_indices else False
            hidden_states, aux_loss = self._create_forward(
                expert,
                hidden_states,
                attention_mask,
                bit_tensor,
                gradient_checkpointing,
            )
            aux_losses.append(aux_loss)

        return hidden_states, sum(aux_losses)

    def _checkpoint_strategy(self, strategy="speed", num_layers=0):
        if strategy == "aggressive":
            # every layer
            return [i for i in range(num_layers)]
        elif strategy == "balanced":
            # every fourth layer
            return [i for i in range(num_layers) if i % 4 == 0]
        else:
            # no gradient checkpointing
            return []

    def _create_forward(
        self,
        expert: nn.Module,
        hidden_states: Tensor,
        attention_mask: Tensor,
        bit_tensor: Tensor,
        gradient_checkpointing=False,
    ):
        def custom_forward(*inputs):
            return expert(*inputs)

        if gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                bit_tensor,
                use_reentrant=False,
            )
        else:
            return custom_forward(hidden_states, attention_mask, bit_tensor)
