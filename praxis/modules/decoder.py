import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from ..configuration_praxis import PraxisConfig
from .controller import PraxisController
from .experts import PraxisBlock
from .router import PraxisMixtureOfDepths


class PraxisDecoder(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.ctrl = PraxisController(config)
        self.experts = nn.ModuleList()
        self.routers = nn.ModuleList() if config.sparse else None
        for i in range(config.n_layer):
            self.experts.append(PraxisBlock(config))
            use_router = i % 2 != 0  # if layer is odd
            if self.routers is not None and use_router:
                self.routers.append(PraxisMixtureOfDepths(config))

    def forward(self, inputs, attention_mask):
        hidden_states = inputs  # Shape: (batch_size, seq_len, n_dim)
        aux_losses = []

        sequence, expert_biases, aux_loss = self.ctrl(hidden_states)
        aux_losses.append(aux_loss)

        for i, choice in enumerate(sequence):
            expert = self.experts[choice]
            residual = hidden_states
            use_router = i % 2 != 0  # if layer is odd
            if self.routers is not None and use_router:
                outputs = self.routers[(i - 1) // 2](
                    hidden_states, expert, attention_mask
                )
            else:
                outputs = expert(hidden_states, attention_mask)
            expert_bias = expert_biases[i]
            hidden_states = outputs["hidden_states"] + expert_bias + residual
            if "aux_loss" in outputs:
                aux_losses.append(outputs["aux_loss"])
        return dict(hidden_states=hidden_states, aux_loss=sum(aux_losses))
