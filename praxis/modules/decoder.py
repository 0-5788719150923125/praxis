import random

import torch.nn as nn
import torch.nn.functional as F

from ..configuration_praxis import PraxisConfig
from .experts import PraxisBlock
from .router import PraxisMixtureOfDepths


class PraxisDecoder(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.experts = nn.ModuleList()
        self.routers = nn.ModuleList() if config.sparse else None
        for i in range(config.n_layer):
            self.experts.append(PraxisBlock(config))
            use_router = i % 2 != 0  # if layer is odd
            if self.routers is not None and use_router:
                self.routers.append(PraxisMixtureOfDepths(config))

    def forward(self, inputs, attention_mask):
        aux_losses = []
        hidden_states = inputs  # Shape: (batch_size, seq_len, n_dim)

        random.shuffle(self.experts)

        for i, expert in enumerate(self.experts):

            use_router = i % 2 != 0  # if layer is odd
            if use_router and self.routers is not None:
                router = self.routers[(i - 1) // 2]
                hidden_states, aux_loss = router(hidden_states, expert, attention_mask)
            else:
                hidden_states, aux_loss = expert(hidden_states, attention_mask)

            aux_losses.append(aux_loss)

        return hidden_states, sum(aux_losses)
