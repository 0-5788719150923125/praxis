import torch
import torch.nn as nn

from ..configuration_praxis import PraxisConfig
from .block import PraxisBlock
from .router import PraxisMixtureOfDepths


class PraxisDecoder(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(config.n_layer):
            use_router = i % 2 != 0  # if layer is odd
            if use_router:
                self.blocks.append(PraxisMixtureOfDepths(PraxisBlock(config), config))
            else:
                self.blocks.append(PraxisBlock(config))

    def forward(self, x, attention_mask):
        hidden_states = x
        aux_losses = []
        for block in self.blocks:
            new_states = block(hidden_states, attention_mask)
            hidden_states = new_states["hidden_states"]
            if "aux_loss" in new_states:
                aux_losses.append(new_states["aux_loss"])
        return dict(hidden_states=hidden_states, aux_loss=sum(aux_losses))
