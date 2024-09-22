import torch
import torch.nn as nn

from ..configuration_praxis import PraxisConfig
from .block import PraxisBlock
from .router import PraxisMixtureOfDepths


class PraxisStack(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        layers = []
        for i in range(config.n_layer):
            odd = i % 2 != 0
            if odd:
                layers.append(PraxisMixtureOfDepths(PraxisBlock(config), config))
            else:
                layers.append(PraxisBlock(config))
        self.blocks = nn.ModuleList(layers)

    def forward(self, x, attention_mask):
        y = x
        aux_losses = []
        for block in self.blocks:
            z = block(y, attention_mask)
            y = z["hidden_states"]
            if self.training and "aux_loss" in z:
                aux_losses.append(z["aux_loss"])
        aux_loss = sum(aux_losses)
        return dict(hidden_states=y, aux_loss=aux_loss)
