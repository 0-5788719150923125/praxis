import torch
import torch.nn as nn

from .block import PraxisBlock
from .router import PraxisMixtureOfDepths


class PraxisStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        for i in range(config.n_layer):
            even = i % 2 == 0
            use_router = False if even else True
            if use_router:
                layers.append(
                    PraxisMixtureOfDepths(
                        PraxisBlock(config), config.n_dim, config.capacity
                    )
                )
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
