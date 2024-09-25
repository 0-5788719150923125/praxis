import torch
import torch.nn as nn

from ..configuration_praxis import PraxisConfig
from .block import PraxisBlock
from .router import PraxisMixtureOfDepths


class PraxisDecoder(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.sparse = config.sparse
        self.switching = nn.ModuleList()
        self.experts = nn.ModuleList()
        for i in range(config.n_layer):
            self.experts.append(PraxisBlock(config))
            use_router = i % 2 != 0  # if layer is odd
            if config.sparse and use_router:
                self.switching.append(PraxisMixtureOfDepths(config))

    def forward(self, inputs, attention_mask):
        hidden_states = inputs
        aux_losses = []
        for i, expert in enumerate(self.experts):
            outputs = expert(hidden_states, attention_mask)
            use_router = i % 2 != 0  # if layer is odd
            if self.sparse and use_router:
                outputs = self.switching[i % 2](hidden_states, expert, attention_mask)
            hidden_states = outputs["hidden_states"]
            if "aux_loss" in outputs:
                aux_losses.append(outputs["aux_loss"])
        return dict(hidden_states=hidden_states, aux_loss=sum(aux_losses))
