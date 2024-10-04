import torch.nn as nn
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

        for i, idx in enumerate(sequence):

            expert = self.experts[idx]
            expert_bias = expert_biases[i]
            residual = hidden_states

            use_router = i % 2 != 0  # if layer is odd
            if use_router and self.routers is not None:
                hidden_states, aux_loss = self.routers[(i - 1) // 2](
                    hidden_states, expert, attention_mask
                )
            else:
                hidden_states, aux_loss = expert(hidden_states, attention_mask)

            hidden_states += expert_bias + residual
            aux_losses.append(aux_loss)

        return hidden_states, sum(aux_losses)
