import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_praxis import PraxisConfig
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
        hidden_states = inputs
        aux_losses = []

        sequence, aux_loss = self.ctrl(hidden_states)
        aux_losses.append(aux_loss)

        for i, choice in enumerate(sequence):
            expert = self.experts[choice]
            use_router = i % 2 != 0  # if layer is odd
            if self.routers is not None and use_router:
                outputs = self.routers[(i - 1) // 2](
                    hidden_states, expert, attention_mask
                )
            else:
                outputs = expert(hidden_states, attention_mask)
            hidden_states = outputs["hidden_states"]
            if "aux_loss" in outputs:
                aux_losses.append(outputs["aux_loss"])
        return dict(hidden_states=hidden_states, aux_loss=sum(aux_losses))


class PraxisController(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_dim = config.n_dim
        self.n_layer = config.n_layer
        self.temperature = 0.5

        self.gru = nn.GRU(config.n_dim, config.n_dim, batch_first=True)
        self.focus = nn.Linear(config.n_dim, config.n_layer)

        # self.batch_reduction = LearnableReduction(config.n_dim, hidden_dim=64)

        self.register_buffer("order_ema", torch.zeros(self.n_layer))
        self.ema_decay = 0.99
        self.aux_weight = 10.0

    def forward(self, inputs):
        # Aggregate inputs (via mean) to create a single context vector
        inputs_reduced = inputs.mean(dim=0)  # Shape: (seq_len, dims)
        # inputs_reduced = self.batch_reduction(inputs)

        # Pass through GRU
        gru_out, _ = self.gru(inputs_reduced)  # Shape: (seq_len, dims)

        # Reduce GRU outputs to just a single time step
        reduced_gru = gru_out.mean(dim=0)  # Shape: (dims)

        # Compute logits for each expert
        logits = self.focus(reduced_gru)  # Shape: (num_experts)

        # Apply Gumbel-Softmax to obtain one-hot encoded selections
        gumbel = F.gumbel_softmax(
            logits, tau=self.temperature, hard=False
        )  # Shape: (num_experts)

        # Return the indices of the order of predicted experts
        sequence = torch.argsort(gumbel, dim=-1, descending=True)

        # Update EMA of expert order
        current_order = F.one_hot(sequence, num_classes=self.n_layer).float()
        self.order_ema = (
            self.ema_decay * self.order_ema + (1 - self.ema_decay) * current_order
        )

        # Compute diversity loss
        aux_loss = F.mse_loss(current_order, self.order_ema) * self.aux_weight

        return sequence, aux_loss


class LearnableReduction(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden_dim, kernel_size=(3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=(3, 1), padding=(1, 0))
        self.activation = nn.GELU()

    def forward(self, x):
        # x shape: (batch_size, seq_len, dim)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, dim)
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        x = x.squeeze(1)  # (batch_size, seq_len, dim)
        return x.mean(dim=0)  # (seq_len, dim)
