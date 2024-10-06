import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from ..configuration_praxis import PraxisConfig
from .experts import PraxisBlock
from .router import PraxisMixtureOfDepths


class PraxisDecoder(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.predictors = nn.ModuleList()
        self.experts = nn.ModuleList()
        self.routers = nn.ModuleList() if config.sparse else None
        for i in range(config.n_layer):
            self.predictors.append(ExpertPredictor(config))
            self.experts.append(PraxisBlock(config))
            use_router = i % 2 != 0  # if layer is odd
            if self.routers is not None and use_router:
                self.routers.append(PraxisMixtureOfDepths(config))

    def forward(self, inputs, attention_mask):
        hidden_states = inputs  # Shape: (batch_size, seq_len, n_dim)
        aux_losses = []

        for i in range(len(self.predictors)):

            predictor = self.predictors[i]
            expert_index, expert_weight = predictor(hidden_states)
            expert = self.experts[expert_index]

            residual = hidden_states

            use_router = i % 2 != 0  # if layer is odd
            if use_router and self.routers is not None:
                router = self.routers[(i - 1) // 2]
                hidden_states, aux_loss = router(hidden_states, expert, attention_mask)
            else:
                hidden_states, aux_loss = expert(hidden_states, attention_mask)

            aux_losses.append(aux_loss)
            hidden_states = (hidden_states * expert_weight) + residual

        return hidden_states, sum(aux_losses)


class ExpertPredictor(nn.Module):
    def __init__(self, config: PraxisConfig):
        super().__init__()
        self.norm = nn.RMSNorm(config.n_dim, eps=config.epsilon)
        self.projector = nn.Linear(config.n_dim, config.n_dim)
        self.predictor = nn.Linear(config.n_dim, config.n_layer)
        self.temperature = 0.5

    def forward(self, inputs):
        # Normalize the states
        normalized_states = self.norm(inputs)

        # Apply linear projection to the original inputs
        formatted_states = self.projector(normalized_states)

        # Reduce the batch dimension to 1
        pooled_states = formatted_states.mean(dim=[0, 1])

        # Apply a second linear projection to get logits
        logits = self.predictor(pooled_states)

        # Apply Gumbel-Softmax trick
        if self.training:
            expert_weights = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
        else:
            expert_weights = logits.softmax(dim=-1)

        # Get top-1 expert index and its weight
        expert_index = expert_weights.argmax(dim=-1)
        expert_weight = expert_weights[expert_index]

        return expert_index, expert_weight
