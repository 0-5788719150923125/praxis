import torch.nn as nn
import torch.nn.functional as F

from ..configuration_praxis import PraxisConfig


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
