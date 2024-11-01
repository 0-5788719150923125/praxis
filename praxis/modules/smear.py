import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

from praxis.activations import ACT2FN


class PraxisSMEAR(nn.Module):
    """
    This module implements Soft-Merging of Experts with Adaptive Routing (SMEAR):
    https://arxiv.org/abs/2306.03745
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__()
        num_experts = config.expert["num_experts"]
        self.num_dims = config.num_dims

        # Router network: simple linear -> softmax
        self.router = nn.Sequential(
            nn.Linear(self.num_dims, num_experts),
            nn.Softmax(dim=-1),
        )

        # Create pool of experts (all sharing same architecture)
        self.act = ACT2FN[config.activation]
        self.dropout = nn.Dropout(config.dropout)
        self.experts_up = nn.ModuleList(
            [nn.Linear(self.num_dims, self.num_dims * 8) for _ in range(num_experts)]
        )
        self.experts_down = nn.ModuleList(
            [nn.Linear(self.num_dims * 4, self.num_dims) for _ in range(num_experts)]
        )

    def forward(self, inputs):
        # Get merged parameters for this batch
        up_weights, up_biases, down_weights, down_biases = (
            self._merge_expert_parameters(inputs)
        )

        # Forward pass through merged expert
        linear, gated = F.linear(inputs, up_weights, up_biases).chunk(2, dim=-1)
        sparsified = self.dropout(linear * self.act(gated))
        outputs = F.linear(sparsified, down_weights, down_biases)

        return outputs

    def _merge_expert_parameters(self, inputs):
        # Average sequence dimension for routing
        reduced_inputs = inputs.mean(dim=1)  # [batch_size, num_dims]

        # Get routing probabilities
        routing_probs = self.router(reduced_inputs)  # [batch_size, num_experts]

        # Initialize merged parameters for both layers
        up_weights = torch.zeros_like(self.experts_up[0].weight)
        up_biases = torch.zeros_like(self.experts_up[0].bias)
        down_weights = torch.zeros_like(self.experts_down[0].weight)
        down_biases = torch.zeros_like(self.experts_down[0].bias)

        # Weighted average of expert parameters
        for i in range(len(self.experts_up)):
            # Get batch-specific routing weights for this expert
            expert_weights = routing_probs[:, i].mean()  # Average over batch

            # Merge parameters for first layer
            up_weights += self.dropout(self.experts_up[i].weight) * expert_weights
            up_biases += self.dropout(self.experts_up[i].bias) * expert_weights

            # Merge parameters for second layer
            down_weights += self.dropout(self.experts_down[i].weight) * expert_weights
            down_biases += self.dropout(self.experts_down[i].bias) * expert_weights

        return (
            up_weights,
            up_biases,
            down_weights,
            down_biases,
        )
