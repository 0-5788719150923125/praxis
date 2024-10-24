import torch
import torch.nn as nn
import torch.nn.functional as F

from praxis import PraxisConfig
from praxis.activations import ACT2FN


class PraxisSMEAR(nn.Module):
    """
    This module implements Soft-Merging of Experts with Adaptive Routing (SMEAR):
    https://arxiv.org/abs/2306.03745
    """

    def __init__(self, config: PraxisConfig):
        super().__init__()
        num_experts = 3
        self.num_dims = config.num_dims

        # Router network: simple linear -> softmax
        self.router = nn.Sequential(
            nn.Linear(self.num_dims, num_experts),
            nn.Softmax(dim=-1),
        )

        # Create pool of experts (all sharing same architecture)
        self.act = ACT2FN[config.activation]
        self.dropout = nn.Dropout(config.dropout)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.num_dims, self.num_dims * 8),
                    nn.Linear(self.num_dims * 4, self.num_dims),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):
        # Get merged parameters for this batch
        up_weights, up_biases, down_weights, down_biases = (
            self._merge_expert_parameters(x)
        )

        # Forward pass through merged expert
        a, b = F.linear(x, up_weights, up_biases).chunk(2, dim=-1)
        y = F.linear(self.dropout(a * self.act(b)), down_weights, down_biases)

        return y

    def _merge_expert_parameters(self, x):
        # Average sequence dimension for routing
        x_mean = x.mean(dim=1)  # [batch_size, num_dims]

        # Get routing probabilities
        routing_probs = self.router(x_mean)  # [batch_size, num_experts]

        # Initialize merged parameters for both layers
        merged_layer1_weight = torch.zeros_like(self.experts[0][0].weight)
        merged_layer1_bias = torch.zeros_like(self.experts[0][0].bias)
        merged_layer2_weight = torch.zeros_like(self.experts[0][1].weight)
        merged_layer2_bias = torch.zeros_like(self.experts[0][1].bias)

        # Weighted average of expert parameters
        for i, expert in enumerate(self.experts):
            # Get batch-specific routing weights for this expert
            expert_weights = routing_probs[:, i].mean()  # Average over batch

            # Merge parameters for first layer
            merged_layer1_weight += expert[0].weight * expert_weights
            merged_layer1_bias += expert[0].bias * expert_weights

            # Merge parameters for second layer
            merged_layer2_weight += expert[1].weight * expert_weights
            merged_layer2_bias += expert[1].bias * expert_weights

        return (
            merged_layer1_weight,
            merged_layer1_bias,
            merged_layer2_weight,
            merged_layer2_bias,
        )
