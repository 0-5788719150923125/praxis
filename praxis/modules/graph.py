import math
import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig

from praxis.activations import ACT2FN


class PraxisGraph(nn.Module):
    """Graph-based expert routing inspired by Graphformer"""

    def __init__(self, config: AutoConfig):
        super().__init__()
        # Core dimensions
        self.num_experts = config.num_experts
        self.hidden_size = config.num_dims
        self.num_context_tokens = 3  # Maintain API compatibility

        # Expert node embeddings
        self.expert_embeddings = nn.Parameter(
            torch.randn(self.num_experts, self.hidden_size)
        )

        # Centrality encoding (from Graphformer)
        self.centrality_embeddings = nn.Parameter(
            torch.randn(self.num_experts, self.hidden_size)
        )

        # Context tokens (for API compatibility)
        self.context_embeddings = nn.Parameter(
            torch.randn(self.num_experts, self.num_context_tokens, self.hidden_size)
        )

        # Spatial encoding (path distances from Graphformer)
        self.spatial_bias = nn.Parameter(
            torch.randn(self.num_experts, self.num_experts)
        )

        # Redefine spatial bias to have clear structural meaning
        num_distance_buckets = 3  # Different types of relationships
        self.spatial_embeddings = nn.Parameter(torch.randn(num_distance_buckets))

        # Define expert relationships matrix (fixed)
        self.register_buffer(
            "expert_distances", self._create_expert_distance_matrix(self.num_experts)
        )

        # Router projection
        self.router = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.temperature = 1.0
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters with Graphformer-inspired values"""
        # Initialize node embeddings
        nn.init.normal_(self.expert_embeddings, std=0.02)

        # Initialize centrality embeddings to zeros
        nn.init.zeros_(self.centrality_embeddings)

        # Initialize context tokens
        nn.init.normal_(self.context_embeddings, std=0.02)

        # Initialize spatial bias for path distances
        nn.init.zeros_(self.spatial_bias)
        with torch.no_grad():
            # Add slight bias for direct connections
            eye = torch.eye(self.num_experts)
            self.spatial_bias.data = self.spatial_bias.data + eye * 0.1

    def add_context(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, expert_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add context tokens from the current expert."""
        context = self.context_embeddings[expert_idx]
        context = context.expand(hidden_states.shape[0], -1, -1)

        # Create attention mask for context tokens
        context_mask = attention_mask.new_ones(
            attention_mask.shape[0], self.num_context_tokens
        )
        extended_mask = torch.cat([context_mask, attention_mask], dim=1)

        # Add context to hidden states
        extended_states = torch.cat([context, hidden_states], dim=1)

        return extended_states, extended_mask

    def remove_context(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remove context tokens."""
        return (
            hidden_states[:, self.num_context_tokens :, :],
            attention_mask[:, self.num_context_tokens :],
        )

    def _create_expert_distance_matrix(self, num_experts: int) -> torch.Tensor:
        """Create a matrix encoding structural relationships between experts.

        Distance values:
        0 = Same expert (self)
        1 = Adjacent expert (neighboring)
        2 = Distant expert (others)
        """
        distances = torch.full((num_experts, num_experts), 2)  # Default: distant

        # Set adjacent relationships
        for i in range(num_experts):
            # Consider experts as circularly arranged
            prev_idx = (i - 1) % num_experts
            next_idx = (i + 1) % num_experts
            distances[i, prev_idx] = 1
            distances[i, next_idx] = 1

        # Set self-relationships
        distances.fill_diagonal_(0)

        return distances

    def compute_attention_scores(
        self,
        hidden_states: torch.Tensor,
        current_expert_idx: int,
        next_indices: list[int],
    ) -> torch.Tensor:
        # Project hidden states for routing query
        query = self.router(hidden_states[:, -1])

        # Get expert node embeddings with centrality
        expert_nodes = (
            self.expert_embeddings[next_indices]
            + self.centrality_embeddings[next_indices]
        )

        # Base attention computation
        attention = torch.matmul(query, expert_nodes.t())

        # Apply structural spatial bias
        distances = self.expert_distances[
            current_expert_idx, next_indices
        ]  # Get distance bucket indices
        spatial_bias = self.spatial_embeddings[
            distances
        ]  # Convert to learnable bias values

        attention = attention + spatial_bias
        attention = attention / math.sqrt(self.hidden_size) / self.temperature

        return attention

    def get_next_expert(
        self,
        hidden_states: torch.Tensor,
        current_idx: int,
        original_experts: List[nn.Module],
        current_experts: List[nn.Module],
        current_expert: nn.Module,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Select next expert using graph attention mechanism"""
        device = hidden_states.device
        current_expert_idx = original_experts.index(current_expert)
        next_indices = list(range(self.num_experts))

        # Compute attention scores
        attention = self.compute_attention_scores(
            hidden_states, current_expert_idx, next_indices
        )

        if self.training:
            # Use Gumbel-Softmax during training
            probs = F.gumbel_softmax(attention, tau=1.0, hard=False)
            next_idx = next_indices[torch.multinomial(probs[0], 1).item()]

            # Compute entropy loss to encourage diversity
            entropy_loss = (
                -(F.softmax(attention, dim=-1) * F.log_softmax(attention, dim=-1))
                .sum(dim=-1)
                .mean()
            )

            return entropy_loss * 0.01, next_idx

        else:
            # Use deterministic selection during inference
            probs = F.softmax(attention, dim=-1)
            next_idx = next_indices[torch.argmax(probs[0]).item()]

            # Debug information
            print(f"Attention shape: {attention.shape}")
            print(f"Raw attention: {attention}")
            print(f"Softmax probs: {probs}")
            print(f"Selected expert: {next_idx}")

            return torch.tensor(0.0, device=device), next_idx
