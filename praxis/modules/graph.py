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
        self.causal = config.causal
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

        # Define expert relationships matrix (fixed)
        self.register_buffer(
            "expert_distances", self._create_expert_distance_matrix(self.num_experts)
        )

        # Graph structure
        max_distance = self.expert_distances.max().item()
        self.spatial_embeddings = nn.Parameter(torch.randn(max_distance + 1))

        # Router projection
        self.router = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.temperature = 1.0
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters with Graphformer-inspired values"""
        nn.init.normal_(self.expert_embeddings)
        nn.init.zeros_(self.centrality_embeddings)
        nn.init.normal_(self.context_embeddings)
        nn.init.uniform_(self.spatial_embeddings)

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
        """Initialize with a ring topology."""
        distances = torch.zeros((num_experts, num_experts), dtype=torch.long)
        for i in range(num_experts):
            for j in range(num_experts):
                dist = min(abs(i - j), num_experts - abs(i - j))
                distances[i, j] = dist
        return distances

    def compute_attention_scores(
        self,
        hidden_states: torch.Tensor,
        current_expert_idx: int,
        next_indices: list[int],
    ) -> torch.Tensor:
        # Project hidden states for routing query
        if self.causal:
            batch_size, seq_len, features = hidden_states.shape
            # Compute cumulative sum of hidden states
            cumsum_hidden_states = torch.cumsum(hidden_states, dim=1)
            counts = torch.arange(1, seq_len + 1, device=hidden_states.device).view(
                1, seq_len, 1
            )
            cumulative_means = cumsum_hidden_states / counts

            # Use the cumulative mean at the last time step
            query = self.router(
                cumulative_means[:, -1, :]
            )  # Shape: (batch_size, hidden_size)
        else:
            # Non-causal case: use mean over all time steps
            query = self.router(
                hidden_states.mean(1)
            )  # Shape: (batch_size, hidden_size)

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
        attention = attention / math.sqrt(self.hidden_size)

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
