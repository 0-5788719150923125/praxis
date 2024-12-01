import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig

from praxis.activations import ACT2FN


class PraxisGraph(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_size = config.num_dims
        self.num_context_tokens = 3

        # Expert Embeddings - represent each expert in latent space
        self.expert_embeddings = nn.Parameter(
            torch.randn(self.num_experts, self.hidden_size)
        )

        # Context Tokens - for each expert
        self.context_embeddings = nn.Parameter(
            torch.randn(self.num_experts, self.num_context_tokens, self.hidden_size)
        )

        # Structural Encodings (Graphormer-inspired)
        self.centrality_bias = nn.Parameter(torch.randn(self.num_experts))
        self.spatial_bias = nn.Parameter(
            torch.randn(self.num_experts, self.num_experts)
        )

        # Routing Network
        self.router = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            ACT2FN["gelu"],
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        # Replace cosine similarity with learned compatibility
        self.compatibility_matrix = nn.Parameter(
            torch.randn(self.num_experts, self.num_experts)
        )

        # Initialize with small values for stability
        self._init_parameters()

    def _init_parameters(self):
        # Initialize embeddings with orthogonal values
        num_experts = self.num_experts
        embedding_dim = self.hidden_size

        # Create orthogonal expert embeddings
        expert_embeddings = torch.empty(num_experts, embedding_dim)
        nn.init.orthogonal_(expert_embeddings)
        self.expert_embeddings.data.copy_(expert_embeddings)

        # Initialize context tokens normally
        nn.init.normal_(self.context_embeddings, mean=0.0, std=0.02)

        # Initialize biases to zero (better starting point)
        nn.init.zeros_(self.spatial_bias)

        # Initialize compatibility matrix
        nn.init.zeros_(self.compatibility_matrix)
        with torch.no_grad():
            # Add slight negative bias for self-connections
            self.compatibility_matrix.fill_diagonal_(-0.1)

    def compute_centrality_scores(self):
        """Compute dynamic centrality based on expert connectivity"""
        # Use softmax of raw centrality_bias for better gradient flow
        scores = F.softmax(self.centrality_bias, dim=-1)
        return scores

    def add_context(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, expert_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add context tokens from the current expert."""
        context = self.context_embeddings[expert_idx]  # [num_tokens, hidden_size]
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

    def project_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to routing space."""
        return self.router(hidden_states)[:, -1]  # [batch_size, hidden_size]

    def get_valid_indices(self, current_expert_idx: int) -> list[int]:
        """Get indices of all possible next experts."""
        return list(range(self.num_experts))

    def compute_attention_scores(
        self,
        projected_state: torch.Tensor,
        current_expert_idx: int,
        next_indices: list[int],
    ) -> torch.Tensor:
        """Compute attention scores with structural information."""
        device = projected_state.device

        # Get expert embeddings
        expert_embeds = self.expert_embeddings[next_indices]

        # Base attention scores
        scale = torch.sqrt(torch.tensor(self.hidden_size, device=device))
        attention = torch.matmul(projected_state, expert_embeds.t()) / scale

        # Add centrality bias
        centrality = self.compute_centrality_scores()[next_indices]
        attention = attention + centrality

        # Add spatial relationships
        for i, target_idx in enumerate(next_indices):
            # Get spatial distance
            distance = self.spatial_bias[current_expert_idx, target_idx]

            # Use learned compatibility instead of cosine similarity
            compatibility = self.compatibility_matrix[current_expert_idx, target_idx]

            # Add slight penalty for self-selection
            if target_idx == current_expert_idx:
                compatibility = compatibility - 0.1

            attention[:, i] = attention[:, i] + distance + compatibility

        return attention

    def get_next_expert(
        self,
        hidden_states: torch.Tensor,
        current_idx: int,
        original_experts: List[nn.Module],
        current_experts: List[nn.Module],
        current_expert: nn.Module,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Compute next expert selection and associated loss."""
        batch_size = hidden_states.size(0)
        device = hidden_states.device

        # Project state
        projected_state = self.project_state(hidden_states)

        # Get current position and valid next experts
        current_expert_idx = original_experts.index(current_expert)
        next_indices = self.get_valid_indices(current_expert_idx)

        if not next_indices:
            return torch.tensor(0.0, device=device), None

        # Compute attention scores
        attention = self.compute_attention_scores(
            projected_state, current_expert_idx, next_indices
        )

        if self.training:
            probs = F.gumbel_softmax(attention, tau=temperature, hard=False)
            next_idx = next_indices[torch.multinomial(probs[0], 1).item()]
            target = torch.zeros(batch_size, device=device, dtype=torch.long)
            loss = F.cross_entropy(attention, target) * 0.001
            return loss, next_idx
        else:

            probs = F.softmax(attention, dim=-1)

            # Find argmax across proper dimension
            max_idx = torch.argmax(probs[0])  # Take first batch, then find max

            next_idx = next_indices[max_idx.item()]
            # # Add debug prints
            # print(f"Attention shape: {attention.shape}")
            # print(f"Raw attention: {attention}")
            # print(f"Softmax probs: {probs}")
            # print(f"Selected index: {max_idx.item()}")
            # print(f"Final expert selection: {next_idx}")

            return torch.tensor(0.0, device=device), next_idx
