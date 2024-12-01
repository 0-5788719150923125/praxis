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
        self.num_context_tokens = 3  # Same as current implementation

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
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        # Initialize with small values for stability
        self._init_parameters()

    def _init_parameters(self):
        # Initialize parameters with small values
        for param in [self.expert_embeddings, self.context_embeddings]:
            nn.init.normal_(param, mean=0.0, std=0.02)
        # Initialize biases with zeros
        nn.init.zeros_(self.centrality_bias)
        nn.init.zeros_(self.spatial_bias)

    def compute_routing_scores(
        self,
        state: torch.Tensor,
        current_expert_idx: int,
        available_indices: list[int],
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing probabilities for next expert selection.
        Returns both routing loss and next expert probabilities.
        """
        batch_size = state.size(0)

        # Project current state
        projected_state = self.router(state.mean(dim=1))  # [batch_size, hidden_size]

        # Get available expert embeddings
        available_embeddings = self.expert_embeddings[available_indices]

        # Compute attention scores
        attention = torch.matmul(
            projected_state, available_embeddings.t()
        )  # [batch_size, num_available]

        # Add structural biases
        attention = attention + self.centrality_bias[available_indices]
        attention = attention + self.spatial_bias[current_expert_idx, available_indices]

        # Convert to probabilities
        if self.training:
            # Use Gumbel-Softmax during training
            probs = F.gumbel_softmax(attention, tau=temperature, hard=False)
            # Compute loss pushing towards current expert (can be modified)
            target = torch.full(
                (batch_size,),
                available_indices.index(current_expert_idx),
                device=state.device,
            )
            loss = (
                F.cross_entropy(attention, target) * 0.001
            )  # Small scale like original
        else:
            # During inference, use standard softmax
            probs = F.softmax(attention, dim=-1)
            loss = torch.tensor(0.0, device=state.device)

        return loss, probs

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

    def get_next_expert(
        self,
        hidden_states: torch.Tensor,
        current_expert_idx: int,
        original_experts: List[nn.Module],
        *args,
        **kwargs
    ) -> tuple[torch.Tensor, Optional[int]]:
        """
        Compute next expert selection and associated loss.
        During inference, returns actual expert index.
        """
        available_indices = range(len(original_experts))
        loss, probs = self.compute_routing_scores(
            hidden_states, current_expert_idx, available_indices
        )

        if not self.training:
            next_idx = torch.argmax(probs, dim=-1)[0].item()
            return loss, available_indices[next_idx]

        return loss, None
