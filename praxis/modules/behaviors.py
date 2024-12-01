import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig

from praxis.activations import ACT2FN


class LayerShuffle(nn.Module):
    """
    This module implements a basic form of LayerShuffle-Position, though we use it
    as a differentiable "context token" and input-manipulation/preparation mechanism,
    rather than a positional encoder.
    https://arxiv.org/abs/2407.04513
    """

    def __init__(self, config: AutoConfig, num_context_tokens: int = 1):
        super().__init__()
        self.num_context_tokens = num_context_tokens

        if self.num_context_tokens < 1:
            return

        # Keep learned embeddings for each position and context token
        self.embeddings = nn.Parameter(
            torch.randn(config.depth, num_context_tokens, config.num_dims)
        )
        # Initialize with small values for stability
        nn.init.normal_(self.embeddings, mean=0.0, std=0.02)

    def add_context(
        self, hidden_states: Tensor, attention_mask: Tensor, position: int
    ) -> tuple[Tensor, Tensor]:

        if self.num_context_tokens < 1:
            return hidden_states, attention_mask

        # Get position-based embeddings
        context = self.embeddings[position]  # [num_tokens, dims]
        # Expand to match batch dimension
        context = context.expand(hidden_states.shape[0], -1, -1)

        # Prepare attention mask for context tokens
        context_mask = attention_mask.new_ones(
            attention_mask.shape[0], self.num_context_tokens
        )
        extended_attention_mask = torch.cat([context_mask, attention_mask], dim=1)

        # Add context to hidden states
        extended_hidden_states = torch.cat([context, hidden_states], dim=1)

        return extended_hidden_states, extended_attention_mask

    def remove_context(
        self, hidden_states: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor]:

        if self.num_context_tokens < 1:
            return hidden_states, attention_mask

        # Remove context tokens from both hidden states and attention mask
        trimmed_states = hidden_states[:, self.num_context_tokens :, :]
        trimmed_mask = attention_mask[:, self.num_context_tokens :]
        return trimmed_states, trimmed_mask

    def shuffle_experts(self, experts: list, allow_resampling: bool = False) -> list:
        depth = self.embeddings.shape[0]
        if allow_resampling:
            return random.choices(experts, k=depth)
        else:
            return random.sample(experts, k=depth)
