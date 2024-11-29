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

    def __init__(self, config: AutoConfig, num_context_tokens: int = 3):
        super().__init__()
        self.num_context_tokens = num_context_tokens
        self.embeddings = nn.Parameter(
            torch.randn(config.depth, num_context_tokens, config.num_dims)
        )
        nn.init.normal_(self.embeddings, mean=0.0, std=0.02)
        # Add mixing components
        bottleneck = config.num_dims // 4
        self.mixer = nn.Sequential(
            nn.Linear(config.num_dims, bottleneck),
            ACT2FN["gelu"],
            nn.Dropout(config.dropout),
            nn.Linear(bottleneck, config.num_dims),
        )
        # Gate for balancing position vs content influence
        self.gate = nn.Linear(config.num_dims, 1)

    def add_context(
        self, hidden_states: Tensor, attention_mask: Tensor, position: int
    ) -> Tensor:
        # Get position-based embeddings
        pos_embeds = self.embeddings[position]  # [num_tokens, dims]

        # Create content-based context
        # Average the hidden states for content representation
        content = hidden_states.mean(dim=1)  # [batch, dims]
        content_context = self.mixer(content)  # [batch, dims]

        # Expand position embeddings
        pos_embeds = pos_embeds.expand(hidden_states.shape[0], -1, -1)
        # Expand content context
        content_context = content_context.unsqueeze(1).expand(
            -1, self.num_context_tokens, -1
        )

        # Compute mixing ratio
        gate = torch.sigmoid(self.gate(content_context))

        # Mix position and content information
        mixed_context = gate * pos_embeds + (1 - gate) * content_context

        # Prepare attention mask for context tokens
        context_mask = attention_mask.new_ones(
            attention_mask.shape[0], self.num_context_tokens
        )
        extended_attention_mask = torch.cat([context_mask, attention_mask], dim=1)

        # Add context to hidden states
        extended_hidden_states = torch.cat([mixed_context, hidden_states], dim=1)

        return extended_hidden_states, extended_attention_mask

    def remove_context(
        self, hidden_states: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
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
