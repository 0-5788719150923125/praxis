import random
from typing import List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.controllers.base import BaseController

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class LayerShuffle(BaseController):
    """
    This module implements a basic form of LayerShuffle-Position, though we use it
    as a differentiable "context token" and input-manipulation/preparation mechanism,
    rather than a positional encoder.
    https://arxiv.org/abs/2407.04513
    """

    def __init__(self, config: ConfigType, num_context_tokens: int = 0):
        super().__init__(config)

        # self.navigator = (
        #     PraxisController(config, max_num_experts=config.num_experts * 3)
        #     if config.autopilot
        #     else False
        # )

        self.num_context_tokens = num_context_tokens
        if self.num_context_tokens < 1:
            return

        # Keep learned embeddings for each position and context token
        self.embeddings = nn.Parameter(
            torch.randn(config.depth, self.num_context_tokens, config.hidden_size)
        )
        # Initialize with small values for stability
        nn.init.normal_(self.embeddings, mean=0.0, std=0.02)

    def add_context(
        self, hidden_states: Tensor, attention_mask: Optional[Tensor], position: int
    ) -> Tuple[Tensor, Optional[Tensor]]:

        if self.num_context_tokens < 1:
            return hidden_states, attention_mask

        # Get position-based embeddings
        context = self.embeddings[position]  # [num_tokens, dims]
        # Expand to match batch dimension
        context = context.expand(hidden_states.shape[0], -1, -1)

        # Prepare attention mask for context tokens
        extended_attention_mask = None
        if attention_mask is not None:
            context_mask = attention_mask.new_ones(
                attention_mask.shape[0], self.num_context_tokens
            )
            extended_attention_mask = torch.cat([context_mask, attention_mask], dim=1)

        # Add context to hidden states
        extended_hidden_states = torch.cat([context, hidden_states], dim=1)

        return extended_hidden_states, extended_attention_mask

    def remove_context(
        self, hidden_states: Tensor, attention_mask: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:

        if self.num_context_tokens < 1:
            return hidden_states, attention_mask

        # Remove context tokens from both hidden states and attention mask
        trimmed_states = hidden_states[:, self.num_context_tokens :, :]

        trimmed_mask = None
        if attention_mask is not None:
            trimmed_mask = attention_mask[:, self.num_context_tokens :]
        return trimmed_states, trimmed_mask

    def sort_experts(
        self, experts: List[nn.Module], allow_resampling: bool = False
    ) -> List[nn.Module]:
        depth = self.depth
        if allow_resampling:
            return random.choices(experts, k=depth)
        else:
            return random.sample(experts, k=depth)

    def get_next_expert(
        self,
        hidden_states: Tensor,
        controller_state: Tensor,
        sequential_experts: List[nn.Module],
        ordered_experts: List[nn.Module],
        current_route: List[int],
        current_depth: int,
    ) -> Tuple[Tensor, Tensor, List[int], Optional[int]]:
        """
        Compute next expert selection and associated loss.
        During inference, returns actual expert index.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            sequential_experts: List of all available experts
            ordered_experts: List of experts in the order they should be executed
            current_route: Current execution path through the network
            current_depth: Current depth in the network

        Returns:
            Tuple containing:
                - Auxiliary loss
                - Updated current route
                - Index of the next expert to use
        """

        aux_loss = 0
        next_expert_idx = current_depth
        return hidden_states, controller_state, aux_loss, current_depth
