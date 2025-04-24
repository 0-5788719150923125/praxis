import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.controllers.base import BaseController


class LayerShuffle(BaseController):
    """
    This module implements a basic form of LayerShuffle-Position, though we use it
    as a differentiable "context token" and input-manipulation/preparation mechanism,
    rather than a positional encoder.
    https://arxiv.org/abs/2407.04513
    """

    def __init__(self, config: "AutoConfig", num_context_tokens: int = 0):
        super().__init__(config)
        self.debug = config.debug
        self.depth = config.depth
        self.current_route = []
        assert (
            config.num_experts == config.depth
        ), "There is no point in making `num_experts` greater than or less than `depth`, when `shuffle != True`. The additional experts would never be used."

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
        self, hidden_states: Tensor, attention_mask: Tensor, position: int
    ) -> tuple[Tensor, Tensor]:

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
        self, hidden_states: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, Tensor]:

        if self.num_context_tokens < 1:
            return hidden_states, attention_mask

        # Remove context tokens from both hidden states and attention mask
        trimmed_states = hidden_states[:, self.num_context_tokens :, :]

        trimmed_mask = None
        if attention_mask is not None:
            trimmed_mask = attention_mask[:, self.num_context_tokens :]
        return trimmed_states, trimmed_mask

    def shuffle_experts(self, experts: list, allow_resampling: bool = False) -> list:
        depth = self.depth
        if allow_resampling:
            return random.choices(experts, k=depth)
        else:
            return random.sample(experts, k=depth)

    def get_next_expert(
        self,
        hidden_states: torch.Tensor,
        current_depth: int,
        original_experts: List[nn.Module],
        current_experts: List[nn.Module],
    ) -> tuple[torch.Tensor, Optional[int]]:
        """
        Compute next expert selection and associated loss.
        During inference, returns actual expert index.
        """

        # if self.navigator:
        #     aux_loss, next_expert_idx = self.navigator(
        #         hidden_states,
        #         current_depth,
        #         original_experts,
        #         current_experts,
        #     )
        # else:
        aux_loss = 0
        next_expert = current_experts[current_depth]
        next_expert_idx = original_experts.index(next_expert)

        self.current_route.append(next_expert_idx)

        return aux_loss, next_expert_idx

    def reset_route(self, hidden_states):
        """Reset the tracking of the current route through layers."""
        if self.debug:
            route = [str(r) for r in self.current_route]
            if not self.training and hidden_states.size(0) == 1:  # not validation
                print(f"DEBUG: inferencing through:  {' -> '.join(route)}")
        self.current_route = []
