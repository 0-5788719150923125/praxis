from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig


class PraxisController(nn.Module):
    """
    This controller implements an expert-prediction mechanism, which trains a small
    router to intelligently-navigate through layers in the network.
    """

    __version__ = "0.1.0"

    def __init__(self, config, max_num_experts: int):
        super().__init__()
        hidden_size = config.num_dims
        self.loss_scale = 0.001

        # Learn embeddings for all possible experts
        self.representations = nn.Parameter(torch.randn(max_num_experts, hidden_size))

        # Simple prediction network
        self.predictor = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        original_experts: List[nn.Module],
        current_experts: List[nn.Module],
        current_expert: nn.Module,
        hidden_states: torch.Tensor,
    ):

        batch_size = hidden_states.size(0)
        device = hidden_states.device

        # Get indices for current expert pool
        current_expert_idx = original_experts.index(current_expert)
        current_indices = [original_experts.index(e) for e in current_experts]

        # Get current state
        state = hidden_states.mean(dim=1)  # [batch_size, hidden_size]

        # Project state to embedding space
        projected_state = self.predictor(state)  # [batch_size, hidden_size]

        # Get embeddings for current expert pool
        available_embeddings = self.representations[
            current_indices
        ]  # [num_current, hidden_size]

        # Compute similarities only with available experts
        logits = torch.matmul(
            projected_state, available_embeddings.t()
        )  # [batch_size, num_current]

        if self.training:
            # Create target for current expert
            target = torch.full((batch_size,), current_expert_idx, device=device)
            # Compute loss only over available experts
            loss = F.cross_entropy(logits, target) * self.loss_scale
            return loss, None
        else:
            # During inference, return index of next recommended expert from current pool
            next_idx = torch.argmax(logits, dim=-1)[0].item()
            recommended_expert_idx = current_indices[next_idx]
            return 0.0, recommended_expert_idx
