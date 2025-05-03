from typing import List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ConfigType = TypeVar("ConfigType", bound=PretrainedConfig)


class Autopilot(nn.Module):
    """
    This controller implements an expert-prediction mechanism, which trains a small
    router to intelligently-navigate through layers in the network.
    """

    __version__ = "0.1.0"

    def __init__(self, config: ConfigType, max_num_experts: int) -> None:
        super().__init__()
        assert config.shuffle, "To use `autopilot`, you must also use `shuffle=True`."

        hidden_size = config.hidden_size
        self.loss_scale = 0.001

        # Learn embeddings for all possible experts
        self.representations = nn.Parameter(torch.randn(max_num_experts, hidden_size))

        # Simple prediction network
        self.predictor = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        current_depth: int,
        original_experts: List[nn.Module],
        current_experts: List[nn.Module],
        current_expert: nn.Module,
    ) -> Tuple[float, Optional[int]]:
        """
        Forward pass of the autopilot controller.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            current_depth: Current depth in the network
            original_experts: Original list of all experts
            current_experts: Current list of available experts
            current_expert: The current expert being processed

        Returns:
            Tuple containing:
                - Loss value (during training) or 0.0 (during inference)
                - Index of the next recommended expert (during inference) or None (during training)
        """

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
