from typing import List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.containers.loss import LossContainer
from praxis.controllers.base import BaseController

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class Pathfinder(BaseController):
    """
    Implements a gating mechanism for dynamic layer selection in transformer models.
    Each layer uses a gating network to decide which layer to process next based on
    the current hidden state.
    """

    def __init__(self, config: ConfigType, allow_early_exits: bool = False) -> None:
        super().__init__(config, allow_visualizer=True)

        # Create a gating network for each layer to decide the next layer
        self.extra_vectors = int(allow_early_exits)
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.hidden_size),
                    nn.Linear(
                        config.hidden_size, self.num_experts + self.extra_vectors
                    ),
                    nn.Dropout(config.dropout),
                )
                for _ in range(self.depth)
            ]
        )

    def get_next_expert(
        self,
        hidden_states: Tensor,
        controller_state: Tensor,
        sequential_experts: List[nn.Module],
        ordered_experts: List[nn.Module],
        current_route: List[int],
        current_depth: int,
    ) -> Tuple[Tensor, Tensor, LossContainer, Optional[int]]:
        # Use the last vector; attention "pushes" information into this vector
        last_hidden = hidden_states[:, -1, :]

        # Apply the gating network for the current layer
        gate_logits = self.gates[current_depth](last_hidden)  # [batch_size, depth]

        # Compute next layer probabilities
        gate_probs = F.softmax(gate_logits, dim=1)

        # Calculate entropy loss for training
        if self.training:
            entropy = -(gate_probs * torch.log(gate_probs + 1e-10)).sum(dim=1).mean()
            entropy_loss = -0.01 * entropy  # Encourage exploration
            loss_container = LossContainer(entropy=entropy_loss)
        else:
            loss_container = LossContainer()

        # Get each example's vote for which expert to use next
        batch_votes = torch.argmax(gate_probs, dim=1)  # [batch_size]

        # Find the most common vote (mode) across the batch
        vote_counts = torch.bincount(
            batch_votes, minlength=self.num_experts + self.extra_vectors
        )
        next_expert_idx = torch.argmax(vote_counts).item()

        # Allow early exits
        if next_expert_idx == self.num_experts:
            return hidden_states, controller_state, loss_container, None

        return hidden_states, controller_state, loss_container, next_expert_idx
