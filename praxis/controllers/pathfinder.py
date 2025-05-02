from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.controllers.base import BaseController


class Pathfinder(BaseController):
    """
    Implements a gating mechanism for dynamic layer selection in transformer models.
    Each layer uses a gating network to decide which layer to process next based on
    the current hidden state.
    """

    def __init__(self, config: "AutoConfig", allow_early_exits: bool = False) -> None:
        super().__init__(config, allow_visualizer=True)

        # Create a gating network for each layer to decide the next layer
        self.extra_vectors = int(allow_early_exits)
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
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
    ) -> Tuple[Tensor, Tensor, List[int], Optional[int]]:
        # Pool the hidden states - using mean pooling for simplicity
        pooled_hidden = hidden_states.mean(dim=1)  # [batch_size, hidden_size]

        # Apply the gating network for the current layer
        gate_logits = self.gates[current_depth](pooled_hidden)  # [batch_size, depth]

        # Compute next layer probabilities
        gate_probs = F.softmax(gate_logits, dim=1)

        # Calculate entropy loss for training
        gating_loss = torch.tensor(0.0, device=hidden_states.device)
        if self.training:
            entropy = -(gate_probs * torch.log(gate_probs + 1e-10)).sum(dim=1).mean()
            gating_loss = -0.01 * entropy  # Encourage exploration

        # Get each example's vote for which expert to use next
        batch_votes = torch.argmax(gate_probs, dim=1)  # [batch_size]

        # Find the most common vote (mode) across the batch
        vote_counts = torch.bincount(
            batch_votes, minlength=self.num_experts + self.extra_vectors
        )
        next_expert_idx = torch.argmax(vote_counts).item()

        # Allow early exits
        if next_expert_idx == self.num_experts:
            return controller_state, gating_loss, current_route, None

        current_route = self._update_route(
            hidden_states, current_route, current_depth, next_expert_idx
        )

        return controller_state, gating_loss, current_route, next_expert_idx
