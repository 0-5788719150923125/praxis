from typing import List, Optional, Tuple

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

    def __init__(self, config: "AutoConfig", allow_early_exits=False):
        super().__init__(config)

        # Create a gating network for each layer to decide the next layer
        extra_vectors = int(allow_early_exits)
        self.gates = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, self.num_experts + extra_vectors)
                for _ in range(self.depth)
            ]
        )

    def get_next_expert(
        self,
        hidden_states: torch.Tensor,
        sequential_experts: List[nn.Module],
        ordered_experts: List[nn.Module],
        current_depth: int,
    ) -> Tuple[torch.Tensor, Optional[int]]:
        # Pool the hidden states - using mean pooling for simplicity
        pooled_hidden = hidden_states.mean(dim=1)  # [batch_size, hidden_size]

        # Apply the gating network for the current layer
        gate_logits = self.gates[current_depth](pooled_hidden)  # [batch_size, depth]

        # Compute next layer probabilities
        gate_probs = F.softmax(gate_logits, dim=1)

        # For training stability, compute a small entropy loss
        # This encourages exploration of different routing paths
        gating_loss = 0
        if self.training:
            entropy = -(gate_probs * torch.log(gate_probs + 1e-10)).sum(dim=1).mean()
            gating_loss = -0.01 * entropy  # Encourage exploration with negative loss

        # Select the next layer (expert) to process
        next_expert_idx = torch.argmax(gate_probs, dim=1)[0].item()

        # Allow early exits
        if next_expert_idx == self.num_experts:
            return gating_loss, None

        self._update_route(hidden_states, current_depth, next_expert_idx)

        return gating_loss, next_expert_idx
