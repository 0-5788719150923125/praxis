from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.controllers.base import BaseController
from praxis.modules.visualization import RouteVisualizer


class Pathfinder(BaseController):
    """
    Implements a gating mechanism for dynamic layer selection in transformer models.
    Each layer uses a gating network to decide which layer to process next based on
    the current hidden state.
    """

    def __init__(self, config: "AutoConfig", allow_early_exits=False):
        super().__init__(config)
        self.debug = config.debug
        self.depth = config.depth

        # Create a gating network for each layer to decide the next layer
        extra = int(allow_early_exits)
        self.gates = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, self.depth + extra)
                for _ in range(self.depth + extra)
            ]
        )

        self.current_route = []

        self.visualizer = (
            RouteVisualizer(
                num_experts=config.num_experts,
                max_history=10000,
                save_rate=100 * config.depth,
            )
            if self.debug
            else False
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
        if next_expert_idx == self.depth:
            return gating_loss, None

        # Record the current layer in the route
        self.current_route.append(next_expert_idx)

        # Update visualizer
        if (
            self.visualizer
            and not self.training
            and hidden_states.size(0) == 1  # not validation
            and current_depth > 0  # not the final layer
        ):
            # Just send the immediate transition
            previous_idx = self.current_route[current_depth - 1]
            self.visualizer.add_transition(previous_idx, next_expert_idx)

        return gating_loss, next_expert_idx

    def reset_route(self, hidden_states):
        """Reset the tracking of the current route through layers."""
        if self.debug:
            route = [str(r) for r in self.current_route]
            if not self.training and hidden_states.size(0) == 1:  # not validation
                print(f"DEBUG: inferencing through:  {' -> '.join(route)}")
        self.current_route = []
