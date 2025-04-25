from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from praxis.controllers.visualization import RouteVisualizer


class BaseController(nn.Module):
    """
    A no-op controller.
    """

    def __init__(self, config: "AutoConfig", use_visualizer=False, *args, **kwargs):
        super().__init__()
        self.debug = config.debug
        self.depth = config.depth
        self.num_experts = config.num_experts
        self.current_route = []
        self.visualizer = (
            RouteVisualizer(
                num_experts=config.num_experts,
                max_history=10000,
                save_rate=100 * config.depth,
            )
            if self.debug and use_visualizer
            else False
        )

    def add_context(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """No-op implementation to maintain API compatibility."""
        return hidden_states, attention_mask

    def remove_context(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """No-op implementation to maintain API compatibility."""
        return hidden_states, attention_mask

    def sort_experts(
        self, experts: List[nn.Module], *args, **kwargs
    ) -> List[nn.Module]:
        """No-op to maintain API compatibility."""
        return experts

    def get_next_expert(
        self,
        hidden_states: torch.Tensor,
        sequential_experts: List[nn.Module],
        ordered_experts: List[nn.Module],
        current_depth: int,
    ) -> Tuple[torch.Tensor, Optional[int]]:
        return 0, current_depth

    def _update_route(self, hidden_states, current_depth, next_expert_idx):
        # Update visualizer
        if (
            self.visualizer
            and not self.training
            and hidden_states.size(0) == 1  # not validation
            and current_depth > 0  # not the final layer
        ):
            self.current_route.append(next_expert_idx)
            previous_idx = self.current_route[current_depth - 1]
            self.visualizer.add_transition(previous_idx, next_expert_idx)

    def reset_route(self, hidden_states):
        """Reset the tracking of the current route through layers."""
        if self.debug:
            route = [str(r) for r in self.current_route]
            if not self.training and hidden_states.size(0) == 1:  # not validation
                print(f"DEBUG: inferencing through:  {' -> '.join(route)}")
        self.current_route = []
