from typing import Any, List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
from torch import Tensor

ConfigType = TypeVar("ConfigType", bound="PretrainedConfig")

from praxis.controllers.visualization import RouteVisualizer


class BaseController(nn.Module):
    """
    A no-op controller.
    """

    def __init__(
        self,
        config: ConfigType,
        allow_visualizer: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__()
        self.debug = config.debug
        self.depth = config.depth
        self.num_experts = config.num_experts
        self.visualizer = (
            RouteVisualizer(
                num_experts=config.num_experts,
                max_history=10000,
                save_rate=100 * config.depth,
            )
            if self.debug and allow_visualizer
            else False
        )

    def add_context(
        self, hidden_states: Tensor, attention_mask: Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[Tensor, Tensor]:
        """No-op implementation to maintain API compatibility."""
        return hidden_states, attention_mask

    def remove_context(
        self, hidden_states: Tensor, attention_mask: Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[Tensor, Tensor]:
        """No-op implementation to maintain API compatibility."""
        return hidden_states, attention_mask

    def sort_experts(
        self, experts: List[nn.Module], *args: Any, **kwargs: Any
    ) -> List[nn.Module]:
        """No-op to maintain API compatibility."""
        return experts

    def get_next_expert(
        self,
        hidden_states: Tensor,
        sequential_experts: List[nn.Module],
        ordered_experts: List[nn.Module],
        current_route: List[int],
        current_depth: int,
    ) -> Tuple[Tensor, List[int], int]:

        next_expert_idx = current_depth
        current_route = self._update_route(
            hidden_states, current_route, current_depth, next_expert_idx
        )

        return 0, current_route, current_depth

    def _update_route(
        self,
        hidden_states: Tensor,
        current_route: List[int],
        current_depth: int,
        next_expert_idx: int,
    ) -> List[int]:
        """Update routes used by the visualizer."""
        if self.debug:
            current_route.append(next_expert_idx)
            if (
                self.visualizer
                and not self.training
                and hidden_states.size(0) == 1  # not validation
                and current_depth > 0  # not the final layer
            ):
                previous_idx = current_route[current_depth - 1]
                self.visualizer.add_transition(previous_idx, next_expert_idx)

        return current_route

    def post_forward(self, hidden_states: Tensor, current_route: List[int]) -> None:
        """Reset the tracking of the current route through layers."""
        if self.debug:
            route = [str(r) for r in current_route]
            if not self.training and hidden_states.size(0) == 1:  # not validation
                print(f"DEBUG: inferencing through:  {' -> '.join(route)}")
