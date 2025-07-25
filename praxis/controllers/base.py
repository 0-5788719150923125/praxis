from typing import Any, List, Optional, Sequence, Tuple, TypeVar

import torch
import torch.nn as nn
from torch import Tensor

ConfigType = TypeVar("ConfigType", bound="PretrainedConfig")

from praxis.containers.loss import LossContainer
from praxis.controllers.visualization import TransitionVisualizer


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
            TransitionVisualizer(
                save_dir="./",
                num_experts=self.num_experts,
                max_depth=self.depth,
                save_every=300,  # 15 minutes
            )
            if self.debug and allow_visualizer
            else False
        )

    def __repr__(self) -> str:
        """String representation of the module."""
        return f"{self.__class__.__name__}()"

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
        controller_state: Tensor,
        sequential_experts: List[nn.Module],
        ordered_experts: List[nn.Module],
        current_route: List[int],
        current_depth: int,
    ) -> Tuple[Tensor, Tensor, LossContainer, Optional[int]]:

        loss_container = LossContainer()
        # Use modulo to wrap around when depth exceeds number of experts
        next_expert_idx = current_depth % len(ordered_experts)
        return hidden_states, controller_state, loss_container, next_expert_idx

    def update_route(
        self,
        hidden_states: Tensor,
        current_route: List[int],
        current_depth: int,
        next_expert_idx: int,
    ) -> List[int]:
        """Update routes used by the visualizer."""
        current_route.append(next_expert_idx)
        return current_route

    def post_forward(self, hidden_states: Tensor, current_route: List[int]) -> None:
        """Reset the tracking of the current route through layers."""
        if (
            not self.debug or self.training or hidden_states.size(0) != 1
        ):  # not training or validation
            return

        route = [str(r) for r in current_route]
        print(f"DEBUG: inferencing through:  {' -> '.join(route)}")
        if self.visualizer:
            self.visualizer.add_full_route(current_route)
