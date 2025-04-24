from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class BaseController(nn.Module):
    """
    A no-op controller.
    """

    def __init__(self, config: "AutoConfig", *args, **kwargs):
        super().__init__()

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

    def shuffle_experts(
        self, experts: List[nn.Module], *args, **kwargs
    ) -> List[nn.Module]:
        """No-op to maintain API compatibility."""
        return experts

    def get_next_expert(
        self,
        hidden_states: torch.Tensor,
        current_depth: int,
        original_experts: List[nn.Module],
        current_experts: List[nn.Module],
    ) -> Tuple[torch.Tensor, Optional[int]]:
        return 0, current_depth

    def reset_route(self, hidden_states):
        """No-op to maintain API compatibility."""
        pass
