from typing import Any, Dict, TypeVar

import torch.nn as nn
from torch import Tensor

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class BaseHalting(nn.Module):
    """No-op halting strategy. Always runs full depth, never halts early."""

    def __init__(self, config: ConfigType) -> None:
        super().__init__()
        self.num_layers = getattr(config, "num_layers", config.depth)
        self.depth = config.depth
        # When False, subclasses must not fold this forward into their reported
        # distributions. The model clears it while an encoder is in its codec
        # pretraining ("preflight") stage, where the decoder runs but its loop
        # count is not yet a meaningful early-exit signal.
        self.record_metrics = True

    def get_depth(self) -> int:
        """Return the effective depth for this forward pass."""
        return self.depth

    def seed(self, hidden_states: Tensor) -> None:
        """Capture a baseline from the pre-loop hidden states."""
        pass

    def check(self, hidden_states: Tensor, current_depth: int) -> bool:
        """Check whether the decoder should halt early."""
        return False

    def get_metrics(self) -> Dict[str, Any]:
        return {}
