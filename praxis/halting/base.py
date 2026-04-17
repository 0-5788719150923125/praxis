from typing import Any, Dict, Optional, TypeVar

import torch.nn as nn
from torch import Tensor

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class BaseHalting(nn.Module):
    """No-op halting strategy. Always runs full depth, never halts early."""

    def __init__(self, config: ConfigType) -> None:
        super().__init__()
        self.num_layers = getattr(config, "num_layers", config.depth)
        self.depth = config.depth

    def get_depth(self) -> int:
        """Return the effective depth for this forward pass.

        Called once before the decoder loop begins. Subclasses may
        return a value smaller than self.depth to shorten the loop
        (e.g. randomized depth during training).
        """
        return self.depth

    def seed(
        self,
        hidden_states: Tensor,
        head: Optional[nn.Module] = None,
    ) -> None:
        """Capture a baseline from the pre-loop hidden states.

        Called once before the decoder loop begins, giving the halting
        strategy a "loop 0" reference point (e.g. raw embeddings).
        No-op during training.

        Args:
            hidden_states: Pre-loop hidden states [batch, seq_len, hidden_size]
            head: The LM head, for computing output distributions
        """
        pass

    def check(
        self,
        hidden_states: Tensor,
        current_depth: int,
        head: Optional[nn.Module] = None,
    ) -> bool:
        """Check whether the decoder should halt early.

        Called after each depth step. Returns True to signal that the
        remaining depth steps should be skipped. No-op during training.

        Args:
            hidden_states: Current hidden states [batch, seq_len, hidden_size]
            current_depth: The depth index just completed (0-based)
            head: The LM head, for computing output distributions
        """
        return False

    def get_metrics(self) -> Dict[str, Any]:
        """Return metrics for dashboards / logging. No-op by default."""
        return {}
