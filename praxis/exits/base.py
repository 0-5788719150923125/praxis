from typing import Optional, TypeVar

import torch.nn as nn
from torch import Tensor

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class BaseExit(nn.Module):
    """No-op exit strategy. Never signals early exit."""

    def __init__(self, config: ConfigType) -> None:
        super().__init__()
        self.num_layers = getattr(config, "num_layers", config.depth)
        self.depth = config.depth

    def reset(self) -> None:
        """Called at the start of each forward pass to clear state."""
        pass

    def seed(
        self,
        hidden_states: Tensor,
        head: Optional[nn.Module] = None,
    ) -> None:
        """Capture a baseline from the pre-loop hidden states.

        Called once before the decoder loop begins, giving the exit
        strategy a "loop 0" reference point (e.g. raw embeddings).

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
        """Check whether the decoder should exit early.

        Called after each depth step during inference. Returns True to
        signal that the remaining depth steps should be skipped.

        Args:
            hidden_states: Current hidden states [batch, seq_len, hidden_size]
            current_depth: The depth index just completed (0-based)
            head: The LM head, for computing output distributions
        """
        return False
