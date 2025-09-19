"""Base classes and protocols for Praxis encoders."""

from dataclasses import dataclass
from typing import Optional, Protocol

import torch


class EncoderInterface(Protocol):
    """Protocol defining the interface for encoders."""

    @property
    def sequence_length_multiplier(self) -> int:
        """
        Return the factor by which sequence length should be multiplied
        when using this encoder. For byte-level encoders, this is typically 8
        to handle UTF-8 byte sequences.
        """
        return 1


@dataclass
class EncoderOutput:
    """
    Standard output format for encoders that communicates their behavior.

    Attributes:
        logits: The output logits from the encoder
        hidden_states: Optional hidden states for downstream processing
        is_aligned: If True, logits are already aligned for loss computation (no shifting needed)
        handles_loss: If True, encoder computes its own loss internally
    """
    logits: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None
    is_aligned: bool = False
    handles_loss: bool = False

    def to(self, device):
        """Move all tensors to the specified device."""
        self.logits = self.logits.to(device)
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.to(device)
        return self