"""Standard forward (causal) language modeling head."""

from typing import Any

import torch.nn as nn
from torch import Tensor

from praxis.heads.base import BaseHead


class ForwardHead(BaseHead):
    """
    Standard forward (causal) language modeling head.
    This is the traditional next-token prediction head.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize the forward head.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Initialize weights
        self.lm_head.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        """
        Forward pass for next-token prediction.

        Args:
            hidden_states: Hidden states from the model [batch, seq_len, hidden_size]
            **kwargs: Additional arguments (ignored)

        Returns:
            Logits tensor [batch, seq_len, vocab_size]
        """
        # Project hidden states to vocabulary
        return self.lm_head(hidden_states)

    @property
    def classifier(self) -> nn.Module:
        """
        Get the classifier module for loss computation.

        Returns:
            The linear layer used as classifier
        """
        return self.lm_head