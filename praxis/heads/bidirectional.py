"""Bidirectional language modeling head."""

from typing import Any, Union

import torch
import torch.nn as nn
from torch import Tensor

from praxis.heads.base import BaseHead


class BidirectionalHead(BaseHead):
    """
    Bidirectional language modeling head that supports both
    forward (next-token) and backward (previous-token) prediction.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize the bidirectional head.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Configuration for bidirectional training
        self.forward_weight = getattr(config, "forward_weight", 0.5)
        self.share_weights = getattr(config, "share_bidirectional_weights", False)
        self.training_mode = getattr(config, "bidirectional_training_mode", "simultaneous")
        
        # Forward head (next-token prediction)
        self.lm_head_forward = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Backward head (previous-token prediction)
        if self.share_weights:
            self.lm_head_backward = self.lm_head_forward
        else:
            self.lm_head_backward = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Initialize weights
        self.lm_head_forward.weight.data.normal_(mean=0.0, std=0.02)
        if not self.share_weights:
            self.lm_head_backward.weight.data.normal_(mean=0.0, std=0.02)
        
        # Track training steps for alternating mode
        self.training_step = 0

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        """
        Forward pass for bidirectional language modeling.

        Args:
            hidden_states: Hidden states from the model [batch, seq_len, hidden_size]
            **kwargs: Additional arguments (ignored for simplicity)

        Returns:
            Logits tensor [batch, seq_len, vocab_size]
        """
        # In simultaneous mode during training, compute weighted average
        if self.training and self.training_mode == "simultaneous":
            forward_logits = self.lm_head_forward(hidden_states)
            backward_logits = self.lm_head_backward(hidden_states)
            
            # Weighted average of forward and backward predictions
            return (self.forward_weight * forward_logits + 
                    (1 - self.forward_weight) * backward_logits)
        
        # In alternating mode, alternate between forward and backward
        elif self.training and self.training_mode == "alternating":
            if self.training_step % 2 == 0:
                logits = self.lm_head_forward(hidden_states)
            else:
                logits = self.lm_head_backward(hidden_states)
            self.training_step += 1
            return logits
        
        # During inference, default to forward prediction
        else:
            return self.lm_head_forward(hidden_states)

    @property
    def classifier(self) -> nn.Module:
        """
        Get the classifier module for loss computation.

        Returns:
            The appropriate linear layer based on training mode
        """
        # In alternating mode, return the classifier that was just used
        if self.training and self.training_mode == "alternating":
            # Note: training_step was already incremented, so we check the previous step
            if (self.training_step - 1) % 2 == 0:
                return self.lm_head_forward
            else:
                return self.lm_head_backward
        
        # For simultaneous mode and inference, return forward classifier
        # This is appropriate since in simultaneous mode we're learning both directions
        # and the loss function will handle the combined predictions
        return self.lm_head_forward