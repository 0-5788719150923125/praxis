"""Tied weight language modeling head that reuses embedding weights."""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.heads.base import BaseHead


class TiedHead(BaseHead):
    """
    Language modeling head with tied weights.
    Instead of having a separate output projection, this head reuses
    the embedding weights for the output layer.
    """

    def __init__(self, config: Any, embedding_weight: Optional[Tensor] = None) -> None:
        """
        Initialize the tied head.

        Args:
            config: Model configuration
            embedding_weight: The embedding weight tensor to tie to
        """
        super().__init__(config)
        self.embedding_weight = embedding_weight
        
        # If embed_size != hidden_size, we need a projection before using embeddings
        self.pre_projection = None
        if config.embed_size != config.hidden_size:
            self.pre_projection = nn.Linear(config.hidden_size, config.embed_size, bias=False)

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        """
        Forward pass using tied weights.

        Args:
            hidden_states: Hidden states from the model [batch, seq_len, hidden_size]
            **kwargs: Additional arguments (ignored)

        Returns:
            Logits tensor [batch, seq_len, vocab_size]
        """
        # Project to embedding dimension if needed
        if self.pre_projection is not None:
            hidden_states = self.pre_projection(hidden_states)
        
        # Use embedding weights as output projection
        # hidden_states: [batch, seq_len, embed_size]
        # embedding_weight: [vocab_size, embed_size]
        # We need to transpose the embedding weight for the linear transformation
        logits = F.linear(hidden_states, self.embedding_weight)
        
        return logits

    @property
    def classifier(self) -> nn.Module:
        """
        Get the classifier module for loss computation.
        For tied weights, we return a dummy module since the actual
        computation uses the embedding weights directly.

        Returns:
            A module that references the embedding weights
        """
        # Create a simple wrapper that holds the embedding weight
        class TiedClassifier(nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight
        
        return TiedClassifier(self.embedding_weight)