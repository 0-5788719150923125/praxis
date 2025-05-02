from functools import partial
from typing import Dict, OrderedDict, Type, TypeVar

import torch
import torch.nn as nn
from torch import Tensor

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class PositionalEmbedding(nn.Sequential):
    """
    Praxis embeddings with learned positional encodings (GPT2-style).
    Uses Sequential organization of layers.
    """

    def __init__(self, config: ConfigType) -> None:
        """
        Initialize learned positional embeddings module.

        Args:
            config: Configuration object with model parameters
        """
        layers = OrderedDict(
            [
                ("wte", nn.Embedding(config.vocab_size, config.embed_size)),
                ("wpe", nn.Embedding(config.max_length, config.embed_size)),
                ("dropout", nn.Dropout(config.dropout)),
                ("reduction", nn.Linear(config.embed_size, config.hidden_size)),
            ]
        )
        super().__init__(layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through learned embeddings.

        Args:
            x: Input tensor of token IDs of shape [batch_size, seq_len]

        Returns:
            Embeddings tensor of shape [batch_size, seq_len, hidden_size]
        """
        B, T = x.shape

        # Token embeddings
        hidden_states = self.wte(x)

        # Add positional embeddings
        position_ids = torch.arange(T, device=x.device)
        positions = self.wpe(position_ids)
        hidden_states = hidden_states + positions

        # Apply remaining sequential layers
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.reduction(hidden_states)

        return hidden_states
