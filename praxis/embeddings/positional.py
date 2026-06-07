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

    def __init__(self, config: ConfigType, encoder=None) -> None:
        """
        Initialize learned positional embeddings module.

        Args:
            config: Configuration object with model parameters
            encoder: Unused; accepted for a uniform registry signature
        """
        layers = OrderedDict(
            [
                ("wte", nn.Embedding(config.vocab_size, config.embed_size)),
                (
                    "wpe",
                    nn.Embedding(config.max_position_embeddings, config.embed_size),
                ),
                ("dropout", nn.Dropout(config.dropout)),
                ("reduction", nn.Linear(config.embed_size, config.hidden_size)),
            ]
        )
        super().__init__(layers)
        # GPT-2 style init; default N(0,1) embeddings explode tied-head logits
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.normal_(self.wpe.weight, std=0.02)

    # Cached decode passes the position offset of the new suffix.
    accepts_offset = True

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """
        Forward pass through learned embeddings.

        Args:
            x: Input tensor of token IDs of shape [batch_size, seq_len]
            offset: Position of the first token (nonzero during cached decode)

        Returns:
            Embeddings tensor of shape [batch_size, seq_len, hidden_size]
        """
        B, T = x.shape

        # Fail loudly here; out-of-range rows surface as async CUDA asserts.
        capacity = self.wpe.num_embeddings
        if offset + T > capacity:
            raise ValueError(
                f"Sequence of length {T} at offset {offset} exceeds learned "
                f"positional capacity {capacity}; raise max_position_embeddings "
                f"(sequence multipliers stretch batches up to 8x block_size)."
            )

        # Token embeddings
        hidden_states = self.wte(x)

        # Add positional embeddings
        position_ids = torch.arange(T, device=x.device) + offset
        positions = self.wpe(position_ids)
        hidden_states = hidden_states + positions

        # Apply remaining sequential layers
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.reduction(hidden_states)

        return hidden_states

    def tie_source(self) -> nn.Embedding:
        """Token table used for weight tying."""
        return self.wte
