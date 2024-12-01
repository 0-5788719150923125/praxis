from functools import partial
from typing import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig


class PraxisLearnedEmbeddings(nn.Sequential):
    """
    Praxis embeddings with learned positional encodings (GPT2-style).
    Uses Sequential organization of layers.
    """

    def __init__(self, config: AutoConfig):
        layers = OrderedDict(
            [
                ("wte", nn.Embedding(config.vocab_size, config.num_embeds)),
                ("wpe", nn.Embedding(config.context_length, config.num_embeds)),
                ("dropout", nn.Dropout(config.dropout)),
                ("reduction", nn.Linear(config.num_embeds, config.num_dims)),
            ]
        )
        super().__init__(layers)

    def forward(self, x: Tensor) -> Tensor:
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


class PraxisFactorizedEmbeddings(nn.Sequential):
    """
    Praxis embeddings using factorized bottleneck architecture.
    Uses Sequential organization of layers.
    """

    def __init__(self, config: AutoConfig):
        bottleneck_dim = config.num_dims // 2
        layers = OrderedDict(
            [
                ("tokens", nn.Embedding(config.vocab_size, config.num_embeds)),
                ("residual", nn.Linear(config.num_embeds, config.num_dims)),
                ("compress", nn.Linear(config.num_embeds, bottleneck_dim)),
                ("decompress", nn.Linear(bottleneck_dim, config.num_dims)),
                ("dropout", nn.Dropout(config.dropout)),
            ]
        )
        super().__init__(layers)

    def forward(self, x: Tensor) -> Tensor:
        # Token embeddings
        hidden_states = self.tokens(x)

        # Store residual
        residual = self.residual(hidden_states)

        # Apply factorized transformation
        compressed_states = self.dropout(self.compress(hidden_states))
        decompressed_states = self.decompress(compressed_states)

        # Add residual connection
        hidden_states = decompressed_states + residual

        return hidden_states


# Registry mapping architecture names to embedding classes
EMBEDDING_REGISTRY = {
    "transformer": PraxisFactorizedEmbeddings,
    "nano": PraxisLearnedEmbeddings,
    "conv": PraxisFactorizedEmbeddings,
    "recurrent": PraxisFactorizedEmbeddings,
}
