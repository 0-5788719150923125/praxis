from functools import partial
from typing import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor


class PraxisEmbeddings(nn.Sequential):
    """
    An embeddings module with optional projection layer and dropout.
    If embed_size differs from hidden_size, a linear projection layer is added
    to map the embeddings to the required hidden dimension.
    """

    def __init__(self, config: "AutoConfig"):
        layers = OrderedDict()

        # Token embeddings using embed_size
        layers["tokens"] = nn.Embedding(config.vocab_size, config.embed_size)

        # Add projection layer if dimensions differ
        if config.embed_size != config.hidden_size:
            layers["projection"] = nn.Linear(config.embed_size, config.hidden_size)

        layers["dropout"] = nn.Dropout(config.dropout)

        super().__init__(layers)


class PraxisLearnedEmbeddings(nn.Sequential):
    """
    Praxis embeddings with learned positional encodings (GPT2-style).
    Uses Sequential organization of layers.
    """

    def __init__(self, config: "AutoConfig"):
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

    def __init__(self, config: "AutoConfig"):
        bottleneck_dim = config.hidden_size // 2
        layers = OrderedDict(
            [
                ("tokens", nn.Embedding(config.vocab_size, config.embed_size)),
                ("residual", nn.Linear(config.embed_size, config.hidden_size)),
                ("compress", nn.Linear(config.embed_size, bottleneck_dim)),
                ("decompress", nn.Linear(bottleneck_dim, config.hidden_size)),
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
    "conv": PraxisFactorizedEmbeddings,
    "min": PraxisEmbeddings,
    "mru": PraxisEmbeddings,
    "nano": PraxisLearnedEmbeddings,
    "recurrent": PraxisFactorizedEmbeddings,
    "transformer": PraxisEmbeddings,
}
