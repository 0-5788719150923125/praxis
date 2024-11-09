from typing import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig


class PraxisEmbedding(nn.Sequential):
    """
    A simple token embeddings layer with linear projection into a reduced dimension.
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__(
            OrderedDict(
                [
                    ("wte", nn.Embedding(config.vocab_size, config.num_embeds)),
                    ("dropout", nn.Dropout(config.dropout)),
                    ("reduction", nn.Linear(config.num_embeds, config.num_dims)),
                ]
            )
        )


class PraxisLearnedPositionEmbedding(nn.Module):
    """
    Learned positional embeddings with a uniform projection.
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.num_embeds)
        self.wpe = nn.Embedding(config.context_length, config.num_embeds)
        self.reduction = nn.Linear(config.num_embeds, config.num_dims)

    def forward(self, x: Tensor):
        B, T = x.shape
        tokens = self.wte(x)
        position = self.wpe(torch.arange(T, device=x.device))
        y = tokens + position
        return self.reduction(y)


EMBEDDING_REGISTRY = {
    "default": PraxisEmbedding,
    "nano": PraxisLearnedPositionEmbedding,
}
