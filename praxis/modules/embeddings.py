from functools import partial
from typing import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig


class PraxisEmbedding(nn.Sequential):
    """
    A flexible token embeddings layer with optional learned positional embeddings.
    """

    __version__ = "0.1.0"

    def __init__(self, config: AutoConfig, learned: bool = False):

        layers = [("wte", nn.Embedding(config.vocab_size, config.num_embeds))]

        if learned:
            layers.append(
                ("wpe", nn.Embedding(config.context_length, config.num_embeds))
            )
        else:
            bottleneck_dim = config.num_dims // 2
            layers.extend(
                [
                    ("down", nn.Linear(config.num_embeds, bottleneck_dim)),
                    ("up", nn.Linear(bottleneck_dim, config.num_dims)),
                ]
            )

        layers.extend(
            [
                ("dropout", nn.Dropout(config.dropout)),
                ("linear", nn.Linear(config.num_embeds, config.num_dims)),
            ]
        )

        super().__init__(OrderedDict(layers))

    def forward(self, x: Tensor):

        hidden_states = self.wte(x)
        if hasattr(self, "wpe"):
            B, T = x.shape
            position_ids = torch.arange(T, device=x.device)
            positions = self.wpe(position_ids)
            hidden_states = hidden_states + positions
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.linear(hidden_states)
        else:
            residual = self.linear(hidden_states)
            hidden_states = self.up(self.down(hidden_states))
            hidden_states = self.dropout(hidden_states)
            hidden_states = hidden_states + residual

        return hidden_states


EMBEDDING_REGISTRY = {
    "transformer": partial(PraxisEmbedding),
    "nano": partial(PraxisEmbedding, learned=True),
    "conv": partial(PraxisEmbedding),
    "recurrent": partial(PraxisEmbedding),
}
