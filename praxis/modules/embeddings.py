from typing import OrderedDict

import torch.nn as nn

from praxis import PraxisConfig


class PraxisEmbedding(nn.Sequential):
    """
    A simple token embeddings layer with a linear projection into
    a reduced dimension.
    """

    def __init__(self, config: PraxisConfig):
        super().__init__(
            OrderedDict(
                [
                    ("wte", nn.Embedding(config.vocab_size, config.n_emb)),
                    ("reduction", nn.Linear(config.n_emb, config.n_dim, bias=False)),
                ]
            )
        )
