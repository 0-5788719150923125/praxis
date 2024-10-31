from typing import OrderedDict

import torch.nn as nn
from transformers import AutoConfig


class PraxisEmbedding(nn.Sequential):
    """
    A simple token embeddings layer with linear projection into a reduced dimension.
    """

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
