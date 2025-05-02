from functools import partial
from typing import Dict, OrderedDict, Type, TypeVar

import torch
import torch.nn as nn
from torch import Tensor

ConfigType = TypeVar("ConfigType", bound="AutoConfig")


class ProjectedEmbedding(nn.Sequential):
    """
    An embeddings module with optional projection layer and dropout.
    If embed_size differs from hidden_size, a linear projection layer is added
    to map the embeddings to the required hidden dimension.
    """

    def __init__(self, config: ConfigType) -> None:
        """
        Initialize embeddings module.

        Args:
            config: Configuration object with model parameters
        """
        layers = OrderedDict()

        # Token embeddings using embed_size
        layers["tokens"] = nn.Embedding(config.vocab_size, config.embed_size)

        # Add projection layer if dimensions differ
        if config.embed_size != config.hidden_size:
            layers["projection"] = nn.Linear(config.embed_size, config.hidden_size)

        layers["dropout"] = nn.Dropout(config.dropout)

        super().__init__(layers)
