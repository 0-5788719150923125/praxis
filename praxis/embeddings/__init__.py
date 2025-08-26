from functools import partial
from typing import Dict, Type

import torch
import torch.nn as nn

from praxis.embeddings.positional import PositionalEmbedding
from praxis.embeddings.projected import ProjectedEmbedding

# Registry mapping architecture names to embedding classes
EMBEDDING_REGISTRY: Dict[str, Type[nn.Module]] = {
    "conv": ProjectedEmbedding,
    "gru": ProjectedEmbedding,
    "min": ProjectedEmbedding,
    "mru": ProjectedEmbedding,
    "nano": ProjectedEmbedding,
    "recurrent": ProjectedEmbedding,
    "ssm": ProjectedEmbedding,
    "transformer": ProjectedEmbedding,
}
