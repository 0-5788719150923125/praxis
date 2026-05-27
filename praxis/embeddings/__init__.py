from functools import partial
from typing import Callable, Dict, List, Tuple

import torch.nn as nn

from praxis.embeddings.byte import ByteEmbedding
from praxis.embeddings.composite import AdditiveEmbedding
from praxis.embeddings.hash import HashEmbedding
from praxis.embeddings.positional import PositionalEmbedding
from praxis.embeddings.projected import ProjectedEmbedding


def _compose(specs: List[Tuple[str, dict]], config, encoder=None) -> nn.Module:
    """Build and sum embedding primitives named by registry key.

    ``specs`` is a list of ``(registry_key, kwargs)``. This is how byte-latent
    embedding profiles are assembled from the low-level primitives.
    """
    mods = [
        EMBEDDING_REGISTRY[key](config, encoder=encoder, **kwargs)
        for key, kwargs in specs
    ]
    return mods[0] if len(mods) == 1 else AdditiveEmbedding(mods)


# Embedding constructors, each called as ``(config, encoder=None)``. Three
# kinds of entries coexist, all returning an nn.Module:
#   - block-type primitives for standard models (keyed by block_type)
#   - byte-latent primitives (low-level, composable)
#   - byte-latent profiles (composed; referenced by encoder profiles via key)
EMBEDDING_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    # Standard models, keyed by block_type.
    "conv": ProjectedEmbedding,
    "gru": ProjectedEmbedding,
    "min": ProjectedEmbedding,
    "mru": PositionalEmbedding,
    "nano": ProjectedEmbedding,
    "recurrent": ProjectedEmbedding,
    "ssm": ProjectedEmbedding,
    "transformer": ProjectedEmbedding,
    "wavelet": ProjectedEmbedding,
    # Byte-latent primitives.
    "tok": ByteEmbedding,
    "hash": HashEmbedding,
    # Byte-latent profiles, referenced by ENCODER_REGISTRY profiles by key.
    "byte": partial(_compose, [("tok", {})]),
    "byte_hash": partial(
        _compose, [("tok", {}), ("hash", {"group_sizes": [3, 4, 5], "functions": 1})]
    ),
}
