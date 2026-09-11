from functools import partial
from typing import List, Tuple

import torch.nn as nn

from praxis import registry
from praxis.embeddings.byte import ByteEmbedding
from praxis.embeddings.composite import AdditiveEmbedding
from praxis.embeddings.hash import HashEmbedding
from praxis.embeddings.positional import PositionalEmbedding
from praxis.embeddings.projected import ProjectedEmbedding
from praxis.registry import Entry


def _compose(specs: List[Tuple[str, dict]], config, encoder=None) -> nn.Module:
    """Build and sum embedding primitives named by registry key.

    ``specs`` is a list of ``(registry_key, kwargs)``. This is how byte-latent
    embedding profiles are assembled from the low-level primitives.
    """
    mods = [
        registry.lookup("embeddings", key)(config, encoder=encoder, **kwargs)
        for key, kwargs in specs
    ]
    return mods[0] if len(mods) == 1 else AdditiveEmbedding(mods)


registry.declare(
    "embeddings",
    title="Token embeddings",
    doc=(
        "Input embedding layers, each called as ``(config, encoder=None)``. Three "
        "kinds of entries coexist: block-type defaults for standard models, keyed by "
        "``--block-type``; low-level byte-latent primitives (``tok``, ``hash``); and "
        "byte-latent profiles composed from those primitives, referenced by key from "
        "encoder profiles. A config-level ``embeddings`` key overrides either default."
    ),
    entries={
        "conv": ProjectedEmbedding,
        "gru": ProjectedEmbedding,
        "min": ProjectedEmbedding,
        "mru": PositionalEmbedding,
        "positional": Entry(
            PositionalEmbedding,
            (
                "Learned absolute position embeddings (GPT2-style), for choosing "
                "explicitly through ``config.embeddings`` rather than by block type."
            ),
        ),
        "nano": ProjectedEmbedding,
        "recurrent": ProjectedEmbedding,
        "ssm": ProjectedEmbedding,
        "transformer": ProjectedEmbedding,
        "wavelet": ProjectedEmbedding,
        "tok": ByteEmbedding,
        "hash": HashEmbedding,
        "byte": Entry(
            partial(_compose, [("tok", {})]),
            "Byte-latent profile: the per-byte token table alone.",
        ),
        "byte_hash": Entry(
            partial(
                _compose,
                [("tok", {}), ("hash", {"group_sizes": [3, 4, 5], "functions": 1})],
            ),
            (
                "Byte-latent profile: the per-byte token table plus one hash function "
                "over byte 3-, 4- and 5-grams, summed."
            ),
        ),
        "byte_multihash": Entry(
            partial(
                _compose,
                [("tok", {}), ("hash", {"group_sizes": [3, 4, 5], "functions": 4})],
            ),
            (
                "byte_hash with four independent hashes per window instead of one, so "
                "an n-gram's code is the 4-tuple of buckets and is ambiguous only when "
                "all four collide at once. How many buckets is set by "
                "``--hash-buckets``, not here. See HashEmbedding for the measurement "
                "that motivates both."
            ),
        ),
    },
)
