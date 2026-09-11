"""Sweeps over every ``embeddings`` entry."""

import pytest
import torch

from praxis import PraxisConfig, registry
from praxis.embeddings import AdditiveEmbedding
from praxis.embeddings.byte import ByteEmbedding
from praxis.embeddings.hash import HashEmbedding
from praxis.embeddings.positional import PositionalEmbedding


@pytest.mark.parametrize("key", sorted(registry.namespace("embeddings")))
def test_every_entry_builds_and_embeds(key):
    config = PraxisConfig(vocab_size=256, embed_size=32, hidden_size=64)
    embedding = registry.lookup("embeddings", key)(config)
    ids = torch.randint(0, 256, (2, 12))
    out = embedding(ids)
    assert out.shape[:2] == ids.shape
    assert torch.isfinite(out).all()


def test_mru_uses_learned_positions():
    """The one block type whose embedding adds positions of its own."""
    assert registry.lookup("embeddings", "mru") is PositionalEmbedding


def test_byte_latent_profiles_build(embedding_config):
    """The byte-latent profiles compose the expected primitives."""
    config = embedding_config()

    tok_only = registry.lookup("embeddings", "byte")(config)
    assert isinstance(tok_only, ByteEmbedding)

    tok_hash = registry.lookup("embeddings", "byte_hash")(config)
    assert isinstance(tok_hash, AdditiveEmbedding)
    kinds = [type(m) for m in tok_hash.embeddings]
    assert kinds == [ByteEmbedding, HashEmbedding]
    # The byte table is the tie source; the hash branch has no single weight.
    assert tok_hash.tie_source() is tok_hash.embeddings[0]
