from dataclasses import dataclass

import pytest


@dataclass
class _EmbeddingConfig:
    """The fields an embedding module reads."""

    vocab_size: int = 1000
    embed_size: int = 64
    hidden_size: int = 128
    max_position_embeddings: int = 512
    dropout: float = 0.1


@pytest.fixture
def embedding_config():
    """``embedding_config(**fields)`` builds a small embedding config."""
    return _EmbeddingConfig
