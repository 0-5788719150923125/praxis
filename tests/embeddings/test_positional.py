"""Tests for praxis/embeddings/positional.py (learned GPT-2 style positions)."""

import pytest
import torch

from praxis import PraxisConfig
from praxis.embeddings.positional import PositionalEmbedding


@pytest.mark.parametrize("embed,hidden", [(64, 128), (32, 256), (512, 64), (128, 128)])
def test_positional_embedding_projects_to_hidden_and_trains(
    embedding_config, embed, hidden
):
    config = embedding_config(embed_size=embed, hidden_size=hidden)
    embedding = PositionalEmbedding(config)
    assert embedding.wte.weight.shape == (config.vocab_size, embed)
    assert embedding.wpe.weight.shape == (config.max_position_embeddings, embed)
    assert (embedding.reduction.in_features, embedding.reduction.out_features) == (
        embed,
        hidden,
    )

    out = embedding(torch.randint(0, config.vocab_size, (4, 32)))
    assert out.shape == (4, 32, hidden)
    out.sum().backward()
    for layer in (embedding.wte, embedding.wpe, embedding.reduction):
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0


def test_max_length_constraint(embedding_config):
    """Capacity is checked on the absolute position, so the cached-decode offset
    counts toward it."""
    config = embedding_config(max_position_embeddings=10)
    embedding = PositionalEmbedding(config)

    assert embedding(torch.randint(0, config.vocab_size, (1, 10))).shape == (
        1,
        10,
        config.hidden_size,
    )
    with pytest.raises(ValueError, match="positional capacity"):
        embedding(torch.randint(0, config.vocab_size, (1, 11)))

    suffix = torch.randint(0, config.vocab_size, (1, 3))
    assert embedding(suffix, offset=7).shape == (1, 3, config.hidden_size)
    with pytest.raises(ValueError, match="positional capacity"):
        embedding(suffix, offset=8)


def test_positional_embedding_offset():
    """A suffix embedded at its offset matches that slice of the full prompt."""
    cfg = PraxisConfig(
        vocab_size=50, hidden_size=32, embed_size=32, device="cpu", dropout=0.0
    )
    emb = PositionalEmbedding(cfg).eval()
    ids = torch.randint(0, 50, (1, 6))
    full = emb(ids)
    suffix = emb(ids[:, 4:], offset=4)
    torch.testing.assert_close(suffix, full[:, 4:])
