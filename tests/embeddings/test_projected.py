"""Tests for praxis/embeddings/projected.py."""

import pytest
import torch

from praxis.embeddings.projected import ProjectedEmbedding


@pytest.mark.parametrize("embed,hidden", [(64, 256), (128, 128), (32, 128), (1024, 32)])
def test_projection_exists_iff_the_widths_differ(embedding_config, embed, hidden):
    config = embedding_config(embed_size=embed, hidden_size=hidden)
    embedding = ProjectedEmbedding(config)
    assert embedding.tokens.weight.shape == (config.vocab_size, embed)
    assert "dropout" in embedding._modules
    assert ("projection" in embedding._modules) is (embed != hidden)
    if embed != hidden:
        assert embedding.projection.in_features == embed
        assert embedding.projection.out_features == hidden

    out = embedding(torch.randint(0, config.vocab_size, (2, 16)))
    assert out.shape == (2, 16, hidden)
    out.sum().backward()
    assert embedding.tokens.weight.grad is not None
    if embed != hidden:
        assert embedding.projection.weight.grad is not None


def test_dropout_is_stochastic_in_training_and_off_in_eval(embedding_config):
    config = embedding_config(dropout=0.5)
    embedding = ProjectedEmbedding(config)
    x = torch.randint(0, config.vocab_size, (10, 20))

    embedding.train()
    first = embedding(x)
    assert not all(torch.allclose(first, embedding(x)) for _ in range(4))

    embedding.eval()
    assert torch.equal(embedding(x), embedding(x))
