"""Tests for praxis.embeddings module."""

from dataclasses import dataclass

import pytest
import torch

from praxis.embeddings.projected import ProjectedEmbedding


@dataclass
class MockConfig:
    """Mock configuration for testing embeddings."""

    vocab_size: int = 1000
    embed_size: int = 64
    hidden_size: int = 128
    max_position_embeddings: int = 512
    dropout: float = 0.1


class TestProjectedEmbedding:
    """Test cases for ProjectedEmbedding class."""

    def test_initialization_with_projection(self):
        """Test initialization when projection is needed."""
        config = MockConfig(embed_size=64, hidden_size=256)
        embedding = ProjectedEmbedding(config)

        # Check layers
        assert "tokens" in embedding._modules
        assert "projection" in embedding._modules
        assert "dropout" in embedding._modules

        # Check dimensions
        assert embedding.tokens.num_embeddings == config.vocab_size
        assert embedding.tokens.embedding_dim == config.embed_size
        assert embedding.projection.in_features == config.embed_size
        assert embedding.projection.out_features == config.hidden_size

    def test_initialization_without_projection(self):
        """Test initialization when no projection is needed."""
        config = MockConfig(embed_size=128, hidden_size=128)
        embedding = ProjectedEmbedding(config)

        # Check layers
        assert "tokens" in embedding._modules
        assert "projection" not in embedding._modules
        assert "dropout" in embedding._modules

        # Check dimensions
        assert embedding.tokens.num_embeddings == config.vocab_size
        assert embedding.tokens.embedding_dim == config.embed_size

    def test_forward_pass_with_projection(self):
        """Test forward pass when projection is needed."""
        config = MockConfig(embed_size=32, hidden_size=128)
        embedding = ProjectedEmbedding(config)

        batch_size = 2
        seq_len = 16
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        output = embedding(x)

        # Check output shape matches hidden_size
        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_forward_pass_without_projection(self):
        """Test forward pass when no projection is needed."""
        config = MockConfig(embed_size=256, hidden_size=256)
        embedding = ProjectedEmbedding(config)

        batch_size = 2
        seq_len = 16
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        output = embedding(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_dimension_projection_upward(self):
        """Test projection from lower to higher dimensions."""
        config = MockConfig(embed_size=16, hidden_size=512)
        embedding = ProjectedEmbedding(config)

        x = torch.randint(0, config.vocab_size, (1, 8))
        output = embedding(x)

        assert output.shape == (1, 8, 512)
        assert embedding.projection is not None
        assert embedding.projection.in_features == 16
        assert embedding.projection.out_features == 512

    def test_dimension_projection_downward(self):
        """Test projection from higher to lower dimensions."""
        config = MockConfig(embed_size=1024, hidden_size=32)
        embedding = ProjectedEmbedding(config)

        x = torch.randint(0, config.vocab_size, (1, 8))
        output = embedding(x)

        assert output.shape == (1, 8, 32)
        assert embedding.projection is not None
        assert embedding.projection.in_features == 1024
        assert embedding.projection.out_features == 32

    def test_dropout_applied(self):
        """Test that dropout is applied during training."""
        config = MockConfig(dropout=0.5)
        embedding = ProjectedEmbedding(config)
        embedding.train()  # Set to training mode

        x = torch.randint(0, config.vocab_size, (10, 20))

        # Run multiple forward passes
        outputs = []
        for _ in range(5):
            output = embedding(x)
            outputs.append(output)

        # In training mode with high dropout, outputs should differ
        all_same = all(torch.allclose(outputs[0], out) for out in outputs[1:])
        assert not all_same

    def test_deterministic_in_eval_mode(self):
        """Test that outputs are deterministic in eval mode."""
        config = MockConfig(dropout=0.5)
        embedding = ProjectedEmbedding(config)
        embedding.eval()  # Set to evaluation mode

        x = torch.randint(0, config.vocab_size, (2, 8))

        # Run multiple forward passes
        output1 = embedding(x)
        output2 = embedding(x)

        # In eval mode, outputs should be identical
        assert torch.allclose(output1, output2)
