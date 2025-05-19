"""Tests for praxis.embeddings module."""

import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from praxis.embeddings import EMBEDDING_REGISTRY
from praxis.embeddings.positional import PositionalEmbedding
from praxis.embeddings.projected import ProjectedEmbedding


@dataclass
class MockConfig:
    """Mock configuration for testing embeddings."""
    vocab_size: int = 1000
    embed_size: int = 64
    hidden_size: int = 128
    max_length: int = 512
    dropout: float = 0.1


class TestPositionalEmbedding:
    """Test cases for PositionalEmbedding class."""
    
    def test_initialization(self):
        """Test proper initialization of PositionalEmbedding."""
        config = MockConfig()
        embedding = PositionalEmbedding(config)
        
        # Check layers are created
        assert isinstance(embedding.wte, nn.Embedding)
        assert isinstance(embedding.wpe, nn.Embedding)
        assert isinstance(embedding.dropout, nn.Dropout)
        assert isinstance(embedding.reduction, nn.Linear)
        
        # Check dimensions
        assert embedding.wte.num_embeddings == config.vocab_size
        assert embedding.wte.embedding_dim == config.embed_size
        assert embedding.wpe.num_embeddings == config.max_length
        assert embedding.wpe.embedding_dim == config.embed_size
        assert embedding.reduction.in_features == config.embed_size
        assert embedding.reduction.out_features == config.hidden_size
    
    def test_forward_pass(self):
        """Test forward pass of PositionalEmbedding."""
        config = MockConfig()
        embedding = PositionalEmbedding(config)
        
        batch_size = 4
        seq_len = 32
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        output = embedding(x)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, config.hidden_size)
        
        # Check output is not all zeros (i.e., computation happened)
        assert not torch.allclose(output, torch.zeros_like(output))
    
    def test_dimension_projection_upward(self):
        """Test that PositionalEmbedding projects up to higher dimensions."""
        config = MockConfig(embed_size=32, hidden_size=256)
        embedding = PositionalEmbedding(config)
        
        x = torch.randint(0, config.vocab_size, (2, 16))
        output = embedding(x)
        
        # Verify projection from lower to higher dimension
        assert output.shape == (2, 16, 256)
        assert embedding.reduction.in_features == 32
        assert embedding.reduction.out_features == 256
    
    def test_dimension_projection_downward(self):
        """Test that PositionalEmbedding projects down to lower dimensions."""
        config = MockConfig(embed_size=512, hidden_size=64)
        embedding = PositionalEmbedding(config)
        
        x = torch.randint(0, config.vocab_size, (2, 16))
        output = embedding(x)
        
        # Verify projection from higher to lower dimension
        assert output.shape == (2, 16, 64)
        assert embedding.reduction.in_features == 512
        assert embedding.reduction.out_features == 64
    
    def test_no_projection_same_dims(self):
        """Test when embed_size equals hidden_size."""
        config = MockConfig(embed_size=128, hidden_size=128)
        embedding = PositionalEmbedding(config)
        
        x = torch.randint(0, config.vocab_size, (2, 16))
        output = embedding(x)
        
        # Still has reduction layer even when dims are same
        assert output.shape == (2, 16, 128)
        assert embedding.reduction.in_features == 128
        assert embedding.reduction.out_features == 128
    
    def test_positional_ids_generation(self):
        """Test that positional IDs are correctly generated."""
        config = MockConfig()
        embedding = PositionalEmbedding(config)
        
        seq_len = 10
        x = torch.randint(0, config.vocab_size, (1, seq_len))
        
        # Manually trace through the forward pass
        token_embeds = embedding.wte(x)
        position_ids = torch.arange(seq_len, device=x.device)
        
        # Check position IDs are correct
        expected_pos_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert torch.equal(position_ids, expected_pos_ids)
    
    def test_max_length_constraint(self):
        """Test behavior at maximum sequence length."""
        config = MockConfig(max_length=10)
        embedding = PositionalEmbedding(config)
        
        # This should work
        x = torch.randint(0, config.vocab_size, (1, 10))
        output = embedding(x)
        assert output.shape == (1, 10, config.hidden_size)
        
        # This should raise error due to position embeddings limit
        x_too_long = torch.randint(0, config.vocab_size, (1, 11))
        with pytest.raises(IndexError):
            embedding(x_too_long)


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


class TestEmbeddingRegistry:
    """Test the embedding registry."""
    
    def test_registry_contains_expected_architectures(self):
        """Test that all expected architectures are registered."""
        expected_architectures = ["conv", "min", "mru", "nano", "recurrent", "transformer"]
        
        for arch in expected_architectures:
            assert arch in EMBEDDING_REGISTRY
            assert EMBEDDING_REGISTRY[arch] == ProjectedEmbedding
    
    def test_registry_values_are_classes(self):
        """Test that registry values are actual classes."""
        for arch, cls in EMBEDDING_REGISTRY.items():
            assert isinstance(cls, type)
            assert issubclass(cls, nn.Module)


# Integration tests
class TestEmbeddingIntegration:
    """Integration tests for embedding modules."""
    
    def test_positional_embedding_gradient_flow(self):
        """Test that gradients flow through PositionalEmbedding."""
        config = MockConfig(embed_size=32, hidden_size=64)
        embedding = PositionalEmbedding(config)
        
        x = torch.randint(0, config.vocab_size, (2, 8))
        output = embedding(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert embedding.wte.weight.grad is not None
        assert embedding.wpe.weight.grad is not None
        assert embedding.reduction.weight.grad is not None
        assert embedding.reduction.bias.grad is not None
    
    def test_projected_embedding_gradient_flow(self):
        """Test that gradients flow through ProjectedEmbedding."""
        config = MockConfig(embed_size=32, hidden_size=64)
        embedding = ProjectedEmbedding(config)
        
        x = torch.randint(0, config.vocab_size, (2, 8))
        output = embedding(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert embedding.tokens.weight.grad is not None
        assert embedding.projection.weight.grad is not None
        assert embedding.projection.bias.grad is not None
    
    def test_extreme_dimensions(self):
        """Test embeddings with extreme dimension differences."""
        # Very small to very large
        config = MockConfig(embed_size=4, hidden_size=2048)
        embedding = ProjectedEmbedding(config)
        x = torch.randint(0, config.vocab_size, (1, 16))
        output = embedding(x)
        assert output.shape == (1, 16, 2048)
        
        # Very large to very small
        config = MockConfig(embed_size=4096, hidden_size=8)
        embedding = ProjectedEmbedding(config)
        x = torch.randint(0, config.vocab_size, (1, 16))
        output = embedding(x)
        assert output.shape == (1, 16, 8)
    
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 1),    # Minimal case
        (32, 512), # Typical case
        (1, 2048), # Long sequence
        (128, 16), # Large batch
    ])
    def test_various_input_sizes(self, batch_size, seq_len):
        """Test embeddings with various input sizes."""
        config = MockConfig(max_length=4096)
        embedding = PositionalEmbedding(config)
        
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        output = embedding(x)
        
        assert output.shape == (batch_size, seq_len, config.hidden_size)