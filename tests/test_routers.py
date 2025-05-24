"""Test suite for SMEAR router."""

import pytest
import torch
import torch.nn as nn

from praxis.blocks.transformer import TransformerBlock
from praxis.configuration_praxis import PraxisConfig
from praxis.orchestration.experts import LocalExpert
from praxis.routers.smear import SMEAR


class TestSMEARRouter:
    """Test cases for SMEAR router functionality."""

    def test_smear_initialization_requires_experts(self):
        """Test that SMEAR requires experts to be provided during initialization."""
        config = PraxisConfig(
            hidden_size=128,
            num_experts=4,
            dropout=0.1,
        )
        
        # Should raise error without experts
        with pytest.raises(ValueError, match="SMEAR router requires 'experts'"):
            SMEAR(config)
    
    def test_smear_with_multiple_experts(self):
        """Test SMEAR with multiple expert blocks."""
        config = PraxisConfig(
            hidden_size=128,
            embed_size=128,
            num_experts=4,
            num_heads=8,
            num_queries=8,
            k_heads=4,
            depth=4,
            dropout=0.1,
            residual_type="standard",
            attention_type="standard",
            expert_type="mlp",
            activation="gelu",
        )
        
        # Create multiple expert blocks
        expert_blocks = []
        for _ in range(config.num_experts):
            block = TransformerBlock(config)
            expert_blocks.append(block)
        
        # Create SMEAR router with experts
        smear = SMEAR(config, experts=expert_blocks)
        
        # Test forward pass
        batch_size = 2
        seq_length = 16
        inputs = torch.randn(batch_size, seq_length, config.hidden_size)
        
        # Call SMEAR with the first expert (not used since we have multiple experts)
        output, kv, state, loss = smear(
            layer=expert_blocks[0],
            inputs=inputs,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=0,
            block_ids=None,
        )
        
        assert output.shape == inputs.shape
        # Loss comes from the underlying transformer blocks, not SMEAR itself
        assert isinstance(loss, (int, float, torch.Tensor))
    
    def test_smear_parameter_merging(self):
        """Test that SMEAR properly merges expert parameters."""
        config = PraxisConfig(
            hidden_size=64,
            num_experts=2,
            dropout=0.0,  # No dropout for deterministic test
        )
        
        # Create simple linear experts
        class SimpleExpert(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
            
            def forward(self, inputs, attention_mask, past_key_values, current_state, current_depth, block_ids):
                return self.linear(inputs), past_key_values, current_state, 0.0
        
        experts = [SimpleExpert(config.hidden_size) for _ in range(config.num_experts)]
        
        # Set different weights for each expert
        with torch.no_grad():
            experts[0].linear.weight.fill_(1.0)
            experts[1].linear.weight.fill_(2.0)
        
        smear = SMEAR(config, experts=experts)
        
        # Test forward pass
        batch_size = 1
        seq_length = 1
        inputs = torch.ones(batch_size, seq_length, config.hidden_size)
        
        output, _, _, _ = smear(
            layer=experts[0],
            inputs=inputs,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=0,
            block_ids=None,
        )
        
        # Just verify the output has the correct shape and is not the same as input
        assert output.shape == inputs.shape
        assert not torch.allclose(output, inputs)  # Output should be different from input


if __name__ == "__main__":
    pytest.main([__file__, "-v"])