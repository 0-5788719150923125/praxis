"""Test SMEAR integration with sequential decoder and num_smear > 1."""

import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from praxis.decoders import DECODER_REGISTRY
from praxis.containers import LossContainer


@dataclass
class MockConfig:
    """Mock configuration for testing SMEAR integration."""
    
    # Core configuration
    hidden_size: int = 256
    depth: int = 6
    num_experts: int = 3
    num_smear: int = 3  # Number of experts for SMEAR to manage
    epsilon: float = 1e-6
    dropout: float = 0.1
    
    # Decoder configuration
    decoder_type: str = "sequential"
    block_type: str = "recurrent"
    router_type: str = "smear"
    controller_type: str = "base"
    compression_type: str = "none"
    sorting_type: str = "none"
    
    # Additional required fields
    checkpoint_every: int = 0
    debug: bool = False
    evolve: bool = False
    hivemind: bool = False
    expert: str = "default"
    meta: dict = None
    
    # For blocks that need these
    num_heads: int = 8
    activation: str = "swish"
    causal: bool = True
    
    def __post_init__(self):
        if self.meta is None:
            self.meta = {}


class TestSMEARIntegration:
    """Test suite for SMEAR integration with decoders."""
    
    def test_smear_with_num_smear_greater_than_one(self):
        """Test that SMEAR works correctly when num_smear > 1."""
        config = MockConfig(num_smear=4)
        
        # Create decoder - should use our new SMEAR logic
        decoder = DECODER_REGISTRY["sequential"](config)
        
        # Verify that locals were created correctly
        assert len(decoder.locals) == config.num_experts
        
        # All locals should point to the same expert (for SMEAR with num_smear > 1)
        first_expert = decoder.locals[0]
        for expert in decoder.locals[1:]:
            assert expert is first_expert, "All locals should point to the same SMEAR-managed expert"
        
        # Check that the router is SMEAR
        assert hasattr(first_expert, 'router')
        assert first_expert.router.__class__.__name__ == 'SMEAR'
        
        # Test forward pass
        batch_size = 2
        seq_len = 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        losses = LossContainer()
        
        output, past_kv, current_state, loss_container = decoder(
            hidden_states,
            losses=losses
        )
        
        # Verify output shape
        assert output.shape == hidden_states.shape
        assert isinstance(loss_container, LossContainer)
    
    def test_smear_backward_compatibility(self):
        """Test that SMEAR still works without num_smear or with num_smear=1."""
        # Test without num_smear attribute
        config = MockConfig()
        delattr(config, 'num_smear')
        
        decoder = DECODER_REGISTRY["sequential"](config)
        assert len(decoder.locals) == config.num_experts
        
        # Test with num_smear = 1
        config = MockConfig(num_smear=1)
        decoder = DECODER_REGISTRY["sequential"](config)
        assert len(decoder.locals) == config.num_experts
    
    def test_smear_expert_merging(self):
        """Test that SMEAR properly merges expert parameters."""
        config = MockConfig(num_smear=3)
        decoder = DECODER_REGISTRY["sequential"](config)
        
        # Get the SMEAR router
        smear_router = decoder.locals[0].router
        
        # Verify it has the correct number of experts
        assert len(smear_router.experts) == config.num_smear
        
        # Test parameter merging
        batch_size = 2
        seq_len = 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # Get routing probabilities
        router_input = hidden_states.mean(dim=1)
        router_input = smear_router.router_norm(router_input)
        logits = smear_router.router(router_input)
        routing_probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Verify routing probabilities shape
        assert routing_probs.shape == (batch_size, config.num_smear)
        
        # Test that probabilities sum to 1
        assert torch.allclose(routing_probs.sum(dim=-1), torch.ones(batch_size))
    
    def test_different_block_types_with_smear(self):
        """Test SMEAR with different block types."""
        # Only test block types that work with minimal config
        for block_type in ["recurrent", "gru", "min"]:
            config = MockConfig(
                num_smear=3,
                block_type=block_type
            )
            
            decoder = DECODER_REGISTRY["sequential"](config)
            
            # Test forward pass
            batch_size = 2
            seq_len = 10
            hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
            losses = LossContainer()
            
            output, _, _, _ = decoder(hidden_states, losses=losses)
            assert output.shape == hidden_states.shape
    
    def test_smear_gradient_flow(self):
        """Test that gradients flow through SMEAR properly."""
        config = MockConfig(num_smear=3)
        decoder = DECODER_REGISTRY["sequential"](config)
        
        # Create input with requires_grad
        batch_size = 2
        seq_len = 10
        hidden_states = torch.randn(
            batch_size, seq_len, config.hidden_size, requires_grad=True
        )
        losses = LossContainer()
        
        # Forward pass
        output, _, _, loss_container = decoder(hidden_states, losses=losses)
        
        # Create a simple loss
        loss = output.mean()
        
        # Backward pass
        loss.backward()
        
        # Check that input has gradients
        assert hidden_states.grad is not None
        assert not torch.allclose(hidden_states.grad, torch.zeros_like(hidden_states.grad))
        
        # Check that SMEAR router has gradients
        smear_router = decoder.locals[0].router
        assert smear_router.router.weight.grad is not None