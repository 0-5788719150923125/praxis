"""Test SMEAR integration with sequential decoder and multiple experts."""

from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import torch.nn as nn

from praxis.containers import LossContainer
from praxis.decoders import DECODER_REGISTRY


@dataclass
class MockConfig:
    """Mock configuration for testing SMEAR integration."""

    # Core configuration
    hidden_size: int = 256
    depth: int = 6
    num_experts: int = 3  # Number of experts for SMEAR to manage
    num_layers: int = 3  # Number of layer components for controllers
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

    def test_smear_with_multiple_experts(self):
        """Test that SMEAR works correctly with multiple experts."""
        config = MockConfig(num_experts=4, num_layers=4)

        # Create decoder - should use our new SMEAR logic
        decoder = DECODER_REGISTRY["sequential"](config)

        # Verify that locals were created correctly
        assert len(decoder.locals) == config.num_experts

        # All locals should point to the same expert (for SMEAR with multiple experts)
        first_expert = decoder.locals[0]
        for expert in decoder.locals[1:]:
            assert (
                expert is first_expert
            ), "All locals should point to the same SMEAR-managed expert"

        # Check that the router is SMEAR
        assert hasattr(first_expert, "router")
        assert first_expert.router.__class__.__name__ == "SMEAR"

        # Test forward pass
        batch_size = 2
        seq_len = 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        losses = LossContainer()

        output, past_kv, current_state, loss_container = decoder(
            hidden_states, losses=losses
        )

        # Verify output shape
        assert output.shape == hidden_states.shape
        assert isinstance(loss_container, LossContainer)

    def test_smear_backward_compatibility(self):
        """Test that SMEAR works with default configuration."""
        # Test with default configuration
        config = MockConfig()

        decoder = DECODER_REGISTRY["sequential"](config)
        assert len(decoder.locals) == config.num_experts

        # Test with single expert
        config = MockConfig(num_experts=1, num_layers=1)
        decoder = DECODER_REGISTRY["sequential"](config)
        assert len(decoder.locals) == config.num_experts

    def test_smear_expert_merging(self):
        """SMEAR emits one coefficient distribution PER TARGET, per example.

        This used to assert ``[batch, num_experts]`` - one distribution for a
        whole block. The router now discovers per-module targets and emits
        ``[batch, targets, num_experts]``; the old shape was the coarseness the
        rewrite removed.
        """
        config = MockConfig(num_experts=3, num_layers=3)
        decoder = DECODER_REGISTRY["sequential"](config)
        router = decoder.locals[0].router

        # One shared block, `num_experts` deviations per target - not N blocks.
        assert len(decoder.locals) == config.num_layers
        assert all(layer is decoder.locals[0] for layer in decoder.locals)
        assert router.num_experts == config.num_experts
        assert router.targets, "no merge targets discovered"

        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        merge, probs = router._coefficients(hidden_states, 0)

        assert probs.shape == (batch_size, len(router.targets), config.num_experts)
        assert torch.allclose(
            probs.sum(dim=-1), torch.ones(batch_size, len(router.targets))
        )

        # A nested ExpertBank must not be re-routed: it already routes itself.
        from praxis.routers.bank import ExpertBank

        assert not any(
            isinstance(decoder.locals[0].block.get_submodule(t.name), ExpertBank)
            for t in router.targets
            if t.name
        )

    def test_different_block_types_with_smear(self):
        """Test SMEAR with different block types."""
        # Only test block types that work with minimal config
        for block_type in ["recurrent", "gru", "min"]:
            config = MockConfig(num_experts=3, num_layers=3, block_type=block_type)

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
        config = MockConfig(num_experts=3, num_layers=3)
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
        assert not torch.allclose(
            hidden_states.grad, torch.zeros_like(hidden_states.grad)
        )

        # Check that SMEAR router has gradients
        smear_router = decoder.locals[0].router
        assert smear_router.router.weight.grad is not None
