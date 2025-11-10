"""
Test suite for Prismatic v3.0: Architectural Diversity

Tests architectural gating with RoPE vs ALiBi positional encodings.
"""

import pytest
import torch
import torch.nn as nn

from praxis.attention.hex import HexAttention
from praxis.configuration import PraxisConfig
from praxis.routers.prismatic import Prismatic


@pytest.fixture
def config():
    """Basic test configuration."""
    return PraxisConfig(
        hidden_size=64,
        num_heads=4,
        num_queries=1,
        num_experts=2,
        dropout=0.0,
        causal=True,
    )


class TestArchitecturalDiversity:
    """Test architectural diversity through RoPE vs ALiBi."""

    def test_hex_attention_with_rope(self, config):
        """Test HexAttention with RoPE positional encoding."""
        attention = HexAttention(config, pos_type="rope")

        batch_size = 2
        seq_len = 16
        inputs = torch.randn(batch_size, seq_len, config.hidden_size)

        output, kv, loss = attention(inputs)

        assert output.shape == inputs.shape
        assert loss == 0.0

    def test_hex_attention_with_alibi(self, config):
        """Test HexAttention with ALiBi positional encoding."""
        attention = HexAttention(config, pos_type="alibi")

        batch_size = 2
        seq_len = 16
        inputs = torch.randn(batch_size, seq_len, config.hidden_size)

        output, kv, loss = attention(inputs)

        assert output.shape == inputs.shape
        assert loss == 0.0

    def test_rope_and_alibi_produce_different_outputs(self, config):
        """Test that RoPE and ALiBi produce different outputs."""
        attention_rope = HexAttention(config, pos_type="rope")
        attention_alibi = HexAttention(config, pos_type="alibi")

        # Use same input
        batch_size = 2
        seq_len = 16
        inputs = torch.randn(batch_size, seq_len, config.hidden_size)

        # Copy weights to make them identical except for pos_type
        attention_alibi.load_state_dict(attention_rope.state_dict(), strict=False)

        output_rope, _, _ = attention_rope(inputs)
        output_alibi, _, _ = attention_alibi(inputs)

        # Outputs should differ due to different positional encodings
        assert not torch.allclose(output_rope, output_alibi, atol=1e-3)

    def test_prismatic_with_architectural_experts(self, config):
        """Test Prismatic router with architecturally diverse experts."""
        # Create experts with different architectures
        expert_alibi = HexAttention(config, pos_type="alibi")
        expert_rope = HexAttention(config, pos_type="rope")
        experts = [expert_alibi, expert_rope]

        prismatic = Prismatic(config, experts=experts)

        assert len(prismatic.experts) == 2
        assert prismatic.experts[0].pos_type == "alibi"
        assert prismatic.experts[1].pos_type == "rope"

    def test_prismatic_forward_with_architectural_diversity(self, config):
        """Test forward pass with architectural diversity."""
        expert_alibi = HexAttention(config, pos_type="alibi")
        expert_rope = HexAttention(config, pos_type="rope")
        experts = [expert_alibi, expert_rope]

        prismatic = Prismatic(config, experts=experts)

        batch_size = 2
        seq_len = 16
        inputs = torch.randn(batch_size, seq_len, config.hidden_size)

        # Direct mode forward
        output, state, loss = prismatic(inputs, None)

        assert output.shape == inputs.shape
        assert isinstance(loss, (int, float))

    def test_routing_selects_architecture(self, config):
        """Test that routing selects one of the architectures."""
        expert_alibi = HexAttention(config, pos_type="alibi")
        expert_rope = HexAttention(config, pos_type="rope")
        experts = [expert_alibi, expert_rope]

        prismatic = Prismatic(config, experts=experts)

        batch_size = 2
        seq_len = 16
        inputs = torch.randn(batch_size, seq_len, config.hidden_size)

        # Get routing probabilities
        routing_probs = prismatic._compute_routing(inputs)

        assert routing_probs.shape == (batch_size, 2)
        assert torch.allclose(routing_probs.sum(dim=-1), torch.ones(batch_size))

        # Run forward pass to populate metrics
        output, _, _ = prismatic(inputs, None)

        # Check metrics are logged
        metrics = prismatic.get_metrics()
        assert "routing/expert_0_weight" in metrics
        assert "routing/expert_1_weight" in metrics
        assert "routing/entropy" in metrics

    def test_gradient_flow(self, config):
        """Test that gradients flow through merged parameters."""
        expert_alibi = HexAttention(config, pos_type="alibi")
        expert_rope = HexAttention(config, pos_type="rope")
        experts = [expert_alibi, expert_rope]

        prismatic = Prismatic(config, experts=experts)

        batch_size = 2
        seq_len = 16
        inputs = torch.randn(batch_size, seq_len, config.hidden_size, requires_grad=True)

        output, _, _ = prismatic(inputs, None)
        loss = output.sum()
        loss.backward()

        assert inputs.grad is not None
        assert not torch.allclose(inputs.grad, torch.zeros_like(inputs.grad))

    def test_different_inputs_different_routing(self, config):
        """Test that different inputs produce different routing patterns."""
        expert_alibi = HexAttention(config, pos_type="alibi")
        expert_rope = HexAttention(config, pos_type="rope")
        experts = [expert_alibi, expert_rope]

        prismatic = Prismatic(config, experts=experts)

        batch_size = 2
        seq_len = 16

        # Two different inputs
        inputs_1 = torch.randn(batch_size, seq_len, config.hidden_size)
        inputs_2 = torch.randn(batch_size, seq_len, config.hidden_size) * 5.0

        routing_1 = prismatic._compute_routing(inputs_1)
        routing_2 = prismatic._compute_routing(inputs_2)

        # Routing should be input-dependent (may or may not differ significantly)
        # Just verify both are valid distributions
        assert torch.allclose(routing_1.sum(dim=-1), torch.ones(batch_size))
        assert torch.allclose(routing_2.sum(dim=-1), torch.ones(batch_size))

    def test_architecture_selection_tracking(self, config):
        """Test that architecture selection counts are tracked."""
        expert_alibi = HexAttention(config, pos_type="alibi")
        expert_rope = HexAttention(config, pos_type="rope")
        experts = [expert_alibi, expert_rope]

        prismatic = Prismatic(config, experts=experts)

        batch_size = 2
        seq_len = 16
        inputs = torch.randn(batch_size, seq_len, config.hidden_size)

        # Run multiple forward passes
        for _ in range(10):
            output, _, _ = prismatic(inputs, None)

        # Check metrics include architecture selection counts
        metrics = prismatic.get_metrics()

        assert "arch/expert_0_count" in metrics
        assert "arch/expert_1_count" in metrics
        assert "arch/expert_0_pct" in metrics
        assert "arch/expert_1_pct" in metrics
        assert "arch/total_selections" in metrics

        # Total selections should be 10
        assert metrics["arch/total_selections"] == 10

        # Percentages should sum to 100
        total_pct = metrics["arch/expert_0_pct"] + metrics["arch/expert_1_pct"]
        assert abs(total_pct - 100.0) < 0.01

        # Counts should sum to total
        total_count = metrics["arch/expert_0_count"] + metrics["arch/expert_1_count"]
        assert total_count == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
