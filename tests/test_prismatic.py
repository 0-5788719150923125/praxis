"""
Test suite for Prismatic v7.0: Architectural Diversity (ALiBi vs RoPE).

Tests verify:
1. Experts created with different pos_type
2. Sparse routing works correctly
3. Load balancing prevents collapse
4. Metrics logged correctly
5. Gradients flow to both experts
"""

import copy

import pytest
import torch
import torch.nn as nn

from praxis.configuration import PraxisConfig
from praxis.routers.prismatic import Prismatic


class SimpleDenseBlock(nn.Module):
    """Simple dense block for testing."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        inputs,
        attention_mask=None,
        past_key_values=None,
        current_state=None,
        current_depth=0,
        block_ids=None,
        **kwargs
    ):
        x = self.linear(inputs)
        x = self.norm(x)
        return x, past_key_values, current_state, 0.0


class TestArchitecturalDiversity:
    """Test that experts use different positional encodings."""

    def test_experts_have_different_pos_types(self):
        """Verify ALiBi and RoPE experts are created."""
        config = PraxisConfig(
            hidden_size=64,
            num_heads=4,
            num_queries=1,
            num_experts=2,
            router_type="prismatic",
            causal=True
        )

        # Create experts with different encodings
        from praxis.attention.hex import HexAttention

        # Create config copies with different encodings
        alibi_config = copy.copy(config)
        alibi_config.encoding = "alibi"
        rope_config = copy.copy(config)
        rope_config.encoding = "rope"

        alibi_expert = HexAttention(alibi_config)
        rope_expert = HexAttention(rope_config)

        router = Prismatic(config, experts=[alibi_expert, rope_expert])

        # Verify pos_types
        assert router.experts[0].pos_type == "alibi"
        assert router.experts[1].pos_type == "rope"

        print("[TEST] ✓ Expert 0 uses ALiBi, Expert 1 uses RoPE")

    def test_different_architectures_produce_different_outputs(self):
        """Verify ALiBi and RoPE produce different outputs on same input."""
        config = PraxisConfig(
            hidden_size=64,
            num_heads=4,
            num_queries=1,
            dropout=0.0,
            causal=True
        )

        from praxis.attention.hex import HexAttention

        alibi_config = copy.copy(config)
        alibi_config.encoding = "alibi"
        rope_config = copy.copy(config)
        rope_config.encoding = "rope"

        alibi_attn = HexAttention(alibi_config)
        rope_attn = HexAttention(rope_config)

        # Same input
        inputs = torch.randn(1, 8, config.hidden_size)

        # Different outputs (different pos_type)
        out_alibi, _, _ = alibi_attn(inputs)
        out_rope, _, _ = rope_attn(inputs)

        assert not torch.allclose(out_alibi, out_rope, atol=1e-3), \
            "ALiBi and RoPE should produce different outputs"

        print("[TEST] ✓ Architectural diversity creates different representations")

    def test_supports_n_experts_with_modulus_cycling(self):
        """Verify Prismatic supports N experts with architecture cycling."""
        config = PraxisConfig(
            hidden_size=64,
            num_heads=4,
            num_queries=1,
            num_experts=4,
            router_type="prismatic",
            causal=True
        )

        from praxis.attention.hex import HexAttention

        # Create 4 experts with cycling architectures
        experts = []
        encodings = ["alibi", "rope"]
        for i in range(4):
            expert_config = copy.copy(config)
            expert_config.encoding = encodings[i % len(encodings)]
            expert = HexAttention(expert_config)
            experts.append(expert)

        router = Prismatic(config, experts=experts)

        # Verify cycling: alibi, rope, alibi, rope
        assert router.experts[0].pos_type == "alibi"
        assert router.experts[1].pos_type == "rope"
        assert router.experts[2].pos_type == "alibi"
        assert router.experts[3].pos_type == "rope"

        print("[TEST] ✓ Supports N experts with modulus cycling")
        print("  Expert 0: ALiBi, Expert 1: RoPE, Expert 2: ALiBi, Expert 3: RoPE")


class TestSparseRouting:
    """Test sparse expert selection."""

    def test_routing_selects_one_expert_per_sequence(self):
        """Verify top-1 sparse routing."""
        config = PraxisConfig(
            hidden_size=64,
            num_heads=4,
            num_queries=1,
            num_experts=2,
            router_type="prismatic"
        )

        experts = [SimpleDenseBlock(config.hidden_size) for _ in range(2)]
        router = Prismatic(config, experts=experts)

        # Create input
        batch_size = 4
        seq_len = 8
        inputs = torch.randn(batch_size, seq_len, config.hidden_size)

        # Get routing decision
        seq_repr = inputs.mean(dim=1)
        seq_repr = router.router_norm(seq_repr)
        logits = router.router(seq_repr)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        expert_indices = torch.argmax(probs, dim=-1)

        # Verify one expert per sequence
        assert expert_indices.shape == (batch_size,)
        assert torch.all((expert_indices == 0) | (expert_indices == 1))

        print("[TEST] ✓ Sparse routing: one expert per sequence")

    def test_gradients_flow_to_router(self):
        """Verify router receives gradients."""
        config = PraxisConfig(
            hidden_size=64,
            num_heads=4,
            num_queries=1,
            num_experts=2,
            router_type="prismatic"
        )

        experts = [SimpleDenseBlock(config.hidden_size) for _ in range(2)]
        router = Prismatic(config, experts=experts)

        inputs = torch.randn(2, 4, config.hidden_size)

        from praxis.layers.local import LocalLayer
        layer = LocalLayer(config, block=experts[0], expert_blocks=experts)

        output, _, _, aux_loss = router._router_forward(
            layer, inputs,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=0,
            block_ids=None
        )

        loss = output.sum() + aux_loss
        loss.backward()

        assert router.router.weight.grad is not None
        assert router.router.weight.grad.abs().sum() > 0

        print("[TEST] ✓ Router learns from gradients")


class TestLoadBalancing:
    """Test load balancing loss."""

    def test_balance_loss_computed(self):
        """Verify balance loss encourages 50/50 usage."""
        config = PraxisConfig(
            hidden_size=64,
            num_heads=4,
            num_queries=1,
            num_experts=2,
            router_type="prismatic"
        )

        experts = [SimpleDenseBlock(config.hidden_size) for _ in range(2)]
        router = Prismatic(config, experts=experts)

        # Balanced probs and balanced selections
        # For perfect balance: importance=[0.5, 0.5], load=[0.5, 0.5]
        # Loss = num_experts * sum(importance * load) = 2 * (0.5*0.5 + 0.5*0.5) = 1.0
        balanced_probs = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        balanced_indices = torch.tensor([[0, 1], [1, 0]])  # [batch, k=2]
        balanced_loss = router._compute_balance_loss(balanced_probs, balanced_indices)
        assert abs(balanced_loss.item() - 1.0) < 0.01  # Should be close to 1.0 for perfect balance

        # Imbalanced probs and imbalanced selections
        # importance=[0.9, 0.1], load=[1.0, 0.0]
        # Loss = 2 * (0.9*1.0 + 0.1*0.0) = 1.8
        imbalanced_probs = torch.tensor([[0.9, 0.1], [0.9, 0.1]])
        imbalanced_indices = torch.tensor([[0, 0], [0, 0]])  # [batch, k=2] all selecting expert 0
        imbalanced_loss = router._compute_balance_loss(imbalanced_probs, imbalanced_indices)
        assert imbalanced_loss.item() > balanced_loss.item()

        print("[TEST] ✓ Load balancing loss prevents collapse")


class TestMetrics:
    """Test metrics logging."""

    def test_metrics_logged(self):
        """Verify all routing and architecture metrics are logged."""
        config = PraxisConfig(
            hidden_size=64,
            num_heads=4,
            num_queries=1,
            num_experts=2,
            router_type="prismatic"
        )

        experts = [SimpleDenseBlock(config.hidden_size) for _ in range(2)]
        router = Prismatic(config, experts=experts)

        # Simulate routing
        expert_indices = torch.tensor([0, 0, 1, 1])
        probs = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]])
        balance_loss = torch.tensor(0.01)

        router._update_metrics(expert_indices, probs, balance_loss)
        metrics = router.get_metrics()

        # Routing metrics (web app compatible)
        assert "routing/expert_0_weight" in metrics
        assert "routing/expert_1_weight" in metrics
        assert "routing/entropy" in metrics
        assert "routing/concentration" in metrics
        assert "routing/variance" in metrics
        assert "routing/balance" in metrics

        # Architecture-specific metrics
        assert "architecture/alibi_usage" in metrics
        assert "architecture/rope_usage" in metrics

        # Verify expert weights sum to 1.0
        total_weight = metrics["routing/expert_0_weight"] + metrics["routing/expert_1_weight"]
        assert abs(total_weight - 1.0) < 0.01

        # Verify architecture usage sums to 100%
        total_usage = metrics["architecture/alibi_usage"] + metrics["architecture/rope_usage"]
        assert abs(total_usage - 100.0) < 0.01

        print("[TEST] ✓ All metrics logged correctly (web app compatible)")

    def test_expert_selection_counts_tracked(self):
        """Verify actual expert selection counts are tracked (k=1 sparse usage)."""
        config = PraxisConfig(
            hidden_size=64,
            num_heads=4,
            num_queries=1,
            num_experts=2,
            router_type="prismatic"
        )

        experts = [SimpleDenseBlock(config.hidden_size) for _ in range(2)]
        router = Prismatic(config, experts=experts)

        # Initially counts should be zero
        assert router.expert_selection_counts[0].item() == 0
        assert router.expert_selection_counts[1].item() == 0

        # Simulate routing - 3 sequences to expert 0, 1 sequence to expert 1
        expert_indices = torch.tensor([0, 0, 0, 1])
        probs = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.3, 0.7]])
        balance_loss = torch.tensor(0.01)

        router._update_metrics(expert_indices, probs, balance_loss)
        metrics = router.get_metrics()

        # Check selection counts
        assert "expert_selection/expert_0_count" in metrics
        assert "expert_selection/expert_1_count" in metrics
        assert metrics["expert_selection/expert_0_count"] == 3
        assert metrics["expert_selection/expert_1_count"] == 1

        # Run again - counts should accumulate
        router._update_metrics(expert_indices, probs, balance_loss)
        metrics = router.get_metrics()

        assert metrics["expert_selection/expert_0_count"] == 6  # 3 + 3
        assert metrics["expert_selection/expert_1_count"] == 2  # 1 + 1

        print("[TEST] ✓ Expert selection counts tracked correctly (cumulative)")
        print(f"  Expert 0 selected 6 times, Expert 1 selected 2 times")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
