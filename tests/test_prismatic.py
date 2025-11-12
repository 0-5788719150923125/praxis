"""
Test suite for Prismatic router with sparse bidirectional temporal routing.

Tests verify:
1. Sparse routing selects one expert per sequence
2. Mask creation (forward vs backward)
3. Load balancing loss
4. Metrics are logged correctly
5. Gradients flow to router
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
        **kwargs  # Accept ghost_position and other parameters
    ):
        x = self.linear(inputs)
        x = self.norm(x)
        return x, past_key_values, current_state, 0.0


class TestExpertCloning:
    """Test that experts are cloned from same initial state."""

    def test_experts_start_identical(self):
        """Verify Expert 1 is cloned from Expert 0 with identical weights."""
        config = PraxisConfig(
            hidden_size=64,
            num_heads=4,
            num_queries=1,
            num_experts=2,
            router_type="prismatic",
            causal=True,
            pos_type="alibi"
        )

        # Create experts
        from praxis.blocks.transformer import TransformerBlock
        base_expert = TransformerBlock(config)
        cloned_expert = copy.deepcopy(base_expert)

        experts = [base_expert, cloned_expert]
        router = Prismatic(config, experts=experts)

        # Verify all parameters are identical at initialization
        for (name0, param0), (name1, param1) in zip(
            router.experts[0].named_parameters(),
            router.experts[1].named_parameters()
        ):
            assert name0 == name1, f"Parameter names differ: {name0} vs {name1}"
            assert torch.allclose(param0, param1), f"Parameters differ for {name0}"

        print("[TEST] ✓ Experts start with identical weights (same reality seed)")

    def test_cloned_experts_maintain_independence(self):
        """Verify cloned experts are independent and can diverge during training."""
        config = PraxisConfig(
            hidden_size=64,
            num_heads=4,
            num_queries=1,
            num_experts=2,
            router_type="prismatic"
        )

        # Create and clone simple experts
        base_expert = SimpleDenseBlock(config.hidden_size)
        cloned_expert = copy.deepcopy(base_expert)

        # Verify they start identical
        for (n0, p0), (n1, p1) in zip(base_expert.named_parameters(), cloned_expert.named_parameters()):
            assert torch.allclose(p0, p1), f"Cloned expert differs at {n0}"

        # Verify they're independent PyTorch modules (different objects)
        assert base_expert is not cloned_expert, "Experts should be different objects"
        assert base_expert.linear is not cloned_expert.linear, "Submodules should be different objects"

        # Verify parameter independence by modifying one
        with torch.no_grad():
            original_weight = base_expert.linear.weight.clone()
            base_expert.linear.weight.add_(0.1)  # Modify base expert

            # Cloned expert should be unaffected
            assert not torch.allclose(base_expert.linear.weight, cloned_expert.linear.weight), \
                "Modifying base expert should not affect clone"

            # Restore
            base_expert.linear.weight.copy_(original_weight)

        print("[TEST] ✓ Cloned experts are independent PyTorch modules")
        print("[TEST] ✓ Constraint-driven divergence possible (from same initial state)")


class TestMaskCreation:
    """Test Prismatic mask creation."""

    def test_forward_mask_creation(self):
        """Verify Prismatic creates correct forward causal mask."""
        config = PraxisConfig(
            hidden_size=64,
            num_heads=4,
            num_queries=1,
            num_experts=2,
            router_type="prismatic"
        )

        # Create dummy experts
        experts = [SimpleDenseBlock(config.hidden_size) for _ in range(2)]
        router = Prismatic(config, experts=experts)

        # Create forward mask
        seq_len = 4
        forward_mask = router._create_causal_mask(
            seq_len, direction="forward", device=torch.device("cpu")
        )

        # Verify shape
        assert forward_mask.shape == (seq_len, seq_len)

        # Forward causal: position i can see j where j <= i
        assert forward_mask[0, 0] == 0.0
        assert forward_mask[0, 1] == float('-inf')

        # Row 3 should see all positions
        assert torch.all(forward_mask[3, :] == 0.0)

    def test_backward_mask_creation(self):
        """Verify Prismatic creates correct backward causal mask."""
        config = PraxisConfig(
            hidden_size=64,
            num_heads=4,
            num_queries=1,
            num_experts=2,
            router_type="prismatic"
        )

        experts = [SimpleDenseBlock(config.hidden_size) for _ in range(2)]
        router = Prismatic(config, experts=experts)

        # Create backward mask
        seq_len = 4
        backward_mask = router._create_causal_mask(
            seq_len, direction="backward", device=torch.device("cpu")
        )

        # Verify shape
        assert backward_mask.shape == (seq_len, seq_len)

        # Backward causal: position i can see j where j >= i
        # Row 0 should see all positions
        assert torch.all(backward_mask[0, :] == 0.0)

        # Row 3 should only see position 3
        assert backward_mask[3, 3] == 0.0
        assert backward_mask[3, 0] == float('-inf')


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

        # Get sequence representation and route
        seq_repr = inputs.mean(dim=1)
        seq_repr = router.router_norm(seq_repr)
        logits = router.router(seq_repr)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        expert_indices = torch.argmax(probs, dim=-1)

        # Verify we have one expert index per sequence
        assert expert_indices.shape == (batch_size,)
        assert expert_indices.dtype == torch.long
        assert torch.all((expert_indices == 0) | (expert_indices == 1))

    def test_gradients_flow_to_router(self):
        """Verify router receives gradients during training."""
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
        batch_size = 2
        seq_len = 4
        inputs = torch.randn(batch_size, seq_len, config.hidden_size)

        # Forward pass (router mode)
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

        # Backward pass
        loss = output.sum() + aux_loss
        loss.backward()

        # Check router has gradients
        assert router.router.weight.grad is not None
        assert router.router.weight.grad.abs().sum() > 0


class TestLoadBalancing:
    """Test load balancing loss."""

    def test_balance_loss_computed(self):
        """Verify balance loss encourages 50/50 expert usage."""
        config = PraxisConfig(
            hidden_size=64,
            num_heads=4,
            num_queries=1,
            num_experts=2,
            router_type="prismatic"
        )

        experts = [SimpleDenseBlock(config.hidden_size) for _ in range(2)]
        router = Prismatic(config, experts=experts)

        # Test balanced probs (should have low loss)
        balanced_probs = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        balanced_loss = router._compute_balance_loss(balanced_probs)
        assert balanced_loss.item() < 0.01  # Nearly zero

        # Test imbalanced probs (should have higher loss)
        imbalanced_probs = torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
        imbalanced_loss = router._compute_balance_loss(imbalanced_probs)
        assert imbalanced_loss.item() > balanced_loss.item()


class TestMetrics:
    """Test metrics logging."""

    def test_metrics_logged(self):
        """Verify routing metrics are logged."""
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
        batch_size = 4
        expert_indices = torch.tensor([0, 0, 1, 1])  # 50/50 split
        probs = torch.tensor([
            [0.8, 0.2],
            [0.7, 0.3],
            [0.3, 0.7],
            [0.2, 0.8]
        ])
        balance_loss = torch.tensor(0.01)

        router._update_metrics(expert_indices, probs, balance_loss)
        metrics = router.get_metrics()

        # Verify metrics exist (web app compatible)
        assert "routing/expert_0_weight" in metrics
        assert "routing/expert_1_weight" in metrics
        assert "routing/entropy" in metrics
        assert "routing/concentration" in metrics
        assert "routing/variance" in metrics
        assert "routing/balance" in metrics
        assert "routing/balance_loss" in metrics
        assert "routing/avg_confidence" in metrics

        # Verify expert weights sum to approximately 1.0 (softmax property)
        total_weight = metrics["routing/expert_0_weight"] + metrics["routing/expert_1_weight"]
        assert abs(total_weight - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
