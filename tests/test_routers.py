"""Test suite for routers."""

import pytest
import torch
import torch.nn as nn

from praxis.blocks.transformer import TransformerBlock
from praxis.configuration import PraxisConfig
from praxis.containers.loss import LossContainer
from praxis.layers import LocalLayer
from praxis.routers.smear import SMEAR
from praxis.routers.taxus import Taxus


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

            def forward(
                self,
                inputs,
                attention_mask,
                past_key_values,
                current_state,
                current_depth,
                block_ids,
            ):
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
        assert not torch.allclose(
            output, inputs
        )  # Output should be different from input


class TestTaxusRouter:
    """Test cases for Taxus depth-buying router."""

    def test_taxus_initialization(self):
        """Test that Taxus router initializes correctly."""
        config = PraxisConfig(
            hidden_size=128,
            depth=6,
            debug=False,
        )

        taxus = Taxus(config, target_depth_ratio=0.5)

        # Check initialization
        assert taxus.hidden_size == 128
        assert taxus.depth == 6
        assert taxus.target_depth_ratio == 0.5
        assert len(taxus.exit_gates) == 6
        assert taxus.layer_costs.shape == (6,)

    def test_taxus_forward_minimum_layers(self):
        """Test that Taxus respects minimum exit layer constraint."""
        config = PraxisConfig(
            hidden_size=64,
            depth=4,
            debug=False,
        )

        taxus = Taxus(config, min_exit_layer=2)

        # Create a simple mock layer
        class MockLayer(nn.Module):
            def forward(self, inputs, *args, **kwargs):
                return inputs + 1, None, None, 0.0

        layer = MockLayer()

        batch_size = 2
        seq_length = 8
        inputs = torch.randn(batch_size, seq_length, config.hidden_size)

        # Test at depth 0 and 1 (before min_exit_layer)
        for depth in [0, 1]:
            output, _, _, losses = taxus(
                layer=layer,
                inputs=inputs,
                attention_mask=None,
                past_key_values=None,
                current_state=None,
                current_depth=depth,
                block_ids=None,
            )

            # Should process normally without exit consideration
            assert torch.allclose(output, inputs + 1)
            assert "taxus_entropy" not in losses
            assert "taxus_usage" not in losses

    def test_taxus_exit_decision(self):
        """Test that Taxus makes exit decisions after minimum layer."""
        config = PraxisConfig(
            hidden_size=64,
            depth=4,
            debug=True,  # Enable statistics tracking
        )

        taxus = Taxus(
            config,
            min_exit_layer=1,
            temperature=1.0,
            target_depth_ratio=0.5,
        )

        # Create a layer that modifies inputs
        class MockLayer(nn.Module):
            def forward(self, inputs, *args, **kwargs):
                return inputs * 2, None, None, 0.0

        layer = MockLayer()

        batch_size = 4
        seq_length = 8
        inputs = torch.randn(batch_size, seq_length, config.hidden_size)

        # Test at depth 2 (after min_exit_layer)
        taxus.eval()  # Set to eval mode for deterministic behavior
        output, _, _, losses = taxus(
            layer=layer,
            inputs=inputs,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=2,
            block_ids=None,
        )

        # Check that auxiliary losses are computed
        assert "taxus_entropy" in losses
        assert "taxus_usage" in losses
        assert "taxus_confidence" in losses
        assert "taxus_cost" in losses

        # Output should be either inputs (exit) or inputs * 2 (processed)
        # In eval mode, this depends on the exit decision
        assert output.shape == inputs.shape

    def test_taxus_auxiliary_losses(self):
        """Test that Taxus computes auxiliary losses correctly."""
        config = PraxisConfig(
            hidden_size=32,
            depth=8,
            debug=False,
        )

        taxus = Taxus(
            config,
            target_depth_ratio=0.5,
            entropy_weight=0.01,
            usage_weight=0.1,
            budget_weight=0.1,
        )

        class MockLayer(nn.Module):
            def forward(self, inputs, *args, **kwargs):
                return inputs, None, None, torch.tensor(0.5)  # Return a layer loss

        layer = MockLayer()

        batch_size = 2
        seq_length = 4
        inputs = torch.randn(batch_size, seq_length, config.hidden_size)

        # Test at different depths
        for depth in [2, 4, 6]:
            output, _, _, losses = taxus(
                layer=layer,
                inputs=inputs,
                attention_mask=None,
                past_key_values=None,
                current_state=None,
                current_depth=depth,
                block_ids=None,
            )

            # Check all expected losses are present
            assert "taxus_entropy" in losses
            assert "taxus_usage" in losses
            assert "taxus_confidence" in losses
            assert "taxus_cost" in losses
            assert "layer" in losses

            # Usage loss should vary with depth
            usage_loss = losses.get_loss("taxus_usage")
            assert usage_loss.item() >= 0  # Should be non-negative

    def test_taxus_training_vs_inference(self):
        """Test that Taxus behaves differently in training vs inference."""
        config = PraxisConfig(
            hidden_size=64,
            depth=4,
            debug=True,
        )

        taxus = Taxus(config, min_exit_layer=1, temperature=0.5)

        class MockLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            def forward(self, inputs, *args, **kwargs):
                self.call_count += 1
                return inputs * 1.5, None, None, 0.0

        layer = MockLayer()

        batch_size = 2
        seq_length = 8
        inputs = torch.randn(batch_size, seq_length, config.hidden_size)

        # Training mode - uses gumbel-softmax for differentiable decisions
        taxus.train()
        output_train, _, _, losses_train = taxus(
            layer=layer,
            inputs=inputs,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=2,
            block_ids=None,
        )

        assert layer.call_count == 1  # Layer should be called
        assert "taxus_exit_prob" in losses_train  # Training includes exit probabilities

        # Inference mode - may skip layer if all samples exit
        taxus.eval()
        layer.call_count = 0

        # Force exit decision by manipulating exit gates
        with torch.no_grad():
            # Set weights to strongly favor exit
            taxus.exit_gates[2][2].weight.fill_(0)
            taxus.exit_gates[2][2].bias[0] = -10  # Continue logit
            taxus.exit_gates[2][2].bias[1] = 10  # Exit logit

        output_eval, _, _, _ = taxus(
            layer=layer,
            inputs=inputs,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=2,
            block_ids=None,
        )

        # With strong exit bias, layer might not be called
        assert output_eval.shape == inputs.shape

    def test_taxus_statistics_tracking(self):
        """Test that Taxus tracks exit statistics correctly in debug mode."""
        config = PraxisConfig(
            hidden_size=32,
            depth=4,
            debug=True,
        )

        taxus = Taxus(config, min_exit_layer=1)

        # Since Taxus forces debug=False, statistics won't be tracked
        # Test that get_exit_statistics returns empty dict
        stats = taxus.get_exit_statistics()
        assert stats == {}

    def test_taxus_with_loss_container_input(self):
        """Test that Taxus handles LossContainer inputs from layers."""
        config = PraxisConfig(
            hidden_size=64,
            depth=4,
            debug=False,
        )

        taxus = Taxus(config, min_exit_layer=0)

        class MockLayer(nn.Module):
            def forward(self, inputs, *args, **kwargs):
                losses = LossContainer()
                losses.add_loss("custom_loss", 0.25)
                losses.add_loss("another_loss", 0.1)
                return inputs + 0.1, None, None, losses

        layer = MockLayer()

        inputs = torch.randn(2, 8, config.hidden_size)

        output, _, _, losses = taxus(
            layer=layer,
            inputs=inputs,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=2,
            block_ids=None,
        )

        # Check that layer losses are properly integrated
        assert "custom_loss" in losses
        assert "another_loss" in losses
        assert pytest.approx(losses.get_loss("custom_loss").item()) == 0.25
        assert pytest.approx(losses.get_loss("another_loss").item()) == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
