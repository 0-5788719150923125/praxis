"""
Tests for reinforcement learning policies in praxis.policies.
"""

import pytest
import torch

from praxis.policies import GRPO, REINFORCE


class TestREINFORCE:
    """Test cases for REINFORCE policy."""

    def test_initialization(self, config):
        """Test REINFORCE policy initialization."""
        policy = REINFORCE(config)

        assert policy.config == config
        assert policy.hidden_size == config.hidden_size
        assert policy.rl_weight == config.rl_weight
        assert hasattr(policy, "value_head")
        assert hasattr(policy, "policy_mlp")
        assert hasattr(policy, "baseline")

    def test_forward_without_rewards(self, config, sample_data):
        """Test forward pass without rewards (inference mode)."""
        policy = REINFORCE(config)
        policy.eval()

        hidden_states = sample_data["hidden_states"]

        with torch.no_grad():
            output_hidden, rl_loss = policy(hidden_states)

        assert output_hidden.shape == hidden_states.shape
        assert rl_loss is None

    def test_forward_with_rewards(self, config, sample_data):
        """Test forward pass with rewards (training mode)."""
        policy = REINFORCE(config)
        policy.train()

        hidden_states = sample_data["hidden_states"]
        rewards = sample_data["rewards"]
        mask = sample_data["attention_mask"]

        output_hidden, rl_loss = policy(hidden_states, rewards=rewards, mask=mask)

        assert output_hidden.shape == hidden_states.shape
        assert rl_loss is not None
        assert isinstance(rl_loss, torch.Tensor)
        assert rl_loss.requires_grad

    def test_baseline_update(self, config, sample_data):
        """Test that baseline gets updated during training."""
        policy = REINFORCE(config)
        policy.train()

        # Set baseline to a different value so we can see the change
        policy.baseline.data.fill_(0.1)
        initial_baseline = policy.baseline.data.clone()

        hidden_states = sample_data["hidden_states"]
        rewards = sample_data["rewards"]

        # Run multiple forward passes to ensure baseline update is noticeable
        for _ in range(5):
            policy(hidden_states, rewards=rewards)

        # Baseline should have changed significantly
        assert abs(policy.baseline.data.item() - initial_baseline.item()) > 1e-6

    def test_gradient_flow(self, config, sample_data):
        """Test that gradients flow through the policy."""
        policy = REINFORCE(config)
        policy.train()

        hidden_states = sample_data["hidden_states"]
        rewards = sample_data["rewards"]

        # Zero gradients first
        policy.zero_grad()

        _, rl_loss = policy(hidden_states, rewards=rewards)

        # rl_loss can be None if not training or no rewards
        if rl_loss is not None:
            rl_loss.backward()

            # Check that gradients exist for trainable parameters
            has_gradients = False
            for param in policy.parameters():
                if param.requires_grad and param.grad is not None:
                    has_gradients = True
                    break

            assert has_gradients, "No gradients found in any trainable parameters"

    # For small batches or zero rewards, losses might be None


class TestPolicyEdgeCases:
    """Edge case tests for RL policies robustness."""

    def test_reinforce_extreme_rewards(self, config):
        """Test REINFORCE with extreme reward values."""
        policy = REINFORCE(config)
        policy.train()

        batch_size, seq_len, hidden_size = 4, 16, config.hidden_size
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Test with very large rewards
        large_rewards = torch.tensor([1000.0, 2000.0, 3000.0, 4000.0])
        output, loss = policy(hidden_states, rewards=large_rewards)
        assert output.shape == hidden_states.shape
        assert loss is not None and torch.isfinite(loss)

        # Test with very small rewards
        small_rewards = torch.tensor([1e-8, 2e-8, 3e-8, 4e-8])
        output, loss = policy(hidden_states, rewards=small_rewards)
        assert output.shape == hidden_states.shape
        assert loss is not None and torch.isfinite(loss)

        # Test with negative rewards
        negative_rewards = torch.tensor([-1.0, -2.0, -3.0, -4.0])
        output, loss = policy(hidden_states, rewards=negative_rewards)
        assert output.shape == hidden_states.shape
        assert loss is not None and torch.isfinite(loss)

    def test_grpo_identical_rewards(self, config):
        """Test GRPO with identical rewards within groups."""
        policy = GRPO(config)
        policy.train()

        batch_size, seq_len, hidden_size, vocab_size = 8, 16, config.hidden_size, 1000
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # All rewards identical (should result in zero advantages)
        identical_rewards = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8])

        output, losses = policy(
            hidden_states, logits, labels, rewards=identical_rewards
        )
        assert output.shape == hidden_states.shape
        # Should handle identical rewards gracefully

    def test_policies_with_masked_sequences(self, config):
        """Test policies with heavily masked sequences."""
        batch_size, seq_len, hidden_size, vocab_size = 4, 16, config.hidden_size, 1000
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        rewards = torch.rand(batch_size)

        # Create mask where most tokens are masked out
        mask = torch.zeros(batch_size, seq_len)
        mask[:, :3] = 1.0  # Only first 3 tokens are valid

        # Test REINFORCE
        reinforce_policy = REINFORCE(config)
        reinforce_policy.train()
        output, loss = reinforce_policy(hidden_states, rewards=rewards, mask=mask)
        assert output.shape == hidden_states.shape

        # Test GRPO
        grpo_policy = GRPO(config)
        grpo_policy.train()
        output, losses = grpo_policy(
            hidden_states, logits, labels, rewards=rewards, mask=mask
        )
        assert output.shape == hidden_states.shape

    def test_policies_numerical_stability(self, config):
        """Test policies with inputs that could cause numerical issues."""
        batch_size, seq_len, hidden_size, vocab_size = 4, 16, config.hidden_size, 1000

        # Test with very large hidden states
        large_hidden = torch.randn(batch_size, seq_len, hidden_size) * 100
        large_logits = torch.randn(batch_size, seq_len, vocab_size) * 100
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        rewards = torch.rand(batch_size)

        # Test REINFORCE numerical stability
        reinforce_policy = REINFORCE(config)
        reinforce_policy.train()
        output, loss = reinforce_policy(large_hidden, rewards=rewards)
        assert output.shape == large_hidden.shape
        assert loss is None or torch.isfinite(loss)

        # Test GRPO numerical stability
        grpo_policy = GRPO(config)
        grpo_policy.train()
        output, losses = grpo_policy(
            large_hidden, large_logits, labels, rewards=rewards
        )
        assert output.shape == large_hidden.shape

    def test_policy_state_preservation(self, config):
        """Test that policies maintain proper state across forward passes."""
        policy = REINFORCE(config)
        policy.train()

        batch_size, seq_len, hidden_size = 4, 16, config.hidden_size
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        rewards = torch.rand(batch_size)

        # Save initial state
        initial_baseline = policy.baseline.data.clone()
        initial_step_count = policy.step_count

        # Multiple forward passes
        for i in range(3):
            output, loss = policy(hidden_states, rewards=rewards)
            assert output.shape == hidden_states.shape
            assert policy.step_count == initial_step_count + i + 1

        # Baseline should have changed
        assert not torch.equal(policy.baseline.data, initial_baseline)
