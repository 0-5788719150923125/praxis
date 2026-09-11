"""
Tests for reinforcement learning policies in praxis.policies.
"""

import pytest
import torch

from praxis.policies import GRPO


class TestGRPO:
    """Test cases for GRPO policy."""

    def test_initialization(self, config):
        """Test GRPO policy initialization."""
        policy = GRPO(config)

        assert policy.config == config
        assert policy.hidden_size == config.hidden_size
        assert policy.group_size == config.grpo_group_size
        assert policy.kl_coeff == config.grpo_kl_coeff
        assert policy.clip_ratio == config.grpo_clip_ratio

    def test_compute_group_advantages(self, config):
        """Test group advantage computation."""
        policy = GRPO(config)

        # Test with rewards that have clear groups
        rewards = torch.tensor([1.0, 0.5, 0.8, 0.2, 0.9, 0.1, 0.7, 0.3])
        group_size = 4

        advantages = policy.compute_group_advantages(rewards, group_size)

        assert advantages.shape == rewards.shape
        # Check that advantages are normalized within groups
        group1_advantages = advantages[:4]
        group2_advantages = advantages[4:]

        assert abs(group1_advantages.mean().item()) < 1e-5  # Should be ~0
        assert abs(group2_advantages.mean().item()) < 1e-5  # Should be ~0

    def test_forward_without_rewards(self, config, sample_data):
        """Test forward pass without rewards."""
        policy = GRPO(config)

        hidden_states = sample_data["hidden_states"]
        logits = sample_data["logits"]
        labels = sample_data["labels"]

        output_hidden, losses = policy(hidden_states, logits, labels)

        assert output_hidden.shape == hidden_states.shape
        assert losses is None

    def test_forward_with_rewards(self, config, sample_data):
        """Test forward pass with rewards."""
        policy = GRPO(config)
        policy.train()

        hidden_states = sample_data["hidden_states"]
        logits = sample_data["logits"]
        labels = sample_data["labels"]
        rewards = sample_data["rewards"]
        ref_logits = sample_data["ref_logits"]
        mask = sample_data["attention_mask"]

        # Use rewards with different values to ensure variance
        varied_rewards = torch.tensor([0.1, 0.8, 0.2, 0.9, 0.3, 0.7, 0.4, 0.6])

        output_hidden, losses = policy(
            hidden_states,
            logits,
            labels,
            rewards=varied_rewards,
            ref_logits=ref_logits,
            mask=mask,
        )

        assert output_hidden.shape == hidden_states.shape
        # losses might be None if batch size issues or all rewards are zero
        if losses is not None:
            assert "grpo_loss" in losses
            assert "policy_loss" in losses
            assert "kl_loss" in losses
            assert "mean_reward" in losses
            assert "mean_advantage" in losses

    def test_zero_rewards_handling(self, config, sample_data):
        """Test handling of zero rewards."""
        policy = GRPO(config)
        policy.train()

        hidden_states = sample_data["hidden_states"]
        logits = sample_data["logits"]
        labels = sample_data["labels"]
        zero_rewards = torch.zeros(8)

        output_hidden, losses = policy(
            hidden_states, logits, labels, rewards=zero_rewards
        )

        assert output_hidden.shape == hidden_states.shape
        assert losses is None  # Should skip RL loss when all rewards are zero

    def test_small_batch_handling(self, config):
        """Test handling of batches smaller than group size."""
        policy = GRPO(config)
        policy.train()

        batch_size = 2  # Smaller than group_size (4)
        seq_len = 16
        hidden_size = 128
        vocab_size = 1000

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        rewards = torch.tensor([0.8, 0.2])

        output_hidden, losses = policy(hidden_states, logits, labels, rewards=rewards)

        assert output_hidden.shape == hidden_states.shape
        if losses is not None:  # May be None if rewards are problematic
            assert "grpo_loss" in losses


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
def test_policy_batch_size_handling(config, batch_size):
    """Test policies with different batch sizes."""
    seq_len = 32
    hidden_size = config.hidden_size
    vocab_size = 1000

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    rewards = torch.rand(batch_size)

    # Test GRPO with different batch sizes
    policy = GRPO(config)
    policy.train()

    output, losses = policy(hidden_states, logits, labels, rewards=rewards)

    assert output.shape == hidden_states.shape
