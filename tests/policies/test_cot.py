"""
Tests for reinforcement learning policies in praxis.policies.
"""

import pytest
import torch

from praxis.policies import ChainOfThought


class TestChainOfThought:
    """Test cases for Chain of Thought policy."""

    def test_initialization(self, config):
        """Test CoT policy initialization."""
        policy = ChainOfThought(config)

        assert policy.config == config
        assert policy.hidden_size == config.hidden_size
        assert hasattr(policy, "quality_head")
        assert hasattr(policy, "step_count")
        assert hasattr(policy, "cot_stats")

    def test_forward_without_training(self, config, sample_data):
        """Test forward pass in eval mode."""
        policy = ChainOfThought(config)
        policy.eval()

        hidden_states = sample_data["hidden_states"]
        logits = sample_data["logits"]
        labels = sample_data["labels"]

        with torch.no_grad():
            output_hidden, losses = policy(hidden_states, logits, labels)

        assert output_hidden.shape == hidden_states.shape
        assert losses is None

    def test_forward_with_training(self, config, sample_data):
        """Test forward pass in training mode."""
        policy = ChainOfThought(config)
        policy.train()

        hidden_states = sample_data["hidden_states"]
        logits = sample_data["logits"]
        labels = sample_data["labels"]
        attention_mask = sample_data["attention_mask"]

        output_hidden, losses = policy(
            hidden_states, logits, labels, attention_mask=attention_mask
        )

        assert output_hidden.shape == hidden_states.shape
        if losses is not None:  # May be None if no valid tokens
            assert "cot_usage_loss" in losses
            assert "quality_loss" in losses

    def test_quality_head(self, config, sample_data):
        """Test quality head computation."""
        policy = ChainOfThought(config)

        hidden_states = sample_data["hidden_states"]
        pooled_hidden = hidden_states.mean(dim=1)

        quality_scores = policy.quality_head(pooled_hidden)

        assert quality_scores.shape == (hidden_states.shape[0], 1)
        assert torch.all(quality_scores >= 0) and torch.all(quality_scores <= 1)
