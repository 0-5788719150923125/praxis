"""Tests for praxis/policies/grpo.py."""

import pytest
import torch

from praxis.policies import GRPO


def _inputs(batch_size, seq_len=16, hidden_size=128, vocab_size=1000):
    return (
        torch.randn(batch_size, seq_len, hidden_size),
        torch.randn(batch_size, seq_len, vocab_size),
        torch.randint(0, vocab_size, (batch_size, seq_len)),
    )


def test_compute_group_advantages(rl_config):
    policy = GRPO(rl_config)
    rewards = torch.tensor([1.0, 0.5, 0.8, 0.2, 0.9, 0.1, 0.7, 0.3])
    advantages = policy.compute_group_advantages(rewards, group_size=4)
    assert advantages.shape == rewards.shape
    # Normalized within each group.
    assert abs(advantages[:4].mean().item()) < 1e-5
    assert abs(advantages[4:].mean().item()) < 1e-5

    # A group with no reward variance has no preference: zero advantage.
    identical = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8])
    assert torch.equal(
        policy.compute_group_advantages(identical, group_size=4), torch.zeros(8)
    )


def test_noop_without_rewards_or_signal(rl_config, sample_data):
    policy = GRPO(rl_config).train()
    hidden, logits, labels = (
        sample_data["hidden_states"],
        sample_data["logits"],
        sample_data["labels"],
    )
    output, losses = policy(hidden, logits, labels)
    assert output.shape == hidden.shape and losses is None
    # All-zero rewards carry no learning signal.
    _, losses = policy(hidden, logits, labels, rewards=torch.zeros(8))
    assert losses is None
    policy.eval()
    _, losses = policy(hidden, logits, labels, rewards=sample_data["rewards"])
    assert losses is None


def test_forward_with_rewards(rl_config, sample_data):
    policy = GRPO(rl_config).train()
    output, losses = policy(
        sample_data["hidden_states"],
        sample_data["logits"],
        sample_data["labels"],
        rewards=torch.tensor([0.1, 0.8, 0.2, 0.9, 0.3, 0.7, 0.4, 0.6]),
        ref_logits=sample_data["ref_logits"],
        mask=sample_data["attention_mask"],
    )
    assert output.shape == sample_data["hidden_states"].shape
    assert set(losses) >= {
        "grpo_loss",
        "policy_loss",
        "kl_loss",
        "mean_reward",
        "mean_advantage",
    }
    assert torch.isfinite(losses["grpo_loss"])


@pytest.mark.parametrize(
    "batch_size",
    [
        pytest.param(1, id="one row"),
        pytest.param(2, id="below group"),
        pytest.param(6, id="not divisible"),
        pytest.param(8, id="whole groups"),
    ],
)
def test_batch_sizes_that_do_not_fill_a_group(rl_config, batch_size):
    """Batches below the group size form one group; batches the group size does
    not divide use the largest divisor under it."""
    policy = GRPO(rl_config).train()
    hidden, logits, labels = _inputs(batch_size)
    rewards = torch.linspace(0.1, 0.9, batch_size)
    output, losses = policy(hidden, logits, labels, rewards=rewards)
    assert output.shape == hidden.shape
    assert torch.isfinite(losses["grpo_loss"])
