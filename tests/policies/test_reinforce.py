"""Tests for praxis/policies/reinforce.py."""

import torch

from praxis.policies import REINFORCE


def test_noop_without_rewards_or_in_eval(rl_config, sample_data):
    policy = REINFORCE(rl_config)
    hidden = sample_data["hidden_states"]

    policy.train()
    output, loss = policy(hidden)
    assert output.shape == hidden.shape and loss is None

    policy.eval()
    with torch.no_grad():
        output, loss = policy(hidden, rewards=sample_data["rewards"])
    assert output.shape == hidden.shape and loss is None
    assert policy.step_count == 0


def test_loss_reaches_policy_and_value_heads(rl_config, sample_data):
    policy = REINFORCE(rl_config).train()
    hidden = sample_data["hidden_states"]
    output, loss = policy(
        hidden, rewards=sample_data["rewards"], mask=sample_data["attention_mask"]
    )
    assert output.shape == hidden.shape
    assert loss.requires_grad and torch.isfinite(loss)

    loss.backward()
    for head in (policy.policy_mlp, policy.value_head):
        assert all(p.grad is not None for p in head.parameters())


def test_baseline_tracks_rewards_and_steps_count(rl_config, sample_data):
    policy = REINFORCE(rl_config).train()
    policy.baseline.data.fill_(0.1)
    rewards = sample_data["rewards"]  # mean 0.5
    for i in range(3):
        policy(sample_data["hidden_states"], rewards=rewards)
        assert policy.step_count == i + 1
    # The EMA moves toward the mean reward.
    assert 0.1 < policy.baseline.item() < rewards.mean().item()
