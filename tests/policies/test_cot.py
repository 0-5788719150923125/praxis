"""Tests for praxis/policies/cot.py."""

import torch

from praxis.policies import ChainOfThought


def test_noop_in_eval(rl_config, sample_data):
    policy = ChainOfThought(rl_config).eval()
    hidden = sample_data["hidden_states"]
    with torch.no_grad():
        output, losses = policy(hidden, sample_data["logits"], sample_data["labels"])
    assert output.shape == hidden.shape and losses is None


def test_training_losses_follow_the_token_weights(rl_config, sample_data):
    policy = ChainOfThought(rl_config).train()
    hidden, logits, labels = (
        sample_data["hidden_states"],
        sample_data["logits"],
        sample_data["labels"],
    )
    # No CoT weights in the batch: an empty container.
    _, losses = policy(hidden, logits, labels)
    assert losses.get_loss("cot_usage_loss").item() == 0.0

    # A quarter of the tokens carry reasoning weights.
    weights = torch.ones(labels.shape)
    weights[:, :8] = 1.5
    _, losses = policy(
        hidden,
        logits,
        labels,
        attention_mask=sample_data["attention_mask"],
        token_weights=weights,
    )
    assert losses.get_loss("cot_usage_loss").item() == 0.75  # 1 - usage ratio
    quality = losses.get_loss("quality_loss")
    assert torch.isfinite(quality) and quality.requires_grad  # via quality_head
