"""Shared fixtures for the forward-path RL policy tests."""

import pytest
import torch

from praxis import PraxisConfig


@pytest.fixture
def rl_config():
    return PraxisConfig(
        hidden_size=128,
        dropout=0.1,
        grpo_group_size=4,
        grpo_kl_coeff=0.04,
        grpo_clip_ratio=0.2,
        rl_weight=0.1,
    )


@pytest.fixture
def sample_data():
    """A batch of 8 sequences of 32 positions, hidden 128, vocab 1000."""
    batch_size, seq_len, hidden_size, vocab_size = 8, 32, 128, 1000
    return {
        "hidden_states": torch.randn(batch_size, seq_len, hidden_size),
        "logits": torch.randn(batch_size, seq_len, vocab_size),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "rewards": torch.tensor([0.8, 0.2, 0.9, 0.1, 0.7, 0.3, 0.6, 0.4]),
        "ref_logits": torch.randn(batch_size, seq_len, vocab_size),
    }
