"""Tests for the engagement-prediction reward (P2) and policy (P3)."""

import torch

from praxis import PraxisConfig
from praxis.policies import EngagementPolicy, needs_rl_datasets, normalize_rl_types
from praxis.policies.engagement_reward import (
    HomeostaticEnergy,
    activation,
    recall,
)


class TestReward:
    def test_activation_fires_on_overlap(self):
        assert activation([1, 2, 3], [3, 9]) == 1.0
        assert activation([1, 2, 3], [7, 8]) == 0.0
        assert activation([], [1]) == 0.0

    def test_recall_is_graded(self):
        assert recall([1, 2, 3, 4], [1, 2, 9]) == 0.5  # 2 of 4 predicted mentioned
        assert recall([1, 2], [1, 2]) == 1.0
        assert recall([], [1]) == 0.0


class TestHomeostaticEnergy:
    def test_accumulates_fast_then_satiates(self):
        e = HomeostaticEnergy()
        first = e.update(1.0)
        # Diminishing returns: each successive full activation adds less.
        gains = [first]
        for _ in range(5):
            before = e.value
            e.update(1.0)
            gains.append(e.value - before)
        assert gains[0] > 0
        assert all(gains[i] >= gains[i + 1] - 1e-9 for i in range(len(gains) - 1))

    def test_decays_without_activation(self):
        e = HomeostaticEnergy(init=0.5)
        before = e.value
        for _ in range(50):
            e.update(0.0)
        assert e.value < before

    def test_energy_stays_bounded(self):
        e = HomeostaticEnergy()
        for _ in range(10_000):
            e.update(1.0)
        assert 0.0 <= e.value <= e.e_max


class TestEngagementPolicy:
    def _config(self):
        return PraxisConfig(hidden_size=64, dropout=0.0, rl_weight=0.1)

    def test_forward_produces_loss_and_metrics(self):
        policy = EngagementPolicy(self._config())
        policy.train()
        b, t, v = 4, 10, 50
        logits = torch.randn(b, t, v, requires_grad=True)
        labels = torch.randint(0, v, (b, t))
        mask = torch.zeros(b, t)
        mask[:, 5:] = 1.0  # last tokens are the "answer" region

        loss, metrics = policy(logits=logits, labels=labels, assistant_mask=mask)

        assert loss is not None and loss.requires_grad and torch.isfinite(loss)
        for key in (
            "engagement_energy",
            "engagement_activation_rate",
            "engagement_recall",
            "engagement_advantage",
        ):
            assert key in metrics
        loss.backward()
        assert logits.grad is not None

    def test_perfect_prediction_drives_energy_up(self):
        policy = EngagementPolicy(self._config())
        policy.train()
        b, t, v = 4, 8, 30
        labels = torch.randint(0, v, (b, t))
        mask = torch.ones(b, t)
        # Logits that argmax exactly to the labels -> full recall every step.
        logits = torch.full((b, t, v), -10.0)
        for i in range(b):
            for j in range(t):
                logits[i, j, labels[i, j]] = 10.0

        e_prev = 0.0
        for _ in range(5):
            _, metrics = policy(logits=logits, labels=labels, assistant_mask=mask)
            assert metrics["engagement_activation_rate"] == 1.0
            assert metrics["engagement_energy"] >= e_prev
            e_prev = metrics["engagement_energy"]
        assert e_prev > 0.0

    def test_noop_when_no_mask_or_eval(self):
        policy = EngagementPolicy(self._config())
        logits = torch.randn(2, 6, 20)
        labels = torch.randint(0, 20, (2, 6))
        policy.eval()
        assert policy(
            logits=logits, labels=labels, assistant_mask=torch.ones(2, 6)
        ) == (
            None,
            {},
        )
        policy.train()
        assert policy(logits=logits, labels=labels, assistant_mask=None) == (None, {})


def test_engagement_needs_no_rl_datasets():
    assert normalize_rl_types("engagement") == ["engagement"]
    assert needs_rl_datasets("engagement") is False
    # Coexists with a weight controller; neither pulls the RL collection.
    assert needs_rl_datasets("harmonic_weight_wave") is False
