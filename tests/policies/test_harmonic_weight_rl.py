"""Tests for praxis/policies/harmonic_weight_rl.py: the controller policy's
gradient estimator and its gate-mask selectors."""

import math
from types import SimpleNamespace

import torch

from praxis.policies.harmonic_weight_rl import HarmonicWeightPolicy


def _cfg(**over):
    base = dict(
        rl_hidden=16,
        rl_lr=0.05,
        rl_entropy_coef=0.0,
        rl_alpha_scale=1.0,
        rl_omega_max=math.pi,
        rl_baseline_decay=0.9,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_reinforce_moves_policy_toward_reward():
    # Clean contextual-bandit: reward = sampled alpha, so the policy gradient
    # should push the alpha mean upward. This checks the estimator's sign.
    torch.manual_seed(0)
    policy = HarmonicWeightPolicy(_cfg())
    state = torch.zeros(3)

    def mean_alpha(n=256):
        with torch.no_grad():
            raws = torch.stack([policy._dist(state).sample() for _ in range(n)])
        return float(policy.map_action(raws)[0].mean())

    before = mean_alpha()
    for _ in range(400):
        raw, (alpha, _, _) = policy.act(state)
        policy.update(state, raw, reward=alpha)  # reward increasing in alpha
    after = mean_alpha()
    assert after > before + 0.05, (before, after)
    # The baseline should track the (now positive) reward.
    assert float(policy.baseline) > 0.0


def test_gate_mask_selectors_are_deterministic():
    from praxis.policies.harmonic_weight_rl import build_gate_mask

    # uniform_hash: same seed -> identical mask; reproducible across calls.
    m1 = build_gate_mask("uniform_hash", 32, 0.0, 1.0, 0.0, seed=7, device="cpu")
    m2 = build_gate_mask("uniform_hash", 32, 0.0, 1.0, 0.0, seed=7, device="cpu")
    assert torch.equal(m1, m2) and m1.dtype == torch.bool and m1.numel() == 32
    # sinusoidal: a sub-threshold-crossing mask, deterministic in the index.
    s = build_gate_mask("sinusoidal", 16, 0.0, math.pi / 2, 0.0, seed=0, device="cpu")
    assert s.dtype == torch.bool and 0 < int(s.sum()) < 16
