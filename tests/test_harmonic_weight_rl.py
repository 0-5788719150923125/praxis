"""Harmonic-weight RL controller: policy-gradient mechanics + callback loop."""

import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from praxis.callbacks.lightning import HarmonicWeightRLCallback
from praxis.policies import RL_POLICIES_REGISTRY
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


class _Trainer:
    def __init__(self):
        self.callback_metrics = {}
        self.global_step = 0


class _PL:
    def __init__(self, model):
        self.model = model


def test_registered_in_rl_registry():
    assert RL_POLICIES_REGISTRY["harmonic_weight"] is HarmonicWeightPolicy
    assert HarmonicWeightPolicy.is_weight_controller is True


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


def test_callback_keeps_helpful_edit_and_updates_policy():
    torch.manual_seed(0)
    policy = HarmonicWeightPolicy(_cfg(rl_alpha_scale=0.3))
    cb = HarmonicWeightRLCallback(
        policy, period=3, horizon=2, warmup_steps=3, keep_threshold=0.0
    )
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    pl, tr = _PL(model), _Trainer()

    before = [p.detach().clone() for p in model.parameters()]
    # Five steps so exactly one episode completes (warmup=3 -> start at step 3,
    # horizon=2 -> finish at step 5) with no new episode left dangling.
    # Loss drops across the episode -> positive reward -> edit kept.
    losses = [5.0, 5.0, 5.0, 4.0, 2.0]
    for ls in losses:
        cb.on_train_batch_end(tr, pl, torch.tensor(ls), None, 0)

    assert cb._metrics, "an episode should have completed"
    assert cb._metrics["rl_edit_kept"] == 1.0
    # The kept edit changed exactly the model (some 2D weight row differs).
    after = list(model.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed
    # rl_* scalars were published to callback_metrics for the logger.
    assert "rl_reward" in tr.callback_metrics and "rl_edit_kept" in tr.callback_metrics


def test_callback_rolls_back_unhelpful_edit():
    torch.manual_seed(0)
    policy = HarmonicWeightPolicy(_cfg(rl_alpha_scale=0.3))
    cb = HarmonicWeightRLCallback(
        policy, period=3, horizon=2, warmup_steps=3, keep_threshold=0.0
    )
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    pl, tr = _PL(model), _Trainer()

    before = [p.detach().clone() for p in model.parameters()]
    # One completed episode (steps 3..5), nothing dangling. Loss rises across
    # the episode -> negative reward -> edit rolled back.
    losses = [3.0, 3.0, 3.0, 4.0, 6.0]
    for ls in losses:
        cb.on_train_batch_end(tr, pl, torch.tensor(ls), None, 0)

    assert cb._metrics["rl_edit_kept"] == 0.0
    # Rollback restored the weights exactly.
    after = list(model.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_gate_mask_selectors_are_deterministic():
    from praxis.policies.harmonic_weight_rl import build_gate_mask

    # uniform_hash: same seed -> identical mask; reproducible across calls.
    m1 = build_gate_mask("uniform_hash", 32, 0.0, 1.0, 0.0, seed=7, device="cpu")
    m2 = build_gate_mask("uniform_hash", 32, 0.0, 1.0, 0.0, seed=7, device="cpu")
    assert torch.equal(m1, m2) and m1.dtype == torch.bool and m1.numel() == 32
    # sinusoidal: a sub-threshold-crossing mask, deterministic in the index.
    s = build_gate_mask("sinusoidal", 16, 0.0, math.pi / 2, 0.0, seed=0, device="cpu")
    assert s.dtype == torch.bool and 0 < int(s.sum()) < 16


def test_anchor_gate_replaces_selected_with_anchor_and_rolls_back():
    torch.manual_seed(0)
    policy = HarmonicWeightPolicy(_cfg())
    # Near-deterministic action so the gate mask is stable for the assertion.
    policy.log_std.data.fill_(-10.0)
    cb = HarmonicWeightRLCallback(
        policy,
        period=3,
        horizon=2,
        warmup_steps=2,
        keep_threshold=0.0,
        edit_mode="anchor_gate",
        selector="sinusoidal",
    )
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    pl, tr = _PL(model), _Trainer()

    # Steps 1,2: anchor snapshot captured at warmup (step 2).
    for ls in (5.0, 5.0):
        cb.on_train_batch_end(tr, pl, torch.tensor(ls), None, 0)
    assert cb._anchor, "anchor should be snapshotted at warmup"

    # Simulate training drift: live weights move away from the anchor.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    # Step 3 starts the episode (gate-replaces a subset back to the anchor),
    # steps 4,5 run the horizon; loss drops -> reward>0 -> kept.
    for ls in (5.0, 4.0, 2.0):
        cb.on_train_batch_end(tr, pl, torch.tensor(ls), None, 0)

    assert cb._metrics["rl_edit_kept"] == 1.0
    assert 0.0 < cb._metrics["rl_gate_frac"] < 1.0
    # The kept edit pulled the gated elements back to the (older) anchor while
    # the ungated elements keep the drifted live value -> a partial reset.
    # Find the row that changed and verify it contains both anchor and live values.
    reverted = False
    for name, p in model.named_parameters():
        if name in cb._anchor and p.dim() == 2:
            anchor = cb._anchor[name]
            eq_anchor = (p.data == anchor).any(dim=1)
            eq_live = (p.data == anchor + 1.0).any(dim=1)
            if (eq_anchor & eq_live).any():
                reverted = True
    assert reverted, "expected a row gated partly to anchor, partly drifted-live"


def test_anchor_gate_rolls_back_unhelpful_edit():
    torch.manual_seed(0)
    policy = HarmonicWeightPolicy(_cfg())
    policy.log_std.data.fill_(-10.0)
    cb = HarmonicWeightRLCallback(
        policy,
        period=3,
        horizon=2,
        warmup_steps=2,
        keep_threshold=0.0,
        edit_mode="anchor_gate",
        selector="uniform_hash",
    )
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    pl, tr = _PL(model), _Trainer()

    for ls in (3.0, 3.0):
        cb.on_train_batch_end(tr, pl, torch.tensor(ls), None, 0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    before = [p.detach().clone() for p in model.parameters()]
    # Loss rises -> reward<0 -> rolled back to the (drifted) pre-edit weights.
    for ls in (3.0, 4.0, 6.0):
        cb.on_train_batch_end(tr, pl, torch.tensor(ls), None, 0)

    assert cb._metrics["rl_edit_kept"] == 0.0
    after = list(model.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))
