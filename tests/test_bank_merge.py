"""The bank merge's fused fast path (praxis/routers/bank.py).

Pins that the fused merge is NUMERICALLY IDENTICAL to the per-parameter loop it
replaced - a performance change, not a behaviour change.

The depth-bias half of this file is gone: ArcSMEAR/ArcVEAR no longer exist as
separate classes, because the per-recurrent-pass bias folded into SMEAR itself
(zero-init, so it is identity until it learns otherwise). Its coverage moved to
tests/test_smear.py::test_depth_bias_is_identity_at_init.
"""

import pytest
import torch
import torch.nn as nn

from praxis.routers import ROUTER_REGISTRY
from praxis.routers.bank import ROUTING_METRICS_INTERVAL
from praxis.routers.bank import ExpertBank as SMEAR
from praxis.routers.bank import SharpenedExpertBank as VEAR


class Expert(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.up = nn.Linear(d, d * 2)
        self.down = nn.Linear(d * 2, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(self.down(torch.relu(self.up(x))))


class Cfg:
    hidden_size = 16
    num_experts = 4
    depth = 6
    expert_dropout = 0.0


def make(router_cls, n_experts=4, depth=6):
    cfg = Cfg()
    cfg.num_experts = n_experts
    cfg.depth = depth
    experts = [Expert(cfg.hidden_size) for _ in range(n_experts)]
    return router_cls(cfg, experts=experts)


def reference_merge(router, expert_weights):
    """The pre-optimisation merge: per-parameter python loop."""
    out = {}
    names = router._collect_parameter_names(router.experts[0])
    for name in names:
        merged = None
        for i, expert in enumerate(router.experts):
            p = router._get_module_parameter(expert, name)
            wp = p * expert_weights[i]
            merged = wp if merged is None else merged + wp
        out[name] = merged
    return out


# ── the merge is a performance change only ──────────────────────────────────


@pytest.mark.parametrize("router_cls", [SMEAR, VEAR])
def test_fused_merge_matches_the_reference_loop(router_cls):
    torch.manual_seed(0)
    r = make(router_cls)
    probs = torch.softmax(torch.randn(3, len(r.experts)), dim=-1)

    got = r._merge_expert_parameters(probs, current_depth=0)
    # VEAR sharpens before merging, so compare against the same weights it used.
    weights = probs
    if isinstance(r, VEAR):
        s = probs.pow(4.0)
        weights = s / s.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    want = reference_merge(r, weights.mean(dim=0))

    assert set(got) == set(want)
    for name in want:
        assert torch.allclose(got[name], want[name], rtol=1e-6, atol=1e-7), name


def test_merge_keeps_gradients_flowing():
    torch.manual_seed(0)
    r = make(SMEAR)
    probs = torch.softmax(torch.randn(2, len(r.experts)), dim=-1)
    merged = r._merge_expert_parameters(probs, 0)
    sum(v.sum() for v in merged.values()).backward()
    grads = [p.grad for e in r.experts for p in e.parameters() if p.grad is not None]
    assert grads, "the merge detached the experts from autograd"
    assert all(torch.isfinite(g).all() for g in grads)


def test_parameter_names_are_cached_not_rebuilt():
    r = make(SMEAR)
    assert r.parameter_names == []
    probs = torch.softmax(torch.randn(2, len(r.experts)), dim=-1)
    r._merge_expert_parameters(probs, 0)
    first = r.parameter_names
    r._merge_expert_parameters(probs, 0)
    assert r.parameter_names is first, "names were rebuilt on the second merge"


# ── diagnostics cadence ─────────────────────────────────────────────────────


def test_metrics_refresh_on_a_cadence_per_depth():
    r = make(SMEAR)
    fired = []
    r._log_routing_metrics = lambda *a, **k: fired.append(k.get("_d", a[2]))
    probs = torch.softmax(torch.randn(2, len(r.experts)), dim=-1)

    for _ in range(ROUTING_METRICS_INTERVAL):
        r._merge_expert_parameters(probs, current_depth=0)
    assert len(fired) == 1, "diagnostics ran more than once inside one period"

    r._merge_expert_parameters(probs, current_depth=0)
    assert len(fired) == 2, "diagnostics never refreshed after the period"


def test_every_depth_gets_its_own_cadence():
    """A single global counter would starve some depths permanently."""
    r = make(SMEAR, depth=3)
    seen = []
    r._log_routing_metrics = lambda ew, rp, d, **k: seen.append(d)
    probs = torch.softmax(torch.randn(2, len(r.experts)), dim=-1)
    for _ in range(2):
        for d in range(3):
            r._merge_expert_parameters(probs, current_depth=d)
    assert sorted(seen) == [0, 1, 2], f"depths refreshed unevenly: {seen}"


def test_metrics_persist_between_refreshes():
    r = make(SMEAR)
    probs = torch.softmax(torch.randn(2, len(r.experts)), dim=-1)
    r._merge_expert_parameters(probs, 0)
    after_first = dict(r._metrics)
    assert after_first, "no metrics on the first merge"
    r._merge_expert_parameters(probs, 0)  # inside the period, skipped
    assert r._metrics, "metrics were cleared between refreshes"


# ── VEAR init latch ─────────────────────────────────────────────────────────


def test_vear_latches_the_expert_init_check():
    r = make(VEAR)
    assert r._experts_ready is False
    x = torch.randn(2, 4, Cfg.hidden_size)
    r._ensure_experts_initialized((x, None))
    assert r._experts_ready is True

    # Once latched the guard must not scan the experts again: make the scan
    # explode, then call it. `experts` is a registered submodule so it cannot be
    # swapped wholesale; shadowing one expert's `parameters` is enough.
    def boom(*a, **k):
        raise AssertionError("re-scanned experts after latching")

    r.experts[0].parameters = boom
    try:
        r._ensure_experts_initialized((x, None))
    finally:
        del r.experts[0].parameters


# ── depth-aware routing ─────────────────────────────────────────────────────
