"""Depth-aware routing (ArcSMEAR/ArcVEAR) and the SMEAR merge fast path.

Two things are pinned here:
  * the fused merge must be NUMERICALLY IDENTICAL to the per-parameter loop it
    replaced - it is a performance change, not a behaviour change;
  * the depth bias must be identity at init, so swapping smear -> arc_smear is a
    clean A/B rather than a different model.
"""

import pytest
import torch
import torch.nn as nn

from praxis.routers import ROUTER_REGISTRY
from praxis.routers.arc_smear import ArcSMEAR, ArcVEAR, DepthBiasedRouting
from praxis.routers.smear import ROUTING_METRICS_INTERVAL, SMEAR
from praxis.routers.vear import VEAR


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


@pytest.mark.parametrize("router_cls", [SMEAR, VEAR, ArcSMEAR, ArcVEAR])
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


@pytest.mark.parametrize("arc_cls,base_cls", [(ArcSMEAR, SMEAR), (ArcVEAR, VEAR)])
def test_identity_at_init(arc_cls, base_cls):
    """Zero-init bias => the depth-aware router IS its parent at step 0."""
    torch.manual_seed(7)
    arc = make(arc_cls)
    torch.manual_seed(7)
    base = make(base_cls)

    assert torch.equal(arc.depth_bias.weight, torch.zeros_like(arc.depth_bias.weight))
    x = torch.randn(3, Cfg.hidden_size)
    for d in range(Cfg.depth):
        assert torch.allclose(
            arc._route_logits(x, d), base._route_logits(x, d), atol=1e-7
        ), f"depth {d} diverged at init"


def test_depth_bias_changes_routing_once_trained():
    torch.manual_seed(0)
    r = make(ArcSMEAR)
    with torch.no_grad():
        r.depth_bias.weight.normal_(0, 1.0)
    x = torch.randn(3, Cfg.hidden_size)
    logits = [r._route_logits(x, d) for d in range(Cfg.depth)]
    for a in range(len(logits)):
        for b in range(a + 1, len(logits)):
            assert not torch.allclose(logits[a], logits[b]), f"depth {a} == {b}"


def test_depth_index_wraps_past_the_table():
    """Halting can sample deeper than `depth`; wrap rather than index-error."""
    r = make(ArcSMEAR, depth=3)
    with torch.no_grad():
        r.depth_bias.weight.normal_(0, 1.0)
    x = torch.randn(2, Cfg.hidden_size)
    assert torch.allclose(r._route_logits(x, 0), r._route_logits(x, 3))
    assert torch.allclose(r._route_logits(x, 1), r._route_logits(x, 4))


def test_depth_bias_is_tiny_relative_to_the_experts():
    r = make(ArcVEAR)
    bias = r.depth_bias.weight.numel()
    experts = sum(p.numel() for e in r.experts for p in e.parameters())
    assert bias == Cfg.depth * len(r.experts)
    assert bias < experts / 100, f"{bias} bias params vs {experts} expert params"


def test_arc_vear_keeps_vear_behaviour():
    r = make(ArcVEAR)
    assert isinstance(r, VEAR)
    assert hasattr(r, "router_aux_loss")
    r.train()
    aux = r.router_aux_loss()
    assert "vear_repulsion" in aux


def test_depth_metrics_ride_the_router_metric_channel():
    """get_metrics is what BaseDecoder collects; training_metrics is not."""
    r = make(ArcVEAR)
    probs = torch.softmax(torch.randn(2, len(r.experts)), dim=-1)
    r._merge_expert_parameters(probs, 0)  # populate the base routing metrics

    m = r.get_metrics()
    assert "router_depth_specialization" in m
    assert "router_depth_similarity" in m
    # the base router's own metrics must survive the override
    assert any(k.endswith("routing_entropy") for k in m), sorted(m)[:6]
    # zero-init => nothing has specialized yet
    assert m["router_depth_specialization"] == pytest.approx(0.0, abs=1e-6)


def test_plain_routers_emit_no_depth_metrics():
    r = make(VEAR)
    probs = torch.softmax(torch.randn(2, len(r.experts)), dim=-1)
    r._merge_expert_parameters(probs, 0)
    assert not [k for k in r.get_metrics() if k.startswith("router_depth_")]


def test_depth_metric_chart_is_registered():
    from praxis.metrics import COMPOSITE_METRIC_REGISTRY
    import re

    entry = next(
        e for e in COMPOSITE_METRIC_REGISTRY if e["key"] == "router_depth_bias"
    )
    pattern = re.compile(entry["key_pattern"])
    assert pattern.match("router_depth_specialization")
    assert pattern.match("router_depth_similarity")
    assert not pattern.match("router_depth_other")


def test_registered_under_both_names():
    assert ROUTER_REGISTRY["arc_smear"] is ArcSMEAR
    assert ROUTER_REGISTRY["arc_vear"] is ArcVEAR
