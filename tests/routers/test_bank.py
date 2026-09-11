"""ExpertBank and SharpenedExpertBank (praxis/routers/bank.py).

Covers the fused merge (numerically identical to the per-parameter loop it
replaced), the per-depth diagnostics cadence, VEAR's expert-init latch, and the
routing diagnostics, which must describe the ROUTER rather than a subclass's
transform.
"""

import math

import pytest
import torch
import torch.nn as nn

from praxis.routers.bank import (
    ROUTING_METRICS_INTERVAL,
    VEAR_SHARPEN,
    ExpertBank,
    SharpenedExpertBank,
)

N_EXPERTS = 8


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


def _sharpen(p):
    s = p.pow(VEAR_SHARPEN)
    return s / s.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def test_construction_requires_experts():
    with pytest.raises(ValueError, match="requires 'experts'"):
        ExpertBank(Cfg())


# --- the merge is a performance change only ----------------------------------


def reference_merge(router, expert_weights):
    """The per-parameter python loop the fused merge replaced."""
    out = {}
    for name in router._collect_parameter_names(router.experts[0]):
        merged = None
        for i, expert in enumerate(router.experts):
            wp = router._get_module_parameter(expert, name) * expert_weights[i]
            merged = wp if merged is None else merged + wp
        out[name] = merged
    return out


@pytest.mark.parametrize("router_cls", [ExpertBank, SharpenedExpertBank])
def test_fused_merge_matches_the_reference_loop(router_cls):
    torch.manual_seed(0)
    r = make(router_cls)
    probs = torch.softmax(torch.randn(3, len(r.experts)), dim=-1)

    got = r._merge_expert_parameters(probs, current_depth=0)
    # VEAR sharpens before merging, so compare against the same weights it used.
    weights = _sharpen(probs) if isinstance(r, SharpenedExpertBank) else probs
    want = reference_merge(r, weights.mean(dim=0))

    assert set(got) == set(want)
    for name in want:
        assert torch.allclose(got[name], want[name], rtol=1e-6, atol=1e-7), name


def test_merge_keeps_gradients_flowing():
    torch.manual_seed(0)
    r = make(ExpertBank)
    probs = torch.softmax(torch.randn(2, len(r.experts)), dim=-1)
    merged = r._merge_expert_parameters(probs, 0)
    sum(v.sum() for v in merged.values()).backward()
    grads = [p.grad for e in r.experts for p in e.parameters() if p.grad is not None]
    assert grads, "the merge detached the experts from autograd"
    assert all(torch.isfinite(g).all() for g in grads)


def test_parameter_names_are_cached_not_rebuilt():
    r = make(ExpertBank)
    assert r.parameter_names == []
    probs = torch.softmax(torch.randn(2, len(r.experts)), dim=-1)
    r._merge_expert_parameters(probs, 0)
    first = r.parameter_names
    r._merge_expert_parameters(probs, 0)
    assert r.parameter_names is first, "names were rebuilt on the second merge"


# --- diagnostics cadence -----------------------------------------------------


def test_metrics_refresh_on_a_cadence_per_depth():
    r = make(ExpertBank)
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
    r = make(ExpertBank, depth=3)
    seen = []
    r._log_routing_metrics = lambda ew, rp, d, **k: seen.append(d)
    probs = torch.softmax(torch.randn(2, len(r.experts)), dim=-1)
    for _ in range(2):
        for d in range(3):
            r._merge_expert_parameters(probs, current_depth=d)
    assert sorted(seen) == [0, 1, 2], f"depths refreshed unevenly: {seen}"


def test_metrics_persist_between_refreshes():
    r = make(ExpertBank)
    probs = torch.softmax(torch.randn(2, len(r.experts)), dim=-1)
    r._merge_expert_parameters(probs, 0)
    assert r._metrics, "no metrics on the first merge"
    r._merge_expert_parameters(probs, 0)  # inside the period, skipped
    assert r._metrics, "metrics were cleared between refreshes"


def test_vear_latches_the_expert_init_check():
    r = make(SharpenedExpertBank)
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


# --- routing diagnostics -----------------------------------------------------
#
# The defect these pin: VEAR sharpens routing probabilities by ``p**4`` before the
# batch-mean merge, and the metrics were computed on the sharpened values. That drove
# the logged entropy to float-exact one-hot, where it stopped responding to the
# weights entirely - bit-identical across four models whose losses differed by 5%.
# ``_log_routing_metrics`` swallows every exception, so a key that vanishes shows up
# here as a KeyError rather than a raise.


class _Probe(ExpertBank):
    """Bare metric surface: exercise the logging path without building experts."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self._metrics = {}


class _VearProbe(SharpenedExpertBank):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self._metrics = {}


def _log(probe, probs, router_probs=None, depth=0):
    """Drive _log_routing_metrics the way _merge_expert_parameters does."""
    merge = probs.mean(dim=0)
    probe._log_routing_metrics(
        merge,
        probs if router_probs is None else router_probs,
        depth,
        merge_weights=merge,
    )
    return {k.replace(f"layer_{depth}_", ""): v for k, v in probe._metrics.items()}


def _diverse(batch=16, n=N_EXPERTS, seed=0):
    """Rows that each prefer a different expert - real input dependence."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, n, generator=g) * 2.0
    return torch.softmax(logits, dim=-1)


def test_entropy_is_never_negative():
    """`+ eps` pushed a weight of exactly 1.0 above 1, so log went positive and
    the reported entropy went negative. That was the saturation tell."""
    one_hot = torch.zeros(4, N_EXPERTS)
    one_hot[:, 0] = 1.0
    m = _log(_Probe(), one_hot)
    assert m["routing_entropy"] >= 0.0
    assert m["routing_entropy_seq"] >= 0.0
    assert m["routing_merge_entropy"] >= 0.0


def test_entropy_bounds_are_sane():
    uniform = torch.full((4, N_EXPERTS), 1.0 / N_EXPERTS)
    m = _log(_Probe(), uniform)
    assert m["routing_entropy"] == pytest.approx(math.log(N_EXPERTS), rel=1e-5)
    assert m["routing_entropy_seq"] == pytest.approx(math.log(N_EXPERTS), rel=1e-5)
    # Uniform routing is identical for every row, so no input dependence.
    assert m["routing_input_dependence"] == pytest.approx(0.0, abs=1e-6)
    assert m["routing_specialization"] == pytest.approx(0.0, abs=1e-6)

    # On realistic probs the MI stays in range: h_mean >= h_seq by concavity.
    m = _log(_Probe(), _diverse(seed=7))
    assert 0.0 <= m["routing_input_dependence"] <= 1.0
    assert m["routing_entropy"] >= m["routing_entropy_seq"] - 1e-6


def test_vear_sharpening_does_not_reach_the_router_diagnostics():
    """Same router output, with and without VEAR's transform."""
    probs = _diverse()
    plain = _log(_Probe(), probs)
    # VEAR merges the sharpened probs but must forward the originals for metrics.
    vear = _log(_VearProbe(), _sharpen(probs), router_probs=probs)

    for key in (
        "routing_entropy",
        "routing_entropy_seq",
        "routing_concentration",
        "routing_variance",
        "routing_peak",
        "routing_specialization",
        "routing_input_dependence",
    ):
        assert vear[key] == pytest.approx(plain[key], rel=1e-6), key


def test_merge_entropy_still_reports_the_transform():
    """The sharpening must remain visible - just not disguised as router state."""
    probs = _diverse()
    vear = _log(_VearProbe(), _sharpen(probs), router_probs=probs)
    # The merge is strictly more concentrated than the router's own opinion.
    assert vear["routing_merge_entropy"] < vear["routing_entropy"]

    plain = _log(_Probe(), probs)
    assert plain["routing_merge_entropy"] == pytest.approx(
        plain["routing_entropy"], rel=1e-6
    ), "for a plain bank the merge uses exactly the router's output"


def test_input_dependence_separates_constant_from_discriminating_routers():
    """A router that sends the WHOLE batch to one expert scores maximum
    specialization while having learned nothing. Only this metric catches it."""
    n = N_EXPERTS

    # Every row commits to the SAME expert: fully specialized, zero information.
    same = torch.zeros(16, n)
    same[:, 3] = 1.0
    m_same = _log(_Probe(), same)
    assert m_same["routing_specialization"] == pytest.approx(1.0, abs=1e-5)
    assert m_same["routing_input_dependence"] == pytest.approx(0.0, abs=1e-5)

    # Each row commits to a DIFFERENT expert: same specialization, max information.
    spread = torch.zeros(n, n)
    spread[torch.arange(n), torch.arange(n)] = 1.0
    m_spread = _log(_Probe(), spread)
    assert m_spread["routing_specialization"] == pytest.approx(1.0, abs=1e-5)
    assert m_spread["routing_input_dependence"] == pytest.approx(1.0, abs=1e-5)


def test_single_expert_router_does_not_divide_by_log_one():
    """log(1) = 0; the normalization must be skipped rather than blow up."""
    m = _log(_Probe(), torch.ones(4, 1))
    assert "routing_input_dependence" not in m
    assert "routing_specialization" not in m
    assert m["routing_entropy"] == pytest.approx(0.0, abs=1e-6)
