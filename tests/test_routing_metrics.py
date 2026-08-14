"""Routing diagnostics must describe the ROUTER, not a subclass's transform.

The defect these pin: VEAR sharpens routing probabilities by ``p**4`` before the
batch-mean merge, and the metrics were computed on the sharpened values. That
drove the logged entropy to float-exact one-hot, where it stopped responding to
the weights entirely - measured bit-identical across four models whose losses
differed by 5%, which reads as a finding rather than as an absent metric.
"""

import math

import pytest
import torch

from praxis.metrics import COMPOSITE_METRIC_REGISTRY

# These pin the BANK merge's diagnostics (praxis/routers/bank.py). The
# targeted router's own metrics live in tests/test_smear.py.
from praxis.routers.bank import ExpertBank as SMEAR
from praxis.routers.bank import VEAR_SHARPEN
from praxis.routers.bank import SharpenedExpertBank as VEAR

N_EXPERTS = 8


class _Probe(SMEAR):
    """Bare metric surface: exercise the logging path without building experts."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self._metrics = {}


class _VearProbe(VEAR):
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


def _sharpen(p):
    s = p.pow(VEAR_SHARPEN)
    return s / s.sum(dim=-1, keepdim=True).clamp_min(1e-8)


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


def test_vear_sharpening_does_not_reach_the_router_diagnostics():
    """THE regression. Same router output, with and without VEAR's transform."""
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
    ), "for plain SMEAR the merge uses exactly the router's output"


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


def test_input_dependence_stays_in_range_on_realistic_probs():
    m = _log(_Probe(), _diverse(seed=7))
    assert 0.0 <= m["routing_input_dependence"] <= 1.0
    # h_mean >= h_seq by concavity, so the raw MI is non-negative.
    assert m["routing_entropy"] >= m["routing_entropy_seq"] - 1e-6


def test_single_expert_router_does_not_divide_by_log_one():
    """log(1) = 0; the normalization must be skipped rather than blow up."""
    m = _log(_Probe(), torch.ones(4, 1))
    assert "routing_input_dependence" not in m
    assert "routing_specialization" not in m
    assert m["routing_entropy"] == pytest.approx(0.0, abs=1e-6)


def test_new_metrics_are_charted():
    """Registry-driven charts: a metric with no key_pattern never renders."""
    patterns = {c.get("key_pattern") for c in COMPOSITE_METRIC_REGISTRY}
    for name in (
        "routing_entropy",
        "routing_entropy_seq",
        "routing_input_dependence",
        "routing_merge_entropy",
    ):
        assert rf"^layer_\d+_{name}$" in patterns, name
