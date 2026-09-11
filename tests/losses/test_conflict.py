"""ObjectiveConflict: cosines between each loss term's trunk gradient and the anchor's.

The measurement multi-task gradient surgery (PCGrad, GradNorm, CAGrad, Nash-MTL)
is motivated by, taken cheaply: one backward per objective to the trunk output
the head classifies. The cases below pin the readings that matter - a term that
opposes the anchor, one that agrees with it, one that pulls in an independent
direction, and one with no path to the shared representation at all.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from praxis import registry
from praxis.losses.conflict import ObjectiveConflict, conflict_metric_descriptions
from praxis.losses.trunk_grads import trunk_gradients
from tests.stubs import Cfg, Enc


def make(interval=1):
    torch.manual_seed(0)
    return ObjectiveConflict(interval=interval), torch.randn(4, 6, requires_grad=True)


def test_exact_opposition_reads_minus_one():
    c, h = make()
    target = torch.randn(4, 6)
    main = (h - target).pow(2).sum()
    out = c.measure({"main": main, "against": -main}, h)
    assert out["conflict_against"] == pytest.approx(-1.0, abs=1e-5)
    assert out["conflict_min"] == pytest.approx(-1.0, abs=1e-5)
    assert c.anchor == "main"


def test_agreement_reads_plus_one():
    c, h = make()
    main = h.pow(2).sum()
    out = c.measure({"main": main, "with": 0.25 * main}, h)
    assert out["conflict_with"] == pytest.approx(1.0, abs=1e-5)


def test_only_terms_that_reach_the_trunk_are_reported():
    """Detached, constant and non-tensor terms are skipped, and a
    parameter-only term (centers_rms, the gate and router repulsions) gets no
    key at all: absence is the answer, not a zero."""
    c, h = make()
    w = torch.randn(6, requires_grad=True)
    out = c.measure(
        {
            "main": h.pow(2).sum(),
            "detached": h.detach().pow(2).sum(),
            "scalar": torch.tensor(0.0),
            "centers_rms": w.pow(2).sum(),
            "real": h.sum(),
        },
        h,
    )
    assert set(out) == {"conflict_real", "conflict_mag_real", "conflict_min"}


def test_magnitude_separates_inert_from_merely_orthogonal():
    """A cosine is scale-invariant, so a term contributing nothing and a term
    contributing a lot in an independent direction read identically (0 on
    disjoint coordinates). Only the magnitude twin tells them apart, and
    without it "no conflict" is not evidence that summing is fine."""
    c, h = make()
    main = h[:, :3].pow(2).sum()  # disjoint coordinates from both terms below
    out = c.measure(
        {"main": main, "loud": h[:, 3:].pow(2).sum(), "inert": 1e-6 * h[:, 3:].sum()},
        h,
    )
    assert out["conflict_loud"] == pytest.approx(0.0, abs=1e-6)
    assert out["conflict_inert"] == pytest.approx(0.0, abs=1e-6)
    # Identical cosines, wildly different participation.
    assert out["conflict_mag_loud"] > 1e-2
    assert out["conflict_mag_inert"] < 1e-4


def test_min_is_the_worst_of_the_terms():
    c, h = make()
    main = h.pow(2).sum()
    out = c.measure({"main": main, "a": 0.5 * main, "b": -main}, h)
    assert out["conflict_min"] == pytest.approx(
        min(out["conflict_a"], out["conflict_b"])
    )
    # Magnitude series must never be mistaken for a cosine when taking the min.
    assert out["conflict_min"] >= -1.0


def test_sampling_holds_its_value_between_measurements():
    """Sampled one step in `interval`, and the standing value is returned in
    between so the chart is a step curve rather than a sparse one."""
    c, h = make(interval=3)
    main = h.pow(2).sum()
    first = dict(c.measure({"main": main, "x": main}, h))  # step 0: measured
    assert first["conflict_x"] == pytest.approx(1.0, abs=1e-5)
    # Steps 1 and 2 are not due; a term that would read -1 must not appear.
    for _ in range(2):
        held = c.measure({"main": main, "x": -main}, h)
        assert held["conflict_x"] == pytest.approx(1.0, abs=1e-5)
    fresh = c.measure({"main": main, "x": -main}, h)  # step 3: due again
    assert fresh["conflict_x"] == pytest.approx(-1.0, abs=1e-5)


def test_no_anchor_or_frozen_input_is_a_no_op():
    c, h = make()
    assert c.measure({"only_aux": h.sum()}, h) == {}
    frozen = torch.randn(4, 6)
    assert c.measure({"main": frozen.sum(), "x": frozen.sum()}, frozen) == {}


def test_conflict_reports_once_the_anchor_resolves():
    """Without a live main term the anchor falls back to the surgical head's
    arm_surgery row, and the cosines are taken against it."""
    conflict = ObjectiveConflict(interval=1)
    h = torch.randn(8, requires_grad=True)
    anchor = (h * 2).sum()
    metrics = conflict.measure(
        {"arm_surgery": anchor, "harmonic_kl": -(h * 2).sum()}, h
    )
    assert metrics["conflict_harmonic_kl"] == pytest.approx(-1.0, abs=1e-5)
    assert metrics["conflict_mag_harmonic_kl"] == pytest.approx(1.0, abs=1e-5)
    assert conflict.anchor == "arm_surgery"


def test_the_real_backward_still_works_afterwards():
    """The sampler retains the graph; the training step that follows must not
    have been consumed by the measurement."""
    c, h = make()
    main = h.pow(2).sum()
    aux = h.sum()
    c.measure({"main": main, "aux": aux}, h)
    (main + aux).backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()


def test_descriptions_cover_the_live_series_only():
    descs = conflict_metric_descriptions(
        ["conflict_mtp", "conflict_mag_mtp", "conflict_min"]
    )
    assert set(descs) == {"conflict_mtp", "conflict_mag_mtp", "conflict_min"}
    # conflict_min owns the cosine chart; the rest ride it via series_group.
    assert "title" in descs["conflict_min"]["chart"]
    assert "title" not in descs["conflict_mtp"]["chart"]
    assert (
        descs["conflict_mtp"]["chart"]["series_group"]
        == descs["conflict_min"]["chart"]["series_group"]
    )
    # The magnitude twins get their own chart.
    assert "title" in descs["conflict_mag_mtp"]["chart"]
    assert descs["conflict_mag_mtp"]["chart"]["series_group"] == "conflict_mag"


def _head_terms(name):
    """A registered head's main CE and an entropy term over the same logits."""
    torch.manual_seed(0)
    head = registry.lookup("heads", name)(Cfg(), encoder=Enc())
    head.train()
    h = torch.randn(2, 5, 48, requires_grad=True)
    logits = head(h)
    labels = torch.randint(0, 32, (2, 5))
    ce = F.cross_entropy(logits.reshape(-1, 32), labels.reshape(-1))
    entropy = -(logits.exp() * logits).sum(-1).mean()
    return {"main": ce, "entropy": entropy}, h


@pytest.mark.parametrize("name", ["prismatic7", "prismatic8"])
def test_head_gradients_are_the_real_shape(name):
    """On an ordinary parallel head the mixture CE reaches the trunk output, so
    it anchors the measurement and the cosine is finite."""
    terms, h = _head_terms(name)
    out = ObjectiveConflict(interval=1).measure(terms, h)
    assert "conflict_entropy" in out
    assert math.isfinite(out["conflict_entropy"])
    assert -1.0 <= out["conflict_entropy"] <= 1.0


def test_a_surgical_head_leaves_main_without_a_trunk_row():
    """prismatic9 detaches every arm and the gate's input in training, so the
    mixture CE has no path to the trunk and anchoring on it measures nothing;
    the task reaches the trunk only through the arm_surgery row."""
    terms, h = _head_terms("prismatic9")
    assert "main" not in trunk_gradients(terms, h)
    assert ObjectiveConflict(interval=1).measure(terms, h) == {}
