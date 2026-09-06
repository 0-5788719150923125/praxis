"""Objective conflict: cosines between each loss term's trunk gradient and the
main loss's.

The measurement multi-task gradient surgery (PCGrad, GradNorm, CAGrad,
Nash-MTL) is motivated by, taken cheaply: rather than a full Jacobian over
shared parameters, one backward per objective to the trunk output the head
classifies. The cases below pin the three readings that matter - an objective
that opposes the main loss, one that agrees with it, and one that has no path
to the shared representation at all.
"""

import math

import pytest
import torch

from praxis.losses.conflict import ANCHOR, ObjectiveConflict


def make(interval=1):
    torch.manual_seed(0)
    return ObjectiveConflict(interval=interval), torch.randn(4, 6, requires_grad=True)


def test_exact_opposition_reads_minus_one():
    c, h = make()
    target = torch.randn(4, 6)
    main = (h - target).pow(2).sum()
    out = c.measure({ANCHOR: main, "against": -main}, h)
    assert out["conflict_against"] == pytest.approx(-1.0, abs=1e-5)
    assert out["conflict_min"] == pytest.approx(-1.0, abs=1e-5)


def test_agreement_reads_plus_one():
    c, h = make()
    main = h.pow(2).sum()
    out = c.measure({ANCHOR: main, "with": 0.25 * main}, h)
    assert out["conflict_with"] == pytest.approx(1.0, abs=1e-5)


def test_orthogonal_objectives_read_zero():
    """Two terms shaping disjoint coordinates do not compete over them, which
    is the reading that says a plain sum is already the right combination."""
    c, h = make()
    main = h[:, :3].pow(2).sum()
    other = h[:, 3:].pow(2).sum()
    out = c.measure({ANCHOR: main, "elsewhere": other}, h)
    assert out["conflict_elsewhere"] == pytest.approx(0.0, abs=1e-6)


def test_parameter_only_term_emits_nothing():
    """centers_rms, the gate repulsion and the router repulsions have no path
    to the shared representation. Absence is the answer, not a zero."""
    c, h = make()
    w = torch.randn(6, requires_grad=True)
    out = c.measure({ANCHOR: h.pow(2).sum(), "centers_rms": w.pow(2).sum()}, h)
    assert "conflict_centers_rms" not in out


def test_constant_and_non_tensor_terms_are_skipped():
    c, h = make()
    out = c.measure(
        {
            ANCHOR: h.pow(2).sum(),
            "detached": h.detach().pow(2).sum(),
            "scalar": torch.tensor(0.0),
            "real": h.sum(),
        },
        h,
    )
    assert set(out) == {"conflict_real", "conflict_min"}


def test_min_is_the_worst_of_the_terms():
    c, h = make()
    main = h.pow(2).sum()
    out = c.measure({ANCHOR: main, "a": 0.5 * main, "b": -main}, h)
    assert out["conflict_min"] == pytest.approx(min(out["conflict_a"], out["conflict_b"]))


def test_sampling_holds_its_value_between_measurements():
    """Sampled one step in `interval`, and the standing value is returned in
    between so the chart is a step curve rather than a sparse one."""
    c, h = make(interval=3)
    main = h.pow(2).sum()
    first = dict(c.measure({ANCHOR: main, "x": main}, h))  # step 0: measured
    assert first["conflict_x"] == pytest.approx(1.0, abs=1e-5)
    # Steps 1 and 2 are not due; a term that would read -1 must not appear.
    for _ in range(2):
        held = c.measure({ANCHOR: main, "x": -main}, h)
        assert held["conflict_x"] == pytest.approx(1.0, abs=1e-5)
    fresh = c.measure({ANCHOR: main, "x": -main}, h)  # step 3: due again
    assert fresh["conflict_x"] == pytest.approx(-1.0, abs=1e-5)


def test_no_anchor_or_frozen_input_is_a_no_op():
    c, h = make()
    assert c.measure({"only_aux": h.sum()}, h) == {}
    frozen = torch.randn(4, 6)
    assert c.measure({ANCHOR: frozen.sum(), "x": frozen.sum()}, frozen) == {}


def test_the_real_backward_still_works_afterwards():
    """The sampler retains the graph; the training step that follows must not
    have been consumed by the measurement."""
    c, h = make()
    main = h.pow(2).sum()
    aux = h.sum()
    c.measure({ANCHOR: main, "aux": aux}, h)
    (main + aux).backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()


def test_descriptions_cover_the_live_series_only():
    from praxis.losses.conflict import conflict_metric_descriptions

    descs = conflict_metric_descriptions(["conflict_mtp", "conflict_min"])
    assert set(descs) == {"conflict_mtp", "conflict_min"}
    # conflict_min owns the chart; the rest ride it via series_group.
    assert "title" in descs["conflict_min"]["chart"]
    assert "title" not in descs["conflict_mtp"]["chart"]
    assert (
        descs["conflict_mtp"]["chart"]["series_group"]
        == descs["conflict_min"]["chart"]["series_group"]
    )


def test_head_gradients_are_the_real_shape():
    """End to end on a prismatic8 head: the main CE and an aux term over the
    same logits both reach the trunk output, and the cosine is finite."""
    from tests.test_prismatic8 import Cfg, Enc
    from praxis.heads import HEAD_REGISTRY

    torch.manual_seed(0)
    head = HEAD_REGISTRY["prismatic8"](Cfg(), encoder=Enc())
    h = torch.randn(2, 5, 48, requires_grad=True)
    logits = head(h)
    labels = torch.randint(0, 32, (2, 5))
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 32), labels.reshape(-1))
    entropy = -(logits.exp() * logits).sum(-1).mean()

    out = ObjectiveConflict(interval=1).measure({ANCHOR: ce, "entropy": entropy}, h)
    assert "conflict_entropy" in out
    assert math.isfinite(out["conflict_entropy"])
    assert -1.0 <= out["conflict_entropy"] <= 1.0
