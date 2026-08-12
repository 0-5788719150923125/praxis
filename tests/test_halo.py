"""HALO honest contract: shared scoring head + composite loss (prismatic5)."""

import math
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from praxis.heads import HEAD_REGISTRY, HaloHead, ParallelHead
from praxis.heads.halo import HaloClassifier
from praxis.losses.halo import HALOLoss


def _cfg(**over):
    base = dict(
        hidden_size=16,
        vocab_size=32,
        max_position_embeddings=64,
        encoder_type="",
        loss_func="halo",
        crystal_n=None,
        crystal_label_smoothing=None,
        tie_word_embeddings=False,
        embed_size=16,
    )
    base.update(over)
    return SimpleNamespace(**base)


# ── HaloClassifier: the scoring function ─────────────────────────────────


def test_classifier_emits_top_zero_distance_logits():
    torch.manual_seed(0)
    clf = HaloClassifier(hidden_size=16, vocab_size=32)
    x = torch.randn(2, 8, 16)
    logits = clf(x)
    assert logits.shape == (2, 8, 32)
    assert torch.isfinite(logits).all()
    # Crystal-style contract: top logit pinned at 0, everything else below.
    top = logits.amax(dim=-1)
    assert torch.allclose(top, torch.zeros_like(top), atol=1e-5)
    assert (logits <= 1e-5).all()


def test_classifier_centroids_are_mean_centered():
    torch.manual_seed(0)
    clf = HaloClassifier(hidden_size=16, vocab_size=32)
    cen = clf.centroids()
    assert torch.allclose(cen.mean(dim=0), torch.zeros(16), atol=1e-6)


def test_classifier_calibration_matches_official_formulas():
    D, K = 16.0, 32.0
    clf = HaloClassifier(hidden_size=16, vocab_size=32)
    r_sq_target = 1.0 - 2.0 / D
    init_gamma = 20.0 / (2.0 - r_sq_target)
    assert math.isclose(float(clf.gamma_value().item()), init_gamma, rel_tol=1e-5)
    ls = 0.1
    margin_ce = math.log((1.0 - ls + ls / K) / (ls / K))
    t_ideal = init_gamma * (1.0 - r_sq_target)
    assert math.isclose(clf.abstain_bias, t_ideal - margin_ce, rel_tol=1e-6)


def test_classifier_scoring_is_scale_invariant():
    """RMS normalization must decouple the ranking from upstream scale."""
    torch.manual_seed(0)
    clf = HaloClassifier(hidden_size=16, vocab_size=32)
    x = torch.randn(4, 16)
    a = clf(x)
    b = clf(x * 37.0)
    assert torch.allclose(a, b, atol=1e-4)


# ── HALOLoss: honest composite mode ──────────────────────────────────────


def _composite_setup(B=2, T=8, H=16, V=32):
    torch.manual_seed(0)
    clf = HaloClassifier(hidden_size=H, vocab_size=V)
    embeddings = torch.randn(B, T, H, requires_grad=True)
    # Stand-in for the blended mixture logits (a leaf so grad is observable).
    logits = torch.randn(B, T, V, requires_grad=True)
    labels = torch.randint(0, V, (B, T))
    loss_fn = HALOLoss(vocab_size=V)
    return loss_fn, clf, embeddings, logits, labels


def test_composite_trains_both_paths():
    loss_fn, clf, embeddings, logits, labels = _composite_setup()
    loss = loss_fn(logits=logits, labels=labels, embeddings=embeddings, classifier=clf)
    assert torch.isfinite(loss)
    loss.backward()
    # CE path reaches the emitted logits (gate + non-HALO arms)...
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    # ...and the geometric path reaches the arm's parameters and the trunk.
    assert clf.centers.grad is not None and clf.centers.grad.abs().sum() > 0
    assert clf.gamma.grad is not None
    assert embeddings.grad is not None and embeddings.grad.abs().sum() > 0


def test_composite_exceeds_plain_ce():
    """The composite must contain the mixture CE exactly (1:1, no knob)."""
    loss_fn, clf, embeddings, logits, labels = _composite_setup()
    loss = loss_fn(logits=logits, labels=labels, embeddings=embeddings, classifier=clf)
    ce = F.cross_entropy(logits.reshape(-1, 32), labels.reshape(-1))
    assert float(loss) > float(ce)


def test_composite_respects_ignore_index_and_weights():
    loss_fn, clf, embeddings, logits, labels = _composite_setup()
    labels[:, 0] = -100
    weights = torch.ones_like(labels, dtype=torch.float32)
    weights[:, 1] = 0.0
    loss = loss_fn(
        logits=logits,
        labels=labels,
        embeddings=embeddings,
        classifier=clf,
        loss_weights=weights,
    )
    assert torch.isfinite(loss)


# ── HALOLoss: legacy side-loss mode ──────────────────────────────────────


def test_legacy_linear_classifier_still_works():
    torch.manual_seed(0)
    H, V = 16, 32
    classifier = nn.Linear(H, V)
    embeddings = torch.randn(2, 8, H, requires_grad=True)
    labels = torch.randint(0, V, (2, 8))
    loss_fn = HALOLoss(vocab_size=V)
    loss = loss_fn(
        logits=None, labels=labels, embeddings=embeddings, classifier=classifier
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert classifier.weight.grad is not None
    assert embeddings.grad is not None


def test_legacy_calibrates_from_measured_geometry_once():
    torch.manual_seed(0)
    H, V = 16, 32
    classifier = nn.Linear(H, V)
    embeddings = torch.randn(2, 8, H)
    labels = torch.randint(0, V, (2, 8))
    loss_fn = HALOLoss(vocab_size=V)
    loss_fn(logits=None, labels=labels, embeddings=embeddings, classifier=classifier)
    assert bool(loss_fn._calibrated)
    # Never sharper than the official default 20/(2 - r_sq_target).
    cap = 20.0 / (2.0 - (1.0 - 2.0 / H))
    assert float(F.softplus(loss_fn.gamma).item()) <= cap + 1e-4
    gamma_after_first = float(loss_fn.gamma.item())
    # Simulate learning, then another step: calibration must not refire.
    with torch.no_grad():
        loss_fn.gamma.fill_(gamma_after_first + 1.0)
    loss_fn(logits=None, labels=labels, embeddings=embeddings, classifier=classifier)
    assert math.isclose(float(loss_fn.gamma.item()), gamma_after_first + 1.0)


def test_legacy_calibration_survives_state_dict_roundtrip():
    """Resume must not clobber the learned gamma with a fresh calibration."""
    torch.manual_seed(0)
    H, V = 16, 32
    classifier = nn.Linear(H, V)
    embeddings = torch.randn(2, 8, H)
    labels = torch.randint(0, V, (2, 8))
    src = HALOLoss(vocab_size=V)
    src(logits=None, labels=labels, embeddings=embeddings, classifier=classifier)
    with torch.no_grad():
        src.gamma.fill_(3.21)
    dst = HALOLoss(vocab_size=V)
    dst.load_state_dict(src.state_dict())
    dst(logits=None, labels=labels, embeddings=embeddings, classifier=classifier)
    assert math.isclose(float(dst.gamma.item()), 3.21, rel_tol=1e-6)


def test_legacy_frozen_centroids_are_not_centered():
    """A frozen instrument (CALM's codec path) must be measured as-is: the
    same embeddings should score differently once the matrix is offset,
    because no centering removes the offset."""
    torch.manual_seed(0)
    H, V = 16, 32
    embeddings = torch.randn(2, 8, H)
    labels = torch.randint(0, V, (2, 8))

    frozen = nn.Linear(H, V)
    frozen.weight.requires_grad_(False)
    loss_a = HALOLoss(vocab_size=V)(
        logits=None, labels=labels, embeddings=embeddings, classifier=frozen
    )
    with torch.no_grad():
        frozen.weight += 0.5  # a uniform offset centering would erase
    loss_b = HALOLoss(vocab_size=V)(
        logits=None, labels=labels, embeddings=embeddings, classifier=frozen
    )
    assert not math.isclose(float(loss_a), float(loss_b), rel_tol=1e-4)


# ── prismatic5 wiring ────────────────────────────────────────────────────


def _prismatic5(cfg):
    return HEAD_REGISTRY["prismatic5"](cfg, encoder=None)


def test_prismatic5_builds_and_classifier_prefers_halo_arm():
    torch.manual_seed(0)
    head = _prismatic5(_cfg())
    assert len(head.branches) == 4
    clf = head.classifier
    assert getattr(clf, "is_halo", False)
    assert isinstance(clf, HaloClassifier)


def test_prismatic5_forward_shape_and_finite():
    torch.manual_seed(0)
    head = _prismatic5(_cfg())
    x = torch.randn(2, 8, 16)
    out = head(x)
    assert out.shape == (2, 8, 32)
    assert torch.isfinite(out).all()


def test_halo_arm_detached_in_blend_but_gate_learns():
    torch.manual_seed(0)
    head = _prismatic5(_cfg())
    head.train()
    x = torch.randn(2, 8, 16)
    out = head(x)
    out.sum().backward()
    halo_arm = head.branches[-1]
    assert isinstance(halo_arm, HaloHead)
    # Blend gradient must not reach the HALO arm (its logits are detached)...
    assert (
        halo_arm.lm_head.centers.grad is None
        or halo_arm.lm_head.centers.grad.abs().sum() == 0
    )
    # ...but the gate still learns how much to trust it.
    assert head.gate.weight.grad is not None
    assert head.gate.weight.grad.abs().sum() > 0


def test_prismatic5_end_to_end_composite_loss():
    """Full honest wiring: trunk features -> prismatic5 logits + HALOLoss."""
    torch.manual_seed(0)
    cfg = _cfg()
    head = _prismatic5(cfg)
    head.train()
    trunk = torch.randn(2, 8, 16, requires_grad=True)
    logits = head(trunk)
    labels = torch.randint(0, cfg.vocab_size, (2, 8))
    loss_fn = HALOLoss(vocab_size=cfg.vocab_size)
    loss = loss_fn(
        logits=logits[..., :-1, :].contiguous(),
        labels=labels[..., 1:].contiguous(),
        embeddings=trunk[..., :-1, :].contiguous(),
        classifier=head.classifier,
    )
    assert torch.isfinite(loss)
    loss.backward()
    halo_arm = head.branches[-1]
    # HALO's geometric term trains the arm (through embeddings/centroids)...
    assert halo_arm.lm_head.centers.grad is not None
    assert halo_arm.lm_head.centers.grad.abs().sum() > 0
    # ...the mixture CE trains the gate and reaches the trunk.
    assert head.gate.weight.grad is not None
    assert trunk.grad is not None and trunk.grad.abs().sum() > 0


# ── detach_in_blend: which objective trains the HALO arm ─────────────────
#
# Not a correctness switch - a measurement one. Detached, the arm's gate share
# is an uncontaminated verdict on HALO's scoring function; attached, CE also
# reaches it and the verdict is traded for the chance the arm becomes useful.
# prismatic5 detaches, prismatic6 attaches, and neither should drift silently.


def _halo_arm(head):
    arms = [b for b in head.branches if isinstance(b, HaloHead)]
    assert len(arms) == 1, f"expected exactly one HALO arm, got {len(arms)}"
    return arms[0]


def _ce_reaches(head, arm):
    """True when the blended CE puts gradient on the arm's own centroids."""
    head.train()
    head.zero_grad(set_to_none=True)
    head(torch.randn(2, 8, head.output_dims()[0])).sum().backward()
    g = arm.lm_head.centers.grad
    return g is not None and bool(g.abs().sum() > 0)


def test_class_default_detaches():
    """The bare head keeps the original honest-contract default."""
    assert HaloHead.detach_in_blend is True
    assert HaloHead(_cfg()).detach_in_blend is True


def test_constructor_overrides_per_instance():
    """Per-instance override, so a profile can choose without moving the
    default out from under the profiles already running."""
    assert HaloHead(_cfg(), detach_in_blend=False).detach_in_blend is False
    assert HaloHead(_cfg(), detach_in_blend=True).detach_in_blend is True
    # None leaves the class default alone.
    assert HaloHead(_cfg(), detach_in_blend=None).detach_in_blend is True


def test_prismatic5_arm_stays_detached():
    """abstractinator-j runs this; its gate share is only a clean verdict
    while CE is kept off the arm."""
    torch.manual_seed(0)
    head = HEAD_REGISTRY["prismatic5"](_cfg())
    arm = _halo_arm(head)
    assert arm.detach_in_blend is True
    assert not _ce_reaches(head, arm)


@pytest.mark.parametrize("name", ["prismatic6", "prismatic6_vear"])
def test_prismatic6_arm_is_attached(name):
    """The detached measurement is complete (0.00125 gate share over 22k
    steps in -j), so prismatic6 lets CE train the arm too."""
    torch.manual_seed(0)
    head = HEAD_REGISTRY[name](_cfg())
    arm = _halo_arm(head)
    assert arm.detach_in_blend is False
    assert _ce_reaches(head, arm)


def test_geometric_objective_runs_either_way():
    """Attaching changes what ALSO trains the arm, never whether HALOLoss
    finds it - composite mode keys off is_halo, not off detachment."""
    for name in ("prismatic5", "prismatic6"):
        head = HEAD_REGISTRY[name](_cfg())
        clf = head.classifier
        assert getattr(clf, "is_halo", False), f"{name} lost composite mode"


def test_detach_is_training_only():
    """Inference always blends the real logits; detachment is a gradient
    concern, so it must not change what the model emits."""
    torch.manual_seed(0)
    head = HEAD_REGISTRY["prismatic5"](_cfg())
    x = torch.randn(2, 8, head.output_dims()[0])
    head.eval()
    with torch.no_grad():
        out = head(x)
    assert out.requires_grad is False
    assert torch.isfinite(out).all()
