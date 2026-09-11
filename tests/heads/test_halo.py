"""HALO honest contract: shared scoring head + composite loss (prismatic5)."""

import math
from types import SimpleNamespace

import pytest
import torch

from praxis import registry
from praxis.heads import HaloHead
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


# ── prismatic5 wiring ────────────────────────────────────────────────────


def _prismatic5(cfg):
    return registry.lookup("heads", "prismatic5")(cfg, encoder=None)


def test_prismatic5_builds_and_classifier_prefers_halo_arm():
    torch.manual_seed(0)
    head = _prismatic5(_cfg())
    assert len(head.branches) == 4
    clf = head.classifier
    assert getattr(clf, "is_halo", False)
    assert isinstance(clf, HaloClassifier)


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
    head = registry.lookup("heads", "prismatic5")(_cfg())
    arm = _halo_arm(head)
    assert arm.detach_in_blend is True
    assert not _ce_reaches(head, arm)


@pytest.mark.parametrize("name", ["prismatic6", "prismatic6_vear"])
def test_prismatic6_arm_is_attached(name):
    """The detached measurement is complete (0.00125 gate share over 22k
    steps in -j), so prismatic6 lets CE train the arm too."""
    torch.manual_seed(0)
    head = registry.lookup("heads", name)(_cfg())
    arm = _halo_arm(head)
    assert arm.detach_in_blend is False
    assert _ce_reaches(head, arm)


def test_geometric_objective_runs_either_way():
    """Attaching changes what ALSO trains the arm, never whether HALOLoss
    finds it - composite mode keys off is_halo, not off detachment."""
    for name in ("prismatic5", "prismatic6"):
        head = registry.lookup("heads", name)(_cfg())
        clf = head.classifier
        assert getattr(clf, "is_halo", False), f"{name} lost composite mode"
