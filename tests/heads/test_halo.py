"""praxis/heads/halo.py: HaloClassifier scoring, HaloHead's blend contract, and
HaloHead's wiring as a prismatic arm. HALOLoss itself is tested in
tests/losses/test_halo.py."""

import math

import pytest
import torch

from praxis import registry
from praxis.heads import HaloHead
from praxis.heads.halo import HaloClassifier
from praxis.losses.halo import HALOLoss
from tests.stubs import Cfg, Enc

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


# ── HaloHead as a prismatic arm ──────────────────────────────────────────


def build(name):
    torch.manual_seed(0)
    return registry.lookup("heads", name)(Cfg(), encoder=Enc())


@pytest.mark.parametrize(
    "name",
    [
        "prismatic5",
        "prismatic6",
        "prismatic6_vear",
        "prismatic7",
        "prismatic8",
        "prismatic9",
    ],
)
def test_classifier_is_the_halo_arm(name):
    """HALOLoss keys composite mode off ``is_halo`` on the head's classifier,
    so every profile with a HALO arm must hand that arm out - whether or not
    it is detached in the blend."""
    clf = build(name).classifier
    assert isinstance(clf, HaloClassifier)
    assert clf.is_halo


def test_prismatic5_end_to_end_composite_loss():
    """Full honest wiring: trunk features -> prismatic5 logits + HALOLoss."""
    head = build("prismatic5").train()
    trunk = torch.randn(2, 8, Cfg.hidden_size, requires_grad=True)
    logits = head(trunk)
    labels = torch.randint(0, Cfg.vocab_size, (2, 8))
    loss_fn = HALOLoss(vocab_size=Cfg.vocab_size)
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
# prismatic5 detaches, prismatic6 onward attach, and neither should drift.


def test_constructor_overrides_per_instance():
    """The bare head keeps the honest-contract default; a profile overrides it
    per instance without moving the default out from under the others."""
    assert HaloHead.detach_in_blend is True
    assert HaloHead(Cfg(), detach_in_blend=False).detach_in_blend is False
    assert HaloHead(Cfg(), detach_in_blend=True).detach_in_blend is True
    # None leaves the class default alone.
    assert HaloHead(Cfg(), detach_in_blend=None).detach_in_blend is True
    assert HaloHead(Cfg()).detach_in_blend is True


@pytest.mark.parametrize(
    "name, detached",
    [
        ("prismatic5", True),
        ("prismatic6", False),
        ("prismatic6_vear", False),
        ("prismatic7", False),
        ("prismatic8", False),
    ],
)
def test_halo_arm_blend_gradient_and_gate(name, detached):
    """A detached arm gets no gradient from the blended CE; an attached one
    does. Either way the gate learns how much to trust it."""
    head = build(name).train()
    arms = [b for b in head.branches if isinstance(b, HaloHead)]
    assert len(arms) == 1, f"expected exactly one HALO arm, got {len(arms)}"
    arm = arms[0]
    assert arm.detach_in_blend is detached

    head(torch.randn(2, 8, Cfg.hidden_size)).sum().backward()
    g = arm.lm_head.centers.grad
    reached = g is not None and bool(g.abs().sum() > 0)
    assert reached is not detached
    assert head.gate.weight.grad is not None
    assert head.gate.weight.grad.abs().sum() > 0
