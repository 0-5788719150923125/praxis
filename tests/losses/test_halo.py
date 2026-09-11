import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from praxis.classifiers import HaloClassifier
from praxis.classifiers.halo import HaloGeometry
from praxis.losses.halo import HALOLoss
from tests.stubs import Cfg, Enc

# ------------------------------------------------------------------------------
# halo
# ------------------------------------------------------------------------------
# HALO honest contract: shared scorer + composite loss (prismatic5).


# ── HALOLoss: honest composite mode ──────────────────────────────────────


def _composite_setup(B=2, T=8, H=16, V=32):
    torch.manual_seed(0)
    scorer = HaloGeometry(hidden_size=H, vocab_size=V)
    embeddings = torch.randn(B, T, H, requires_grad=True)
    # Stand-in for the blended mixture logits (a leaf so grad is observable).
    logits = torch.randn(B, T, V, requires_grad=True)
    labels = torch.randint(0, V, (B, T))
    loss_fn = HALOLoss(vocab_size=V)
    return loss_fn, scorer, embeddings, logits, labels


def test_composite_trains_both_paths():
    loss_fn, scorer, embeddings, logits, labels = _composite_setup()
    loss = loss_fn(logits=logits, labels=labels, embeddings=embeddings, scorer=scorer)
    assert torch.isfinite(loss)
    loss.backward()
    # CE path reaches the emitted logits (gate + non-HALO arms)...
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    # ...and the geometric path reaches the arm's parameters and the trunk.
    assert scorer.centers.grad is not None and scorer.centers.grad.abs().sum() > 0
    assert scorer.gamma.grad is not None
    assert embeddings.grad is not None and embeddings.grad.abs().sum() > 0


def test_composite_exceeds_plain_ce():
    """The composite must contain the mixture CE exactly (1:1, no knob)."""
    loss_fn, scorer, embeddings, logits, labels = _composite_setup()
    loss = loss_fn(logits=logits, labels=labels, embeddings=embeddings, scorer=scorer)
    ce = F.cross_entropy(logits.reshape(-1, 32), labels.reshape(-1))
    assert loss.item() > ce.item()


def test_composite_respects_ignore_index_and_weights():
    """Perturbing the -100 column and the zero-weight column changes nothing;
    perturbing a live column does."""
    loss_fn, scorer, embeddings, logits, labels = _composite_setup()
    labels[:, 0] = -100
    weights = torch.ones_like(labels, dtype=torch.float32)
    weights[:, 1] = 0.0

    def score(col=None):
        e, l = embeddings.detach().clone(), logits.detach().clone()
        if col is not None:
            e[:, col] += 5.0 * torch.randn_like(e[:, col])
            l[:, col] += 5.0 * torch.randn_like(l[:, col])
        return loss_fn(
            logits=l, labels=labels, embeddings=e, scorer=scorer, loss_weights=weights
        ).item()

    base = score()
    assert math.isfinite(base)
    assert score(0) == pytest.approx(base, rel=1e-6)
    assert score(1) == pytest.approx(base, rel=1e-6)
    assert score(2) != pytest.approx(base, rel=1e-3)


# ── HALOLoss: legacy side-loss mode ──────────────────────────────────────


def test_legacy_linear_scorer_trains():
    torch.manual_seed(0)
    H, V = 16, 32
    scorer = nn.Linear(H, V)
    embeddings = torch.randn(2, 8, H, requires_grad=True)
    labels = torch.randint(0, V, (2, 8))
    loss_fn = HALOLoss(vocab_size=V)
    loss = loss_fn(logits=None, labels=labels, embeddings=embeddings, scorer=scorer)
    assert torch.isfinite(loss)
    loss.backward()
    assert scorer.weight.grad is not None
    assert embeddings.grad is not None


def test_legacy_calibrates_from_measured_geometry_once():
    torch.manual_seed(0)
    H, V = 16, 32
    scorer = nn.Linear(H, V)
    embeddings = torch.randn(2, 8, H)
    labels = torch.randint(0, V, (2, 8))
    loss_fn = HALOLoss(vocab_size=V)
    loss_fn(logits=None, labels=labels, embeddings=embeddings, scorer=scorer)
    assert bool(loss_fn._calibrated)
    # Never sharper than the official default 20/(2 - r_sq_target).
    cap = 20.0 / (2.0 - (1.0 - 2.0 / H))
    assert float(F.softplus(loss_fn.gamma).item()) <= cap + 1e-4
    gamma_after_first = float(loss_fn.gamma.item())
    # Simulate learning, then another step: calibration must not refire.
    with torch.no_grad():
        loss_fn.gamma.fill_(gamma_after_first + 1.0)
    loss_fn(logits=None, labels=labels, embeddings=embeddings, scorer=scorer)
    assert math.isclose(float(loss_fn.gamma.item()), gamma_after_first + 1.0)


def test_legacy_calibration_survives_state_dict_roundtrip():
    """Resume must not clobber the learned gamma with a fresh calibration."""
    torch.manual_seed(0)
    H, V = 16, 32
    scorer = nn.Linear(H, V)
    embeddings = torch.randn(2, 8, H)
    labels = torch.randint(0, V, (2, 8))
    src = HALOLoss(vocab_size=V)
    src(logits=None, labels=labels, embeddings=embeddings, scorer=scorer)
    with torch.no_grad():
        src.gamma.fill_(3.21)
    dst = HALOLoss(vocab_size=V)
    dst.load_state_dict(src.state_dict())
    dst(logits=None, labels=labels, embeddings=embeddings, scorer=scorer)
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
        logits=None, labels=labels, embeddings=embeddings, scorer=frozen
    )
    with torch.no_grad():
        frozen.weight += 0.5  # a uniform offset centering would erase
    loss_b = HALOLoss(vocab_size=V)(
        logits=None, labels=labels, embeddings=embeddings, scorer=frozen
    )
    assert not math.isclose(float(loss_a), float(loss_b), rel_tol=1e-4)


# ------------------------------------------------------------------------------
# prismatic9
# ------------------------------------------------------------------------------
# prismatic9: the arms train on their own objectives, PCGrad on the trunk.
#
# The mixture of softmaxes is a standard and correct way to COMBINE predictions, and
# prismatic9 does not change it. What it changes is that the mixture also decided, as a
# side effect, how much each arm got TRAINED - cross-entropy reaches arm i scaled by its
# posterior responsibility, so an arm the gate stops trusting stops receiving gradient.
# abstractinator-n drove one to 6.3e-08.
#
# Here each arm gets its own cross-entropy and the trunk receives one PCGrad-combined
# gradient over those objectives instead of their plain sum.


def test_geometry_is_still_suppressed_during_training():
    """The other half: the double-count the flag exists to prevent."""
    from praxis.classifiers.halo import HaloClassifier
    from praxis.losses.halo import HALOLoss
    from tests.stubs import Cfg, Enc

    torch.manual_seed(0)
    arm = HaloClassifier(Cfg(), encoder=Enc())
    crit = HALOLoss(vocab_size=32)
    x = torch.randn(2, 5, 48)
    y = torch.randint(0, 32, (2, 4))
    kw = dict(
        logits=arm(x)[..., :-1, :].contiguous(),
        labels=y,
        embeddings=x[..., :-1, :].contiguous(),
        scorer=arm.scorer,
    )
    crit.train()
    full = float(crit(**kw))
    crit.composite_geometry = False
    ce_only = float(crit(**kw))
    assert ce_only < full, "training-mode suppression stopped working"
    crit.eval()
    assert float(crit(**kw)) == pytest.approx(full, rel=1e-4)
