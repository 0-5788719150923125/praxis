"""Level-repulsion on ParallelHead gate weights.

The gate's mean per-branch weights should settle at DISTINCT tiers (e.g.
70/20/10) rather than degenerate ties (70/15/15). The ``gate_repulsion``
constructor arg (bound by the prismatic3_repel head-registry profile) adds a
pairwise log-gap penalty that repels equal weights apart, like energy levels.
Off by default; the min-gap diagnostic is always exposed.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from praxis.heads.base import BaseHead
from praxis.heads.parallel import ParallelHead


class IdBranch(BaseHead):
    """Identity branch - the gate is what we're testing, not the branches."""

    def transform(self, h):
        return h

    def forward(self, h, **k):
        return h

    @property
    def classifier(self):
        return None


def _cfg(hidden=8, vocab=8):
    return SimpleNamespace(hidden_size=hidden, vocab_size=vocab)


def _head(lam, n=3):
    # gate_repulsion is a constructor kwarg (bound by the head-registry profile),
    # not a config flag.
    return ParallelHead(
        _cfg(), encoder=None, branches=[IdBranch for _ in range(n)], gate_repulsion=lam
    )


def test_repulsion_off_by_default():
    h = _head(lam=0.0).train()
    h.transform(torch.randn(4, 16, 8))
    assert "gate_repulsion" not in h.aux_losses()


def test_min_gap_diagnostic_always_present():
    # Even with repulsion off, the degeneracy diagnostic is charted.
    h = _head(lam=0.0).train()
    h.transform(torch.randn(4, 16, 8))
    tm = h.training_metrics()
    assert "gate_min_gap" in tm and "gate_entropy" in tm
    assert all(f"gate_weight_{i}" in tm for i in range(3))


def test_aux_loss_is_prescaled_by_lambda():
    x = torch.randn(4, 16, 8)
    torch.manual_seed(0)
    h1 = _head(lam=1.0).train()
    torch.manual_seed(0)
    h2 = _head(lam=3.0).train()
    # identical gate init (same seed) -> identical repulsion energy, scaled by lam
    h1.transform(x)
    h2.transform(x)
    r1 = float(h1.aux_losses()["gate_repulsion"].detach())
    r2 = float(h2.aux_losses()["gate_repulsion"].detach())
    assert abs(r2 - 3.0 * r1) < 1e-4


def test_repulsion_breaks_a_tie_into_distinct_tiers():
    """Start from a near-degenerate gate (all branches ~equal) and optimize the
    repulsion alone: the mean weights must separate into distinct tiers."""
    torch.manual_seed(0)
    h = _head(lam=1.0, n=3).train()
    # Near-tie start: tiny gate weights -> softmax ~ uniform (min_gap ~ 0).
    with torch.no_grad():
        h.gate.weight.mul_(0.01)
    x = torch.randn(8, 32, 8)

    h.transform(x)
    start_gap = h._gate_min_gap
    assert start_gap < 0.02  # genuinely tied to begin with

    opt = torch.optim.SGD(h.gate.parameters(), lr=0.5)
    for _ in range(300):
        opt.zero_grad()
        h.transform(x)
        h.aux_losses()["gate_repulsion"].backward()
        opt.step()

    h.transform(x)
    end_gap = h._gate_min_gap
    weights = sorted(h._gate_mean.tolist(), reverse=True)
    # The tiers are now clearly separated, not degenerate.
    assert end_gap > start_gap + 0.1, (start_gap, end_gap)
    # All three remain distinct (no two within the floor of each other).
    assert weights[0] - weights[1] > 0.02 and weights[1] - weights[2] > 0.02
