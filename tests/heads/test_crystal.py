"""praxis/heads/crystal.py: CrystalSmearHead (prismatic7's bank) and its parent
CrystalVearHead.

The SMEAR bank routes per example (each row's logits read only that row) and is
one shared geometry plus low-rank deviations, so it starts EXACTLY equal to a
single-geometry head and the shared base trains every step.
"""

import pytest
import torch
import torch.nn as nn

from praxis.heads.crystal import CrystalSmearHead, CrystalVearHead
from tests.stubs import Cfg, Enc


def _make(cls=CrystalSmearHead, n=4):
    torch.manual_seed(0)
    return cls(Cfg(), encoder=Enc(), n_experts=n)


def test_deviations_are_exactly_zero_at_init():
    """LoRA init: b is zero, so every expert IS the base and prismatic7 starts
    bit-identical to a single-geometry head. That is what makes the swap an
    A/B rather than a reroll."""
    head = _make()
    stack = head._expert_centers()
    base = head.bank.experts[0].centers
    for e in range(stack.shape[0]):
        torch.testing.assert_close(stack[e], base, rtol=0, atol=0)


def test_bank_is_base_plus_deviations_not_n_center_sets():
    smear, vear = _make(CrystalSmearHead), _make(CrystalVearHead)
    assert len(smear.bank.experts) == 1, "more than one full center set retained"
    assert len(vear.bank.experts) == 4
    n_smear = sum(p.numel() for p in smear.parameters())
    n_vear = sum(p.numel() for p in vear.parameters())
    assert n_smear < n_vear, f"smear {n_smear} is not cheaper than vear {n_vear}"


def _liven(head):
    """Give the routing and the geometries something to say."""
    with torch.no_grad():
        nn.init.normal_(head.bank.router.weight, std=3.0)
        if isinstance(head, CrystalSmearHead):
            nn.init.normal_(head.lora_b, std=0.3)
        else:
            for e in head.bank.experts:
                nn.init.normal_(e.centers, std=0.5)
    head.bank.dropout_rate = 0.0  # deterministic
    return head.train()


@pytest.mark.parametrize("cls", [CrystalSmearHead, CrystalVearHead])
def test_training_routes_per_example(cls):
    """Row 0 holds still while row 1 changes, and row 1's own logits move,
    which keeps the check from being vacuous."""
    head = _liven(_make(cls))
    same = torch.randn(1, 5, Cfg.hidden_size)
    a = torch.cat([same, torch.randn(1, 5, Cfg.hidden_size) * 8], dim=0)
    b = torch.cat([same, torch.randn(1, 5, Cfg.hidden_size) * 8], dim=0)
    with torch.no_grad():
        la, lb = head(a), head(b)
    torch.testing.assert_close(la[0], lb[0], rtol=1e-4, atol=1e-4)
    assert not torch.allclose(la[1], lb[1], rtol=1e-3, atol=1e-3)


def test_shared_base_and_deviations_receive_gradient():
    head = _make()
    head.train()
    head(torch.randn(3, 6, Cfg.hidden_size)).sum().backward()
    assert head.bank.experts[0].centers.grad is not None
    assert head.bank.experts[0].centers.grad.abs().sum() > 0
    assert head.lora_a.grad is not None and head.lora_b.grad is not None


def test_every_declared_pca_card_is_emitted():
    """The bank declares one Center PCA Density card per EXPERT, and the
    snapshot loop has to fill all of them, although ``bank.experts`` holds only
    the shared trunk."""
    head = _make()
    declared = {k for k in head.all_metric_descriptions() if "centers_pca" in k}
    emitted = {k for k in head.dashboard_snapshots() if "centers_pca" in k}
    assert len(declared) == 4
    assert declared == emitted, f"blank cards: {sorted(declared - emitted)}"


def test_pca_panels_share_one_frame_and_are_deterministic():
    """The panels exist to be compared, so they must be drawn in the same
    projection: identical geometries (the LoRA init) render identically, and a
    trained deviation shows up as displacement rather than as a re-fit. Repeat
    calls must also agree, without drawing from the training RNG stream."""
    head = _make()
    first = head.dashboard_snapshots()
    assert first == head.dashboard_snapshots()

    grids = [first[f"crystal_centers_pca_{k}"]["grid"] for k in range(4)]
    assert all(g == grids[0] for g in grids), "identical experts drew differently"

    with torch.no_grad():  # give expert 2 a deviation to show
        head.lora_b[2].normal_(0.0, 0.5)
    moved = [
        head.dashboard_snapshots()[f"crystal_centers_pca_{k}"]["grid"] for k in range(4)
    ]
    assert moved[2] != moved[1], "the deviation left no mark on its own panel"
    # The frame spans every expert, so the untouched panels are re-drawn too;
    # what must hold is that they still agree with EACH OTHER.
    assert all(g == moved[0] for g in (moved[1], moved[3]))


def test_smear_paper_config_repulsion_off_dropout_on():
    """Repulsion is VEAR's, not the paper's, and meaningless for deviations off
    a shared base. Expert dropout IS the paper's balancing mechanism."""
    head = _make().train()
    assert head._rep_scale == 0.0
    assert "crystal_bank_repulsion" not in head.aux_losses()
    assert head.bank.dropout_rate > 0


@pytest.mark.parametrize("dims", [(2, 7), (1, 1)])
def test_eval_forward_runs_for_rows_and_single_tokens(dims):
    head = _make()
    head.eval()
    with torch.no_grad():
        out = head(torch.randn(*dims, Cfg.hidden_size))
    assert out.shape[:-1] == torch.Size(dims)
    assert torch.isfinite(out).all()
