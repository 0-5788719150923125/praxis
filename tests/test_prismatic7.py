"""prismatic7: the crystal bank merged the way the SMEAR paper merges.

prismatic6 is already SMEAR in its routing EXPONENT (sharpen 1.0) and in
nothing else. Two things separate it from the paper, and both are pinned here:

  * routing is per EXAMPLE, not on the batch mean. Under a batch mean the loss
    reaches the coefficients only through ``mean(dim=0)``, so every example
    contributes the identical routing gradient and a constant router is the
    fixed point - which is what smear_input_dependence sitting near zero
    through abstractinator-m/n/p has been;
  * the bank is one shared geometry plus low-rank deviations, not N independent
    center sets, so it is EXACTLY prismatic6 at init and the shared trunk keeps
    full gradient however the routing collapses.
"""

import pytest
import torch
import torch.nn as nn

from praxis.heads import HEAD_REGISTRY
from praxis.heads.crystal import CrystalSmearHead, CrystalVearHead


class Cfg:
    hidden_size = 48
    embed_size = 48
    vocab_size = 32
    loss_func = "cross_entropy"
    tie_word_embeddings = False
    crystal_n = None
    crystal_label_smoothing = 0.0
    embedding_rms_lambda = 0.0
    causal = True
    debug = False


class Enc(nn.Module):
    """Minimal encoder declaring an output layout, which the bank requires."""

    def __init__(self, d=48, v=32):
        super().__init__()
        self.output_dim = d
        self.output_vocab_size = v


def make(cls=CrystalSmearHead, n=4):
    torch.manual_seed(0)
    return cls(Cfg(), encoder=Enc(), n_experts=n)


def test_registered_and_distinct_from_prismatic6():
    assert "prismatic7" in HEAD_REGISTRY
    assert HEAD_REGISTRY["prismatic7"] is not HEAD_REGISTRY["prismatic6"]


def test_deviations_are_exactly_zero_at_init():
    """LoRA init: b is zero, so every expert IS the base and prismatic7 starts
    bit-identical to a single-geometry head. That is what makes the swap an
    A/B rather than a reroll."""
    head = make()
    stack = head._expert_centers()
    base = head.bank.experts[0].centers
    for e in range(stack.shape[0]):
        torch.testing.assert_close(stack[e], base, rtol=0, atol=0)


def test_bank_is_base_plus_deviations_not_n_center_sets():
    smear, vear = make(CrystalSmearHead), make(CrystalVearHead)
    assert len(smear.bank.experts) == 1, "more than one full center set retained"
    assert len(vear.bank.experts) == 4
    n_smear = sum(p.numel() for p in smear.parameters())
    n_vear = sum(p.numel() for p in vear.parameters())
    assert n_smear < n_vear, f"smear {n_smear} is not cheaper than vear {n_vear}"


def test_training_routes_per_example_not_on_the_batch_mean():
    """THE property. Two examples whose routing differs must get different
    logits for the same hidden state; under a batch mean they cannot."""
    head = make()
    with torch.no_grad():  # give the deviations something to say
        nn.init.normal_(head.lora_b, std=0.3)
        nn.init.normal_(head.router.weight if hasattr(head, "router")
                        else head.bank.router.weight, std=3.0)
    head.train()

    same = torch.randn(1, 5, Cfg.hidden_size)
    a = torch.cat([same, torch.randn(1, 5, Cfg.hidden_size) * 8], dim=0)
    b = torch.cat([same, torch.randn(1, 5, Cfg.hidden_size) * 8], dim=0)
    head.bank.dropout_rate = 0.0  # deterministic

    with torch.no_grad():
        la, lb = head(a), head(b)
    # Row 0 is the SAME input in both batches. Its logits may only differ if
    # the merge is per example... and must NOT differ, since row 0 routes on
    # itself. A batch-mean merge would let row 1 move row 0.
    torch.testing.assert_close(la[0], lb[0], rtol=1e-4, atol=1e-4)


def test_batch_mean_parent_fails_that_same_property():
    """Proves the test above is not vacuous: prismatic6's bank DOES let one
    example's routing move another's logits."""
    head = make(CrystalVearHead)
    with torch.no_grad():
        for e in head.bank.experts:
            nn.init.normal_(e.centers, std=0.5)
        nn.init.normal_(head.bank.router.weight, std=3.0)
    head.train()
    head.bank.dropout_rate = 0.0

    same = torch.randn(1, 5, Cfg.hidden_size)
    a = torch.cat([same, torch.randn(1, 5, Cfg.hidden_size) * 8], dim=0)
    b = torch.cat([same, torch.randn(1, 5, Cfg.hidden_size) * 8], dim=0)
    with torch.no_grad():
        la, lb = head(a), head(b)
    assert not torch.allclose(la[0], lb[0], rtol=1e-3, atol=1e-3), (
        "the parent no longer merges on the batch mean; this control is stale"
    )


def test_shared_trunk_receives_gradient_however_routing_falls():
    head = make()
    head.train()
    head(torch.randn(3, 6, Cfg.hidden_size)).sum().backward()
    assert head.bank.experts[0].centers.grad is not None
    assert head.bank.experts[0].centers.grad.abs().sum() > 0
    assert head.lora_a.grad is not None and head.lora_b.grad is not None


def test_repulsion_is_off():
    """VEAR's, not the paper's, and meaningless for deviations off a shared base."""
    head = make()
    head.train()
    assert head._rep_scale == 0.0
    assert "crystal_bank_repulsion" not in head.aux_losses()


def test_expert_dropout_is_retained():
    """Dropout IS the paper's balancing mechanism, so it stays."""
    assert make().bank.dropout_rate > 0


@pytest.mark.parametrize("dims", [(2, 7), (1, 1)])
def test_inference_paths_still_run(dims):
    head = make()
    head.eval()
    with torch.no_grad():
        out = head(torch.randn(*dims, Cfg.hidden_size))
    assert out.shape[:-1] == torch.Size(dims)
    assert torch.isfinite(out).all()
