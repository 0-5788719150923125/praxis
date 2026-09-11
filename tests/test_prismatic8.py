"""prismatic8: the crystal bank retired, three fixed arms over one stem.

The head keeps its three-way choice between a geometric, a direct and a
hyperspherical readout. What it gives up is the arm that re-picked its own
output geometry per example: by the time features reach the classifier the
trunk has already routed them at every level it offers, and the last stage of
the model should be the predictable one.

The measurement that retired it (abstractinator-m, step 17343): the routed
bank did not perturb one geometry, it grew four unequal and partly degenerate
ones - LoRA deviations at 0.36x / 2.61x / 1.27x / 0.85x the base's Frobenius
norm, per-expert effective_dim of 13 / 4 / 9 / 21 against the base's 23, and
expert 1 collapsed into four dimensions with 84% of its variance in the top
two PCs.
"""

import pytest
import torch
import torch.nn as nn

from praxis import registry
from praxis.heads.crystal import CrystalHead, CrystalVearHead
from praxis.heads.forward import ForwardHead
from praxis.heads.halo import HaloHead
from praxis.heads.parallel import ParallelHead


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
    """Minimal encoder declaring an output layout."""

    def __init__(self, d=48, v=32):
        super().__init__()
        self.output_dim = d
        self.output_vocab_size = v


def build(name="prismatic8"):
    torch.manual_seed(0)
    return registry.lookup("heads", name)(Cfg(), encoder=Enc())


def test_registered_and_distinct_from_prismatic7():
    assert "prismatic8" in registry.namespace("heads")
    assert registry.lookup("heads", "prismatic8") is not registry.lookup(
        "heads", "prismatic7"
    )


def test_arms_are_crystal_forward_halo_with_no_bank():
    """The geometric arm is a plain CrystalHead. Not a bank of one - no router
    and no experts exist on this path at all."""
    head = build()
    assert isinstance(head, ParallelHead)
    kinds = [type(b) for b in head.branches]
    assert kinds == [CrystalHead, ForwardHead, HaloHead]
    assert not any(isinstance(b, CrystalVearHead) for b in head.branches)
    for b in head.branches:
        assert not hasattr(b, "bank")
        assert not hasattr(b, "lora_a")
        assert not hasattr(b, "router")


def test_stem_is_shared_and_other_arms_match_prismatic7():
    """Only the geometric arm changes, so a prismatic7 -> prismatic8 delta
    attributes to the bank and to nothing else."""
    eight, seven = build("prismatic8"), build("prismatic7")
    assert eight.stem is not None and seven.stem is not None
    assert type(eight.stem) is type(seven.stem)
    assert [type(b) for b in eight.branches[1:]] == [
        type(b) for b in seven.branches[1:]
    ]
    # The HALO arm stays ATTACHED, as prismatic6/7 made it.
    assert eight.branches[2].detach_in_blend is False


def test_causal_readout_restored():
    """The bank pooled the sequence to route, which cost the speculative
    decoder a re-encode per candidate. A single crystal reads position t from
    position t."""
    head = build()
    assert head.branches[0].causal_readout is True
    assert head.causal_readout is True


def test_logits_shape_and_mixture_is_normalized():
    head = build().eval()
    x = torch.randn(2, 5, 48)
    with torch.no_grad():
        out = head(x)
    assert out.shape == (2, 5, 32)
    # _gate_combine_logits emits a log-prob mixture: rows exponentiate to 1.
    assert torch.allclose(out.exp().sum(-1), torch.ones(2, 5), atol=1e-4)


def test_geometry_is_one_table_in_its_own_frame():
    """The pre-bank view: a single PCA card, not four sharing a frame that the
    largest deviation stretches."""
    head = build()
    snaps = head.dashboard_snapshots()
    pca = [k for k in snaps if "crystal_centers_pca" in k]
    assert pca == ["p0_crystal_centers_pca"], pca
    assert "p0_crystal_bank_distinctness" not in head.training_metrics()


def test_gradient_reaches_the_crystal_centers():
    head = build()
    x = torch.randn(2, 5, 48, requires_grad=True)
    head(x).sum().backward()
    centers = head.branches[0].classifier.centers
    assert centers.grad is not None
    assert torch.isfinite(centers.grad).all()
    assert centers.grad.abs().sum() > 0


@pytest.mark.parametrize("n_pos", [1, 7])
def test_prefix_invariance(n_pos):
    """Reading position t out of a longer row equals running the prefix that
    ends at t - the property the bank's sequence pooling broke."""
    head = build().eval()
    torch.manual_seed(1)
    x = torch.randn(1, 8, 48)
    with torch.no_grad():
        full = head(x)[0, n_pos]
        prefix = head(x[:, : n_pos + 1])[0, n_pos]
    assert torch.allclose(full, prefix, atol=1e-5)
