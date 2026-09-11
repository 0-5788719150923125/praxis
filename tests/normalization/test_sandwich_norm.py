"""Tests for praxis/normalization/sandwich_norm.py: the paired (two-position) norms."""

import torch

from praxis import registry


def test_tied_sandwich_normalizes_to_unit_rms():
    norm = registry.lookup("normalization", "sandwich_tied")(64)
    out = norm(torch.randn(10, 20, 64), mode="direct")
    rms = torch.sqrt(torch.mean(out**2, dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_hero_norm_positions_differ():
    """Hero centers on the read and only rescales on the write."""
    hidden_size = 64
    torch.manual_seed(0)
    x = torch.randn(10, 20, hidden_size)

    hero = registry.lookup("normalization", "hero")(hidden_size)

    assert hero.pre_norm
    assert hero.post_norm

    # Pre position is a LayerNorm: mean is removed.
    pre_output = hero(x, mode="pre")
    pre_mean = pre_output.mean(dim=-1)
    assert torch.allclose(pre_mean, torch.zeros_like(pre_mean), atol=1e-5)

    # Post position is an RMSNorm: unit RMS, but the mean survives.
    post_output = hero(x, mode="post")
    post_rms = torch.sqrt(torch.mean(post_output**2, dim=-1))
    post_mean = post_output.mean(dim=-1)
    assert torch.allclose(post_rms, torch.ones_like(post_rms), atol=1e-4)
    assert not torch.allclose(post_mean, torch.zeros_like(post_mean), atol=1e-5)

    # The two positions therefore disagree.
    assert not torch.equal(pre_output, post_output)

    # "none" stays a no-op, "direct" falls to the read norm.
    assert torch.equal(hero(x, mode="none"), x)
    assert torch.equal(hero(x, mode="direct"), pre_output)


def test_hero_inverted_mirrors_hero():
    """The inverted hero swaps which position centers."""
    hidden_size = 64
    torch.manual_seed(0)
    x = torch.randn(10, 20, hidden_size)

    inverted = registry.lookup("normalization", "hero_inverted")(hidden_size)

    post_mean = inverted(x, mode="post").mean(dim=-1)
    pre_mean = inverted(x, mode="pre").mean(dim=-1)

    assert torch.allclose(post_mean, torch.zeros_like(post_mean), atol=1e-5)
    assert not torch.allclose(pre_mean, torch.zeros_like(pre_mean), atol=1e-5)


def test_sandwich_weight_tying():
    """`sandwich` gives each position its own weight; `sandwich_tied` shares one."""
    hidden_size = 64

    untied = registry.lookup("normalization", "sandwich")(hidden_size)
    tied = registry.lookup("normalization", "sandwich_tied")(hidden_size)
    hero = registry.lookup("normalization", "hero")(hidden_size)

    # Default: two independent RMSNorm weights. Also the control for `hero`,
    # since hero-vs-sandwich then isolates the LayerNorm/RMSNorm swap alone.
    assert len(list(untied.parameters())) == 2
    assert untied.pre.weight is not untied.post.weight

    # The pre-2026-09-09 behaviour: one weight, reused at both positions.
    assert len(list(tied.parameters())) == 1

    # LayerNorm (weight + bias) on the read, RMSNorm (weight) on the write.
    assert len(list(hero.parameters())) == 3


def test_paired_norm_post_is_unused_at_direct_only_sites():
    """A direct-only call site never reaches the post norm.

    Blocks call pre then post, but the MTP heads call `mode="direct"` only, so
    the post norm's weight gets no gradient there. Pinned because it decides
    whether a paired norm is free to drop into every site in the registry.
    """
    hidden_size = 64
    hero = registry.lookup("normalization", "hero")(hidden_size)

    x = torch.randn(4, 8, hidden_size)
    hero(x, mode="direct").sum().backward()

    assert hero.pre.weight.grad is not None
    assert hero.post.weight.grad is None

    # Both positions do receive gradient when a site actually sandwiches.
    hero.zero_grad(set_to_none=True)
    h = hero(x, mode="pre")
    hero(h, mode="post").sum().backward()

    assert hero.pre.weight.grad is not None
    assert hero.post.weight.grad is not None
