import pytest
import torch

from praxis import registry


def test_sandwich_norm_behavior():
    """Test sandwich normalization (both pre and post norm enabled)."""
    hidden_size = 64
    x = torch.randn(10, 20, hidden_size)

    sandwich_norm = registry.lookup("normalization", "sandwich_tied")(hidden_size)

    # Verify flags are set correctly
    assert sandwich_norm.pre_norm == True
    assert sandwich_norm.post_norm == True

    # Test pre mode - should apply normalization (pre_norm=True)
    pre_output = sandwich_norm(x, mode="pre")
    assert not torch.equal(pre_output, x)  # Should be normalized

    # Test post mode - should apply normalization (post_norm=True)
    post_output = sandwich_norm(x, mode="post")
    assert not torch.equal(post_output, x)  # Should be normalized

    # Test both mode - should apply normalization (both flags True)
    both_output = sandwich_norm(x, mode="both")
    assert not torch.equal(both_output, x)  # Should be normalized

    # Test direct mode - should always apply normalization
    direct_output = sandwich_norm(x, mode="direct")
    assert not torch.equal(direct_output, x)  # Should be normalized

    # Test none mode - should always be no-op
    none_output = sandwich_norm(x, mode="none")
    assert torch.equal(none_output, x)  # Should be unchanged

    # Verify it actually normalizes correctly (RMS should be ~1)
    normalized = sandwich_norm(x, mode="direct")
    rms = torch.sqrt(torch.mean(normalized**2, dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_hero_norm_positions_differ():
    """Hero centers on the read and only rescales on the write."""
    hidden_size = 64
    torch.manual_seed(0)
    x = torch.randn(10, 20, hidden_size)

    hero = registry.lookup("normalization", "hero")(hidden_size)

    assert hero.pre_norm == True
    assert hero.post_norm == True

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
