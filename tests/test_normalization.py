import pytest
import torch
import torch.nn as nn

from praxis import registry
from praxis.normalization import LayerNorm, NoNorm, RMSNorm

MODULE_CLASSES = list(registry.namespace("normalization").values())


@pytest.fixture(params=MODULE_CLASSES)
def norm_module(request):
    module_class = request.param
    normalized_shape = 64
    eps = 1e-5

    try:
        return module_class(normalized_shape, eps=eps)
    except Exception as e:
        pytest.skip(f"Failed to initialize normalization module: {str(e)}")


def test_forward_pass(norm_module):
    """Test forward pass with valid input."""
    batch_size = 32
    seq_len = 16
    hidden_size = 64

    x = torch.randn(batch_size, seq_len, hidden_size)

    try:
        output = norm_module(x)

        assert output.shape == (batch_size, seq_len, hidden_size)
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains infinite values"

    except Exception as e:
        pytest.fail(f"Forward pass failed: {str(e)}")


def test_registry_keys():
    """Test that all expected keys are present in the registry."""
    expected_keys = {
        "layer_norm",
        "rms_norm",
        "none",
        "post_rms_norm",
        "sandwich",
        "sandwich_tied",
        "hero",
        "hero_inverted",
    }
    actual_keys = set(registry.namespace("normalization").keys())

    assert (
        actual_keys == expected_keys
    ), f"Registry keys mismatch. Expected: {expected_keys}, Got: {actual_keys}"


def test_registry_values():
    """Test that registry values are callable functions."""
    for key, value in registry.namespace("normalization").items():
        assert callable(value), f"Registry value for '{key}' should be callable"


def test_no_normalization_identity():
    """Test that NoNorm returns input unchanged."""
    normalized_shape = 64
    norm = NoNorm(normalized_shape)

    x = torch.randn(10, 20, 64)
    output = norm(x)

    assert torch.equal(x, output), "NoNorm should return input unchanged"


def test_no_normalization_parameters():
    """Test that NoNorm stores parameters correctly."""
    normalized_shape = 128
    eps = 1e-6
    norm = NoNorm(normalized_shape, eps=eps)

    assert norm.normalized_shape == normalized_shape
    assert norm.eps == eps


def test_layer_norm_behavior():
    """Test LayerNorm basic functionality."""
    hidden_size = 64
    layer_norm = registry.lookup("normalization", "layer_norm")(hidden_size)

    x = torch.randn(32, 16, hidden_size)
    output = layer_norm(x)

    assert output.shape == x.shape

    # Check that output is normalized (mean ≈ 0, std ≈ 1)
    mean = output.mean(dim=-1)
    std = output.std(dim=-1)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-2)


def test_rms_norm_behavior():
    """Test RMSNorm basic functionality."""
    hidden_size = 64
    rms_norm = registry.lookup("normalization", "rms_norm")(hidden_size)

    x = torch.randn(32, 16, hidden_size)
    output = rms_norm(x)

    assert output.shape == x.shape

    # Check that RMS is normalized
    rms = torch.sqrt(torch.mean(output**2, dim=-1))
    expected_rms = torch.ones_like(rms)

    assert torch.allclose(rms, expected_rms, atol=1e-4)


def test_pre_post_norm_flags():
    """Test that normalization flags are set correctly."""
    hidden_size = 64

    # Test default configurations
    layer_norm = registry.lookup("normalization", "layer_norm")(hidden_size)
    rms_norm = registry.lookup("normalization", "rms_norm")(hidden_size)
    post_rms_norm = registry.lookup("normalization", "post_rms_norm")(hidden_size)
    sandwich_norm = registry.lookup("normalization", "sandwich_tied")(hidden_size)

    # Check default flags (pre_norm=True, post_norm=False)
    assert layer_norm.pre_norm == True
    assert layer_norm.post_norm == False
    assert rms_norm.pre_norm == True
    assert rms_norm.post_norm == False

    # Check post-norm configuration (pre_norm=False, post_norm=True)
    assert post_rms_norm.pre_norm == False
    assert post_rms_norm.post_norm == True

    # Check sandwich configuration (pre_norm=True, post_norm=True)
    assert sandwich_norm.pre_norm == True
    assert sandwich_norm.post_norm == True


def test_mode_based_forward():
    """Test the forward method with different mode parameters."""
    hidden_size = 64
    x = torch.randn(10, 20, hidden_size)

    # Test pre-norm configuration (default)
    layer_norm = registry.lookup("normalization", "layer_norm")(hidden_size)

    # Pre mode should apply normalization (pre_norm=True)
    pre_output = layer_norm(x, mode="pre")
    assert not torch.equal(pre_output, x)  # Should be normalized

    # Post mode should be no-op (post_norm=False)
    post_output = layer_norm(x, mode="post")
    assert torch.equal(post_output, x)  # Should be unchanged

    # Direct mode should always apply normalization
    direct_output = layer_norm(x, mode="direct")
    assert not torch.equal(direct_output, x)  # Should be normalized

    # None mode should always be no-op
    none_output = layer_norm(x, mode="none")
    assert torch.equal(none_output, x)  # Should be unchanged

    # Test post-norm configuration
    post_rms_norm = registry.lookup("normalization", "post_rms_norm")(hidden_size)

    # Pre mode should be no-op (pre_norm=False)
    pre_output = post_rms_norm(x, mode="pre")
    assert torch.equal(pre_output, x)  # Should be unchanged

    # Post mode should apply normalization (post_norm=True)
    post_output = post_rms_norm(x, mode="post")
    assert not torch.equal(post_output, x)  # Should be normalized


def test_both_mode():
    """Test the 'both' mode for normalization."""
    hidden_size = 64
    x = torch.randn(10, 20, hidden_size)

    # Create a custom normalization with both flags enabled
    both_norm = LayerNorm(hidden_size, pre_norm=True, post_norm=True)

    # Both mode should apply normalization when either flag is True
    both_output = both_norm(x, mode="both")
    assert not torch.equal(both_output, x)  # Should be normalized

    # Test with no flags enabled
    no_norm = LayerNorm(hidden_size, pre_norm=False, post_norm=False)
    no_output = no_norm(x, mode="both")
    assert torch.equal(no_output, x)  # Should be unchanged


def test_no_normalization_all_modes():
    """Test that NoNorm always returns input unchanged regardless of mode."""
    hidden_size = 64
    x = torch.randn(10, 20, hidden_size)

    none_norm = registry.lookup("normalization", "none")(hidden_size)

    # All modes should return input unchanged
    assert torch.equal(none_norm(x, mode="pre"), x)
    assert torch.equal(none_norm(x, mode="post"), x)
    assert torch.equal(none_norm(x, mode="both"), x)
    assert torch.equal(none_norm(x, mode="none"), x)
    assert torch.equal(none_norm(x, mode="direct"), x)


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
