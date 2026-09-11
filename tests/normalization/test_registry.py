import pytest
import torch

from praxis import registry

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
