import pytest
import torch

from praxis import PraxisConfig
from praxis.blocks.wavelet import WaveletBlock


def make_block(hidden_size: int = 128, **kwargs) -> WaveletBlock:
    config = PraxisConfig(hidden_size=hidden_size, **kwargs)
    return WaveletBlock(config)


@pytest.mark.parametrize("hidden_size", [64, 96, 128])
@pytest.mark.parametrize("seq_len", [1, 7, 16, 64])
def test_shape_preserved(hidden_size, seq_len):
    """Output keeps the input shape, even at non-power-of-2 widths/lengths."""
    block = make_block(hidden_size).eval()
    x = torch.randn(2, seq_len, hidden_size)
    out, kv, state, aux = block(x)
    assert out.shape == x.shape
    assert kv is None and state is None
    assert float(aux) == 0.0


def test_causality():
    """Perturbing position t must not affect any output before t."""
    torch.manual_seed(0)
    block = make_block(128).eval()
    x = torch.randn(1, 32, 128)
    t = 20
    with torch.no_grad():
        base = block(x)[0]
        x2 = x.clone()
        x2[:, t, :] = torch.randn(128)
        pert = block(x2)[0]
    assert torch.allclose(base[:, :t], pert[:, :t], atol=1e-5)
    assert not torch.allclose(base[:, t], pert[:, t], atol=1e-5)


def test_gradient_flow():
    """All parameters receive finite gradients."""
    block = make_block(64)
    x = torch.randn(2, 16, 64, requires_grad=True)
    block(x)[0].sum().backward()
    grads = [p.grad for p in block.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert not any(torch.isnan(g).any() for g in grads)
