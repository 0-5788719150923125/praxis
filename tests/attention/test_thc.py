"""TemporalHealthComplex: a gated residual branch that runs a causal complex-valued
convolution over a reduced width. Live through ModularAttention when ``use_thc`` is
in ``config.meta``."""

import math

import pytest
import torch

from praxis.attention.thc import TemporalHealthComplex


@pytest.fixture
def thc():
    torch.manual_seed(0)
    return TemporalHealthComplex(d_model=64, reduction_factor=8)


@pytest.mark.parametrize(
    "d_model, reduction_factor, d_complex",
    [
        (128, 8, 16),  # the default reduction
        (256, 4, 64),
        (8, 16, 1),  # clamped: never zero
        (128, 256, 1),
    ],
)
def test_complex_width_is_the_reduced_width(d_model, reduction_factor, d_complex):
    thc = TemporalHealthComplex(d_model=d_model, reduction_factor=reduction_factor)
    assert thc.d_model == d_model and thc.d_complex == d_complex
    x = torch.randn(2, 4, d_model)
    assert thc(x).shape == x.shape


def test_gate_initialization_strategies():
    zeros = TemporalHealthComplex(d_model=128, gate_init="zeros")
    assert torch.equal(zeros.gate.weight, torch.zeros_like(zeros.gate.weight))
    assert torch.equal(zeros.gate.bias, torch.zeros_like(zeros.gate.bias))

    small = TemporalHealthComplex(d_model=128, gate_init="small")
    assert not torch.equal(small.gate.weight, torch.zeros_like(small.gate.weight))
    assert torch.equal(small.gate.bias, torch.zeros_like(small.gate.bias))

    ones = TemporalHealthComplex(d_model=128, gate_init="ones")
    assert torch.equal(ones.gate.weight, torch.zeros_like(ones.gate.weight))
    assert torch.equal(ones.gate.bias, torch.ones_like(ones.gate.bias))

    with pytest.raises(ValueError, match="Unknown gate initialization"):
        TemporalHealthComplex(d_model=128, gate_init="invalid")


@pytest.mark.parametrize("seq_len", [1, 16, 64])
def test_forward_preserves_shape_and_dtype(thc, seq_len):
    x = torch.randn(2, seq_len, 64)
    out = thc(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.isfinite(out).all()


def test_is_causal(thc):
    """The complex convolution pads on the left only: editing a token moves no
    earlier output."""
    thc.eval()
    x = torch.randn(2, 24, 64)
    with torch.no_grad():
        base = thc(x)
        for position in (5, 12, 20):
            edited = x.clone()
            edited[:, position] += 1.0
            moved = thc(edited)
            assert torch.equal(moved[:, :position], base[:, :position]), position
            assert not torch.equal(moved[:, position:], base[:, position:]), position


def test_gradient_flow(thc):
    x = torch.randn(4, 32, 64, requires_grad=True)
    thc.train()
    thc(x).sum().backward()
    assert x.grad.abs().sum() > 0
    for name, param in thc.named_parameters():
        assert param.grad is not None, name
        if "complex_conv" in name:
            assert param.grad.abs().sum() > 0, name


def test_complex_representations(thc):
    x = torch.randn(4, 32, 64)
    real, imag = thc.get_complex_representations(x)
    assert real.shape == imag.shape == (4, 32, thc.d_complex)
    assert real.dtype == imag.dtype == torch.float32
    assert real.abs().sum() > 0 and imag.abs().sum() > 0


def test_phase_statistics(thc):
    stats = thc.get_phase_statistics(torch.randn(4, 32, 64))
    assert set(stats) == {
        "mean_magnitude",
        "magnitude_std",
        "mean_phase",
        "phase_std",
        "mean_phase_diff",
        "phase_diff_std",
        "phase_coherence",
    }
    assert all(math.isfinite(v) for v in stats.values())
    assert stats["magnitude_std"] >= 0
    assert stats["phase_diff_std"] > 0  # the phase actually moves along time
    assert 0 <= stats["phase_coherence"] <= 1


def test_residual_branch_is_half_open_at_zero_gate_init():
    """``gate_init="zeros"`` gives sigmoid(0) = 0.5, not a closed gate: the output
    stays near the input, and a ones-init gate does move it."""
    x = torch.randn(2, 16, 128)
    with torch.no_grad():
        half = TemporalHealthComplex(d_model=128, gate_init="zeros").eval()(x)
        wide = TemporalHealthComplex(d_model=128, gate_init="ones").eval()(x)
    assert torch.norm(half - x) / torch.norm(x) < 0.3
    assert not torch.allclose(wide, x, atol=1e-3)


def test_large_inputs_stay_finite():
    thc = TemporalHealthComplex(d_model=128)
    x = torch.randn(2, 16, 128) * 100
    assert torch.isfinite(thc(x)).all()
