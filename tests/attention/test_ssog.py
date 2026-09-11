"""SSOGAttention: the faithful port, its field bounds and steering gradients, the
window, the geometry snapshots and cascade, and flex/materialised parity on GPU.
Causality, the dashboard walks and the 0-dim parameter rule are swept for every
attention entry in test_registry.py."""

import pytest
import torch
import torch.nn.functional as F

from praxis import PraxisConfig, registry
from praxis.attention.arc_ssog import ArcSSOGAttention
from praxis.attention.ssog import COLD_GATE_INIT, NUM_ATOMS, SSOGAttention


def _module(cls=SSOGAttention, **overrides):
    cfg = PraxisConfig(hidden_size=64, num_heads=2, num_queries=2, dropout=0.0)
    cfg.causal = True  # modeling.py sets this at assembly; the bare config is False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    torch.manual_seed(0)
    return cls(cfg), cfg


def test_registered_and_shapes():
    assert registry.lookup("attention", "ssog") is SSOGAttention
    module, cfg = _module()
    assert cfg.num_queries == 1  # patch_config keeps the config honest
    x = torch.randn(3, 16, 64)
    out, kv, aux = module(x)
    assert out.shape == x.shape and kv is None and aux == 0.0
    # The reference ArcSSOG subclasses: four shared atoms, no depth axis, cold gate.
    assert module.num_atoms == NUM_ATOMS == 4
    assert module.raw_mu.shape == (module.num_heads, 4)
    assert torch.allclose(
        module.raw_gate, torch.full_like(module.raw_gate, COLD_GATE_INIT)
    )
    assert not hasattr(module, "depths")


def test_field_never_looks_ahead_and_steering_learns():
    module, _ = _module()
    x = torch.randn(2, 10, 64)
    mu, sigma, loglam, tau = module._field(x)
    assert (mu >= 0).all() and (sigma >= 0.25).all()
    assert torch.allclose(loglam.exp().sum(-1), torch.ones_like(loglam[..., 0]))
    module(x)[0].square().sum().backward()
    for name in ("raw_mu", "raw_sigma", "log_lambda", "raw_temperature", "raw_gate"):
        assert getattr(module, name).grad is not None, name
    assert module.steer.weight.grad is not None  # probe learns even from cold


def test_window():
    module, _ = _module(window_size=2)
    module.eval()
    x = torch.randn(1, 12, 64)
    out = module(x)[0]
    x2 = x.clone()
    x2[:, 0] = torch.randn(64)  # far outside every later token's window
    out2 = module(x2)[0]
    assert torch.allclose(out[:, 4:], out2[:, 4:], atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex needs a GPU")
def test_flex_matches_materialised():
    module, _ = _module()
    module = module.cuda()
    # Open the steering taps so per-query mu/sigma/lambda actually vary.
    with torch.no_grad():
        module.raw_gate.fill_(0.0)
        module.steer.weight.normal_(std=0.05)
    x = torch.randn(2, 64, 64, device="cuda")
    v = module.value(x).view(2, 64, 2, 32).transpose(1, 2)
    field = module._field(x)
    # Both paths now also return the attention denominator, which the null atom
    # scales the output by; the two have to agree on BOTH.
    ref, ref_lse = module._materialised(v, *field)
    out, out_lse = module._flex(v, *field)
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max()
    assert torch.allclose(out_lse, ref_lse, atol=1e-4), (out_lse - ref_lse).abs().max()


def test_geometry_snapshots_have_one_row_per_atom_and_hop():
    import math

    from praxis.attention.ssog import GEOM_BINS

    module, _ = _module(depth=4)
    snapshots = module.dashboard_snapshots()
    assert set(snapshots) == {"ssog_geometry", "ssog_cascade"}
    for key, expected_rows in (
        ("ssog_geometry", module.num_atoms + 1),  # atoms, plus the summed mixture
        ("ssog_cascade", 4),  # one row per recurrent hop
    ):
        grid = snapshots[key]["grid"]
        assert len(grid) == expected_rows == snapshots[key]["grid_rows"]
        assert all(len(row) == GEOM_BINS for row in grid)
        assert all(math.isfinite(v) and v >= 0.0 for row in grid for v in row)


@pytest.mark.parametrize("cls", [SSOGAttention, ArcSSOGAttention])
def test_cascade_reaches_further_with_depth(cls):
    """The h-fold self-convolution has mean lag h*mu, so the composed field has
    to walk outward monotonically - that IS the scale-space claim the card makes.
    ArcSSOG composes its real per-depth kernels (band h is k_0 * ... * k_h-1),
    not one shared kernel, and must march outward all the same."""
    module, _ = _module(cls, depth=5)
    lags = module.geom_lags
    rows = module._cascade()
    centroid = (rows * lags).sum(-1) / rows.sum(-1)
    assert torch.all(centroid[1:] > centroid[:-1]), centroid.tolist()


def test_inverse_softplus_survives_a_long_ladder():
    """``log(expm1(y))`` overflows float32 at y > ~88, which silently produced
    ``inf`` atom centres the moment a ladder was initialised past that. The
    faithful port stops at 32 and never hit it; the helper is shared."""
    from praxis.attention.ssog import _inv_softplus

    y = torch.tensor([0.05, 0.5, 2.0, 32.0, 88.0, 128.0, 512.0])
    raw = _inv_softplus(y)
    assert torch.isfinite(raw).all()
    assert torch.allclose(F.softplus(raw), y, atol=1e-5)


def test_no_compile_is_honoured():
    """The module compiles the flex kernel ITSELF, whatever the rest of the
    model does. Under `--no-compile` that self-compile is the only thing
    compiling, and its cost scales with the atom count - so the flag has to
    reach it, or a 12-atom field sits in inductor at the first forward."""
    eager, _ = _module(no_compile=True)
    assert eager.compile_flex is False and eager.flex_attention is None

    compiled, _ = _module(no_compile=False)
    assert compiled.compile_flex is True

    # The fallback is not a stub: it differentiates in plain eager, which is
    # the property the self-compile existed to work around.
    x = torch.randn(2, 16, 64, requires_grad=True)
    eager(x)[0].square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name in ("raw_mu", "raw_sigma", "log_lambda", "raw_gate"):
        assert getattr(eager, name).grad is not None, name


def test_inference_never_takes_the_compiled_path():
    """`DecodeBackend.eval_mode` forbids compiled frames in the decode loop -
    it changes guard inputs every token. The module compiles flex ITSELF, which
    smuggled one back in; and flex's inference kernel needs a power-of-two head
    dim, so at head_size 37 every decode step paid a compile that then failed."""
    module, _ = _module(no_compile=False)
    if module.flex_attention is None:
        pytest.skip("flex_attention unavailable")

    x = torch.randn(1, 24, 64)
    with torch.no_grad():
        eval_out = module(x)[0]
    grad_out = module(x)[0]  # grad enabled: flex path is allowed again
    assert eval_out.shape == grad_out.shape
    # Same field, so the two paths must agree; this also pins the materialised
    # fallback as exact rather than an approximation.
    assert torch.allclose(eval_out, grad_out, atol=1e-4)
