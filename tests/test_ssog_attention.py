"""SSOG attention: shape, causality, field non-negativity, steering gradients,
and flex/materialised parity on GPU."""

import pytest
import torch

from praxis import PraxisConfig
from praxis.attention import ATTENTION_REGISTRY
from praxis.attention.ssog import SSOGAttention


def _module(**overrides):
    cfg = PraxisConfig(hidden_size=64, num_heads=2, num_queries=2, dropout=0.0)
    cfg.causal = True  # modeling.py sets this at assembly; the bare config is False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    torch.manual_seed(0)
    return SSOGAttention(cfg), cfg


def test_registered_and_shapes():
    assert ATTENTION_REGISTRY["ssog"] is SSOGAttention
    module, cfg = _module()
    assert cfg.num_queries == 1  # patch_config keeps the config honest
    x = torch.randn(3, 16, 64)
    out, kv, aux = module(x)
    assert out.shape == x.shape and kv is None and aux == 0.0


def test_causal():
    module, _ = _module()
    module.eval()
    x = torch.randn(1, 12, 64)
    out = module(x)[0]
    x2 = x.clone()
    x2[:, 8:] = torch.randn(1, 4, 64)
    out2 = module(x2)[0]
    assert torch.allclose(out[:, :8], out2[:, :8], atol=1e-6)
    assert not torch.allclose(out[:, 8:], out2[:, 8:])


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
    ref = module._materialised(v, *field)
    out = module._flex(v, *field)
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max()
