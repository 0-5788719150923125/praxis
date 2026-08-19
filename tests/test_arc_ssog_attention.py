"""ArcSSOG: the per-depth field, the warm gate, and the populated bank.

Each test pins one of the three deviations, plus the invariant that matters
most: the faithful ``ssog`` port next door must not move.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from praxis import PraxisConfig
from praxis.attention import ATTENTION_REGISTRY
from praxis.attention.arc_ssog import ARC_GATE_INIT, ArcSSOGAttention
from praxis.attention.ssog import COLD_GATE_INIT, NUM_ATOMS, SSOGAttention


def _module(cls=ArcSSOGAttention, **overrides):
    cfg = PraxisConfig(hidden_size=64, num_heads=2, num_queries=2, dropout=0.0, depth=4)
    cfg.causal = True  # modeling.py sets this at assembly; the bare config is False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    torch.manual_seed(0)
    return cls(cfg), cfg


def test_registered_profiles():
    assert ATTENTION_REGISTRY["arc_ssog"] is ArcSSOGAttention
    # The atom count is a registry PROFILE, not a config field or a CLI flag.
    wide = ATTENTION_REGISTRY["arc_ssog_wide"]
    assert wide.func is ArcSSOGAttention and wide.keywords["num_atoms"] == 32
    cfg = PraxisConfig(hidden_size=64, num_heads=2, dropout=0.0, depth=3)
    cfg.causal = True
    assert wide(cfg).num_atoms == 32


def test_the_faithful_port_did_not_move():
    """The whole point of the subclass is that ``ssog`` stays the reference."""
    base, _ = _module(SSOGAttention)
    assert base.num_atoms == NUM_ATOMS == 4
    assert base.raw_mu.shape == (base.num_heads, 4)  # no depth axis
    assert torch.allclose(
        base.raw_gate, torch.full_like(base.raw_gate, COLD_GATE_INIT)
    )
    assert not hasattr(base, "depths")


def test_field_is_per_depth():
    module, cfg = _module()
    D, H, R = cfg.depth, module.num_heads, module.num_atoms
    assert module.raw_mu.shape == (D, H, R)
    assert module.raw_sigma.shape == (D, H, R)
    assert module.log_lambda.shape == (D, H, R)
    assert module.raw_gate.shape == (D, 3)
    assert module.raw_temperature.shape == (D, 1)
    # Independent jitter, so the passes break symmetry from each other.
    assert not torch.allclose(module.raw_mu[0], module.raw_mu[-1])

    # Each pass reads its OWN slice, and only that slice takes gradient.
    module.eval()
    x = torch.randn(2, 12, 64)
    out0, out2 = module(x, None, None, None, 0)[0], module(x, None, None, None, 2)[0]
    assert not torch.allclose(out0, out2)

    module.zero_grad()
    module(x, None, None, None, 2)[0].square().sum().backward()
    grad = module.raw_mu.grad
    assert grad[2].abs().sum() > 0
    assert grad[0].abs().sum() == 0 and grad[3].abs().sum() == 0


def test_depth_index_is_clamped():
    """Recurrent depth can exceed the registered passes; that must not index
    out of bounds mid-run."""
    module, cfg = _module()
    x = torch.randn(1, 8, 64)
    out = module(x, None, None, None, cfg.depth + 5)[0]
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_gate_is_warm_but_the_field_still_starts_frozen():
    module, _ = _module()
    gate = F.softplus(module.raw_gate)
    assert torch.allclose(gate, torch.full_like(gate, F.softplus(torch.tensor(ARC_GATE_INIT))))
    assert 0.1 < gate.mean().item() < 0.2  # warm, not open
    # Zero-init on the probe is what actually freezes the field at step 0, and
    # is the reason the second (gate) barrier was redundant.
    assert module.steer.weight.abs().sum() == 0
    assert module.steer.bias.abs().sum() == 0
    assert module.steer_bias.abs().sum() == 0
    x = torch.randn(2, 10, 64)
    mu, sigma, _, _ = module._field(x, 1)
    base_mu = F.softplus(module.raw_mu[1].float())[None, :, None, :]
    assert torch.allclose(mu, base_mu.expand_as(mu), atol=1e-6)


def test_atom_bank_is_populated_and_the_ladder_is_finite():
    """The ladder now reaches past 88 tokens, where the direct
    ``log(expm1(y))`` inverse softplus overflows float32 and silently produced
    ``inf`` centres."""
    module, _ = _module()
    assert module.num_atoms == 12
    mu = F.softplus(module.raw_mu)
    sigma = F.softplus(module.raw_sigma) + 0.25
    assert torch.isfinite(mu).all() and torch.isfinite(sigma).all()
    assert mu.max() > 88.0  # past the overflow point, which is the regression
    ladder = mu[0, 0]
    assert torch.all(ladder[1:] > ladder[:-1])  # geometric, ordered
    assert (sigma / mu.clamp_min(1e-6)).max() < 1.0  # constant-Q, no atom wider than its lag


def test_metrics_and_snapshots_are_per_depth_and_declared():
    module, cfg = _module()
    metrics = module.training_metrics()
    for d in range(cfg.depth):
        for key in ("gate_mu", "gate_sigma", "gate_lambda", "temperature", "reach", "far_mass"):
            assert f"ssog_{key}_d{d}" in metrics
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    # Nothing averaged over depth: that average is what -r reported, and it hid
    # six passes agreeing to do nothing.
    assert not any(k.endswith(("_mean", "_a0")) for k in metrics)

    declared = type(module).metric_descriptions
    assert set(metrics) <= set(declared)
    assert all(declared[k].get("chart") for k in metrics)

    snapshots = module.dashboard_snapshots()
    assert set(snapshots) == {"ssog_geometry", "ssog_cascade"}
    for key, snap in snapshots.items():
        assert declared[key].get("snapshot")
        assert snap["grid_rows"] == cfg.depth == len(snap["grid"])


def test_cascade_composes_the_real_per_depth_kernels():
    """Not a self-convolution of one shared kernel: band h is k_0 * ... * k_h-1,
    so it must still march outward monotonically."""
    module, _ = _module()
    rows = module._cascade()
    lags = module.geom_lags
    centroid = (rows * lags).sum(-1) / rows.sum(-1)
    assert torch.all(centroid[1:] > centroid[:-1]), centroid.tolist()


def test_reaches_the_dashboard_through_the_precompute_recipe():
    from praxis.metrics.descriptions import get_metric_descriptions
    from praxis.web.snapshots import _recipe_head_snapshots

    module, _ = _module()
    model = nn.Sequential(nn.Identity(), module)
    served = _recipe_head_snapshots(model)["snapshots"]
    assert set(module.dashboard_snapshots()) <= set(served)
    descriptions = get_metric_descriptions(model)
    for key in module.training_metrics():
        assert descriptions.get(key, {}).get("chart"), key
    assert descriptions["ssog_reach_d0"]["caller"] == "ArcSSOGAttention"
