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
from praxis.attention.ssog import (
    COLD_GATE_INIT,
    NULL_LOGIT_INIT,
    NUM_ATOMS,
    SSOGAttention,
)


def _module(cls=ArcSSOGAttention, num_atoms=None, mu_init_max=None, null_atom=False, **overrides):
    cfg = PraxisConfig(hidden_size=64, num_heads=2, num_queries=2, dropout=0.0, depth=4)
    cfg.causal = True  # modeling.py sets this at assembly; the bare config is False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    torch.manual_seed(0)
    if cls is ArcSSOGAttention:
        return cls(
            cfg, num_atoms=num_atoms, mu_init_max=mu_init_max, null_atom=null_atom
        ), cfg
    return cls(cfg), cfg


def test_registered_profiles():
    assert ATTENTION_REGISTRY["arc_ssog"] is ArcSSOGAttention
    # Bank size and ladder span are ONE decision and both are registry
    # PROFILE arguments, never config fields or CLI flags.
    wide = ATTENTION_REGISTRY["arc_ssog_wide"]
    assert wide.func is ArcSSOGAttention
    assert wide.keywords == {"num_atoms": 12, "mu_init_max": 128.0}
    cfg = PraxisConfig(hidden_size=64, num_heads=2, dropout=0.0, depth=3)
    cfg.causal = True
    built = wide(cfg)
    assert built.num_atoms == 12 and built.mu_init_max == 128.0


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


def test_default_ladder_stays_inside_the_window():
    """Every atom inside the sequence window at every curriculum tier. An atom
    centred outside it has its truncated mass renormalised onto the oldest
    tokens - the softmax forces attention somewhere - which is a sink rather
    than a long-range read. The wide profile put three atoms out there."""
    module, _ = _module()
    assert module.num_atoms == 4 and module.mu_init_max == 32.0
    mu = F.softplus(module.raw_mu)
    assert mu.max() <= 64.0
    ladder = mu[0, 0]
    assert torch.all(ladder[1:] > ladder[:-1])  # geometric, ordered
    sigma = F.softplus(module.raw_sigma) + 0.25
    assert (sigma / mu.clamp_min(1e-6)).max() < 1.0  # constant-Q


def test_wide_profile_ladder_is_finite_past_the_overflow_point():
    """``log(expm1(y))`` overflows float32 at y > ~88 and silently produced
    ``inf`` centres; the wide ladder is the configuration that reaches there."""
    module, _ = _module(num_atoms=12, mu_init_max=128.0)
    mu = F.softplus(module.raw_mu)
    sigma = F.softplus(module.raw_sigma) + 0.25
    assert torch.isfinite(mu).all() and torch.isfinite(sigma).all()
    assert mu.max() > 88.0


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


def test_snapshots_issue_no_gpu_work():
    """``dashboard_snapshots`` runs in the web layer's snapshot producer
    thread, concurrently with training. GPU work there contends with the
    training stream and can block on it forever - it wedged a run for six
    hours, with every watchdog dump parked on the CUDA softmax in ``_atoms``.
    The field is a few dozen floats, so all of this belongs on the host."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU to observe device traffic")

    module, _ = _module()
    module = module.cuda()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()

    snapshots = module.dashboard_snapshots()
    metrics = module.training_metrics()

    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before, "snapshot path allocated on device"
    assert snapshots and metrics
    # And the intermediates really are host tensors, not just freed ones.
    for tensor in module._atoms(0):
        assert tensor.device.type == "cpu"
    assert module._cascade().device.type == "cpu"


def test_producer_thread_never_touches_the_model():
    """The snapshot producer runs on a background thread. Any parameter read
    there is a device call - ``.cpu()`` blocks just as a softmax does - and it
    contends with training for the CUDA context. So the training thread
    publishes and the producer only reads a stash of plain lists."""
    import threading

    module, _ = _module()
    module.raw_mu.data.fill_(0.5)
    module.training_metrics()  # training thread publishes
    published = module.dashboard_snapshots()

    # A producer read must not consult the parameters at all: mutate them and
    # the stash must be unchanged until the next publish.
    module.raw_mu.data.fill_(3.0)
    assert module.dashboard_snapshots() == published
    module.training_metrics()
    assert module.dashboard_snapshots() != published

    # And nothing in the read path can block on another thread.
    done = threading.Event()
    threading.Thread(
        target=lambda: (module.dashboard_snapshots(), done.set()), daemon=True
    ).start()
    assert done.wait(timeout=5), "producer-thread read blocked"

    payload = module.dashboard_snapshots()
    assert all(isinstance(v["grid"], list) for v in payload.values())


def test_null_atom_is_opt_in_and_off_by_default():
    plain, _ = _module()
    assert plain.raw_null is None
    assert not any("null" in k for k in plain.training_metrics())

    nulled, _ = _module(null_atom=True)
    assert nulled.raw_null.shape == (4, nulled.num_heads)  # per depth
    assert torch.allclose(
        nulled.raw_null, torch.full_like(nulled.raw_null, NULL_LOGIT_INIT)
    )
    x = torch.randn(2, 20, 64)
    assert not torch.allclose(plain(x)[0], nulled(x)[0])


def test_null_share_falls_with_query_position():
    """The whole argument for the null atom in THIS stack: nothing below the
    head knows absolute position, so a query near the start cannot otherwise
    say "there is nothing that far back" - its atom's truncated tail gets
    renormalised onto the oldest token instead. One constant logit fixes that,
    because the denominator it competes against is itself position-dependent."""
    module, _ = _module(null_atom=True)
    module.eval()
    x = torch.randn(1, 96, 64)
    with torch.no_grad():
        B, T, _ = x.shape
        v = module.value(x).view(B, T, module.num_heads, module.head_dim).transpose(1, 2)
        _, lse = module._materialised(v, *module._field(x, 0))
        share = (1.0 - torch.sigmoid(lse - module._null_logit(0)[None, :, None]))[0, 0]

    assert share[0] > 3 * share[-1], (share[0].item(), share[-1].item())
    assert share[0] > share[4] > share[32]  # monotone falloff, not a step
    assert share.mean() < 0.15  # near-identity at init, per NULL_LOGIT_INIT


def test_null_logit_is_not_initialised_into_saturation():
    """A strongly negative init would pin sigmoid(lse - l_0) at 1.0 with a dead
    gradient, which is exactly how the reference's softplus(-8) steering gate
    spent 11.7k steps welded shut."""
    module, _ = _module(null_atom=True)
    module(torch.randn(2, 24, 64), None, None, None, 1)[0].square().sum().backward()
    grad = module.raw_null.grad
    assert grad is not None and grad[1].abs().sum() > 1e-8, grad
    # and only the depth that ran takes gradient
    assert grad[0].abs().sum() == 0


def test_null_share_is_tracked_per_depth():
    module, cfg = _module(null_atom=True)
    assert module.null_share.shape == (cfg.depth,)
    module(torch.randn(2, 16, 64), None, None, None, 2)
    metrics = module.training_metrics()
    assert metrics["ssog_null_share_d2"] > 0.0
    assert metrics["ssog_null_share_d0"] == 0.0  # depth 0 never ran
    assert metrics["ssog_null_logit_d2"] == pytest.approx(NULL_LOGIT_INIT)
