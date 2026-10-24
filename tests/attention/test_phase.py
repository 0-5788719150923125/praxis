"""Kuramoto Attention: the phase projection and tangent step stages, the learned
rotary rates, and the per-layer diagnostics they publish."""

import math

import pytest
import torch
from torch import nn

from praxis import PraxisConfig, registry
from praxis.attention import patch_attention_config
from praxis.attention.phase import PhaseProjection, PhaseState, TangentStep
from praxis.metrics.specialization import (
    collect_attention_descriptions,
    collect_attention_metrics,
    layer_key,
)


def _config(**fields):
    config = PraxisConfig(
        hidden_size=32, num_heads=2, num_queries=2, head_size=8, dropout=0.0
    )
    for name, value in fields.items():
        setattr(config, name, value)
    return config


def test_patch_keeps_width_and_derives_count():
    config = _config(hidden_size=256, num_heads=3, head_size=30)
    PhaseProjection.patch_config(config)
    assert (config.head_size, config.num_heads, config.num_queries) == (30, 17, 1)
    PhaseProjection.patch_config(config)
    assert (config.head_size, config.num_heads, config.num_queries) == (30, 17, 1)


def test_patch_pins_an_unset_width():
    config = _config(hidden_size=64, num_heads=4, head_size=None)
    PhaseProjection.patch_config(config)
    assert (config.head_size, config.num_heads) == (16, 8)


def test_registry_patch_reaches_the_profile_stage():
    config = _config(attention_type="kuramoto")
    patch_attention_config(config)
    assert (config.num_heads, config.num_queries) == (8, 1)
    # Profiles with unrelated keywords still patch.
    config = _config(attention_type="arc_single_dropoff")
    patch_attention_config(config)
    assert config.num_heads == 1


@pytest.mark.parametrize("encoding", ["rope", "rope_learned_rates", "nope"])
def test_forward_backward_and_metrics(encoding):
    torch.manual_seed(0)
    config = _config(encoding=encoding)
    module = registry.lookup("attention", "kuramoto")(config).train()
    x = torch.randn(2, 12, config.hidden_size, requires_grad=True)
    out, _, _ = module(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
    out.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    report = module.training_metrics()
    for key in ("kuramoto_local_order", "kuramoto_global_order", "kuramoto_repel"):
        assert key in report
    assert 0.0 <= report["kuramoto_local_order"] <= 1.0 + 1e-4
    assert 0.0 < report["kuramoto_decoupled"] < 1.0
    if encoding == "rope_learned_rates":
        assert report["rope_rate_mean"] > 0


def test_tangent_direction_is_kuramoto_coupling():
    """With a fixed attention matrix, the step is sum_u A_tu sin(theta_u - theta_t)
    (the paper's Lemma 2), here with a unit gate and an unbounded radius."""
    torch.manual_seed(0)
    config = _config()
    PhaseProjection.patch_config(config)
    H, kh, T = config.num_heads, config.head_size // 2, 5
    step = TangentStep(config).eval()
    nn.init.zeros_(step.value_gate.weight)
    nn.init.ones_(step.value_gate.bias)
    step.log_radius.data.fill_(20.0)

    theta = torch.rand(1, T, H, kh) * 2 * math.pi
    A = torch.softmax(torch.randn(H, T, T), dim=-1)  # [H, t, u]
    re = torch.einsum("htu,buhj->bthj", A, theta.cos())
    im = torch.einsum("htu,buhj->bthj", A, theta.sin())
    attended = torch.cat([re, im], dim=-1).flatten(2)  # per head [Re | Im]
    state = PhaseState(theta.cos(), theta.sin(), torch.ones(H * kh))
    out = step(attended, state)

    diff = theta.unsqueeze(1) - theta.unsqueeze(2)  # [b, t, u, h, j]: u minus t
    expected = torch.einsum("htu,btuhj->bthj", A, diff.sin()).flatten(2)
    assert torch.allclose(out[..., : H * kh], expected, atol=1e-5)
    assert (out[..., H * kh :] == 0).all()


def test_score_is_gated_phase_similarity():
    """q.k / sqrt(d) = tau / sqrt(d) * sum_j g^q g^k cos(theta_t - theta_u + w_j (t - u))."""
    torch.manual_seed(0)
    config = _config(encoding="rope_learned_rates")
    PhaseProjection.patch_config(config)
    proj = PhaseProjection(config).eval()
    rope = registry.lookup("encoding", "rope_learned_rates")(config)
    with torch.no_grad():
        rope.log_rates.add_(torch.randn_like(rope.log_rates))
        proj.log_tau.fill_(0.3)
    H, kh, T = config.num_heads, config.head_size // 2, 6
    x = torch.randn(1, T, config.hidden_size)
    q, k, _, state = proj(x)
    q = q.view(1, T, H, -1).transpose(1, 2)
    k = k.view(1, T, H, -1).transpose(1, 2)
    gq = q[..., :kh] / state.cos.transpose(1, 2)
    gk = k[..., :kh] / state.cos.transpose(1, 2)
    qr, kr, _ = rope.before_scores(q, k, k)
    scores = qr @ kr.transpose(-2, -1)

    theta = torch.atan2(state.sin, state.cos).transpose(1, 2)  # [1, H, T, kh]
    rates = rope.rates(0)  # [H, kh]
    pos = torch.arange(T).float()
    drift = (pos[:, None] - pos[None, :])[None, :, :, None] * rates[:, None, None, :]
    delta = theta.unsqueeze(3) - theta.unsqueeze(2)  # [1, H, t, u, kh]: t minus u
    expected = (gq.unsqueeze(3) * gk.unsqueeze(2) * (delta + drift).cos()).sum(-1)
    assert torch.allclose(scores, expected, atol=1e-4)


def test_learned_rates_turn_each_pair_at_its_own_rate():
    config = _config(num_heads=1, num_queries=1, head_size=8, depth=1)
    rope = registry.lookup("encoding", "rope_learned_rates")(config)
    x = torch.zeros(1, 1, 5, 8)
    x[..., :4] = 1.0
    q, _, _ = rope.before_scores(x, x, x)
    angle = torch.atan2(q[0, 0, :, 4:], q[0, 0, :, :4])
    expected = torch.arange(5).float()[:, None] * rope.rates(0)[0]
    wrapped = torch.atan2(expected.sin(), expected.cos())
    assert torch.allclose(angle, wrapped, atol=1e-5)
    assert torch.allclose(rope.rates(0)[0], 10000.0 ** (-torch.arange(4) * 2 / 8.0))


def test_metrics_are_reported_per_layer():
    torch.manual_seed(0)
    config = _config(encoding="rope_learned_rates")
    model = nn.Sequential(
        registry.lookup("attention", "kuramoto")(config),
        registry.lookup("attention", "kuramoto")(config),
    ).train()
    x = torch.randn(2, 8, config.hidden_size)
    for layer in model:
        layer(x)

    metrics = collect_attention_metrics(model)
    for layer in (0, 1):
        assert layer_key("kuramoto_local_order", layer) in metrics
        assert layer_key("rope_rate_mean", layer) in metrics
    assert "kuramoto_local_order" not in metrics

    descriptions = collect_attention_descriptions(model)
    entry = descriptions[layer_key("kuramoto_global_order", 1)]
    assert entry["chart"]["series_group"] == "kuramoto_global_order"
    assert entry["chart"]["series_label"] == "layer 1"
    assert set(metrics) <= set(descriptions)


def test_plain_self_attention_declares_nothing_new():
    module = registry.lookup("attention", "self_attention")(_config())
    assert module.training_metrics() == {}
    assert "metric_descriptions" not in vars(module)
