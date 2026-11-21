"""kuramoto_capsules: routing by agreement over transported harmonic phase poses."""

import math

import pytest
import torch

from praxis import PraxisConfig, registry
from praxis.attention.phase import CapsuleTangentStep, TangentStep
from praxis.metrics.specialization import collect_attention_metrics


def _module(**fields):
    torch.manual_seed(0)
    config = PraxisConfig(
        hidden_size=32,
        num_heads=2,
        num_queries=2,
        head_size=8,
        dropout=0.0,
        encoding="rope_learned_rates",
    )
    for name, value in fields.items():
        setattr(config, name, value)
    return registry.lookup("attention", "kuramoto_capsules")(config)


def _route(module, x, positions):
    """The module's forward with explicit positions for the rotary frame."""
    q, k, v, state = module.qkv(x)
    B, T, _ = x.shape
    shape = (B, T, module.num_heads, module.head_dim)
    q, k, v = (t.view(*shape).transpose(1, 2) for t in (q, k, v))
    q, k, v = module.encoding.before_scores(q, k, v, positions=positions)
    return module.output.route(
        lambda q_, k_, v_, scale, with_ghost: module._attend(
            q_, k_, v_, scale=scale, with_ghost=with_ghost
        ),
        q,
        k,
        state,
        rates=module.encoding.rates(0),
        positions=positions,
    )


def test_forward_backward_and_capsule_metrics():
    module = _module().train()
    x = torch.randn(2, 12, 32, requires_grad=True)
    out, _, _ = module(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
    out.square().sum().backward()
    assert module.output.vote_weight.grad.abs().sum() > 0
    assert module.encoding.log_rates.grad.abs().sum() > 0

    metrics = collect_attention_metrics(module)
    for key in (
        "capsule_routing_shift",
        "capsule_whole_length",
        "capsule_vote_harmonic_share",
        "kuramoto_local_order",
    ):
        assert math.isfinite(metrics[f"{key}_layer0"]), key
    assert metrics["capsule_routing_shift_layer0"] > 0
    assert 0 < metrics["capsule_whole_length_layer0"] < 1


def test_selects_learned_rates():
    from praxis.attention import patch_attention_config

    config = PraxisConfig(
        hidden_size=32, num_heads=2, head_size=8, encoding="rope",
        attention_type="kuramoto_capsules",
    )
    patch_attention_config(config)
    assert config.encoding == "rope_learned_rates"
    assert _module(encoding="rope").encoding.__class__.__name__ == "LearnedRateRoPE"
    with pytest.raises(ValueError, match="rope_learned_rates"):
        CapsuleTangentStep(PraxisConfig(hidden_size=32, head_size=8, num_heads=8, encoding="rope"))


def test_free_running_oscillators_are_a_fixed_point():
    """Phases advancing at exactly their natural frequencies (theta_t = theta_0
    - omega t) agree once carried into one frame, so every vote matches the
    token's own phase and the step is zero. Raw, untransported phases would
    disagree and pull."""
    module = _module().eval()
    with torch.no_grad():
        module.encoding.log_rates.uniform_(-3.0, -0.5)
    H, m, T = module.num_heads, module.output.head_oscillators, 10
    rates = module.encoding.rates(0).detach()  # [H, m]
    theta0 = torch.rand(H, m) * 2 * math.pi
    t = torch.arange(T).float().view(T, 1, 1)
    theta = (theta0 - rates * t).reshape(1, T, H * m)
    out = _route(module, theta, torch.arange(T))
    assert out.abs().max() < 1e-5

    # The same sequence without transport is not at rest.
    disagree = _route(module, theta + rates.reshape(1, 1, -1) * t.view(1, T, 1), torch.arange(T))
    assert disagree.abs().max() > 1e-2


def test_shifting_every_position_changes_nothing():
    module = _module().eval()
    with torch.no_grad():
        module.encoding.log_rates.uniform_(-3.0, -0.5)
    x = torch.randn(2, 9, 32)
    T = x.shape[1]
    base = _route(module, x, torch.arange(T))
    shifted = _route(module, x, torch.arange(T) + 37)
    # Phases are float32 by design; the shift costs only rounding.
    torch.testing.assert_close(shifted, base, atol=1e-4, rtol=1e-3)


def test_route_matches_the_forward():
    module = _module().eval()
    x = torch.randn(2, 7, 32)
    out, _, _ = module(x)
    torch.testing.assert_close(_route(module, x, torch.arange(7)), out)


def test_one_round_one_harmonic_without_drift_is_the_paper_up_to_length(monkeypatch):
    """With no drift, one harmonic of weight 1 and a single round, the whole is
    the paper's resultant squashed per head: the same direction per oscillator,
    rescaled by one positive factor per (token, head)."""
    torch.manual_seed(0)
    config = PraxisConfig(
        hidden_size=32, num_heads=2, num_queries=2, head_size=8, dropout=0.0,
        encoding="rope_learned_rates",
    )
    paper = registry.lookup("attention", "kuramoto")(config).eval()
    capsule = registry.lookup("attention", "kuramoto_capsules")(config).eval()
    capsule.load_state_dict(paper.state_dict(), strict=False)
    capsule.output = CapsuleTangentStep(config, harmonics=1, rounds=1).eval()
    capsule.output.load_state_dict(paper.output.state_dict(), strict=False)
    for module in (paper, capsule):
        with torch.no_grad():
            module.encoding.log_rates.fill_(-1e4)

    seen = []
    original = TangentStep.forward

    def spy(self, attended, *args, **kwargs):
        seen.append(attended.detach())
        return original(self, attended, *args, **kwargs)

    monkeypatch.setattr(TangentStep, "forward", spy)
    x = torch.randn(2, 6, 32)
    paper(x)
    capsule(x)
    H = paper.num_heads
    G_paper, G_capsule = (g.view(2, 6, H, 2, -1) for g in seen)  # [B, T, H, re/im, m]
    ratio = G_capsule / G_paper
    torch.testing.assert_close(ratio, ratio[..., :1, :1].expand_as(ratio))
    assert (ratio > 0).all()
