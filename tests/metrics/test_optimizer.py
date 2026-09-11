import torch
import torch.nn as nn

from praxis.metrics.optimizer import extract_optimizer_dynamics
from praxis.optimization import get_optimizer, get_optimizer_profile
from praxis.optimization.half_lion import HalfLion
from praxis.optimization.low_rank_moment import LowRankSecondMoment
from praxis.optimization.wrappers import SequentialWrapper

# ------------------------------------------------------------------------------
# optimizer_metrics
# ------------------------------------------------------------------------------
# Optimizer-state telemetry suite (praxis.metrics.optimizer).


def _stepped(optimizer, model, steps=5):
    if hasattr(optimizer, "train"):
        optimizer.train()
    X = torch.randn(16, 8)
    Y = X @ torch.randn(8, 4)
    for _ in range(steps):
        optimizer.zero_grad()
        ((model(X) - Y) ** 2).mean().backward()
        optimizer.step()
    optimizer.zero_grad()  # leave grads populated (extraction runs pre-step)
    ((model(X) - Y) ** 2).mean().backward()
    return optimizer


def _build(profile_name, wrappers=()):
    model = nn.Linear(8, 4, bias=False)
    profile, _ = get_optimizer_profile(profile_name)
    profile.pop("wd_ban_list", None)
    return model, get_optimizer(model, list(wrappers), **profile)


def test_always_emits_lr_and_grad_rms():
    model, opt = _build("AdamW")
    m = extract_optimizer_dynamics(_stepped(opt, model))
    assert "opt_lr" in m and "opt_grad_rms" in m and m["opt_grad_rms"] > 0


def test_adam_emits_full_suite():
    model, opt = _build("AdamW")
    m = extract_optimizer_dynamics(_stepped(opt, model))
    for k in [
        "opt_momentum_rms",
        "opt_momentum_grad_cos",
        "opt_update_rms",
        "opt_update_weight_ratio",
        "opt_second_moment_rms",
    ]:
        assert k in m, k
    assert -1.0 <= m["opt_momentum_grad_cos"] <= 1.0
    assert m["opt_update_rms"] > 0


def test_lion_emits_momentum_but_not_second_moment():
    # Lion is the praxis default: sign momentum, no exp_avg_sq. The momentum/
    # cosine cards must still emit; the second-moment/update cards must not.
    # (This is the regression that left most cards blank on calm-e.)
    model, opt = _build("Lion")
    m = extract_optimizer_dynamics(_stepped(opt, model))
    assert "opt_momentum_rms" in m and "opt_momentum_grad_cos" in m
    assert "opt_grad_rms" in m
    assert "opt_second_moment_rms" not in m
    assert "opt_update_rms" not in m and "opt_update_weight_ratio" not in m


def test_schedule_free_adds_spread_over_any_base():
    model, opt = _build("Lion", ["schedule_free"])
    m = extract_optimizer_dynamics(_stepped(opt, model))
    assert "opt_sf_spread" in m and m["opt_sf_spread"] >= 0.0
    assert "opt_momentum_grad_cos" in m  # base momentum still reachable
    assert "opt_gate_mean" not in m  # plain schedule-free has no gate


def test_wave_and_gated_expose_gate_with_lion_base():
    # calm-e's actual stack: Lion + wave_schedule_free.
    for key in ["wave_schedule_free", "gated_schedule_free"]:
        model, opt = _build("Lion", [key])
        m = extract_optimizer_dynamics(_stepped(opt, model))
        assert "opt_gate_mean" in m and 0.0 <= m["opt_gate_mean"] <= 1.0
        assert "opt_sf_spread" in m
        assert "opt_momentum_grad_cos" in m


def test_sgd_no_momentum_emits_only_universal():
    model = nn.Linear(8, 4, bias=False)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)  # momentum=0 -> no state
    m = extract_optimizer_dynamics(_stepped(opt, model))
    assert m["opt_lr"] == 0.05 and "opt_grad_rms" in m
    assert "opt_momentum_rms" not in m and "opt_second_moment_rms" not in m


def test_handles_none():
    assert extract_optimizer_dynamics(None) == {}


# ------------------------------------------------------------------------------
# low_rank_moment
# ------------------------------------------------------------------------------
# LowRankSecondMoment: factored second-moment telemetry, passthrough update.


def _problem():
    torch.manual_seed(0)
    model = nn.Linear(8, 4, bias=False)
    X = torch.randn(64, 8)
    Y = X @ torch.randn(8, 4)
    return model, X, Y


def _step(opt, model, X, Y):
    opt.zero_grad()
    loss = ((model(X) - Y) ** 2).mean()
    loss.backward()
    opt.step()
    return float(loss.detach())


def test_metrics_surface_second_moment_under_momentum_sgd():
    # SGD-momentum supplies m1 but no exp_avg_sq; the factored estimate should
    # fill in v so all three second-moment cards emit.
    model, X, Y = _problem()
    opt = LowRankSecondMoment(torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9))
    for _ in range(10):
        _step(opt, model, X, Y)
    out = extract_optimizer_dynamics(opt)
    for k in ["opt_second_moment_rms", "opt_update_rms", "opt_update_weight_ratio"]:
        assert k in out and out[k] > 0.0, (k, out)


def test_composes_with_half_lion():
    model, X, Y = _problem()
    opt = SequentialWrapper(["low_rank_moment", "half_lion"])(
        torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    )
    opt.train()
    for _ in range(20):
        _step(opt, model, X, Y)
    out = extract_optimizer_dynamics(opt)
    assert "opt_second_moment_rms" in out  # factored estimate reached through HalfLion
    assert "opt_gate_mean" in out  # HalfLion's wave gate also surfaces


# ------------------------------------------------------------------------------
# half_lion
# ------------------------------------------------------------------------------
# HalfLion: blend live weights with a frozen init via a traveling index wave.


def _quadratic_problem():
    # Learnable target (y = X @ W_true), so the optimum is ~0 and a real
    # optimizer drives the (eval) loss down sharply.
    torch.manual_seed(0)
    model = nn.Linear(8, 4, bias=False)
    X = torch.randn(64, 8)
    Y = X @ torch.randn(8, 4)
    return model, X, Y


def _train(optimizer, model, X, Y, steps=200):
    optimizer.train()
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = ((model(X) - Y) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def test_gate_mean_is_surfaced_for_dynamics():
    from praxis.metrics.optimizer import extract_optimizer_dynamics

    model, X, Y = _quadratic_problem()
    opt = HalfLion(torch.optim.SGD(model.parameters(), lr=0.05))
    _train(opt, model, X, Y, steps=5)
    out = extract_optimizer_dynamics(opt)
    assert "opt_gate_mean" in out and 0.0 <= out["opt_gate_mean"] <= 1.0
    assert "opt_sf_spread" not in out  # no z/momentum => spread path no-ops
