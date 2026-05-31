"""HalfLion: blend live weights with a frozen init via a traveling index wave."""

import torch
import torch.nn as nn

from praxis.optimizers.half_lion import HalfLion
from praxis.optimizers.wrappers import WRAPPER_REGISTRY, wrappers_disable_schedule


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


def test_registered_and_keeps_schedule():
    assert "half_lion" in WRAPPER_REGISTRY
    # Not a schedule-free method: the LR schedule still applies.
    assert wrappers_disable_schedule(["half_lion"]) is False


def test_eval_loss_reduces():
    model, X, Y = _quadratic_problem()
    opt = HalfLion(torch.optim.SGD(model.parameters(), lr=0.05))
    losses = _train(opt, model, X, Y)
    # Eval deploys 100% current weights: that is the solution that should be good.
    opt.eval()
    with torch.no_grad():
        eval_loss = float(((model(X) - Y) ** 2).mean())
    assert eval_loss < losses[0] * 0.2, (losses[0], eval_loss)


def test_eval_deploys_pure_current_train_deploys_blend():
    model, X, Y = _quadratic_problem()
    opt = HalfLion(torch.optim.SGD(model.parameters(), lr=0.05))
    _train(opt, model, X, Y, steps=30)
    blend = model.weight.detach().clone()  # train mode => the blend is deployed
    opt.eval()
    current = model.weight.detach().clone()  # eval => 100% current
    p = next(iter(model.parameters()))
    assert torch.equal(current, opt.state[p]["w"])  # eval weights are exactly w
    assert not torch.allclose(blend, current)  # the blend pulled toward the prior


def test_frozen_prior_never_changes():
    model, X, Y = _quadratic_problem()
    init = model.weight.detach().clone()
    opt = HalfLion(torch.optim.SGD(model.parameters(), lr=0.05))
    _train(opt, model, X, Y, steps=50)
    p = next(iter(model.parameters()))
    assert torch.equal(opt.state[p]["w0"], init)  # the prior == init, untouched


def test_mix_is_per_coordinate_and_bounded():
    model, _, _ = _quadratic_problem()
    opt = HalfLion(torch.optim.SGD(model.parameters(), lr=0.05))
    p = next(iter(model.parameters()))
    mix = opt._mix(p)
    assert float(mix.min()) >= 0.0 and float(mix.max()) <= opt.wave_amp + 1e-6
    assert float(mix.max() - mix.min()) > 0.05  # genuinely oscillates across index


def test_blend_keeps_a_current_core():
    # With amp < 1 the blend never fully replaces current with the prior:
    # deployed = (1 - mix)*current + mix*prior, mix <= amp, so current always
    # retains weight >= 1 - amp at every coordinate.
    model, X, Y = _quadratic_problem()
    opt = HalfLion(torch.optim.SGD(model.parameters(), lr=0.05))
    _train(opt, model, X, Y, steps=20)
    p = next(iter(model.parameters()))
    w, w0 = opt.state[p]["w"], opt.state[p]["w0"]
    mix = opt._mix(p)
    expected = torch.lerp(w, w0, mix)
    assert torch.allclose(p.data, expected, atol=1e-5)
    assert float(mix.max()) <= opt.wave_amp + 1e-6


def test_phase_travels_over_steps():
    model, X, Y = _quadratic_problem()
    opt = HalfLion(torch.optim.SGD(model.parameters(), lr=0.05))
    opt.train()
    p0 = opt.wave_phase
    _train(opt, model, X, Y, steps=10)
    assert opt.wave_phase > p0  # the bands drift each step


def test_set_wave_changes_the_gate():
    model, _, _ = _quadratic_problem()
    opt = HalfLion(torch.optim.SGD(model.parameters(), lr=0.05))
    p = next(iter(model.parameters()))
    before = opt._mix(p).clone()
    opt.set_wave(amp=0.3, cycles=7.0, phase=1.0)  # RL-controller-style override
    after = opt._mix(p)
    assert not torch.allclose(before, after)
    assert opt.wave_cycles == 7.0 and opt.wave_amp == 0.3


def test_train_eval_reversible_at_init():
    # Before any step there is no state; train()/eval() must be safe no-ops.
    model, _, _ = _quadratic_problem()
    w0 = model.weight.detach().clone()
    opt = HalfLion(torch.optim.SGD(model.parameters(), lr=0.05))
    opt.train()
    opt.eval()
    assert torch.equal(model.weight, w0)


def test_gate_mean_is_surfaced_for_dynamics():
    from praxis.metrics.optimizer import extract_optimizer_dynamics

    model, X, Y = _quadratic_problem()
    opt = HalfLion(torch.optim.SGD(model.parameters(), lr=0.05))
    _train(opt, model, X, Y, steps=5)
    out = extract_optimizer_dynamics(opt)
    assert "opt_gate_mean" in out and 0.0 <= out["opt_gate_mean"] <= 1.0
    assert "opt_sf_spread" not in out  # no z/momentum => spread path no-ops
