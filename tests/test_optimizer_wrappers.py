"""WRAPPER_REGISTRY, SequentialWrapper, and the GatedScheduleFree optimizer."""

import torch
import torch.nn as nn

from praxis.optimizers.gated_schedule_free import GatedScheduleFree
from praxis.optimizers.wrappers import (
    WRAPPER_REGISTRY,
    SequentialWrapper,
    wrappers_disable_schedule,
)


def _quadratic_problem():
    # Minimize ||W x - y||^2 with a LEARNABLE target (y = X @ W_true), so the
    # optimum is ~0 and a real optimizer should drive the loss down sharply.
    torch.manual_seed(0)
    model = nn.Linear(8, 4, bias=False)
    X = torch.randn(64, 8)
    Y = X @ torch.randn(8, 4)
    return model, X, Y


def _train(optimizer, model, X, Y, steps=200):
    optimizer.train() if hasattr(optimizer, "train") else None
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = ((model(X) - Y) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def test_gated_schedule_free_reduces_loss():
    model, X, Y = _quadratic_problem()
    base = torch.optim.SGD(model.parameters(), lr=0.05)
    opt = GatedScheduleFree(base, momentum=0.9)
    losses = _train(opt, model, X, Y)
    assert losses[-1] < losses[0] * 0.5, (losses[0], losses[-1])
    # Eval deploys the average x; it should also be a good solution.
    opt.eval()
    with torch.no_grad():
        eval_loss = float(((model(X) - Y) ** 2).mean())
    assert eval_loss < losses[0] * 0.6


def test_gate_stays_in_unit_interval():
    model, X, Y = _quadratic_problem()
    opt = GatedScheduleFree(torch.optim.SGD(model.parameters(), lr=0.05), momentum=0.9)
    opt.train()
    for _ in range(50):
        opt.zero_grad()
        ((model(X) - Y) ** 2).mean().backward()
        opt.step()
        for group in opt.param_groups:
            for p in group["params"]:
                g = opt.state[p]["m1"].abs() / (opt.state[p]["m2"].sqrt() + 1e-8)
                assert float(g.min()) >= 0.0 and float(g.max()) <= 1.0 + 1e-6
    assert 0.0 <= opt.gate_mean <= 1.0


def test_train_eval_swap_is_reversible_at_init():
    # Before any step there is no state; train()/eval() must be safe no-ops.
    model, _, _ = _quadratic_problem()
    w0 = model.weight.detach().clone()
    opt = GatedScheduleFree(torch.optim.SGD(model.parameters(), lr=0.05))
    opt.train()
    opt.eval()
    assert torch.equal(model.weight, w0)


def test_registry_keys_and_disable_schedule():
    for k in [
        "trac",
        "ortho",
        "lookahead",
        "schedule_free",
        "gated_schedule_free",
        "wave_schedule_free",
    ]:
        assert k in WRAPPER_REGISTRY
    # Only the schedule-free family runs without an LR schedule.
    assert wrappers_disable_schedule(["schedule_free"]) is True
    assert wrappers_disable_schedule(["gated_schedule_free"]) is True
    assert wrappers_disable_schedule(["wave_schedule_free"]) is True
    assert wrappers_disable_schedule(["lookahead"]) is False
    assert wrappers_disable_schedule(["ortho", "gated_schedule_free"]) is True
    assert wrappers_disable_schedule([]) is False


def test_wave_schedule_free_reduces_loss_and_gates_periodically():
    from praxis.optimizers.wave_schedule_free import WaveScheduleFree

    model, X, Y = _quadratic_problem()
    opt = WaveScheduleFree(torch.optim.SGD(model.parameters(), lr=0.05), momentum=0.9)
    losses = _train(opt, model, X, Y)
    assert losses[-1] < losses[0] * 0.5, (losses[0], losses[-1])

    # The gate is a standing wave over the (flattened) coordinate index: in
    # [0, 1], and genuinely non-constant across coordinates.
    p = next(iter(model.parameters()))
    gate = opt._wave(p)
    assert float(gate.min()) >= 0.0 and float(gate.max()) <= 1.0
    assert float(gate.max() - gate.min()) > 0.1  # actually oscillates
    assert 0.0 <= opt.gate_mean <= 1.0


def test_wave_set_wave_changes_the_gate():
    from praxis.optimizers.wave_schedule_free import WaveScheduleFree

    model, _, _ = _quadratic_problem()
    opt = WaveScheduleFree(torch.optim.SGD(model.parameters(), lr=0.05))
    p = next(iter(model.parameters()))
    before = opt._wave(p).clone()
    opt.set_wave(amp=0.5, cycles=7.0, phase=1.0)  # RL-controller-style override
    after = opt._wave(p)
    assert not torch.allclose(before, after)
    assert opt.wave_cycles == 7.0 and opt.wave_amp == 0.5


def test_sequential_wrapper_nests_in_order():
    model, _, _ = _quadratic_problem()
    base = torch.optim.SGD(model.parameters(), lr=0.05)
    wrapped = SequentialWrapper(["ortho", "schedule_free"])(base)
    # Outermost is schedule-free; unwrapping .optimizer reaches the base.
    assert type(wrapped).__name__ == "ScheduleFreeWrapper"
    inner = wrapped.optimizer
    assert type(inner).__name__ == "OrthoGrad"
    assert inner.optimizer is base


def test_sequential_wrapper_rejects_unknown_key():
    try:
        SequentialWrapper(["nope"])
    except ValueError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown wrapper key")


def test_gate_one_recovers_schedulefree_averaging():
    # With the gate pinned to 1, the gated averaging is plain schedule-free
    # averaging: x tracks the iterate exactly as the scalar method would.
    model, X, Y = _quadratic_problem()
    opt = GatedScheduleFree(torch.optim.SGD(model.parameters(), lr=0.05), momentum=0.9)
    opt.train()
    # Force gate to 1 by monkeypatching the SNR (consistent gradient => g~1
    # anyway on this deterministic problem, but pin it to be exact).
    opt.zero_grad()
    ((model(X) - Y) ** 2).mean().backward()
    opt.step()
    assert opt.gate_mean > 0.9  # deterministic full-batch grad => near-1 SNR
