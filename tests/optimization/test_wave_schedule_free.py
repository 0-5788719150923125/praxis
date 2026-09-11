"""WaveScheduleFree: schedule-free averaging gated by a standing wave over the index."""

import torch
import torch.nn as nn


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


def test_wave_schedule_free_reduces_loss_and_gates_periodically():
    from praxis.optimization.wave_schedule_free import WaveScheduleFree

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
    from praxis.optimization.wave_schedule_free import WaveScheduleFree

    model, _, _ = _quadratic_problem()
    opt = WaveScheduleFree(torch.optim.SGD(model.parameters(), lr=0.05))
    p = next(iter(model.parameters()))
    before = opt._wave(p).clone()
    opt.set_wave(amp=0.5, cycles=7.0, phase=1.0)  # RL-controller-style override
    after = opt._wave(p)
    assert not torch.allclose(before, after)
    assert opt.wave_cycles == 7.0 and opt.wave_amp == 0.5
