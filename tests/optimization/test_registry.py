import torch
import torch.nn as nn

from praxis import registry
from praxis.optimization.gated_schedule_free import GatedScheduleFree
from praxis.optimization.half_lion import HalfLion
from praxis.optimization.wrappers import wrappers_disable_schedule

# ------------------------------------------------------------------------------
# optimizer_wrappers
# ------------------------------------------------------------------------------
# The ``wrappers`` registry, SequentialWrapper, and the GatedScheduleFree optimizer.


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
        assert k in registry.namespace("wrappers")
    # Only the schedule-free family runs without an LR schedule.
    assert wrappers_disable_schedule(["schedule_free"]) is True
    assert wrappers_disable_schedule(["gated_schedule_free"]) is True
    assert wrappers_disable_schedule(["wave_schedule_free"]) is True
    assert wrappers_disable_schedule(["lookahead"]) is False
    assert wrappers_disable_schedule(["ortho", "gated_schedule_free"]) is True
    assert wrappers_disable_schedule([]) is False


# ------------------------------------------------------------------------------
# half_lion
# ------------------------------------------------------------------------------
# HalfLion: blend live weights with a frozen init via a traveling index wave.


def test_train_eval_reversible_at_init():
    # Before any step there is no state; train()/eval() must be safe no-ops.
    model, _, _ = _quadratic_problem()
    w0 = model.weight.detach().clone()
    opt = HalfLion(torch.optim.SGD(model.parameters(), lr=0.05))
    opt.train()
    opt.eval()
    assert torch.equal(model.weight, w0)
