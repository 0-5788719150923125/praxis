"""Scheduler tests: stage-aware re-warmup for multi-stage training."""

import torch

from praxis.schedulers import get_scheduler_func


def _run(warmup, anchor_getter, steps):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = get_scheduler_func(
        {"lr": 1.0},
        disable_schedule=True,
        warmup_steps=warmup,
        stage_anchor=anchor_getter,
    )(opt)
    lrs = []
    for _ in range(steps):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return lrs


def test_stage1_warmup_then_hold():
    # No stage boundary: ramp 0->1 over warmup, then hold at peak.
    lrs = _run(10, lambda: -1, 25)
    assert abs(lrs[0]) < 1e-9
    assert abs(lrs[10] - 1.0) < 1e-9
    assert all(abs(x - 1.0) < 1e-9 for x in lrs[10:])


def test_stage2_rewarmup_fires_at_anchor():
    # Boundary reported at step 20: LR re-ramps from there over warmup steps.
    anchor = {"v": -1}
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = get_scheduler_func(
        {"lr": 1.0},
        disable_schedule=True,
        warmup_steps=10,
        stage_anchor=lambda: anchor["v"],
    )(opt)
    lrs = []
    for step in range(40):
        lrs.append(opt.param_groups[0]["lr"])
        if step == 20:
            anchor["v"] = 20  # codec freeze
        opt.step()
        sched.step()
    assert abs(lrs[19] - 1.0) < 1e-9  # held at peak before the boundary
    assert lrs[22] < 0.5  # dropped into the re-warmup
    assert abs(lrs[31] - 1.0) < 1e-9  # back to peak ~warmup steps later


def test_no_anchor_getter_is_plain_warmup():
    # Omitting stage_anchor entirely must behave like the original warmup.
    lrs = _run(8, None, 20)
    assert abs(lrs[0]) < 1e-9
    assert abs(lrs[8] - 1.0) < 1e-9
    assert all(abs(x - 1.0) < 1e-9 for x in lrs[8:])
