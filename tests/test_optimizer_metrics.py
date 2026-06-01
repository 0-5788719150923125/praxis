"""Optimizer-state telemetry suite (praxis.metrics.optimizer)."""

import torch
import torch.nn as nn

from praxis.metrics.optimizer import (
    OPTIMIZER_METRIC_DESCRIPTIONS,
    extract_optimizer_dynamics,
)
from praxis.optimizers import get_optimizer, get_optimizer_profile
from praxis.optimizers.wrappers import SequentialWrapper


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


def test_descriptions_in_dynamics_manifest():
    from praxis.metrics.descriptions import get_metric_descriptions

    descs = get_metric_descriptions(nn.Linear(2, 2))
    for k in OPTIMIZER_METRIC_DESCRIPTIONS:
        assert k in descs and descs[k]["chart"]["group"] == "optimizer"


def test_handles_none():
    assert extract_optimizer_dynamics(None) == {}


def test_descriptions_stamp_producing_caller():
    """Each entry carries the class name of the module that raised it."""
    from praxis.metrics.descriptions import get_metric_descriptions

    class Field(nn.Module):
        metric_descriptions = {"field_amp": "amplitude"}

    class Head(nn.Module):
        metric_descriptions = {"head_loss": "aux loss"}

        def __init__(self):
            super().__init__()
            self.field = Field()

        def all_metric_descriptions(self):
            out = {}
            for mod in self.modules():
                descs = getattr(type(mod), "metric_descriptions", None)
                if isinstance(descs, dict):
                    out.update(descs)
            return out

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = Head()

    descs = get_metric_descriptions(Model())
    assert descs["field_amp"]["caller"] == "Field"
    assert descs["head_loss"]["caller"] == "Head"
    # Optimizer telemetry is universal and attributed generically.
    for k in OPTIMIZER_METRIC_DESCRIPTIONS:
        assert descs[k]["caller"] == "Optimizer"
