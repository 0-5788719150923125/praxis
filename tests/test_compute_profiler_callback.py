"""ComputeProfilerCallback lifecycle: arming, closing, stashing, compile guard."""

import pytest
import torch
import torch.nn as nn

from praxis.callbacks.lightning.compute_profiler import (
    _MAX_WINDOW_BATCHES,
    ComputeProfilerCallback,
)


class Tiny(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.fc = nn.Linear(d, d)

    def forward(self, x):
        return self.fc(x).relu()


class Stack(nn.Module):
    """Two levels deep with EXECUTING direct children.

    Shaped like the real model (encoder / decoder / head are called directly)
    rather than a bare ModuleList, which never executes and so would never fire
    a depth-0 hook.
    """

    def __init__(self, d=16):
        super().__init__()
        self.encoder = Tiny(d)
        self.decoder = Tiny(d)
        self.head = nn.Linear(d, d)

    def forward(self, x):
        return self.head(self.decoder(self.encoder(x)))


class FakeCompiled(nn.Module):
    """Stands in for torch.compile's OptimizedModule wrapper."""

    def __init__(self, inner):
        super().__init__()
        self._orig_mod = inner

    def forward(self, *a, **k):
        return self._orig_mod(*a, **k)


class FakeModule:
    """Stands in for the LightningModule wrapper around the model."""

    def __init__(self, model):
        self.model = model
        self.device = torch.device("cpu")


class FakeTrainer:
    def __init__(self, step=0, zero=True):
        self.global_step = step
        self.is_global_zero = zero


@pytest.fixture
def wired():
    cb = ComputeProfilerCallback({"warmup_steps": 0, "interval": 10})
    pl = FakeModule(Tiny())
    cb.on_train_start(FakeTrainer(), pl)
    return cb, pl


def _run_step(cb, pl, trainer, batch_idx=0):
    cb.on_train_batch_start(trainer, pl, None, batch_idx)
    pl.model(torch.randn(4, 16)).sum().backward()
    pl.model.zero_grad(set_to_none=True)
    cb.on_before_optimizer_step(trainer, pl, None)
    cb.on_train_batch_end(trainer, pl, None, None, batch_idx)


# ── install guard ───────────────────────────────────────────────────────────


def test_compiled_model_gets_a_coarse_forward_only_profile(capsys):
    """Compiled runs are profiled, just coarsely and forward-only."""
    inner = Stack()
    wrapper = FakeCompiled(inner)
    cb = ComputeProfilerCallback()
    cb.on_train_start(FakeTrainer(), FakeModule(wrapper))

    assert cb._installed and not cb._disabled
    assert cb.profiler.forward_only is True
    # only the model's direct children carry scopes
    hooked = {
        m._praxis_scope.split("|")[0]
        for _, m in inner.named_modules()
        if hasattr(m, "_praxis_scope")
    }
    assert hooked == {"encoder", "decoder", "head"}
    out = capsys.readouterr().out
    assert "forward only" in out and "top-level" in out


def test_eager_model_gets_the_full_profile(capsys):
    cb = ComputeProfilerCallback()
    model = Stack()
    cb.on_train_start(FakeTrainer(), FakeModule(model))

    assert cb._installed and cb.profiler.forward_only is False
    deep = [
        m for _, m in model.named_modules()
        if getattr(m, "_praxis_scope", "").startswith("encoder.fc|")
    ]
    assert deep, "eager mode must instrument leaf modules"
    assert "forward+backward" in capsys.readouterr().out


def test_compiled_snapshot_is_labelled_forward_only():
    cb = ComputeProfilerCallback({"warmup_steps": 0, "interval": 10})
    inner = Stack()
    pl = FakeModule(FakeCompiled(inner))
    cb.on_train_start(FakeTrainer(), pl)

    trainer = FakeTrainer(step=0)
    cb.on_train_batch_start(trainer, pl, None, 0)
    inner(torch.randn(4, 16)).sum().backward()
    inner.zero_grad(set_to_none=True)
    cb.on_before_optimizer_step(trainer, pl, None)

    stash = getattr(inner, "_compute_profile", None)
    assert stash is not None, "no sample landed"
    assert stash["compute_profile"]["mode"] == "forward"


def test_installs_on_an_eager_model(wired):
    cb, _ = wired
    assert cb._installed and not cb._disabled


def test_a_disabled_callback_never_arms(wired):
    cb, pl = wired
    cb._disabled = True
    cb.on_train_batch_start(FakeTrainer(step=100), pl, None, 0)
    assert cb._active is None


def test_only_global_zero_profiles(wired):
    cb, pl = wired
    cb.on_train_batch_start(FakeTrainer(step=100, zero=False), pl, None, 0)
    assert cb._active is None


# ── cadence ─────────────────────────────────────────────────────────────────


def test_respects_warmup():
    cb = ComputeProfilerCallback({"warmup_steps": 50, "interval": 10})
    pl = FakeModule(Tiny())
    cb.on_train_start(FakeTrainer(), pl)
    cb.on_train_batch_start(FakeTrainer(step=10), pl, None, 0)
    assert cb._active is None
    cb.on_train_batch_start(FakeTrainer(step=50), pl, None, 0)
    assert cb._active is not None
    cb._close(pl)


def test_samples_on_the_interval(wired):
    cb, pl = wired
    _run_step(cb, pl, FakeTrainer(step=0))
    assert cb.profiler.samples == 1
    # too soon: interval is 10
    _run_step(cb, pl, FakeTrainer(step=5))
    assert cb.profiler.samples == 1
    _run_step(cb, pl, FakeTrainer(step=10))
    assert cb.profiler.samples == 2


def test_accumulation_microbatches_do_not_each_arm(wired):
    """global_step does not advance between microbatches; one window per step."""
    cb, pl = wired
    trainer = FakeTrainer(step=0)
    cb.on_train_batch_start(trainer, pl, None, 0)
    first = cb._active
    assert first is not None
    cb.on_train_batch_start(trainer, pl, None, 1)
    assert cb._active is first, "second microbatch opened a second profiler"
    cb._close(pl)


# ── window closing ──────────────────────────────────────────────────────────


def test_before_optimizer_step_closes_the_window(wired):
    cb, pl = wired
    _run_step(cb, pl, FakeTrainer(step=0))
    assert cb._active is None


def test_window_closes_after_one_microbatch(wired):
    """Under accumulation the window must NOT span the whole cycle.

    Every microbatch runs the same graph, so extra ones add no information and
    multiply the profiled step's cost by the accumulation factor.
    """
    cb, pl = wired
    trainer = FakeTrainer(step=0)
    cb.on_train_batch_start(trainer, pl, None, 0)
    assert cb._active is not None
    pl.model(torch.randn(2, 16)).sum().backward()
    pl.model.zero_grad(set_to_none=True)
    cb.on_train_batch_end(trainer, pl, None, None, 0)
    assert cb._active is None, "window survived the first microbatch"
    assert cb._window_batches == 1


def test_later_microbatches_do_not_reopen_the_window(wired):
    cb, pl = wired
    trainer = FakeTrainer(step=0)
    cb.on_train_batch_start(trainer, pl, None, 0)
    pl.model(torch.randn(2, 16)).sum().backward()
    pl.model.zero_grad(set_to_none=True)
    cb.on_train_batch_end(trainer, pl, None, None, 0)
    # global_step has not advanced: the interval gate must keep it shut
    for i in range(1, 4):
        cb.on_train_batch_start(trainer, pl, None, i)
        assert cb._active is None, f"re-armed on microbatch {i}"


def test_train_end_closes_and_detaches(wired):
    cb, pl = wired
    cb.on_train_batch_start(FakeTrainer(step=0), pl, None, 0)
    cb.on_train_end(FakeTrainer(step=1), pl)
    assert cb._active is None
    assert not cb.profiler._handles


# ── stashing ────────────────────────────────────────────────────────────────


def test_stashes_where_the_dashboard_reads(wired):
    cb, pl = wired
    _run_step(cb, pl, FakeTrainer(step=0))
    assert isinstance(getattr(pl.model, "_compute_profile", None), dict)
    assert "compute_profile" in pl.model._compute_profile
    metrics = getattr(pl.model, "_compute_metrics", None)
    assert isinstance(metrics, dict) and "compute_coverage" in metrics


def test_nothing_is_stashed_before_a_sample(wired):
    cb, pl = wired
    assert not hasattr(pl.model, "_compute_profile")


def test_dynamics_callback_drains_the_stash(wired):
    from praxis.callbacks.lightning.dynamics import DynamicsLoggerCallback

    cb, pl = wired
    _run_step(cb, pl, FakeTrainer(step=0))
    drained = DynamicsLoggerCallback._extract_compute_dynamics(None, pl.model)
    assert "compute_coverage" in drained
    assert all(isinstance(v, (int, float)) for v in drained.values())


def test_dynamics_drain_is_empty_without_the_profiler():
    from praxis.callbacks.lightning.dynamics import DynamicsLoggerCallback

    assert DynamicsLoggerCallback._extract_compute_dynamics(None, Tiny()) == {}


def test_metric_keys_are_sql_safe(wired):
    """dynamics.db does an unquoted ALTER TABLE ADD COLUMN per key."""
    import re

    cb, pl = wired
    _run_step(cb, pl, FakeTrainer(step=0))
    for key in pl.model._compute_metrics:
        assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key), key


def test_announces_once(capsys, wired):
    cb, pl = wired
    _run_step(cb, pl, FakeTrainer(step=0))
    first = capsys.readouterr().out
    _run_step(cb, pl, FakeTrainer(step=10))
    second = capsys.readouterr().out
    assert "ComputeProfiler" in first
    assert "CLASS" not in second, "summary printed more than once"
