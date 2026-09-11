import io
import math
import random
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch import nn

from praxis import PraxisConfig, PraxisForCausalLM
from praxis.callbacks.lightning import (
    HarmonicWeightRLCallback,
    SnapshotPumpCallback,
    StallWatchdogCallback,
)
from praxis.callbacks.lightning.compute_profiler import ComputeProfilerCallback
from praxis.callbacks.lightning.governor import GNSBatchGovernor
from praxis.callbacks.lightning.signal_handler import SignalHandlerCallback
from praxis.callbacks.lightning.terminal import TerminalInterface
from praxis.data.batch_schedule import BatchSchedule, plan_cycle
from praxis.data.seq_probe import SequenceProbe
from praxis.environments import EnvironmentFeatures
from praxis.generation.decode_backend import ModelBackend
from praxis.governors.gns import GradientNoiseEstimator
from praxis.memory.neural_memory import NeuralMemory
from praxis.policies import EngagementPolicy
from praxis.policies.harmonic_weight_rl import HarmonicWeightPolicy
from praxis.trainers import BackpropagationTrainer
from praxis.web.snapshots import Recipe, SnapshotProducer, SnapshotStore

# ------------------------------------------------------------------------------
# governor
# ------------------------------------------------------------------------------
# GNS batch governor: estimator math, tier control, Lightning wiring.


@pytest.fixture(autouse=True)
def _clean_schedule():
    """The schedule is class-level state a governor writes on construction."""
    BatchSchedule.reset()
    yield
    BatchSchedule.reset()


def test_terminal_reports_live_effective_batch():
    """The info panel derives target_batch from the trainer itself, not the
    decision-cadence stash: zero staleness when the governor moves a tier."""
    from praxis.callbacks.lightning.terminal import TerminalInterface

    t = SimpleNamespace(accumulate_grad_batches=4, world_size=1)
    assert TerminalInterface._effective_batch(t, 16) == 64
    t.accumulate_grad_batches = 8  # a committed tier change is visible at once
    assert TerminalInterface._effective_batch(t, 16) == 128
    assert TerminalInterface._effective_batch(SimpleNamespace(), 16) is None
    assert TerminalInterface._effective_batch(t, None) is None


# ── Lightning wiring (simulated hooks) ───────────────────────────────────


def _fake_trainer(factor=1):
    return SimpleNamespace(
        accumulate_grad_batches=factor,
        world_size=1,
        global_step=0,
        val_check_batch=None,
        fit_loop=SimpleNamespace(
            epoch_loop=SimpleNamespace(
                batch_progress=SimpleNamespace(current=SimpleNamespace(ready=0))
            )
        ),
    )


def _set_grads(module, value):
    for p in module.parameters():
        p.grad = torch.full_like(p, value)


def _run_cycle(gov, trainer, module, k, rows=None):
    """Simulate one full accumulation cycle of k microbatches, mirroring
    Lightning's ordering: batch_start -> backward -> (step + global_step bump)
    -> batch_end with the within-epoch batch index.

    ``rows`` is the microbatch row count the data pipeline produced; the
    governor reads it from the batch rather than assuming the plan's value."""
    if rows is None:
        rows = plan_cycle(gov._rows, gov.row_ceiling).micro_rows
    cur = trainer.fit_loop.epoch_loop.batch_progress.current
    batch = torch.zeros(rows, 8, dtype=torch.long)
    for i in range(k):
        gov.on_train_batch_start(trainer, module, batch, cur.ready)
        _set_grads(module, 0.01 * (i + 1))
        gov.on_after_backward(trainer, module)
        cur.ready += 1
        if i == k - 1:
            gov.on_before_optimizer_step(trainer, module, None)
            trainer.global_step += 1
        gov.on_train_batch_end(trainer, module, None, None, cur.ready - 1)


def test_callback_undoes_lightning_loss_scaling():
    """First-microbatch grads arrive scaled by 1/K; the estimator must see
    the unscaled squared norm (x K^2)."""
    gov = GNSBatchGovernor(batch_size=16, target_batch_size=512)
    trainer = _fake_trainer()
    module = nn.Linear(3, 3, bias=False)
    gov.on_train_start(trainer, module)
    # Starts at one full microbatch per point: 2 x 16 = 32 rows.
    assert trainer.accumulate_grad_batches == 2
    assert gov._rows == 32

    k = 2
    batch = torch.zeros(16, 8, dtype=torch.long)
    gov.on_train_batch_start(trainer, module, batch, 0)
    _set_grads(module, 0.6 / k)  # Lightning-scaled first microbatch
    gov.on_after_backward(trainer, module)
    gov.on_train_batch_start(trainer, module, batch, 1)
    gov.on_after_backward(trainer, module)  # second microbatch (count -> k)
    _set_grads(module, 0.5)  # accumulated gradient
    gov.on_before_optimizer_step(trainer, module, None)

    state = gov.estimator.state_dict()
    assert state["updates"] == 1
    # small_sq = 9 * 0.6^2 = 3.24 (after the K^2 correction), big_sq = 2.25:
    # g = (32*2.25 - 16*3.24)/16 = 1.26; s = (3.24 - 2.25)/(1/16 - 1/32).
    assert state["g_sq_ema"] == pytest.approx(1.26, rel=1e-4)
    assert state["s_ema"] == pytest.approx(31.68, rel=1e-4)


def test_up_move_defers_until_aligned_boundary():
    """Lightning steps at ready % factor == 0, so a 2->4 move at ready=2 must
    wait for ready=4 - otherwise the first new cycle is short and mis-scaled."""
    gov = GNSBatchGovernor(batch_size=16, target_batch_size=512)
    trainer = _fake_trainer()
    module = nn.Linear(4, 4)
    gov.on_train_start(trainer, module)

    # Warm the estimator with measurements that say B_noise ~ 1584 rows.
    for _ in range(gov.estimator.min_updates):
        gov.estimator.update(small_sq=1000.0, big_sq=505.0, b_small=16, b_big=32)
    gov._steps = gov.decide_every - 1  # next step triggers a decision

    _run_cycle(gov, trainer, module, k=2)  # decision fires at ready=2
    # (rows, measuring): 32 -> 64, which needs accum 4. Above the ceiling the
    # split is forced by memory, so the window stays a measuring one.
    assert gov._pending == (64, True)
    assert trainer.accumulate_grad_batches == 2  # deferred: 2 % 4 != 0

    _run_cycle(gov, trainer, module, k=2)  # boundary at ready=4 commits
    assert trainer.accumulate_grad_batches == 4
    assert gov._rows == 64
    assert gov._pending is None


def test_irregular_cycle_is_not_measured():
    """A cycle whose microbatch count != K (epoch-end early step) must not
    feed the estimator - its loss scaling doesn't match."""
    gov = GNSBatchGovernor(batch_size=16, target_batch_size=512)
    trainer = _fake_trainer()
    module = nn.Linear(4, 4)
    gov.on_train_start(trainer, module)
    gov.on_train_batch_start(trainer, module, torch.zeros(16, 8, dtype=torch.long), 0)
    _set_grads(module, 0.1)
    gov.on_after_backward(trainer, module)  # only 1 of K=2 microbatches
    gov.on_before_optimizer_step(trainer, module, None)
    assert gov.estimator._updates == 0


def test_validation_fires_on_exact_step_cadence():
    """Every batch end the governor repoints Lightning's batch-modulo check
    at the raw batch where the next val_every optimizer-step boundary lands,
    so validation fires at global_step multiples regardless of the factor."""
    gov = GNSBatchGovernor(batch_size=16, target_batch_size=512, val_every=3)
    trainer = _fake_trainer()
    module = nn.Linear(4, 4)
    gov.on_train_start(trainer, module)
    assert trainer.val_check_batch == GNSBatchGovernor.VAL_PARKED

    _run_cycle(gov, trainer, module, k=2)  # global_step 1
    _run_cycle(gov, trainer, module, k=2)  # global_step 2
    # Mid-interval the target already points at the boundary batch (step 3
    # at factor 2 lands on raw batch 6) - strictly ahead, so no early fire.
    assert trainer.val_check_batch == 6

    _run_cycle(gov, trainer, module, k=2)  # global_step 3: boundary
    # Target equals the just-finished batch count: (5+1) % 6 == 0 fires NOW.
    assert trainer.val_check_batch == 6

    # After the boundary, the target moves a whole interval ahead - even
    # across a factor change (steps 4-6 at factor 4 end on raw batch 18).
    trainer.accumulate_grad_batches = 4
    _run_cycle(gov, trainer, module, k=4)  # global_step 4
    assert trainer.val_check_batch == 18
    _run_cycle(gov, trainer, module, k=4)  # global_step 5
    assert trainer.val_check_batch == 18
    _run_cycle(gov, trainer, module, k=4)  # global_step 6: boundary
    assert trainer.val_check_batch == 18  # == batches done: fires


def test_validation_target_recovers_after_resume_gap():
    """Cadence anchors to absolute global_step multiples: resuming at an
    arbitrary step targets the NEXT boundary, no state carried over."""
    gov = GNSBatchGovernor(batch_size=16, target_batch_size=512, val_every=1000)
    trainer = _fake_trainer(factor=2)
    trainer.global_step = 6100  # resumed mid-interval
    trainer.fit_loop.epoch_loop.batch_progress.current.ready = 42000
    module = nn.Linear(4, 4)
    gov.on_train_start(trainer, module)
    _run_cycle(gov, trainer, module, k=2)  # global_step 6101
    # 899 steps to the 7000 boundary, at factor 2 from 42002 batches done.
    assert trainer.val_check_batch == 42002 + 899 * 2


def test_validation_cadence_disabled_without_val_every():
    gov = GNSBatchGovernor(batch_size=16, target_batch_size=512)
    trainer = _fake_trainer()
    trainer.val_check_batch = "untouched"
    module = nn.Linear(4, 4)
    gov.on_train_start(trainer, module)
    for _ in range(3):
        _run_cycle(gov, trainer, module, k=2)
    assert trainer.val_check_batch == "untouched"


def test_state_dict_roundtrip_restores_effective_rows():
    src = GNSBatchGovernor(batch_size=16, target_batch_size=512)
    src._rows = 128
    src._steps = 33
    src.estimator.update(small_sq=10.0, big_sq=6.0, b_small=16, b_big=32)
    dst = GNSBatchGovernor(batch_size=16, target_batch_size=512)
    dst.load_state_dict(src.state_dict())
    trainer = _fake_trainer()
    dst.on_train_start(trainer, nn.Linear(2, 2))
    assert dst._rows == 128
    assert trainer.accumulate_grad_batches == 8  # 128 rows / 16-row ceiling
    assert dst._steps == 33
    assert dst.estimator.state_dict()["updates"] == 1


def test_state_dict_accepts_legacy_factor_key():
    """Pre-rows checkpoints stored an accumulation factor against the fixed
    microbatch; resuming one must land on the same effective batch."""
    gov = GNSBatchGovernor(batch_size=16, target_batch_size=512)
    gov.load_state_dict({"factor": 4, "steps": 10, "estimator": {}})
    assert gov._rows == 64  # 4 x 16
    trainer = _fake_trainer()
    gov.on_train_start(trainer, nn.Linear(2, 2))
    assert trainer.accumulate_grad_batches == 4


def test_metrics_stash_and_descriptions_fold():
    gov = GNSBatchGovernor(batch_size=16, target_batch_size=512)
    trainer = _fake_trainer()
    module = nn.Linear(4, 4)
    gov.on_train_start(trainer, module)
    stash = module._governor_metrics
    assert stash["gov_effective_batch"] == 32.0
    assert stash["gov_target_batch"] == 32.0

    from praxis.metrics.descriptions import get_metric_descriptions

    class _Bare:
        pass

    plain = _Bare()
    assert "gov_noise_scale" not in get_metric_descriptions(plain)

    governed = _Bare()
    governed._governor_metrics = {"gov_effective_batch": 32.0}
    descs = get_metric_descriptions(governed)
    assert "gov_noise_scale" in descs
    assert descs["gov_effective_batch"]["caller"] == "GNSBatchGovernor"
    # Series companions render on the lead metric's chart.
    assert (
        descs["gov_effective_batch"]["chart"]["series_group"]
        == descs["gov_noise_scale"]["chart"]["series_group"]
    )


# ── governor + schedule together ─────────────────────────────────────────


def test_governor_publishes_the_plan_on_start():
    gov = GNSBatchGovernor(batch_size=64, target_batch_size=512)
    trainer = _fake_trainer()
    gov.on_train_start(trainer, nn.Linear(2, 2))
    assert BatchSchedule.enabled
    assert BatchSchedule.row_ceiling == 64
    assert BatchSchedule.effective_rows == 128  # 2 x the ceiling
    assert trainer.accumulate_grad_batches == 2
    assert BatchSchedule.next_microbatch().micro_rows == 64


def test_governor_can_descend_below_the_microbatch_ceiling():
    """End to end: a sustained small noise scale walks the effective batch
    below batch_size, which the old floor made unreachable."""
    gov = GNSBatchGovernor(batch_size=64, target_batch_size=512)
    trainer = _fake_trainer()
    module = nn.Linear(4, 4)
    gov.on_train_start(trainer, module)
    assert gov._rows == 128

    for _ in range(6):
        # Measurements that put B_noise far below the current batch.
        gov.estimator = GradientNoiseEstimator()
        for _ in range(gov.estimator.min_updates):
            gov.estimator.update(small_sq=4.0, big_sq=3.98, b_small=2, b_big=4)
        gov._steps = gov.decide_every - 1
        k = int(trainer.accumulate_grad_batches)
        _run_cycle(gov, trainer, module, k=k)
        if gov._pending is not None:
            # Let the deferred move land on its aligned boundary.
            for _ in range(4):
                if gov._pending is None:
                    break
                _run_cycle(gov, trainer, module, k=int(trainer.accumulate_grad_batches))

    assert gov._rows < 64, gov._rows
    plan = plan_cycle(gov._rows, gov.row_ceiling)
    assert plan.delivered_rows == gov._rows
    assert plan.accum == 2  # still measurable at the bottom


def test_governor_measures_observed_rows_not_planned_rows():
    """The plan is shared state the pipeline can lag on; the estimator's two
    points must come from the batches that actually arrived."""
    gov = GNSBatchGovernor(batch_size=64, target_batch_size=512)
    trainer = _fake_trainer()
    module = nn.Linear(3, 3, bias=False)
    gov.on_train_start(trainer, module)

    k = 2
    stale = torch.zeros(8, 4, dtype=torch.long)  # pipeline still on 8 rows
    gov.on_train_batch_start(trainer, module, stale, 0)
    _set_grads(module, 0.6 / k)
    gov.on_after_backward(trainer, module)
    gov.on_train_batch_start(trainer, module, stale, 1)
    gov.on_after_backward(trainer, module)
    _set_grads(module, 0.5)
    gov.on_before_optimizer_step(trainer, module, None)

    assert gov.estimator._updates == 1
    # small_sq = 9*(0.3)^2 * k^2 = 3.24, big_sq = 9*0.5^2 = 2.25. The S term is
    # what distinguishes the two readings: at the observed (8, 16) rows it is
    # 0.99/(1/8 - 1/16) = 15.84, where the planned (64, 128) would give 126.72.
    # |G|^2 is scale-free across proportional pairs, so it cannot tell them
    # apart - assert on S.
    state = gov.estimator.state_dict()
    assert state["s_ema"] == pytest.approx(15.84, rel=1e-4)
    assert state["g_sq_ema"] == pytest.approx(1.26, rel=1e-4)


def test_mixed_shape_cycle_is_not_measured():
    """A cycle whose microbatches differ in size breaks both the uniform
    1/accum loss scaling and the estimator's pairing - skip the pair."""
    gov = GNSBatchGovernor(batch_size=64, target_batch_size=512)
    trainer = _fake_trainer()
    module = nn.Linear(3, 3)
    gov.on_train_start(trainer, module)

    gov.on_train_batch_start(trainer, module, torch.zeros(64, 4), 0)
    _set_grads(module, 0.1)
    gov.on_after_backward(trainer, module)
    gov.on_train_batch_start(trainer, module, torch.zeros(16, 4), 1)  # shape changed
    gov.on_after_backward(trainer, module)
    _set_grads(module, 0.2)
    gov.on_before_optimizer_step(trainer, module, None)
    assert gov.estimator._updates == 0


# ------------------------------------------------- measurement duty cycle
#
# Splitting a step into two microbatches is forced by memory above the row
# ceiling and free there. At or below it the split exists ONLY to give the
# two-point estimator its second point - so paying it on every step ran two
# forwards where one would do (an effective batch of 4 rows against a 64-row
# ceiling ran 2x2 instead of 1x4). The governor pays it on the windows it
# actually measures instead.


def test_small_batch_uses_one_microbatch_when_not_measuring():
    """The reported bug: 4 governed rows under a 64-row ceiling ran 2x2."""
    gov = GNSBatchGovernor(batch_size=64, target_batch_size=1024)

    exploit = gov._plan(4, measuring=False)
    assert (exploit.accum, exploit.micro_rows) == (1, 4)
    assert exploit.delivered_rows == 4

    measure = gov._plan(4, measuring=True)
    assert (measure.accum, measure.micro_rows) == (2, 2)
    assert measure.delivered_rows == 4  # same data either way


def test_split_above_the_ceiling_is_unchanged():
    """Where memory forces >= 2 microbatches the second point is free, so the
    duty cycle must never reduce it - and never stop measuring."""
    gov = GNSBatchGovernor(batch_size=64, target_batch_size=1024)
    for rows in (128, 256, 1024):
        assert gov._plan(rows, measuring=False) == gov._plan(rows, measuring=True)
        assert gov._split_is_free(rows)
        assert gov._should_measure(rows)


def test_duty_cycle_only_applies_below_the_ceiling():
    gov = GNSBatchGovernor(batch_size=64, target_batch_size=1024)
    # Small batch: measures on 1 window in MEASURE_EVERY.
    measured = []
    for _ in range(gov.MEASURE_EVERY * 3):
        gov._windows += 1
        measured.append(gov._should_measure(4))
    assert sum(measured) == 3
    # Large batch: every window, regardless of the counter.
    assert all(gov._should_measure(256) for _ in range(gov.MEASURE_EVERY * 3))


def test_estimator_is_not_fed_by_unsplit_steps():
    """With accum=1 there is no pair; folding one in would corrupt the EMA."""
    gov = GNSBatchGovernor(batch_size=64, target_batch_size=1024)
    trainer = _fake_trainer()
    module = nn.Linear(4, 4)
    gov.on_train_start(trainer, module)
    gov._rows, gov._measuring = 4, False
    gov._publish(trainer)
    assert trainer.accumulate_grad_batches == 1

    _run_cycle(gov, trainer, module, k=1, rows=4)
    assert gov.estimator._updates == 0
    # ...but the delivered rows still reach the controller, so a window with no
    # measurement is not a window with no bookkeeping.
    assert gov._delivered_mean() == 4


def test_measuring_change_goes_through_the_aligned_commit():
    """Turning measuring on/off moves accum, so it must not bypass alignment -
    an unaligned factor change produces a short, mis-scaled cycle."""
    gov = GNSBatchGovernor(batch_size=64, target_batch_size=1024)
    trainer = _fake_trainer()
    module = nn.Linear(4, 4)
    gov.on_train_start(trainer, module)

    gov._rows, gov._measuring = 8, True
    gov._publish(trainer)
    assert trainer.accumulate_grad_batches == 2

    # Pend a switch to exploiting (accum 2 -> 1) at an odd `ready`.
    cur = trainer.fit_loop.epoch_loop.batch_progress.current
    cur.ready = 3
    gov._pending = (8, False)
    gov._stepped = True
    gov.on_train_batch_end(trainer, module, None, None, cur.ready - 1)
    # accum 1 divides everything, so ready=3 IS aligned and it commits.
    assert trainer.accumulate_grad_batches == 1
    assert gov._measuring is False


def test_schedule_agrees_with_the_governor_about_accum():
    """The data pipeline and Lightning read the split minimum from the same
    place; if they disagreed the pipeline would build cycles of the wrong
    length for the factor the trainer steps on."""
    from praxis.data.batch_schedule import BatchSchedule

    gov = GNSBatchGovernor(batch_size=64, target_batch_size=1024)
    trainer = _fake_trainer()
    try:
        for rows, measuring in ((4, False), (4, True), (256, False)):
            gov._rows, gov._measuring = rows, measuring
            gov._publish(trainer)
            assert BatchSchedule.accum() == trainer.accumulate_grad_batches
            plan = BatchSchedule.next_microbatch()
            assert plan.accum == trainer.accumulate_grad_batches
    finally:
        BatchSchedule.reset()


def test_resume_returns_to_measuring():
    """The restored EMA was fitted on gradients this run never saw; taking a
    fresh window before the first decision costs one window."""
    gov = GNSBatchGovernor(batch_size=64, target_batch_size=1024)
    gov._rows, gov._measuring = 4, False
    state = gov.state_dict()

    fresh = GNSBatchGovernor(batch_size=64, target_batch_size=1024)
    fresh.load_state_dict(state)
    assert fresh._rows == 4
    assert fresh._measuring is True


# ------------------------------------------------------------------------------
# compute_profiler_callback
# ------------------------------------------------------------------------------
# ComputeProfilerCallback lifecycle: arming, closing, stashing, compile guard.


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
        m
        for _, m in model.named_modules()
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


# ------------------------------------------------------------------------------
# harmonic_weight_rl
# ------------------------------------------------------------------------------
# Harmonic-weight RL controller: policy-gradient mechanics + callback loop.


def _cfg(**over):
    base = dict(
        rl_hidden=16,
        rl_lr=0.05,
        rl_entropy_coef=0.0,
        rl_alpha_scale=1.0,
        rl_omega_max=math.pi,
        rl_baseline_decay=0.9,
    )
    base.update(over)
    return SimpleNamespace(**base)


class _Trainer:
    def __init__(self):
        self.callback_metrics = {}
        self.global_step = 0


class _PL:
    def __init__(self, model):
        self.model = model


def test_callback_keeps_helpful_edit_and_updates_policy():
    torch.manual_seed(0)
    policy = HarmonicWeightPolicy(_cfg(rl_alpha_scale=0.3))
    cb = HarmonicWeightRLCallback(
        policy, period=3, horizon=2, warmup_steps=3, keep_threshold=0.0
    )
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    pl, tr = _PL(model), _Trainer()

    before = [p.detach().clone() for p in model.parameters()]
    # Five steps so exactly one episode completes (warmup=3 -> start at step 3,
    # horizon=2 -> finish at step 5) with no new episode left dangling.
    # Loss drops across the episode -> positive reward -> edit kept.
    losses = [5.0, 5.0, 5.0, 4.0, 2.0]
    for ls in losses:
        cb.on_train_batch_end(tr, pl, torch.tensor(ls), None, 0)

    assert cb._metrics, "an episode should have completed"
    assert cb._metrics["rl_edit_kept"] == 1.0
    # The kept edit changed exactly the model (some 2D weight row differs).
    after = list(model.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed
    # rl_* scalars were published to callback_metrics for the logger.
    assert "rl_reward" in tr.callback_metrics and "rl_edit_kept" in tr.callback_metrics


def test_reward_is_ema_return_over_horizon():
    # The per-edit reward is an EMA-integrated return over the horizon, not the
    # one-step endpoint delta. With loss_ema_decay=0 the smoothed loss equals
    # the raw loss, so the arithmetic is exact.
    torch.manual_seed(0)
    policy = HarmonicWeightPolicy(_cfg(rl_alpha_scale=0.3))
    cb = HarmonicWeightRLCallback(
        policy,
        period=3,
        horizon=3,
        warmup_steps=3,
        keep_threshold=0.0,
        loss_ema_decay=0.0,
        reward_decay=0.5,
    )
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    pl, tr = _PL(model), _Trainer()

    # Episode starts at step 3 (L_before=5), edit takes effect on steps 4..6.
    # Post-edit improvements vs L_before=5 are 1, 2, 3; EMA(d=0.5): 1 -> 1.5 -> 2.25.
    for ls in [5.0, 5.0, 5.0, 4.0, 3.0, 2.0]:
        cb.on_train_batch_end(tr, pl, torch.tensor(ls), None, 0)

    assert cb._metrics["rl_reward"] == pytest.approx(2.25)  # EMA return
    assert cb._metrics["rl_reward_instant"] == pytest.approx(3.0)  # endpoint delta
    assert cb._metrics["rl_edit_kept"] == 1.0


def test_callback_rolls_back_unhelpful_edit():
    torch.manual_seed(0)
    policy = HarmonicWeightPolicy(_cfg(rl_alpha_scale=0.3))
    cb = HarmonicWeightRLCallback(
        policy, period=3, horizon=2, warmup_steps=3, keep_threshold=0.0
    )
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    pl, tr = _PL(model), _Trainer()

    before = [p.detach().clone() for p in model.parameters()]
    # One completed episode (steps 3..5), nothing dangling. Loss rises across
    # the episode -> negative reward -> edit rolled back.
    losses = [3.0, 3.0, 3.0, 4.0, 6.0]
    for ls in losses:
        cb.on_train_batch_end(tr, pl, torch.tensor(ls), None, 0)

    assert cb._metrics["rl_edit_kept"] == 0.0
    # Rollback restored the weights exactly.
    after = list(model.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_edit_kept_reports_rolling_rate_not_binary():
    # rl_edit_kept is an EMA of the per-episode keep decision, so over multiple
    # mixed episodes it lands strictly between 0 and 1 (the chart is a rate, not
    # a 0/1 line). loss_ema_decay=0 makes L_before exactly the start-step loss.
    torch.manual_seed(0)
    policy = HarmonicWeightPolicy(_cfg(rl_alpha_scale=0.3))
    cb = HarmonicWeightRLCallback(
        policy,
        period=3,
        horizon=2,
        warmup_steps=3,
        keep_threshold=0.0,
        loss_ema_decay=0.0,
    )
    pl, tr = _PL(nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))), _Trainer()

    # Episode 1 (steps 3-5): loss falls -> kept -> rate seeds to 1.0.
    # Episode 2 (steps 6-8): loss rises -> rolled back -> rate = 0.9*1 + 0.1*0.
    for ls in [5.0, 5.0, 5.0, 4.0, 3.0, 3.0, 4.0, 5.0]:
        cb.on_train_batch_end(tr, pl, torch.tensor(ls), None, 0)

    kept = cb._metrics["rl_edit_kept"]
    assert 0.0 < kept < 1.0, kept  # the whole point: an in-between value
    assert kept == pytest.approx(0.9)


def _schedulefree(model):
    # Outermost wrapper, as praxis builds it; one step to create the z iterate.
    from pytorch_optimizer.optimizer import ScheduleFreeWrapper

    base = torch.optim.SGD(model.parameters(), lr=1e-3)
    sf = ScheduleFreeWrapper(base, momentum=0.98)
    sf.train()
    model(torch.randn(2, 8)).sum().backward()
    sf.step()
    sf.zero_grad(set_to_none=True)
    return sf


def test_schedulefree_edit_mirrors_onto_z():
    # Under schedule-free the edit must also land on the carried iterate z, not
    # just p.data - else it gets smeared by the x/z reconstruction. Dropping
    # loss -> edit kept -> the chosen param's z row changed.
    torch.manual_seed(0)
    policy = HarmonicWeightPolicy(_cfg(rl_alpha_scale=0.3))
    cb = HarmonicWeightRLCallback(
        policy, period=3, horizon=2, warmup_steps=3, keep_threshold=0.0
    )
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    sf = _schedulefree(model)
    z_before = {p: sf.state[p]["z"].clone() for p in model.parameters()}

    tr, pl = _Trainer(), _PL(model)
    tr.optimizers = [sf]
    for ls in [5.0, 5.0, 5.0, 4.0, 2.0]:  # loss drops -> kept
        cb.on_train_batch_end(tr, pl, torch.tensor(ls), None, 0)

    assert cb._metrics["rl_edit_kept"] == 1.0
    # The callback found schedule-free and a z row was edited (and kept).
    assert cb._schedulefree(tr) is sf
    changed = any(
        not torch.equal(sf.state[p]["z"], z_before[p]) for p in model.parameters()
    )
    assert changed, "a schedule-free z iterate row should have been edited"


def test_schedulefree_rollback_restores_both_weight_and_z():
    # Rising loss -> edit rolled back -> both p.data and z restored exactly.
    torch.manual_seed(0)
    policy = HarmonicWeightPolicy(_cfg(rl_alpha_scale=0.3))
    cb = HarmonicWeightRLCallback(
        policy, period=3, horizon=2, warmup_steps=3, keep_threshold=0.0
    )
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    sf = _schedulefree(model)
    z_before = {p: sf.state[p]["z"].clone() for p in model.parameters()}
    w_before = [p.detach().clone() for p in model.parameters()]

    tr, pl = _Trainer(), _PL(model)
    tr.optimizers = [sf]
    for ls in [3.0, 3.0, 3.0, 4.0, 6.0]:  # loss rises -> rolled back
        cb.on_train_batch_end(tr, pl, torch.tensor(ls), None, 0)

    assert cb._metrics["rl_edit_kept"] == 0.0
    assert all(torch.equal(sf.state[p]["z"], z_before[p]) for p in model.parameters())
    assert all(torch.equal(b, p) for b, p in zip(w_before, model.parameters()))


def test_wave_mode_drives_and_rolls_back_the_optimizer_wave():
    # edit_mode="wave": the controller's action sets the WaveScheduleFree wave
    # (amp, cycles, phase); a non-helpful change restores the three scalars.
    from praxis.optimization.wave_schedule_free import WaveScheduleFree

    torch.manual_seed(0)
    policy = HarmonicWeightPolicy(_cfg())
    cb = HarmonicWeightRLCallback(
        policy,
        period=3,
        horizon=2,
        warmup_steps=3,
        keep_threshold=0.0,
        edit_mode="wave",
    )
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    sf = WaveScheduleFree(torch.optim.SGD(model.parameters(), lr=1e-3), momentum=0.9)
    sf.train()
    model(torch.randn(2, 8)).sum().backward()  # populate grads for the state
    wave_before = (sf.wave_amp, sf.wave_cycles, sf.wave_phase)

    tr, pl = _Trainer(), _PL(model)
    tr.optimizers = [sf]
    # Rising loss -> negative return -> the wave change is rolled back.
    for ls in [3.0, 3.0, 3.0, 4.0, 6.0]:
        cb.on_train_batch_end(tr, pl, torch.tensor(ls), None, 0)

    assert cb._metrics["rl_edit_kept"] == 0.0
    assert (sf.wave_amp, sf.wave_cycles, sf.wave_phase) == wave_before
    # The action was published under the reused rl_action_* keys.
    assert "rl_action_alpha" in cb._metrics  # = amp


def test_wave_mode_keeps_helpful_wave_change():
    from praxis.optimization.wave_schedule_free import WaveScheduleFree

    torch.manual_seed(0)
    policy = HarmonicWeightPolicy(_cfg())
    cb = HarmonicWeightRLCallback(
        policy,
        period=3,
        horizon=2,
        warmup_steps=3,
        keep_threshold=0.0,
        edit_mode="wave",
    )
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    sf = WaveScheduleFree(torch.optim.SGD(model.parameters(), lr=1e-3), momentum=0.9)
    sf.train()
    model(torch.randn(2, 8)).sum().backward()
    wave_before = (sf.wave_amp, sf.wave_cycles, sf.wave_phase)

    tr, pl = _Trainer(), _PL(model)
    tr.optimizers = [sf]
    for ls in [5.0, 5.0, 5.0, 4.0, 2.0]:  # dropping loss -> kept
        cb.on_train_batch_end(tr, pl, torch.tensor(ls), None, 0)

    assert cb._metrics["rl_edit_kept"] == 1.0
    assert (sf.wave_amp, sf.wave_cycles, sf.wave_phase) != wave_before  # wave moved


def test_anchor_gate_replaces_selected_with_anchor_and_rolls_back():
    torch.manual_seed(0)
    policy = HarmonicWeightPolicy(_cfg())
    # Near-deterministic action so the gate mask is stable for the assertion.
    policy.log_std.data.fill_(-10.0)
    cb = HarmonicWeightRLCallback(
        policy,
        period=3,
        horizon=2,
        warmup_steps=2,
        keep_threshold=0.0,
        edit_mode="anchor_gate",
        selector="sinusoidal",
    )
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    pl, tr = _PL(model), _Trainer()

    # Steps 1,2: anchor snapshot captured at warmup (step 2).
    for ls in (5.0, 5.0):
        cb.on_train_batch_end(tr, pl, torch.tensor(ls), None, 0)
    assert cb._anchor, "anchor should be snapshotted at warmup"

    # Simulate training drift: live weights move away from the anchor.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    # Step 3 starts the episode (gate-replaces a subset back to the anchor),
    # steps 4,5 run the horizon; loss drops -> reward>0 -> kept.
    for ls in (5.0, 4.0, 2.0):
        cb.on_train_batch_end(tr, pl, torch.tensor(ls), None, 0)

    assert cb._metrics["rl_edit_kept"] == 1.0
    assert 0.0 < cb._metrics["rl_gate_frac"] < 1.0
    # The kept edit pulled the gated elements back to the (older) anchor while
    # the ungated elements keep the drifted live value -> a partial reset.
    # Find the row that changed and verify it contains both anchor and live values.
    reverted = False
    for name, p in model.named_parameters():
        if name in cb._anchor and p.dim() == 2:
            anchor = cb._anchor[name]
            eq_anchor = (p.data == anchor).any(dim=1)
            eq_live = (p.data == anchor + 1.0).any(dim=1)
            if (eq_anchor & eq_live).any():
                reverted = True
    assert reverted, "expected a row gated partly to anchor, partly drifted-live"


def test_anchor_gate_rolls_back_unhelpful_edit():
    torch.manual_seed(0)
    policy = HarmonicWeightPolicy(_cfg())
    policy.log_std.data.fill_(-10.0)
    cb = HarmonicWeightRLCallback(
        policy,
        period=3,
        horizon=2,
        warmup_steps=2,
        keep_threshold=0.0,
        edit_mode="anchor_gate",
        selector="uniform_hash",
    )
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    pl, tr = _PL(model), _Trainer()

    for ls in (3.0, 3.0):
        cb.on_train_batch_end(tr, pl, torch.tensor(ls), None, 0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    before = [p.detach().clone() for p in model.parameters()]
    # Loss rises -> reward<0 -> rolled back to the (drifted) pre-edit weights.
    for ls in (3.0, 4.0, 6.0):
        cb.on_train_batch_end(tr, pl, torch.tensor(ls), None, 0)

    assert cb._metrics["rl_edit_kept"] == 0.0
    after = list(model.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))


# ------------------------------------------------------------------------------
# shutdown
# ------------------------------------------------------------------------------
# Cancelling a run must look like a cancellation, not a crash.
#
# Three things went wrong when a run was stopped cleanly:
#
# 1. The signal handler printed unguarded. Python delivers a handler on the MAIN thread
# between bytecodes, so a ValueError from a closed stdout surfaced as an exception
# rooted in whatever the main thread was executing - a traceback through an optimizer
# scan that had nothing to do with the cause. 2. That exception reached the generic
# ``except Exception`` in ``run_training`` and was reported as "fatal error" with a full
# traceback. 3. The cleanup thread nulled the terminal's dashboard while training was
# still stepping, so the inference hook's print fallback dumped rolling-context text
# onto the terminal the dashboard had just released.


class _DeadStream(io.StringIO):
    """A stream that raises exactly as a closed stdout does."""

    def write(self, s):
        raise ValueError("I/O operation on closed file.")

    def flush(self):
        raise ValueError("I/O operation on closed file.")

    @property
    def closed(self):
        return True


@pytest.fixture
def handler(monkeypatch):
    cb = SignalHandlerCallback()
    cb.trainer_ref = SimpleNamespace(should_stop=False)
    # The cleanup thread is not what these tests are about, and it would race
    # the assertions.
    monkeypatch.setattr(cb, "_deferred_cleanup", lambda: None)
    yield cb
    cb.cuda_manager._shutdown_requested = False


# ── the handler must be total ────────────────────────────────────────────


def test_handler_survives_a_closed_stdout(handler, monkeypatch):
    """The reported failure: ValueError escaping into unrelated main-thread
    code, which then gets classified as a crash."""
    monkeypatch.setattr(sys, "stdout", _DeadStream())
    monkeypatch.setattr(sys, "__stderr__", _DeadStream())

    handler._handle_signal(2, None)  # must not raise

    # And the shutdown still happened - the announcement is not load-bearing.
    assert handler.trainer_ref.should_stop is True
    assert handler.cuda_manager.is_shutting_down() is True


def test_handler_still_flags_shutdown_when_the_trainer_is_gone(handler):
    """Each step is independently guarded, so one failure cannot skip the
    others. The flag is what teardown reads to tell cancel from crash."""

    class Exploding:
        @property
        def should_stop(self):
            raise RuntimeError("trainer already torn down")

        @should_stop.setter
        def should_stop(self, value):
            raise RuntimeError("trainer already torn down")

    handler.trainer_ref = Exploding()
    handler._handle_signal(2, None)
    assert handler.cuda_manager.is_shutting_down() is True


def test_message_reaches_a_live_stdout(handler, monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    handler._handle_signal(2, None)
    assert "Gracefully stopping training" in buf.getvalue()


def test_message_falls_back_to_real_stderr(handler, monkeypatch):
    """stdout is captured into the dashboard's log panel; when it is dead the
    process's own stderr is the next best surface."""
    err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", _DeadStream())
    monkeypatch.setattr(sys, "__stderr__", err)
    handler._handle_signal(2, None)
    assert "Gracefully stopping training" in err.getvalue()


def test_cleanup_runs_before_the_fit_starts(monkeypatch):
    """A signal during dataset setup arrives before on_fit_start binds the
    terminal interface. Reading it raised AttributeError inside the cleanup
    thread, skipping the dataloader, CUDA and wandb steps AND the force-exit
    watchdog - the last defence against a hung shutdown."""
    cb = SignalHandlerCallback()
    assert cb.terminal_interface is None  # bound in __init__, not on_fit_start

    started = []
    monkeypatch.setattr(
        "praxis.callbacks.lightning.signal_handler.threading.Thread",
        lambda *a, **k: SimpleNamespace(start=lambda: started.append(k.get("target"))),
    )
    cb._deferred_cleanup()  # must not raise
    assert started, "the force-exit watchdog must still be armed"


# ── no internal text on the way out ──────────────────────────────────────


class _Interface(TerminalInterface):
    """Only the inference-display branch is under test."""

    def __init__(self, use_dashboard, dashboard):
        self.use_dashboard = use_dashboard
        self.dashboard = dashboard
        self.headless = False
        self.printed = []

    def print(self, text):
        self.printed.append(text)


def test_torn_down_dashboard_does_not_print_the_context():
    """Cleanup nulls the dashboard while training still steps; the fallback
    print would put the rolling context on the restored terminal."""
    iface = _Interface(use_dashboard=True, dashboard=None)
    iface._show_context("rolling context text")
    assert iface.printed == []


def test_run_without_a_dashboard_still_prints():
    """The fallback is not removed - a run that never had a dashboard is the
    case it exists for."""
    iface = _Interface(use_dashboard=False, dashboard=None)
    iface._show_context("rolling context text")
    assert iface.printed == ["rolling context text"]


def test_live_dashboard_gets_the_text_and_nothing_is_printed():
    class _Dash:
        def __init__(self):
            self.status = None

        def update_status(self, text):
            self.status = text

        def force_redraw(self):
            pass

    dash = _Dash()
    iface = _Interface(use_dashboard=True, dashboard=dash)
    iface._show_context("rolling context text")
    assert dash.status == "rolling context text"
    assert iface.printed == []


def test_shutdown_silences_even_a_live_dashboard():
    """Once shutdown starts nothing more is emitted anywhere - the dashboard
    may be mid-teardown in the cleanup thread."""
    iface = _Interface(use_dashboard=False, dashboard=None)
    iface.begin_shutdown()
    iface._show_context("rolling context text")
    assert iface.printed == []


def test_begin_shutdown_stops_generation():
    """Belt and braces: the hook bails before generating at all, so nothing
    downstream of it can emit either."""
    calls = []

    class _Gen(TerminalInterface):
        def __init__(self):
            self.generator = object()
            self.interval = 10
            self.last_time = None

        def _is_trigger_passed(self, *a):
            calls.append(a)
            return False

    cb = _Gen()
    lm = SimpleNamespace(
        trainer=SimpleNamespace(accumulate_grad_batches=1, global_step=9999)
    )
    cb._generate_text(lm, batch_idx=0, interval=10)
    assert calls, "sanity: normally it gets as far as the trigger check"

    calls.clear()
    cb.begin_shutdown()
    cb._generate_text(lm, batch_idx=0, interval=10)
    assert calls == []


def test_signal_cleanup_flags_the_interface_before_nulling(monkeypatch):
    """Order matters: flag first, then tear down. Nulling first leaves a
    window where the hook sees no dashboard and falls through to print."""
    order = []

    class _Dash:
        def stop(self):
            order.append("dashboard.stop")

        def __exit__(self, *a):
            order.append("dashboard.exit")

    class _Iface:
        dashboard = _Dash()

        def begin_shutdown(self):
            order.append("begin_shutdown")

    cb = SignalHandlerCallback()
    cb.terminal_interface = _Iface()
    monkeypatch.setattr(
        "praxis.callbacks.lightning.signal_handler.threading.Thread",
        lambda *a, **k: SimpleNamespace(start=lambda: None),
    )
    cb._deferred_cleanup()
    assert order[0] == "begin_shutdown"
    assert "dashboard.stop" in order


# ------------------------------------------------------------------------------
# seq_probe
# ------------------------------------------------------------------------------
# Probe-attribution sequence curriculum (praxis/data/seq_probe.py).
#
# The invariants that matter are the ones the previous controller failed:
#
# - an arm's coefficient must recover its true value from the regression, - an arm with
# no measurable edge must not be handed a confident share, - the fit must track a change
# in which arm is best rather than average over all of history, - the fixed per-tier
# roll must remain the cold-start path.
#
# The controller this replaced (a learning-progress bandit scoring each arm by the loss
# drop between two visits to it) is gone rather than deprecated: that drop measures how
# much the WHOLE model improved in the interval, so it carried no information about arm
# quality - a worthless arm still earned a full share, and sampling an arm more often
# shortened its own interval, making the mechanism negative feedback on visit rate that
# drove the mix to uniform.


TIERS = ((4, 0.01), (2, 0.1))
ARMS = [1, 2, 4]


@pytest.fixture(autouse=True)
def _clean():
    SequenceProbe.reset()
    yield
    SequenceProbe.reset()


def feed(values, windows=400, noise=5.0, seed=0, max_visits=40):
    """Run windows whose probe delta is a linear function of the arm counts."""
    rng = random.Random(seed)
    for _ in range(windows):
        visits = {m: rng.randint(0, max_visits) for m in ARMS}
        delta = sum(values[m] * c for m, c in visits.items()) + rng.gauss(0.0, noise)
        SequenceProbe.observe_window(visits, delta)
    return dict(zip(SequenceProbe.arms, SequenceProbe._beta))


def test_warmup_stays_short_enough_to_be_visible():
    """A guard on the constants, not the code: the cards have to arrive early
    enough in a run that their absence is not mistaken for a missing feature."""
    from praxis.callbacks.lightning.seq_probe import SequenceProbeCallback as cb

    assert cb.first_report_step() <= 64, cb.first_report_step()
    assert cb.first_mix_step() <= 320, cb.first_mix_step()


def test_advertised_arrival_matches_actual_arrival():
    """The printed estimate has to be the truth. The first window only anchors
    the probe's loss level - it produces no delta to regress - so an estimate
    that forgets it is off by a whole window, which is how a working feature
    gets reported as broken."""
    from types import SimpleNamespace

    import torch

    from praxis.callbacks.lightning.seq_probe import SequenceProbeCallback

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Linear(2, 2)
            self.loss = 3.0

        def forward(self, input_ids=None, labels=None, **kw):
            self.loss *= 0.98
            return SimpleNamespace(loss=torch.tensor(self.loss))

    module = SimpleNamespace(model=Inner(), device="cpu", outputs_are_aligned=False)
    cb = SequenceProbeCallback(block_size=8, sequence_multiplier_tiers=TIERS)
    cb._probe = [(1, torch.zeros(2, 8, dtype=torch.long))]
    SequenceProbe.enable(8, TIERS)

    batch = torch.zeros(2, 8, dtype=torch.long)
    first_report = first_mix = None
    for step in range(1, cb.first_mix_step() + 2):
        cb.on_train_batch_start(None, module, batch, step)
        cb.on_before_optimizer_step(None, module, None)
        m = SequenceProbe.metrics()
        if m and first_report is None:
            first_report = step
        if any(k.startswith("seq_prob_x") for k in m) and first_mix is None:
            first_mix = step

    assert first_report == cb.first_report_step(), (
        first_report,
        cb.first_report_step(),
    )
    assert first_mix == cb.first_mix_step(), (first_mix, cb.first_mix_step())


def test_window_ramps_so_the_first_window_is_short():
    from praxis.callbacks.lightning.seq_probe import SequenceProbeCallback as cb

    lengths = cb.window_lengths(5)
    assert lengths[0] == cb.warmup_window
    assert lengths[-1] == cb.window
    assert lengths == sorted(lengths)  # monotone ramp, never a shrink


def test_dynamics_extractor_surfaces_the_card_keys():
    """The seq_mix card pattern-matches ^seq_prob_x\\d+$ off the dynamics
    payload, so the extractor is the contract that matters."""
    from praxis.callbacks.lightning.dynamics import DynamicsLoggerCallback

    extract = DynamicsLoggerCallback._extract_seq_curriculum_dynamics
    assert extract(object()) == {}  # disarmed: no keys, no card

    SequenceProbe.enable(64, TIERS)
    feed({1: 1.0, 2: 3.0, 4: 0.0}, windows=50)
    payload = extract(object())
    assert [k for k in payload if k.startswith("seq_prob_x")], sorted(payload)


def test_callback_disarms_loudly_without_validation_data(capsys):
    """A silently inert controller looks exactly like a missing card."""
    from types import SimpleNamespace

    from praxis.callbacks.lightning.seq_probe import SequenceProbeCallback

    cb = SequenceProbeCallback(block_size=64, sequence_multiplier_tiers=TIERS)
    trainer = SimpleNamespace(datamodule=SimpleNamespace(val_datasets=False))
    cb.on_train_start(trainer, SimpleNamespace(device="cpu"))
    out = capsys.readouterr().out
    assert "DISARMED" in out
    assert "will" in out and "not appear" in out  # names the consequence
    assert cb._failed
    assert not SequenceProbe.enabled


# ------------------------------------------------------------------------------
# decode_compile
# ------------------------------------------------------------------------------
# Decode-time compilation of NeuralMemory: plumbing, scoping, and fallback.
#
# The measured payoff (1.7x on a 128-byte generation, byte-identical output) needs a GPU
# and several minutes of Inductor, so it is not asserted here. What IS asserted is
# everything that could silently break it or, worse, leak a compiled body into training:
# installation, dispatch, restoration, the ``no_compile`` gate, and degrading to eager
# when compilation raises.
#
# ``torch.compile`` is stubbed throughout - compiling for real would make this test
# minutes long and would test Inductor rather than this wiring.


def build_memory_model(**overrides):
    torch.manual_seed(0)
    cfg = PraxisConfig(
        vocab_size=200,
        hidden_size=64,
        embed_size=64,
        depth=2,
        num_layers=2,
        num_heads=4,
        device="cpu",
        block_type="transformer",
        max_position_embeddings=256,
        attention_type="causal",
        encoding="rope",
        memory_type="mal_energy",
        **overrides,
    )
    return PraxisForCausalLM(cfg).eval()


def neural_memories(model):
    return [m for m in model.modules() if isinstance(m, NeuralMemory)]


@pytest.fixture
def feature_on():
    """The decode compile is opt-in per environment; most tests want it on."""
    EnvironmentFeatures.set_from_environment({"compile_decode_memory": True})
    try:
        yield
    finally:
        EnvironmentFeatures.clear()


@pytest.fixture
def stub_compile(monkeypatch):
    """Replace torch.compile with a counting pass-through."""
    calls = {"n": 0}

    def fake(fn, **kwargs):
        calls["n"] += 1

        def wrapper(*args, **kw):
            calls.setdefault("invoked", 0)
            calls["invoked"] += 1
            return fn(*args, **kw)

        return wrapper

    monkeypatch.setattr(torch, "compile", fake)
    return calls


def test_queue_callback_warms_the_backend(feature_on, stub_compile):
    from praxis.callbacks.lightning.generation_queue import GenerationQueueCallback

    class FakeGenerator:
        def __init__(self, backend):
            self.backend = backend

    model = build_memory_model()
    backend = ModelBackend(model, tokenizer=None)
    GenerationQueueCallback(FakeGenerator(backend)).on_train_start(None, None)
    assert stub_compile["n"] == len(neural_memories(model))


def test_queue_callback_tolerates_a_backend_without_warmup():
    from praxis.callbacks.lightning.generation_queue import GenerationQueueCallback

    class Bare:
        backend = object()

    GenerationQueueCallback(Bare()).on_train_start(None, None)


# ------------------------------------------------------------------------------
# snapshot_pump
# ------------------------------------------------------------------------------
# Model-touching snapshots must run on the training thread, never beside it.
#
# The producer's own thread running a torch op on the live model is what wedged
# abstractinator-s: an ABBA deadlock on (GIL, AutogradMeta.mutex_) that not even the
# stall watchdog could report. These tests pin the structural rule that replaced it,
# since the deadlock itself is timing-dependent and cannot be reproduced reliably in a
# unit test.


def _producer(recipes, tick=0.01):
    return SnapshotProducer(
        store=SnapshotStore(),
        model_fn=lambda: "MODEL",
        shutdown_event=threading.Event(),
        recipes=recipes,
        tick=tick,
    )


def _recorder():
    """A recipe that records the thread that ran it."""
    seen = []
    return (
        seen,
        lambda model: (seen.append(threading.current_thread().name), {"n": len(seen)})[
            1
        ],
    )


def test_callback_attaches_and_pumps():
    seen, recipe = _recorder()
    p = _producer({"probe": Recipe(recipe, 0.0)})
    cb = SnapshotPumpCallback(p)

    cb.on_fit_start(None, None)
    assert p._pumped is True

    cb.on_train_batch_end(None, None, None, None, 0)
    assert len(seen) == 1

    cb.on_fit_end(None, None)
    assert p._pumped is False


def test_callback_survives_a_broken_producer():
    class Broken:
        def attach_pump(self):
            pass

        def pump(self):
            raise RuntimeError("nope")

    cb = SnapshotPumpCallback(Broken())
    cb.on_train_batch_end(None, None, None, None, 0)  # must not raise


# ------------------------------------------------------------------------------
# stall_watchdog
# ------------------------------------------------------------------------------
# The wedge detector has to work when nothing else does, so test that it actually writes
# stacks rather than just that it constructs.


class _Trainer_stall_watchdog(SimpleNamespace):
    is_global_zero = True
    global_step = 7


def test_dumps_stacks_when_a_step_overruns(tmp_path):
    wd = StallWatchdogCallback(run_dir=tmp_path, timeout_s=0.2)
    trainer = _Trainer_stall_watchdog()
    wd.on_fit_start(trainer, None)
    wd.on_train_batch_start(trainer, None, None, 0)
    time.sleep(0.6)  # overrun: the C timer thread fires while we sit here
    wd.on_train_batch_end(trainer, None, None, None, 0)
    wd.on_train_end(trainer, None)

    log = (tmp_path / "stalls.log").read_text()
    assert "watchdog armed" in log
    assert "Thread" in log or "File " in log, log  # a real traceback landed
    assert "step 7 took" in log


def test_priority_dump_lands_without_the_training_thread(tmp_path):
    """The case that matters: the step never ends, so no Lightning hook runs
    again. The watch thread has to notice and dump the MAIN thread by itself -
    faulthandler alone caps at 100 threads and drops main off the end."""
    wd = StallWatchdogCallback(run_dir=tmp_path, timeout_s=0.2)
    wd.POLL_S = 0.05
    trainer = _Trainer_stall_watchdog()
    wd.on_fit_start(trainer, None)
    wd.on_train_batch_start(trainer, None, None, 0)
    time.sleep(1.0)  # never call another hook: this is the deadlock shape
    log = (tmp_path / "stalls.log").read_text()
    wd.on_train_end(trainer, None)

    assert "priority dump" in log, log
    assert "[MAIN]" in log, log
    assert "has been running" in log
    # And it dumps once per stuck step, not once per poll.
    assert log.count("priority dump (") == 1


def test_a_normal_step_dumps_nothing(tmp_path):
    wd = StallWatchdogCallback(run_dir=tmp_path, timeout_s=30.0)
    trainer = _Trainer_stall_watchdog()
    wd.on_fit_start(trainer, None)
    for i in range(3):
        wd.on_train_batch_start(trainer, None, None, i)
        wd.on_train_batch_end(trainer, None, None, None, i)
    wd.on_train_end(trainer, None)

    log = (tmp_path / "stalls.log").read_text()
    assert "watchdog armed" in log
    assert "Traceback" not in log and "took" not in log


# ------------------------------------------------------------------------------
# terminal_warmup
# ------------------------------------------------------------------------------
# Terminal inference must not re-enter warmup during validation.


class _NoGen(TerminalInterface):
    def __init__(self):  # bypass full init; only warmup math is exercised
        self.generator = object()
        self.interval = 10
        self.last_time = None
        self.captured = None

    def _is_trigger_passed(self, last_time, interval):
        self.captured = interval
        return False  # stop after recording the interval


def test_validation_batches_keep_trained_interval():
    cb = _NoGen()
    lm = SimpleNamespace(
        trainer=SimpleNamespace(accumulate_grad_batches=1, global_step=6127)
    )
    cb._generate_text(lm, batch_idx=0, interval=10)  # validation: batch_idx resets
    assert cb.captured == cb.interval  # past warmup; no 120s stall


def test_fresh_run_still_warms_up():
    cb = _NoGen()
    lm = SimpleNamespace(
        trainer=SimpleNamespace(accumulate_grad_batches=1, global_step=0)
    )
    cb._generate_text(lm, batch_idx=0, interval=10)
    assert cb.captured > cb.interval  # step 0 starts at the slow end


# ------------------------------------------------------------------------------
# dashboard_shutdown
# ------------------------------------------------------------------------------
# Stopping the dashboard must leave the terminal alone.
#
# The reported failure: Ctrl+C during a run, and the dashboard's box drawing and charts
# render *into* the shell's scrollback, interleaved with the shutdown's own messages.
# The cause was an ordering bug, not a rendering one. ``stop()`` flipped a flag and
# immediately left the alternate screen, while the render thread was still mid-frame or
# asleep in its 100ms tick - and that thread writes through a private handle on the real
# stdout, bypassing every redirection. The frame it painted next landed, absolutely
# positioned, on the restored terminal.


class _Tty(io.StringIO):
    """A stand-in terminal that records everything written to it."""

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.chunks = []

    def write(self, s):
        with self.lock:
            self.chunks.append(s)
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return True

    @property
    def text(self):
        with self.lock:
            return "".join(self.chunks)


# ── the force-exit paths bypass every cleanup hook ───────────────────────


def test_release_terminal_leaves_the_alternate_screen(dashboard):
    """os._exit runs no atexit handler, so these paths restore inline."""
    from praxis.callbacks.lightning.signal_handler import SignalHandlerCallback

    cb = SignalHandlerCallback()
    cb.terminal_interface = SimpleNamespace(dashboard=dashboard)

    dashboard.start()
    time.sleep(0.15)
    assert dashboard.terminal_manager.in_fullscreen

    real_stderr = _Tty()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "__stderr__", real_stderr)
        cb._release_terminal()

    assert "\033[?1049l".encode().decode("unicode_escape") in real_stderr.text
    assert "\033[?25h".encode().decode("unicode_escape") in real_stderr.text
    assert not dashboard.running
    assert not dashboard.dashboard_output.enabled


def test_release_terminal_does_not_emit_an_unmatched_rmcup(dashboard):
    """An rmcup we never matched with smcup jumps the cursor into scrollback."""
    from praxis.callbacks.lightning.signal_handler import SignalHandlerCallback

    cb = SignalHandlerCallback()
    cb.terminal_interface = SimpleNamespace(dashboard=dashboard)  # never started

    real_stderr = _Tty()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "__stderr__", real_stderr)
        cb._release_terminal()

    assert "\033[?1049l".encode().decode("unicode_escape") not in real_stderr.text
    assert "\033[?25h".encode().decode("unicode_escape") in real_stderr.text


# ------------------------------------------------------------------------------
# engagement
# ------------------------------------------------------------------------------
# Tests for the engagement-prediction reward (P2) and policy (P3).


class TestLiveDrainCallback:
    """The training-loop seam: live web rewards -> policy energy baseline."""

    def _setup(self, period=1):
        import types

        from praxis.callbacks.lightning import EngagementLiveRewardCallback
        from praxis.policies.engagement_channel import LIVE_ENGAGEMENT

        LIVE_ENGAGEMENT.drain()  # start clean
        policy = EngagementPolicy(PraxisConfig(hidden_size=32, dropout=0.0))
        pl = types.SimpleNamespace(model=types.SimpleNamespace(policy=policy))
        trainer = types.SimpleNamespace(callback_metrics={})
        cb = EngagementLiveRewardCallback(period=period)
        return cb, trainer, pl, policy, LIVE_ENGAGEMENT

    def test_drain_folds_live_reward_into_energy(self):
        cb, trainer, pl, policy, channel = self._setup(period=1)
        assert policy.energy.value == 0.0
        channel.submit(["paris"], ["i", "think", "paris"])  # activation 1.0
        cb.on_train_batch_end(trainer, pl, None, None, 0)
        assert policy.energy.value > 0.0
        assert trainer.callback_metrics["engagement_live_count"].item() == 1.0
        assert channel.snapshot()["buffered"] == 0  # drained

    def test_respects_period(self):
        cb, trainer, pl, policy, channel = self._setup(period=3)
        channel.submit(["paris"], ["paris"])
        cb.on_train_batch_end(trainer, pl, None, None, 0)  # step 1: no drain
        assert policy.energy.value == 0.0
        cb.on_train_batch_end(trainer, pl, None, None, 1)  # step 2: no drain
        cb.on_train_batch_end(trainer, pl, None, None, 2)  # step 3: drains
        assert policy.energy.value > 0.0


# ------------------------------------------------------------------------------
# trainers
# ------------------------------------------------------------------------------
# Tests for the trainers module.


class TestRLCTProbeUnpacksTheBatch:
    """The RLCT callback re-uses the trainer's batch unpacking.

    `on_train_batch_end` swallows probe exceptions and prints them, so a break
    here is non-blocking and invisible to every other test - which is how an
    UnboundLocalError for `rewards` survived a whole training run. `_probe`
    does its unpacking OUTSIDE any try, so calling it directly makes that
    class of failure loud.
    """

    def test_probe_unpacks_without_unbound_names(self):
        from praxis.callbacks.lightning.rlct import RLCTLandscapeCallback

        config = PraxisConfig(
            depth=2,
            hidden_size=64,
            embed_size=32,
            vocab_size=256,
            num_heads=2,
            num_queries=2,
            device_map="cpu",
        )
        model = PraxisForCausalLM(config)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        pl_module = BackpropagationTrainer(
            model, optimizer, None, {"batch_size": 4}, None, byte_level=True
        )

        batch = {
            "input_ids": torch.randint(0, 256, (4, 16)),
            "task_type_ids": torch.zeros(4, 16, dtype=torch.uint8),
            "assistant_mask": torch.ones(4, 16, dtype=torch.uint8),
            "block_ids": torch.ones(4, 16, dtype=torch.long),
        }

        callback = RLCTLandscapeCallback(
            {"probe_seqs": 2, "probe_len": 8, "manifold_grid": 2, "field_grid": 2}
        )
        # Any NameError/UnboundLocalError in the unpacking propagates from here.
        callback._probe(pl_module, batch, 0, 0)

    def test_probe_forwards_block_ids(self):
        """block_ids must survive sub-batching, or the probe measures a model
        whose packed documents can read each other while the real step's cannot.
        """
        from praxis.callbacks.lightning import rlct

        captured = {}
        config = PraxisConfig(
            depth=2,
            hidden_size=64,
            embed_size=32,
            vocab_size=256,
            num_heads=2,
            num_queries=2,
            device_map="cpu",
        )
        model = PraxisForCausalLM(config)
        original = model.forward

        def spy(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)

        model.forward = spy
        pl_module = BackpropagationTrainer(
            model,
            torch.optim.SGD(model.parameters(), lr=1e-4),
            None,
            {"batch_size": 4},
            None,
            byte_level=True,
        )
        batch = {
            "input_ids": torch.randint(0, 256, (4, 16)),
            "block_ids": torch.ones(4, 16, dtype=torch.long),
        }
        rlct.RLCTLandscapeCallback(
            {"probe_seqs": 2, "probe_len": 8, "manifold_grid": 2, "field_grid": 2}
        )._probe(pl_module, batch, 0, 0)

        assert captured.get("block_ids") is not None
        assert captured["block_ids"].shape == captured["input_ids"].shape


# ------------------------------------------------------------------------------
# stdout_safety
# ------------------------------------------------------------------------------
# Nothing on a background thread may swap the process-global ``sys.stdout``.
#
# ``contextlib.redirect_stdout`` mutates a PROCESS-GLOBAL. Used from the Flask API
# thread or a build thread, it silently redirects every other thread's output for the
# width of the block, and any thread that reads ``sys.stdout`` before the block ends and
# writes to it after gets ``ValueError: I/O operation on closed file``.
#
# That killed abstractinator-m at its first step: the snapshot publisher requested the
# spec payload (which printed the model repr under a redirect) at the same moment the
# compute profiler flushed stdout on the training thread. The profiler's own error
# handler then used ``print``, failed identically, and escaped its ``except`` - turning
# optional telemetry into a fatal error.


def test_profiler_callback_logger_never_raises():
    """Its whole job is to report failures, so it must not become one."""
    from praxis.callbacks.lightning.compute_profiler import _log_quietly

    original = sys.stdout
    closed = io.StringIO()
    closed.close()
    sys.stdout = closed
    try:
        _log_quietly("this must not raise")
    finally:
        sys.stdout = original
