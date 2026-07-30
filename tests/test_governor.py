"""GNS batch governor: estimator math, tier control, Lightning wiring."""

import math
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from praxis.callbacks.lightning.governor import GNSBatchGovernor
from praxis.data.batch_schedule import BatchSchedule, plan_cycle
from praxis.governors import GOVERNOR_REGISTRY
from praxis.governors.gns import BatchTierController, GradientNoiseEstimator


@pytest.fixture(autouse=True)
def _clean_schedule():
    """The schedule is class-level state a governor writes on construction."""
    BatchSchedule.reset()
    yield
    BatchSchedule.reset()


# ── estimator ────────────────────────────────────────────────────────────


def test_estimator_recovers_known_noise_scale():
    """Feed squared norms of synthetic small/big-batch mean gradients whose
    true B_noise = tr(cov)/|mu|^2 is known; the EMA ratio must land near it."""
    torch.manual_seed(0)
    d, sigma, b_small, b_big = 512, 0.5, 4, 64
    mu = torch.randn(d) / math.sqrt(d)  # |mu|^2 ~ 1
    true_noise = d * sigma**2 / float(mu.pow(2).sum())
    est = GradientNoiseEstimator()
    for _ in range(600):
        g_small = mu + sigma * torch.randn(d) / math.sqrt(b_small)
        g_big = mu + sigma * torch.randn(d) / math.sqrt(b_big)
        est.update(
            small_sq=float(g_small.pow(2).sum()),
            big_sq=float(g_big.pow(2).sum()),
            b_small=b_small,
            b_big=b_big,
        )
    measured = est.noise_scale()
    assert measured is not None
    assert abs(measured - true_noise) / true_noise < 0.25


def test_estimator_not_ready_early_and_rejects_negative_signal():
    est = GradientNoiseEstimator()
    assert est.noise_scale() is None
    # small_sq >> big_sq at these batch sizes drives the |G|^2 estimate
    # negative: (2*1 - 1*10)/1 = -8. Must refuse to report a scale.
    for _ in range(est.min_updates):
        est.update(small_sq=10.0, big_sq=1.0, b_small=1, b_big=2)
    assert est.ready
    assert est.noise_scale() is None


def test_estimator_single_point_is_a_noop():
    est = GradientNoiseEstimator()
    est.update(small_sq=1.0, big_sq=1.0, b_small=16, b_big=16)
    assert est._updates == 0


# ── tier controller ──────────────────────────────────────────────────────


def test_controller_moves_one_tier_with_deadband_and_clamps():
    ctl = BatchTierController(max_rows=512)
    # Far above: one doubling per decision, never a jump.
    assert ctl.desired_rows(32, 1584.0) == 64
    # Inside the deadband (log2(70/64) ~ 0.13): hold.
    assert ctl.desired_rows(64, 70.0) == 64
    # Far below: halve.
    assert ctl.desired_rows(64, 10.0) == 32
    # Ceiling and floor.
    assert ctl.desired_rows(512, 1e9) == 512
    assert ctl.desired_rows(2, 0.001) == 2
    # No measurement: hold.
    assert ctl.desired_rows(128, None) == 128


def test_controller_floor_is_two_rows_not_a_microbatch():
    """The floor exists to keep the two-point estimator alive (one row per
    microbatch at the two-microbatch minimum), NOT to respect batch_size -
    which is a per-microbatch ceiling and must not bound the batch below."""
    ctl = BatchTierController(max_rows=512)
    assert ctl.min_rows == 2
    rows = 512
    for _ in range(20):  # a sustained tiny noise scale must descend all the way
        rows = ctl.desired_rows(rows, 1.0)
    assert rows == 2


def test_controller_regulates_against_delivered_rows():
    """When a long-sequence cycle's attention budget caps rows below the
    target, the decision must use what ran - not what was asked for."""
    ctl = BatchTierController(max_rows=512)
    # Target 128, but only 32 rows actually landed; B_noise of 100 is ABOVE
    # what ran, so the honest move is up even though 100 < 128.
    assert ctl.desired_rows(128, 100.0, measured=32.0) == 256
    # Same noise judged against the requested target would have held.
    assert ctl.desired_rows(128, 100.0) == 128


def test_controller_hysteresis_prevents_immediate_flap():
    """Moving a tier shifts the reference an octave, so net hysteresis is
    2*deadband - 1. A measurement that just triggered an up-move must not be
    able to trigger the down-move from the new tier (the 32<->64 flapping
    observed in abstractinator-f under the old 0.75 deadband)."""
    ctl = BatchTierController(max_rows=512)
    # Up-threshold of the 32-row tier is the next tier (64 rows).
    assert ctl.desired_rows(32, 63.0) == 32
    assert ctl.desired_rows(32, 65.0) == 64
    # The same measurement holds at 64 rows...
    assert ctl.desired_rows(64, 65.0) == 64
    # ...as does anything above 64's down-threshold (32 rows).
    assert ctl.desired_rows(64, 33.0) == 64
    assert ctl.desired_rows(64, 31.0) == 32


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


def test_registry_builds_callback_with_ceiling():
    gov = GOVERNOR_REGISTRY["gns_batch"](
        batch_size=16, target_batch_size=512, val_every=1024
    )
    assert isinstance(gov, GNSBatchGovernor)
    assert gov.controller.max_rows == 512  # target_batch_size, in rows
    assert gov.controller.min_rows == 2  # NOT batch_size
    assert gov.row_ceiling == 16  # batch_size, a per-microbatch ceiling
    assert gov.val_every == 1024


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
    assert gov._pending == 64  # rows: 32 -> 64, which needs accum 4
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
    gov.on_train_batch_start(
        trainer, module, torch.zeros(16, 8, dtype=torch.long), 0
    )
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


def test_validation_target_matches_abstractinator_f_resume():
    """Replay the real resume shape from the 2026-07-28 run (checkpoint at
    global_step 6656, raw batch 47104, factor 8, val_every 1024): the target
    must point at step 7168's raw batch, 47104 + 512*8 = 51200."""
    gov = GNSBatchGovernor(batch_size=16, target_batch_size=512, val_every=1024)
    # Legacy checkpoint key: an accumulation factor against the old fixed
    # microbatch, i.e. 8 x 16 = 128 rows -> accum 8 at ceiling 16.
    gov.load_state_dict({"factor": 8, "steps": 6656, "estimator": {}})
    trainer = _fake_trainer()
    trainer.global_step = 6656
    trainer.fit_loop.epoch_loop.batch_progress.current.ready = 47104
    module = nn.Linear(4, 4)
    gov.on_train_start(trainer, module)
    assert trainer.accumulate_grad_batches == 8

    _run_cycle(gov, trainer, module, k=8)  # global_step 6657
    assert trainer.val_check_batch == 51200


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


# ── batch schedule: the factorization ────────────────────────────────────


def test_batch_size_is_a_ceiling_not_a_floor():
    """The ask that motivated the rework. With batch_size=64 the governor must
    still be able to run an 8-row step; the old wiring floored it at 128."""
    ceiling = 64
    for rows in (2, 4, 8, 16, 32, 64):
        plan = plan_cycle(rows, ceiling)
        assert plan.delivered_rows == rows, rows
        assert plan.micro_rows <= ceiling
        assert plan.accum >= 2


def test_microbatch_fills_the_ceiling_at_large_batches():
    """The other half: a large effective batch must not underuse the GPU. Rows
    per microbatch sit AT the ceiling once there are enough rows to fill it."""
    ceiling = 64
    for rows in (128, 256, 512, 1024):
        plan = plan_cycle(rows, ceiling)
        assert plan.micro_rows == ceiling, rows
        assert plan.delivered_rows == rows


def test_plan_is_exact_and_respects_the_estimator_minimum():
    for rows in (2, 8, 64, 128, 512):
        for ceiling in (1, 8, 16, 64, 256):
            plan = plan_cycle(rows, ceiling)
            assert plan.micro_rows >= 1
            assert plan.micro_rows <= max(1, ceiling)
            assert plan.accum >= 2  # two points for the estimator, always
            assert plan.micro_rows * plan.accum == plan.delivered_rows


def test_accum_ignores_the_sequence_multiplier():
    """Lightning steps at ready % accum == 0. An accum that moved with the
    curriculum's dice would land mid-cycle, so it must be a function of the
    governed rows and the ceiling alone."""
    base = plan_cycle(256, 64).accum
    for m in (1, 2, 4, 8):
        assert plan_cycle(256, 64, m).accum == base


def test_multiplier_holds_attention_cost_constant():
    """rows ~ 1/m^2 against length ~ m keeps rows*len^2 invariant. The nominal
    row count handed to get_batch (which re-applies the m^2 division) must
    round-trip to the planned microbatch rows."""
    costs = set()
    for m in (1, 2, 4):
        plan = plan_cycle(256, 64, m)
        costs.add(plan.micro_rows * m * m)
        assert plan.nominal_rows // (m * m) == plan.micro_rows
    assert len(costs) == 1


def test_schedule_holds_the_multiplier_across_a_cycle():
    """Lightning divides every microbatch loss by accum uniformly, which only
    averages correctly when the microbatches are the same size - so the
    multiplier is rolled once per cycle, not once per microbatch."""
    # Force the 2x tier every roll so the assertion exercises a real change.
    BatchSchedule.enable(row_ceiling=64, effective_rows=256, tiers=((2, 1.0),))
    accum = BatchSchedule.accum()
    assert accum == 4
    plans = [BatchSchedule.next_microbatch() for _ in range(2 * accum)]
    first, second = plans[:accum], plans[accum:]
    assert len({(p.micro_rows, p.multiplier) for p in first}) == 1
    assert len({(p.micro_rows, p.multiplier) for p in second}) == 1
    assert all(p.multiplier == 2 for p in plans)


def test_schedule_retarget_takes_effect_at_the_next_cycle():
    BatchSchedule.enable(row_ceiling=64, effective_rows=128, tiers=())
    first = BatchSchedule.next_microbatch()
    assert first.micro_rows == 64 and first.accum == 2
    BatchSchedule.set_effective_rows(8)
    # Mid-cycle: the open cycle keeps its shape.
    assert BatchSchedule.next_microbatch().micro_rows == 64
    # Next cycle picks up the new target.
    assert BatchSchedule.next_microbatch().micro_rows == 4


def test_long_sequences_are_only_drawn_when_the_step_can_afford_them():
    """Eligibility is budgeted against the governed rows, not the fixed
    ceiling: a 4-row step has nothing to trade for a 4x length."""
    from praxis.data.datasets.manager import SEQUENCE_MULTIPLIER_TIERS

    def eligible(rows):
        base = plan_cycle(rows, 64).micro_rows
        return {m for m, _ in SEQUENCE_MULTIPLIER_TIERS if base >= m * m}

    assert eligible(4) == set()  # base_rows 2: m=1 only
    assert eligible(8) == {2}
    assert eligible(128) == {2, 4, 8}


def test_schedule_is_inert_until_enabled():
    """A run without a governor must keep its static batch behaviour."""
    assert BatchSchedule.next_microbatch() is None
    assert BatchSchedule.current() is None
    assert BatchSchedule.metrics() == {}


def test_dataset_falls_back_to_static_shape_without_a_governor():
    from praxis.data.datasets.weighted import WeightedIterableDataset

    ds = object.__new__(WeightedIterableDataset)
    ds.batch_size = 16
    ds.sequence_multiplier_tiers = ()
    ds.governed = True  # governed, but no governor has enabled the schedule
    assert ds._next_shape() == (16, 1)


def test_dataset_reads_the_governed_shape():
    from praxis.data.datasets.weighted import WeightedIterableDataset

    ds = object.__new__(WeightedIterableDataset)
    ds.batch_size = 64  # now only a ceiling; the schedule decides
    ds.sequence_multiplier_tiers = ()
    ds.governed = True
    BatchSchedule.enable(row_ceiling=64, effective_rows=8, tiers=())
    nominal, multiplier = ds._next_shape()
    assert (nominal, multiplier) == (4, 1)  # 8 rows over 2 microbatches


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


def test_terminal_reports_governed_microbatch_rows():
    """batch_size is only a ceiling now, so the info panel must read the live
    plan or it reports a batch the run never used."""
    from praxis.callbacks.lightning.terminal import TerminalInterface

    trainer = SimpleNamespace(accumulate_grad_batches=2, world_size=1)
    BatchSchedule.enable(row_ceiling=64, effective_rows=8, tiers=())
    BatchSchedule.next_microbatch()  # open a cycle: 2 x 4 rows
    assert TerminalInterface._effective_batch(trainer, 64) == 8


# ── dynamic sequence lengths under the governor ──────────────────────────


def _arm_probe_toward(best, tiers, windows=120):
    """Fit the probe curriculum so ``best`` is the clear winner."""
    from praxis.data.seq_probe import SequenceProbe

    SequenceProbe.reset()
    SequenceProbe.enable(64, tiers)
    values = {m: (3.0 if m == best else 0.0) for m in SequenceProbe.arms}
    rng = random.Random(0)
    for _ in range(windows):
        visits = {m: rng.randint(0, 40) for m in SequenceProbe.arms}
        delta = sum(values[m] * c for m, c in visits.items()) + rng.gauss(0.0, 5.0)
        SequenceProbe.observe_window(visits, delta)
    return SequenceProbe


def test_learned_curriculum_still_drives_the_multiplier():
    """The governor rolls the multiplier now, but the roll still goes through the
    curriculum controller - its fitted distribution must reach the batches."""
    from praxis.data.datasets.manager import SEQUENCE_MULTIPLIER_TIERS

    probe = _arm_probe_toward(4, SEQUENCE_MULTIPLIER_TIERS)
    try:
        assert probe.shared_probs[4] > 0.5

        BatchSchedule.enable(
            row_ceiling=64, effective_rows=512, tiers=SEQUENCE_MULTIPLIER_TIERS
        )
        rng = random.Random(1)
        drawn = [BatchSchedule.next_microbatch(rng).multiplier for _ in range(400)]
        # The fitted preference shows up in what the pipeline actually builds,
        # not just in the controller's table.
        assert drawn.count(4) / len(drawn) > 0.5
        assert len(set(drawn)) > 1  # the explore floor keeps other arms alive
    finally:
        probe.reset()


def test_multiplier_arms_narrow_as_the_batch_descends():
    """Documented consequence of budgeting eligibility against governed rows: a
    small step cannot afford long sequences under the constant-attention rule,
    so the reachable lengths shrink with the batch. The bandit keeps its
    estimates - only what is drawn changes.

    Armed with the probe curriculum, whose uniform explore floor gives every
    ELIGIBLE arm a real share; the raw tier chances (1%, 0.1%) are too thin to
    read an eligibility set off a finite sample."""
    from praxis.data.datasets.manager import SEQUENCE_MULTIPLIER_TIERS as tiers

    def drawn_lengths(effective_rows, ceiling):
        BatchSchedule.enable(
            row_ceiling=ceiling, effective_rows=effective_rows, tiers=tiers
        )
        rng = random.Random(2)
        return {BatchSchedule.next_microbatch(rng).multiplier for _ in range(300)}

    # No arm is better than any other here, so the fit stays diffuse and every
    # eligible arm keeps a real share - which is what makes the eligibility set
    # readable off a finite sample.
    probe = _arm_probe_toward(None, tiers)
    try:
        assert probe.shared_probs is not None

        assert drawn_lengths(128, 64) == {1, 2, 4, 8}
        assert drawn_lengths(32, 64) == {1, 2, 4}
        assert drawn_lengths(8, 64) == {1, 2}
        # At the floor only the base length is affordable.
        assert drawn_lengths(2, 64) == {1}
    finally:
        probe.reset()


def test_positional_capacity_still_covers_every_drawn_length():
    """max_position_embeddings is sized at config time as block_size *
    max_sequence_multiplier(batch_size). Eligibility is budgeted against
    base_rows, which never exceeds batch_size, so the governed pipeline can
    never draw a length the model has no positions for."""
    from praxis.data.datasets.manager import (
        SEQUENCE_MULTIPLIER_TIERS as tiers,
        max_sequence_multiplier,
    )

    for ceiling in (16, 64, 256):
        sized_for = max_sequence_multiplier(ceiling, tiers)
        rows = 2
        while rows <= 4096:
            base = plan_cycle(rows, ceiling).micro_rows
            assert max_sequence_multiplier(base, tiers) <= sized_for
            rows *= 2


def test_validation_loader_keeps_a_fixed_shape():
    """The val loader is the same dataset class. Following the governed plan
    would make val loss incomparable across steps AND, because the plan counts
    microbatches to find a cycle boundary, shift the training cycle."""
    from praxis.data.datasets.weighted import WeightedIterableDataset

    train = object.__new__(WeightedIterableDataset)
    train.batch_size, train.sequence_multiplier_tiers, train.governed = 64, (), True
    val = object.__new__(WeightedIterableDataset)
    val.batch_size, val.sequence_multiplier_tiers, val.governed = 64, (), False

    BatchSchedule.enable(row_ceiling=64, effective_rows=8, tiers=())
    assert val._next_shape() == (64, 1)  # untouched by the governor
    assert BatchSchedule.current() is None  # and it did not open a cycle
    assert train._next_shape() == (4, 1)  # the governed shape


def test_validation_draws_do_not_shift_the_training_cycle():
    from praxis.data.datasets.weighted import WeightedIterableDataset

    train = object.__new__(WeightedIterableDataset)
    train.batch_size, train.sequence_multiplier_tiers, train.governed = 64, (), True
    val = object.__new__(WeightedIterableDataset)
    val.batch_size, val.sequence_multiplier_tiers, val.governed = 64, (), False

    # accum 4: a cycle is four microbatches.
    BatchSchedule.enable(row_ceiling=64, effective_rows=256, tiers=())
    train._next_shape()
    train._next_shape()
    for _ in range(10):  # a whole validation pass mid-cycle
        val._next_shape()
    assert BatchSchedule._micro_index == 2  # still two into the open cycle
