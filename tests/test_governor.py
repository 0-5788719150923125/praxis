"""GNS batch governor: estimator math, tier control, Lightning wiring."""

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from praxis.callbacks.lightning.governor import GNSBatchGovernor
from praxis.governors import GOVERNOR_REGISTRY
from praxis.governors.gns import BatchTierController, GradientNoiseEstimator


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
    ctl = BatchTierController(micro_batch=16, max_factor=32)
    # Far above: one doubling per decision, never a jump.
    assert ctl.desired_factor(2, 1584.0) == 4
    # Inside the deadband (log2(70/64) ~ 0.13): hold.
    assert ctl.desired_factor(4, 70.0) == 4
    # Far below: halve, floored at 2.
    assert ctl.desired_factor(4, 10.0) == 2
    assert ctl.desired_factor(2, 5.0) == 2
    # Ceiling.
    assert ctl.desired_factor(32, 1e9) == 32
    # No measurement: hold.
    assert ctl.desired_factor(8, None) == 8


def test_controller_hysteresis_prevents_immediate_flap():
    """Moving a tier shifts the reference an octave, so net hysteresis is
    2*deadband - 1. A measurement that just triggered an up-move must not be
    able to trigger the down-move from the new tier (the 32<->64 flapping
    observed in abstractinator-f under the old 0.75 deadband)."""
    ctl = BatchTierController(micro_batch=16, max_factor=32)
    # Up-threshold of tier 2 is the next tier's batch (64 rows).
    assert ctl.desired_factor(2, 63.0) == 2
    assert ctl.desired_factor(2, 65.0) == 4
    # The same measurement holds at tier 4...
    assert ctl.desired_factor(4, 65.0) == 4
    # ...as does anything above tier 4's down-threshold (64/2 = 32 rows).
    assert ctl.desired_factor(4, 33.0) == 4
    assert ctl.desired_factor(4, 31.0) == 2


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
    assert gov.controller.max_factor == 32
    assert gov.controller.min_factor == 2
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


def _run_cycle(gov, trainer, module, k):
    """Simulate one full accumulation cycle of k microbatches, mirroring
    Lightning's ordering: backward -> (step + global_step bump) -> batch_end
    with the within-epoch batch index."""
    cur = trainer.fit_loop.epoch_loop.batch_progress.current
    for i in range(k):
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
    assert trainer.accumulate_grad_batches == 2  # starts at the floor

    k = 2
    _set_grads(module, 0.6 / k)  # Lightning-scaled first microbatch
    gov.on_after_backward(trainer, module)
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
    assert gov._pending == 4
    assert trainer.accumulate_grad_batches == 2  # deferred: 2 % 4 != 0

    _run_cycle(gov, trainer, module, k=2)  # boundary at ready=4 commits
    assert trainer.accumulate_grad_batches == 4
    assert gov._pending is None


def test_irregular_cycle_is_not_measured():
    """A cycle whose microbatch count != K (epoch-end early step) must not
    feed the estimator - its loss scaling doesn't match."""
    gov = GNSBatchGovernor(batch_size=16, target_batch_size=512)
    trainer = _fake_trainer()
    module = nn.Linear(4, 4)
    gov.on_train_start(trainer, module)
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
    must point at step 7168's raw batch, 47104 + 512*8 = 51200, and the
    telemetry stash must expose both sides of the cadence pair."""
    gov = GNSBatchGovernor(batch_size=16, target_batch_size=512, val_every=1024)
    gov.load_state_dict({"factor": 8, "steps": 6656, "estimator": {}})
    trainer = _fake_trainer()
    trainer.global_step = 6656
    trainer.fit_loop.epoch_loop.batch_progress.current.ready = 47104
    module = nn.Linear(4, 4)
    gov.on_train_start(trainer, module)
    assert trainer.accumulate_grad_batches == 8

    _run_cycle(gov, trainer, module, k=8)  # global_step 6657
    assert trainer.val_check_batch == 51200

    stash = module._governor_metrics
    # Stash refreshes on the decision cadence; force one to check the keys.
    gov._steps = gov.decide_every - 1
    for _ in range(gov.estimator.min_updates):
        gov.estimator.update(small_sq=10.0, big_sq=6.0, b_small=16, b_big=128)
    _run_cycle(gov, trainer, module, k=8)  # decision fires, stash updates
    stash = module._governor_metrics
    assert stash["gov_next_val_batch"] == 51200.0
    assert stash["gov_raw_batches"] == 47120.0  # 47104 + 2 cycles of 8


def test_validation_cadence_disabled_without_val_every():
    gov = GNSBatchGovernor(batch_size=16, target_batch_size=512)
    trainer = _fake_trainer()
    trainer.val_check_batch = "untouched"
    module = nn.Linear(4, 4)
    gov.on_train_start(trainer, module)
    for _ in range(3):
        _run_cycle(gov, trainer, module, k=2)
    assert trainer.val_check_batch == "untouched"


def test_state_dict_roundtrip_restores_factor():
    src = GNSBatchGovernor(batch_size=16, target_batch_size=512)
    src._factor = 8
    src._steps = 33
    src.estimator.update(small_sq=10.0, big_sq=6.0, b_small=16, b_big=32)
    dst = GNSBatchGovernor(batch_size=16, target_batch_size=512)
    dst.load_state_dict(src.state_dict())
    trainer = _fake_trainer()
    dst.on_train_start(trainer, nn.Linear(2, 2))
    assert trainer.accumulate_grad_batches == 8
    assert dst._steps == 33
    assert dst.estimator.state_dict()["updates"] == 1


def test_metrics_stash_and_descriptions_fold():
    gov = GNSBatchGovernor(batch_size=16, target_batch_size=512)
    trainer = _fake_trainer()
    module = nn.Linear(4, 4)
    gov.on_train_start(trainer, module)
    stash = module._governor_metrics
    assert stash["gov_effective_batch"] == 32.0

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
