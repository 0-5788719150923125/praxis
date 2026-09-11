"""GNS batch governor: estimator math, tier control, Lightning wiring."""

import random

import pytest

from praxis.data.batch_schedule import BatchSchedule, plan_cycle

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


def test_schedule_is_inert_until_enabled():
    """A run without a governor must keep its static batch behaviour."""
    assert BatchSchedule.next_microbatch() is None
    assert BatchSchedule.current() is None
    assert BatchSchedule.metrics() == {}


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
    from praxis.data.datasets.manager import SEQUENCE_MULTIPLIER_TIERS as tiers
    from praxis.data.datasets.manager import (
        max_sequence_multiplier,
    )

    for ceiling in (16, 64, 256):
        sized_for = max_sequence_multiplier(ceiling, tiers)
        rows = 2
        while rows <= 4096:
            base = plan_cycle(rows, ceiling).micro_rows
            assert max_sequence_multiplier(base, tiers) <= sized_for
            rows *= 2


# ---------------------------------------------------- the fetcher's lead
#
# Lightning pre-fetches one batch whenever the loader has no length, which an
# infinite streaming dataset never does, and refills the moment it hands one
# over. So the pipeline builds microbatch N+1 before the hooks for microbatch N
# run, and anything reading the shared plan at consumption time is a step ahead
# unless the plan is explicitly walked forward with the trainer.


def test_current_plan_follows_the_trained_batch_not_the_built_one():
    BatchSchedule.enable(row_ceiling=64, effective_rows=256, tiers=())
    first = BatchSchedule.next_microbatch()
    assert BatchSchedule.current() is None  # nothing has been trained on yet
    BatchSchedule.next_microbatch()  # the fetcher runs ahead
    assert BatchSchedule.in_flight() == 2

    assert BatchSchedule.consume() is first  # oldest built = the one in hand
    assert BatchSchedule.current() is first
    assert BatchSchedule.in_flight() == 1


def test_metrics_report_the_trained_shape():
    """The gov_* shape cards ride alongside the step's loss, so they must
    describe that step's microbatch, not the next one's."""
    BatchSchedule.enable(row_ceiling=64, effective_rows=8, tiers=())
    assert BatchSchedule.metrics() == {}  # nothing trained on yet
    BatchSchedule.next_microbatch()
    assert BatchSchedule.metrics() == {}  # built, not yet consumed
    BatchSchedule.consume()
    assert BatchSchedule.metrics()["gov_micro_rows"] == 4.0


def test_retarget_keeps_production_in_phase_with_the_trainer():
    """A commit truncates the open production cycle. The microbatches already
    in flight belong to the trainer's NEXT cycle, so the fresh cycle owes them;
    ignoring them shifted the phase by one for the rest of the run, and every
    later step then straddled two cycles - two multipliers, two row counts,
    inside one step Lightning scales by a single 1/accum."""
    BatchSchedule.enable(row_ceiling=64, effective_rows=256, tiers=())
    accum = BatchSchedule.accum()
    assert accum == 4

    queue = [BatchSchedule.next_microbatch()]  # the fetcher's initial prefetch
    ready, step, cycle = 0, 0, []
    straddled = []
    for _ in range(24):
        queue.append(BatchSchedule.next_microbatch())  # refills on handover
        cycle.append(BatchSchedule.consume())  # the batch now being trained on
        ready += 1
        queue.pop(0)
        if ready % accum:
            continue
        step += 1
        # Plans are shared per cycle, so identity says which cycle a
        # microbatch came from - even when two cycles roll the same shape.
        if len({id(p) for p in cycle}) > 1:
            straddled.append(step)
        cycle = []
        if step == 2:  # a committed tier change
            BatchSchedule.set_effective_rows(128)
            BatchSchedule.restart_cycle()
            assert BatchSchedule.in_flight() == 1
            accum, ready = BatchSchedule.accum(), 0
    # Exactly one step straddles: the one holding the microbatch the fetcher
    # had already built when the commit landed. That batch cannot be unbuilt.
    # Before the carry it was every step from the commit onward.
    assert straddled == [3]
