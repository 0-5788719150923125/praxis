"""One factorization for the three dynamic batch knobs.

Praxis varies three things about a batch during training, each driven by its
own signal:

* the **sequence-length multiplier** ``m`` - chosen by the probe curriculum in
  :mod:`praxis.data.seq_probe`, deciding what LENGTH to train on;
* the **effective batch** ``B_eff`` (rows consumed per optimizer step) - chosen
  by the gradient-noise-scale governor in :mod:`praxis.governors.gns`, deciding
  how many SAMPLES the step averages over;
* the **accumulation factor** - historically a static
  ``target_batch_size / batch_size``.

They used to be wired together in a way that made ``batch_size`` do two
unrelated jobs. It set the rows per microbatch (a memory/compute property of
the GPU) AND, because the effective batch could only be a whole multiple of it,
it set a FLOOR under the governed quantity: with ``batch_size=64`` and the
estimator's two-microbatch minimum, the smallest reachable effective batch was
128 rows. Reaching smaller batches meant shrinking the microbatch, which taxes
throughput at every batch size - the run pays for downward range forever.

Here the two jobs separate. ``batch_size`` is a CEILING on rows per microbatch,
nothing more; ``B_eff`` is governed independently and factorized against that
ceiling:

    accum      = smallest power of two >= 2 with B_eff / accum <= row_ceiling
    base_rows  = B_eff // accum                 (rows per microbatch at m=1)
    micro_rows = max(1, base_rows // m**2)      (attention budget at length m)

Three properties matter, and each is a constraint the old wiring violated:

1. **``accum`` depends only on ``B_eff``, never on ``m``.** Lightning steps when
   ``ready % accum == 0``, so an ``accum`` that moved with the curriculum's dice
   would land mid-cycle and produce short, mis-scaled cycles. Keeping it a
   function of the governed quantity alone means it changes only when the
   governor moves a tier, which is the one case the governor already defers to
   an aligned boundary.

2. **At least two microbatches, always.** The noise-scale estimator is a
   two-point method: it needs a first-microbatch gradient AND an accumulated
   one. The old code protected this by flooring the effective batch at
   ``2 * batch_size``; the guard belongs on the microbatch COUNT, which is what
   it was really about. Flooring the count instead is what frees the effective
   batch to reach 2 rows.

3. **The multiplier is fixed for a whole accumulation cycle.** ``m`` divides
   rows by ``m**2``, so sampling it per microbatch made the microbatches within
   one cycle different sizes. Lightning divides every microbatch loss by
   ``accum`` uniformly, which only averages correctly when they are the same
   size, and the estimator's two points are only comparable when they are drawn
   from the same distribution. Resampling per cycle instead of per microbatch
   fixes both, and gives the curriculum a clean unit to count: one arm visit per
   optimizer step, which is the regressor its fit is built on.

On the ``m**2``: it holds ATTENTION COST constant, not token count. Rows scale
as ``1/m**2`` while length scales as ``m``, so ``rows * len**2`` is invariant
while ``rows * len`` falls as ``1/m``. A consequence worth reading off the
cards rather than assuming away: at large ``m`` the attention budget caps
``micro_rows`` hard enough that ``accum * micro_rows`` lands below the governed
``B_eff``. The plan reports what it actually delivers, and the governor
regulates against the delivered mean, so neither the controller nor the chart
believes a batch size that never ran.

Cross-process note: like the sampler weights and the probe curriculum, the
schedule is class-level state - the governor writes it, the data pipeline reads
it. Praxis forces the ``spawn`` start method (``configure_multiprocessing``),
which makes ``num_workers=0``, so the dataset iterates in the same process and
sees writes immediately. A forked worker would see the snapshot it was forked
with; the governor therefore measures rows from the batch it actually received
rather than from what it asked for, so a stale plan can degrade throughput but
cannot corrupt the estimator.
"""

import random
from typing import Dict, NamedTuple, Optional, Tuple


class BatchPlan(NamedTuple):
    """How one accumulation cycle is laid out.

    Attributes:
        micro_rows: Rows in each microbatch.
        accum: Microbatches per optimizer step.
        multiplier: Sequence-length multiplier for every microbatch in the cycle.
        delivered_rows: ``micro_rows * accum`` - the effective batch that will
            actually run, which is at most the governed target.
        nominal_rows: What to pass ``InterleaveDataManager.get_batch``, which
            applies the ``m**2`` division itself; ``micro_rows * m**2``.
    """

    micro_rows: int
    accum: int
    multiplier: int
    delivered_rows: int
    nominal_rows: int


# Fixed, model-agnostic constant (no per-experiment tuning): the two-point
# noise-scale estimator needs a second measurement point, so a MEASURING step is
# never a single microbatch. This is the ONLY reason the minimum is 2 rather
# than 1 - and it is the only reason a step ever splits when the whole effective
# batch already fits under the ceiling.
#
# That distinction matters. Above the ceiling the split is forced by memory and
# the second point is free. At or below it, forcing accum=2 buys the estimator
# its pair at the price of running two forwards where one would do: at an
# effective batch of 4 rows with a 64-row ceiling, 2 microbatches of 2 rows
# instead of 1 microbatch of 4 - double the step cost for the same data, on a
# GPU with 16x the headroom. The governor therefore pays this only on the
# windows it actually measures (see GNSBatchGovernor.MEASURE_EVERY) and passes
# ``min_micro_batches=1`` the rest of the time.
MIN_MICRO_BATCHES = 2


def plan_cycle(
    effective_rows: int,
    row_ceiling: int,
    multiplier: int = 1,
    min_micro_batches: int = MIN_MICRO_BATCHES,
) -> BatchPlan:
    """Factorize a governed effective batch into (rows per microbatch, accum).

    Args:
        effective_rows: Governed ``B_eff`` - target rows per optimizer step.
        row_ceiling: ``batch_size``; the most rows one microbatch may hold at
            ``multiplier == 1``. A ceiling, never a floor.
        multiplier: Sequence-length multiplier for this cycle.
        min_micro_batches: Microbatches per step, minimum.

    ``accum`` is derived by doubling from the minimum until a microbatch fits
    under the ceiling, so it is a power-of-two multiple of the minimum and
    depends on nothing but ``effective_rows`` and ``row_ceiling``.
    """
    row_ceiling = max(1, int(row_ceiling))
    multiplier = max(1, int(multiplier))
    min_micro = max(1, int(min_micro_batches))
    effective_rows = max(min_micro, int(effective_rows))

    accum = min_micro
    # Doubling (rather than a division) keeps accum a clean power-of-two
    # multiple of the minimum, so micro_rows * accum hits effective_rows
    # exactly whenever effective_rows is itself a power of two - which the
    # governor's doubling tiers guarantee.
    while effective_rows // accum > row_ceiling:
        accum *= 2

    base_rows = max(1, effective_rows // accum)
    micro_rows = max(1, base_rows // (multiplier * multiplier))
    return BatchPlan(
        micro_rows=micro_rows,
        accum=accum,
        multiplier=multiplier,
        delivered_rows=micro_rows * accum,
        nominal_rows=micro_rows * multiplier * multiplier,
    )


class BatchSchedule:
    """Shared batch plan: the governor writes it, the data pipeline reads it.

    Inert until ``enable``; while inert the data pipeline keeps its static
    behaviour, so a run without a governor is byte-for-byte unchanged.
    """

    enabled: bool = False
    row_ceiling: int = 0  # batch_size: rows per microbatch, at most
    effective_rows: int = 0  # governed B_eff, in rows per optimizer step
    tiers: Tuple = ()  # sequence-multiplier tiers, for resampling
    # Microbatches per step, minimum. MIN_MICRO_BATCHES while the governor is
    # measuring the noise scale, 1 while it is not - a step only splits below
    # the ceiling to buy the estimator's second point, and that is not needed
    # on every step. See MIN_MICRO_BATCHES above.
    min_micro: int = MIN_MICRO_BATCHES

    _plan: Optional[BatchPlan] = None
    _micro_index: int = 0  # microbatches emitted in the open cycle

    @classmethod
    def enable(
        cls,
        row_ceiling: int,
        effective_rows: int,
        tiers=(),
        min_micro: int = MIN_MICRO_BATCHES,
    ) -> None:
        cls.enabled = True
        cls.row_ceiling = max(1, int(row_ceiling))
        cls.effective_rows = max(MIN_MICRO_BATCHES, int(effective_rows))
        cls.tiers = tuple(tiers)
        cls.min_micro = max(1, int(min_micro))
        cls._plan = None
        cls._micro_index = 0

    @classmethod
    def reset(cls) -> None:
        cls.enabled = False
        cls.row_ceiling = 0
        cls.effective_rows = 0
        cls.tiers = ()
        cls.min_micro = MIN_MICRO_BATCHES
        cls._plan = None
        cls._micro_index = 0

    @classmethod
    def set_effective_rows(cls, rows: int, min_micro: Optional[int] = None) -> None:
        """Governor hook: retarget the effective batch (and the split minimum).

        Takes effect at the next cycle boundary, never mid-cycle - changing
        microbatch rows partway through would break both the uniform ``1/accum``
        loss scaling and the estimator's pairing.
        """
        if not cls.enabled:
            return
        cls.effective_rows = max(MIN_MICRO_BATCHES, int(rows))
        if min_micro is not None:
            cls.min_micro = max(1, int(min_micro))

    @classmethod
    def accum(cls) -> int:
        """Microbatches per optimizer step under the current target.

        Readable without advancing anything, because the governor needs it to
        set ``trainer.accumulate_grad_batches`` before any batch is drawn.
        """
        if not cls.enabled:
            return MIN_MICRO_BATCHES
        return cls._plan_now().accum

    @classmethod
    def _plan_now(cls, multiplier: int = 1) -> BatchPlan:
        """Factorize the current target under the current split minimum."""
        return plan_cycle(
            cls.effective_rows,
            cls.row_ceiling,
            multiplier,
            min_micro_batches=cls.min_micro,
        )

    @classmethod
    def current(cls) -> Optional[BatchPlan]:
        """The open cycle's plan, without advancing. None before the first."""
        return cls._plan

    @classmethod
    def next_microbatch(cls, rng=random) -> Optional[BatchPlan]:
        """Plan for the next microbatch, starting a new cycle when one is due.

        Returns None while inert so the caller falls back to its static path.
        The multiplier is resampled only at a cycle start; every microbatch in
        a cycle then reports the same plan.
        """
        if not cls.enabled:
            return None
        if cls._plan is None or cls._micro_index >= cls._plan.accum:
            cls._plan = cls._new_cycle(rng)
            cls._micro_index = 0
        cls._micro_index += 1
        return cls._plan

    @classmethod
    def _new_cycle(cls, rng) -> BatchPlan:
        """Sample this cycle's multiplier and factorize against the target.

        Eligibility is budgeted against the rows this step actually has to
        spend (``base_rows``, i.e. effective rows per microbatch at ``m=1``),
        not against ``row_ceiling``. The tiers' ``m**2`` rule divides rows to
        hold attention cost constant, so a length is only affordable if the
        step has ``m**2`` rows to give up; budgeting against the fixed ceiling
        would let a 4-row step draw the 8x length and deliver a 2-row optimizer
        step. The bandit keeps its estimates for every arm either way - the
        sampler renormalizes over the eligible ones - so a small batch narrows
        what gets drawn without the controller losing state.
        """
        from praxis.data.datasets.manager import sample_sequence_multiplier

        base = cls._plan_now().micro_rows
        multiplier = sample_sequence_multiplier(base, cls.tiers, rng)
        return cls._plan_now(multiplier)

    @classmethod
    def restart_cycle(cls) -> None:
        """Drop the open cycle so the next microbatch starts a fresh one.

        Called when the governor commits a new accumulation factor: the old
        cycle's remaining microbatches would carry the previous factor's row
        count into a step counted under the new one.
        """
        cls._plan = None
        cls._micro_index = 0

    @classmethod
    def metrics(cls) -> Dict[str, float]:
        """Live plan, for the governor's telemetry stash."""
        if not cls.enabled or cls._plan is None:
            return {}
        return {
            "gov_micro_rows": float(cls._plan.micro_rows),
            "gov_accum": float(cls._plan.accum),
            "gov_seq_multiplier": float(cls._plan.multiplier),
        }
