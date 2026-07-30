"""Lightning wiring for the gradient-noise-scale batch governor.

Replaces the static ``AccumulationSchedule`` when ``governor: gns_batch`` is
set. The governed quantity is the EFFECTIVE BATCH in rows per optimizer step,
tracking the measured gradient noise scale (see ``praxis/governors/gns.py``);
``praxis/data/batch_schedule.py`` factorizes it into rows-per-microbatch and an
accumulation factor. ``target_batch_size`` is the ceiling on the effective
batch, ``batch_size`` the ceiling on one microbatch - neither is a floor.

Timing contract with Lightning 2.x automatic optimization:

* ``on_after_backward`` fires once per microbatch, gradients accumulated so
  far in ``.grad``. After the FIRST microbatch, ``.grad`` holds exactly that
  microbatch's gradient - scaled by 1/K, because Lightning divides each
  microbatch loss by ``accumulate_grad_batches``. We record its squared norm
  and undo the scaling (x K^2) at estimation time.
* ``on_before_optimizer_step`` fires once per completed cycle, pre-clip
  (clipping happens inside the optimizer step closure afterwards), which is
  exactly the accumulated gradient the estimator wants.
* Lightning steps when ``batch_progress.current.ready % factor == 0``, so a
  factor change only produces correctly-scaled full cycles if it lands on a
  boundary aligned to the NEW factor. Down-moves (new divides old) are always
  aligned at a step boundary; up-moves may need to wait one extra cycle. The
  commit logic defers until ``ready % new == 0``.

Row counts feeding the estimator are read from the batches that actually
arrived, never from the plan: the plan is shared state the data pipeline may
lag on, and a two-point estimate built from assumed batch sizes would be
silently wrong rather than merely stale.

Distributed note: with multiple ranks the first-microbatch gradient is
rank-local while the accumulated one is all-reduced, so the two-point pair
is inconsistent - the governor is built for the single-process runs Praxis
actually does; it logs a warning and holds the initial batch otherwise.
"""

from typing import Any, Dict, Optional

import torch
from lightning.pytorch.callbacks import Callback

from praxis.data.batch_schedule import MIN_MICRO_BATCHES, BatchSchedule, plan_cycle
from praxis.governors.gns import BatchTierController, GradientNoiseEstimator


def _batch_rows(batch) -> Optional[int]:
    """Row count of a produced batch, whatever container the pipeline used."""
    if torch.is_tensor(batch):
        return int(batch.size(0))
    if isinstance(batch, dict):
        ids = batch.get("input_ids")
        if torch.is_tensor(ids):
            return int(ids.size(0))
    if isinstance(batch, (list, tuple)) and batch:
        return _batch_rows(batch[0])
    return None


class GNSBatchGovernor(Callback):
    """Governs the effective batch (rows/step) from the gradient noise scale."""

    # Optimizer steps between tier decisions. Each decision is the one host
    # sync (.item()) the governor performs; per-step work stays on device.
    decide_every = 16

    # Initial placeholder for Lightning's batch-modulo validation check,
    # in force only until the first batch ends and per-batch repointing takes
    # over. A huge int (not float("inf"): inf flips Lightning's
    # is_infinite_dataset branch, which validates on epoch's last batch).
    VAL_PARKED = 1_000_000_000

    def __init__(
        self,
        batch_size: int,
        target_batch_size: int,
        val_every: Optional[int] = None,
        sequence_multiplier_tiers=(),
    ) -> None:
        super().__init__()
        # Rows per microbatch, at most. Sized to what the GPU holds; it no
        # longer bounds the governed range in either direction.
        self.row_ceiling = max(1, int(batch_size))
        # Validation cadence in OPTIMIZER steps. The static trainer setup
        # converts val_every to raw batches with the fixed factor, which a
        # dynamic factor breaks (target/batch=32 put validation ~5000 steps
        # apart while the governor ran at factor 2-4). The governor owns the
        # cadence instead, AccumulationSchedule-style: every batch end it
        # repoints ``trainer.val_check_batch`` at the raw-batch index where
        # the next val_every boundary of global_step lands, so val points
        # fall on the same steps as every static run.
        self.val_every = int(val_every) if val_every else None
        self.controller = BatchTierController(max_rows=int(target_batch_size))
        self.estimator = GradientNoiseEstimator()
        # Start where one microbatch fills the ceiling: the smallest effective
        # batch that still uses the whole GPU per forward, given the
        # estimator's two-microbatch minimum. Below this the step simply has
        # fewer rows to spend, so utilization must fall - that is inherent, not
        # a policy. Starting here preserves the pre-governor step-0 behaviour
        # while leaving the range open in both directions.
        self._rows = self.controller.clamp(self.row_ceiling * MIN_MICRO_BATCHES)
        self._pending: Optional[int] = None
        self._steps = 0
        self._micro_count = 0
        self._micro_rows: Optional[int] = None  # observed rows, microbatch 1
        self._cycle_rows = 0  # observed rows accumulated this cycle
        self._small_sq: Optional[torch.Tensor] = None
        self._stepped = False
        self._distributed = False
        # Delivered-rows mean over the decision window; the controller
        # regulates against this rather than the requested target.
        self._delivered_sum = 0.0
        self._delivered_n = 0
        BatchSchedule.enable(
            row_ceiling=self.row_ceiling,
            effective_rows=self._rows,
            tiers=sequence_multiplier_tiers,
        )
        plan = plan_cycle(self._rows, self.row_ceiling)
        print(
            f"[Governor] gns_batch: effective batch in "
            f"[{self.controller.min_rows}, {self.controller.max_rows}] rows, "
            f"microbatch <= {self.row_ceiling} rows; starting at "
            f"{self._rows} rows ({plan.accum} x {plan.micro_rows})"
        )

    # ── lifecycle ─────────────────────────────────────────────────────────

    def on_train_start(self, trainer, pl_module) -> None:
        self._distributed = getattr(trainer, "world_size", 1) > 1
        if self._distributed:
            print(
                "[Governor] world_size > 1: two-point estimator pairs are "
                "rank-inconsistent; holding the initial batch."
            )
        self._publish(trainer)
        if self.val_every:
            trainer.val_check_batch = self.VAL_PARKED
            step = int(getattr(trainer, "global_step", 0) or 0)
            next_boundary = (step // self.val_every + 1) * self.val_every
            print(
                f"[Governor] validation every {self.val_every} optimizer "
                f"steps; resuming at step {step}, next boundary at step "
                f"{next_boundary}"
            )
        self._reset_cycle()
        self._stash(pl_module, noise_scale=None)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        # ``ready`` resets each epoch; a stale partial cycle must not pair a
        # first-microbatch norm with the wrong accumulated gradient.
        self._reset_cycle()
        self._stepped = False

    def _reset_cycle(self) -> None:
        self._micro_count = 0
        self._micro_rows = None
        self._cycle_rows = 0
        self._small_sq = None

    def _publish(self, trainer) -> None:
        """Push the governed row count into the shared plan and Lightning.

        Both sides read from the same factorization, so the accumulation factor
        Lightning steps on and the microbatch rows the data pipeline builds can
        never disagree about how many rows a step contains.
        """
        BatchSchedule.set_effective_rows(self._rows)
        BatchSchedule.restart_cycle()
        trainer.accumulate_grad_batches = plan_cycle(
            self._rows, self.row_ceiling
        ).accum

    # ── measurement ───────────────────────────────────────────────────────

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        """Record the rows this microbatch actually carries.

        The estimator's ``b_small``/``b_big`` must be true row counts. Reading
        them here rather than assuming the plan's value keeps the estimate
        correct even when the data pipeline is a cycle behind (a forked worker,
        or the dataloader's one-batch lead).
        """
        rows = _batch_rows(batch)
        if rows is None:
            return
        if self._micro_count == 0:
            self._micro_rows = rows
        self._cycle_rows += rows

    def on_after_backward(self, trainer, pl_module) -> None:
        self._micro_count += 1
        if self._micro_count == 1:
            self._small_sq = self._grad_sq_norm(pl_module)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        k = int(trainer.accumulate_grad_batches)
        if (
            not self._distributed
            and k >= MIN_MICRO_BATCHES
            and self._small_sq is not None
            and self._micro_rows
            and self._micro_count == k  # full, regular cycle only
        ):
            big_sq = self._grad_sq_norm(pl_module)
            # Homogeneous cycles are the estimator's precondition: the two
            # points differ only in how many rows they average over, which the
            # per-cycle multiplier makes true (see batch_schedule). A cycle
            # whose rows are not a clean k-multiple of the first microbatch
            # means the pipeline changed shape mid-cycle, so skip the pair
            # rather than fold a mismatched measurement into the EMA.
            homogeneous = self._cycle_rows == self._micro_rows * k
            if big_sq is not None and homogeneous:
                # Undo Lightning's 1/K loss scaling on the first microbatch.
                self.estimator.update(
                    small_sq=self._small_sq * (k * k),
                    big_sq=big_sq,
                    b_small=float(self._micro_rows),
                    b_big=float(self._cycle_rows),
                )
        if self._cycle_rows:
            self._delivered_sum += float(self._cycle_rows)
            self._delivered_n += 1
        self._stepped = True
        self._steps += 1
        if self._steps % self.decide_every == 0 and self.estimator.ready:
            noise = self.estimator.noise_scale()  # the one host sync
            desired = self.controller.desired_rows(
                self._rows, noise, measured=self._delivered_mean()
            )
            self._pending = desired if desired != self._rows else None
            self._stash(pl_module, noise_scale=noise)
            self._delivered_sum = 0.0
            self._delivered_n = 0

    def _delivered_mean(self) -> Optional[float]:
        if self._delivered_n <= 0:
            return None
        return self._delivered_sum / self._delivered_n

    # ── actuation ─────────────────────────────────────────────────────────

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Runs on EVERY batch (before the stepped-only bookkeeping, which
        # still consumes the pre-reset ``_stepped``/``_micro_count`` state).
        self._patch_val_cadence(trainer, batch_idx)
        if not self._stepped:
            return
        self._stepped = False
        last_cycle_rows = self._cycle_rows
        self._reset_cycle()
        if self._pending is not None:
            accum = plan_cycle(self._pending, self.row_ceiling).accum
            if self._aligned(trainer, accum):
                self._rows = self._pending
                self._pending = None
                self._publish(trainer)
                self._stash(
                    pl_module,
                    noise_scale=self.estimator.noise_scale(),
                    delivered=last_cycle_rows,
                )

    def _patch_val_cadence(self, trainer, batch_idx) -> None:
        """Live-repoint Lightning's validation trigger every batch.

        Lightning fires validation when ``(batch_idx + 1) %
        trainer.val_check_batch == 0``, checked right AFTER this hook - so
        keeping ``val_check_batch`` equal to the absolute raw-batch index of
        the next ``val_every`` optimizer-step boundary makes the modulo hit
        exactly there (never at a multiple: the target always sits at most
        one interval ahead). Stateless: recomputed each batch from
        ``global_step`` and the live factor, so tier changes just move the
        target, checkpoint resume needs no special handling, and the value
        is always readable as "the batch validation will run on".
        """
        if not self.val_every:
            return
        step = int(getattr(trainer, "global_step", 0) or 0)
        done = batch_idx + 1  # raw batches completed this epoch
        if self._stepped and step > 0 and step % self.val_every == 0:
            trainer.val_check_batch = done  # boundary lands on this batch
            return
        # Predict the boundary batch: remaining optimizer steps x current
        # factor, minus the microbatches already consumed of the open cycle.
        remaining = self.val_every - (step % self.val_every)
        factor = max(int(trainer.accumulate_grad_batches or 1), 1)
        into_cycle = 0 if self._stepped else self._micro_count
        target = done + remaining * factor - into_cycle
        # Strictly future: a stale estimate must never fire early.
        trainer.val_check_batch = max(target, done + 1)

    @staticmethod
    def _aligned(trainer, factor: int) -> bool:
        """True when Lightning's within-epoch batch counter sits on a
        boundary of the NEW factor, so every future cycle is full-length and
        correctly loss-scaled."""
        try:
            ready = int(trainer.fit_loop.epoch_loop.batch_progress.current.ready)
        except AttributeError:
            return True  # exotic trainer; take the boundary we know we're on
        return ready % int(factor) == 0

    # ── internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _grad_sq_norm(pl_module) -> Optional[torch.Tensor]:
        """Squared global L2 norm of all present gradients, as a 0-dim tensor
        (no host sync)."""
        grads = [p.grad for p in pl_module.parameters() if p.grad is not None]
        if not grads:
            return None
        norms = torch._foreach_norm(grads)
        stacked = torch.stack([n.to(norms[0].device) for n in norms])
        return (stacked * stacked).sum().detach()

    def _stash(
        self,
        pl_module,
        noise_scale: Optional[float],
        delivered: Optional[int] = None,
    ) -> None:
        """Publish telemetry on the core model; DynamicsLoggerCallback drains
        it on its own cadence (mirrors the RLCT stash pattern).

        ``gov_effective_batch`` reports rows that actually ran when we have a
        completed cycle to report, falling back to the target. The two differ
        when a long-sequence cycle's attention budget caps microbatch rows, and
        the chart should show the batch the optimizer saw.
        """
        model = getattr(pl_module, "model", pl_module)
        core = getattr(model, "_orig_mod", model)
        rows = delivered if delivered else self._delivered_mean()
        metrics: Dict[str, float] = {
            "gov_effective_batch": float(rows if rows else self._rows),
            "gov_target_batch": float(self._rows),
        }
        if noise_scale is not None:
            metrics["gov_noise_scale"] = float(noise_scale)
        metrics.update(BatchSchedule.metrics())
        metrics.update(self.estimator.internals())
        core._governor_metrics = metrics

    # ── resume ────────────────────────────────────────────────────────────

    def state_dict(self) -> Dict[str, Any]:
        return {
            "effective_rows": self._rows,
            "steps": self._steps,
            "estimator": self.estimator.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        # Accept the pre-rows checkpoint key: older runs stored an accumulation
        # factor against the then-fixed microbatch, which is the same thing
        # expressed in units of batch_size.
        if "effective_rows" in state:
            rows = int(state["effective_rows"])
        elif "factor" in state:
            rows = int(state["factor"]) * self.row_ceiling
        else:
            rows = self._rows
        self._rows = self.controller.clamp(rows)
        self._steps = int(state.get("steps", 0))
        est = state.get("estimator")
        if isinstance(est, dict):
            self.estimator.load_state_dict(est)
        BatchSchedule.set_effective_rows(self._rows)
        BatchSchedule.restart_cycle()
