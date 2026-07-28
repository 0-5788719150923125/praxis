"""Lightning wiring for the gradient-noise-scale batch governor.

Replaces the static ``AccumulationSchedule`` when ``governor: gns_batch`` is
set: instead of a fixed accumulation factor derived from target_batch_size,
the factor tracks the measured gradient noise scale (see
``praxis/governors/gns.py``), with target_batch_size reinterpreted as the
ceiling.

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

Distributed note: with multiple ranks the first-microbatch gradient is
rank-local while the accumulated one is all-reduced, so the two-point pair
is inconsistent - the governor is built for the single-process runs Praxis
actually does; it logs a warning and holds the initial factor otherwise.
"""

from typing import Any, Dict, Optional

import torch
from lightning.pytorch.callbacks import Callback

from praxis.governors.gns import BatchTierController, GradientNoiseEstimator


class GNSBatchGovernor(Callback):
    """Governs ``trainer.accumulate_grad_batches`` from the gradient noise scale."""

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
    ) -> None:
        super().__init__()
        self.micro_batch = max(1, int(batch_size))
        # Validation cadence in OPTIMIZER steps. The static trainer setup
        # converts val_every to raw batches with the fixed factor, which a
        # dynamic factor breaks (target/batch=32 put validation ~5000 steps
        # apart while the governor ran at factor 2-4). The governor owns the
        # cadence instead, AccumulationSchedule-style: every batch end it
        # repoints ``trainer.val_check_batch`` at the raw-batch index where
        # the next val_every boundary of global_step lands, so val points
        # fall on the same steps as every static run.
        self.val_every = int(val_every) if val_every else None
        self.controller = BatchTierController(
            micro_batch=self.micro_batch,
            max_factor=max(1, -(-int(target_batch_size) // self.micro_batch)),
        )
        self.estimator = GradientNoiseEstimator()
        # Start at the floor: early training is the noise-dominated regime
        # where small batches are the efficient ones, and the estimator will
        # raise the tier as soon as the measurements say otherwise.
        self._factor = self.controller.min_factor
        self._pending: Optional[int] = None
        self._steps = 0
        self._micro_count = 0
        self._small_sq: Optional[torch.Tensor] = None
        self._stepped = False
        self._distributed = False
        print(
            f"[Governor] gns_batch: effective batch in "
            f"[{self.controller.min_factor * self.micro_batch}, "
            f"{self.controller.max_factor * self.micro_batch}] rows "
            f"(microbatch {self.micro_batch}), starting at "
            f"{self._factor * self.micro_batch}"
        )

    # ── lifecycle ─────────────────────────────────────────────────────────

    def on_train_start(self, trainer, pl_module) -> None:
        self._distributed = getattr(trainer, "world_size", 1) > 1
        if self._distributed:
            print(
                "[Governor] world_size > 1: two-point estimator pairs are "
                "rank-inconsistent; holding the initial factor."
            )
        trainer.accumulate_grad_batches = self._factor
        if self.val_every:
            trainer.val_check_batch = self.VAL_PARKED
        self._micro_count = 0
        self._small_sq = None
        self._stash(pl_module, noise_scale=None)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        # ``ready`` resets each epoch; a stale partial cycle must not pair a
        # first-microbatch norm with the wrong accumulated gradient.
        self._micro_count = 0
        self._small_sq = None
        self._stepped = False

    # ── measurement ───────────────────────────────────────────────────────

    def on_after_backward(self, trainer, pl_module) -> None:
        self._micro_count += 1
        if self._micro_count == 1:
            self._small_sq = self._grad_sq_norm(pl_module)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        k = int(trainer.accumulate_grad_batches)
        if (
            not self._distributed
            and k >= 2
            and self._small_sq is not None
            and self._micro_count == k  # full, regular cycle only
        ):
            big_sq = self._grad_sq_norm(pl_module)
            if big_sq is not None:
                # Undo Lightning's 1/K loss scaling on the first microbatch.
                self.estimator.update(
                    small_sq=self._small_sq * (k * k),
                    big_sq=big_sq,
                    b_small=float(self.micro_batch),
                    b_big=float(k * self.micro_batch),
                )
        self._stepped = True
        self._steps += 1
        if self._steps % self.decide_every == 0 and self.estimator.ready:
            noise = self.estimator.noise_scale()  # the one host sync
            desired = self.controller.desired_factor(self._factor, noise)
            self._pending = desired if desired != self._factor else None
            self._stash(pl_module, noise_scale=noise)

    # ── actuation ─────────────────────────────────────────────────────────

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Runs on EVERY batch (before the stepped-only bookkeeping, which
        # still consumes the pre-reset ``_stepped``/``_micro_count`` state).
        self._patch_val_cadence(trainer, batch_idx)
        if not self._stepped:
            return
        self._stepped = False
        self._micro_count = 0
        self._small_sq = None
        if self._pending is not None and self._aligned(trainer, self._pending):
            self._factor = self._pending
            self._pending = None
            trainer.accumulate_grad_batches = self._factor
            self._stash(pl_module, noise_scale=self.estimator.noise_scale())

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

    def _stash(self, pl_module, noise_scale: Optional[float]) -> None:
        """Publish telemetry on the core model; DynamicsLoggerCallback drains
        it on its own cadence (mirrors the RLCT stash pattern)."""
        model = getattr(pl_module, "model", pl_module)
        core = getattr(model, "_orig_mod", model)
        metrics: Dict[str, float] = {
            "gov_effective_batch": float(self._factor * self.micro_batch),
        }
        if noise_scale is not None:
            metrics["gov_noise_scale"] = float(noise_scale)
        metrics.update(self.estimator.internals())
        core._governor_metrics = metrics

    # ── resume ────────────────────────────────────────────────────────────

    def state_dict(self) -> Dict[str, Any]:
        return {
            "factor": self._factor,
            "steps": self._steps,
            "estimator": self.estimator.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._factor = self.controller.clamp(int(state.get("factor", self._factor)))
        self._steps = int(state.get("steps", 0))
        est = state.get("estimator")
        if isinstance(est, dict):
            self.estimator.load_state_dict(est)
