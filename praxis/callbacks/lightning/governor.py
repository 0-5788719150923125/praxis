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

    def __init__(self, batch_size: int, target_batch_size: int) -> None:
        super().__init__()
        self.micro_batch = max(1, int(batch_size))
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
