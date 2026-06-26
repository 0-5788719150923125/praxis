"""Lightning callback: the RLCT loss-landscape probe.

Every ``period`` optimizer steps this perturbs the live weights through a fixed
2D slice and samples the loss (the terrain), plus a best-effort SGLD estimate of
the local learning coefficient lambda-hat. It runs in ``on_train_batch_end`` -
on the TRAINING thread, never the snapshot-producer thread - so perturbing
weights and re-running forwards can't race the train/eval-mode guard that the
background dashboard probes are bound by (see ``praxis/web/snapshots.py``).

Results are stashed on the (uncompiled) model:

* ``model._rlct_landscape`` - the grid payload, merged into ``/api/head_snapshots``
  by the snapshot recipe/route and rendered by the ``rlct_mesh`` frontend
  renderer.
* ``model._rlct_metrics`` - the scalar cards (lambda-hat + LLC mean/max/min/std),
  drained into ``dynamics.db`` by :class:`DynamicsLoggerCallback`.

The probe snapshots params + buffers and restores them exactly, so it leaves the
optimizer's view of the model untouched.
"""

import time

import torch
from lightning.pytorch.callbacks import Callback

from praxis.metrics.rlct import (
    RLCT_DEFAULTS,
    compute_param_field,
    compute_param_manifold,
    probe_landscape,
)


class RLCTLandscapeCallback(Callback):
    """Periodically probe the loss-landscape geometry around the live weights.

    Args:
        cfg: override constants (see :data:`praxis.metrics.rlct.RLCT_DEFAULTS`).
            Defaults are model-agnostic; this exists for tests, not per-run
            tuning.
    """

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.cfg = {**RLCT_DEFAULTS, **(cfg or {})}
        self._last_probe_step = -1
        self._announced = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step < int(self.cfg["warmup_steps"]):
            return
        # Fire the first probe as soon as warmup ends, then every `period` steps.
        # Anchoring to warmup (rather than step % period) avoids a long dead wait
        # when warmup isn't a multiple of period - and probing once per step also
        # de-dupes the several on_train_batch_end calls per optimizer step under
        # gradient accumulation (step - last == 0 < period).
        if self._last_probe_step >= 0 and step - self._last_probe_step < int(
            self.cfg["period"]
        ):
            return
        self._last_probe_step = step

        try:
            self._probe(pl_module, batch, batch_idx, step)
        except Exception as e:
            print(f"[RLCT] probe failed at step {step}: {e}")
            import traceback

            traceback.print_exc()

    def _probe(self, pl_module, batch, batch_idx, step):
        model = getattr(pl_module, "model", pl_module)
        # The trainer may hand us a torch.compile wrapper; perturb and stash on
        # the underlying module, which is the object the generator/snapshot
        # recipe reads (main.py builds the Generator on the uncompiled model).
        core = getattr(model, "_orig_mod", model)

        # Reuse the trainer's own batch unpacking, then take a small sub-batch to
        # keep G^2 forward passes cheap.
        (
            input_ids,
            rewards,
            token_weights,
            task_type_ids,
            assistant_mask,
            should_skip,
        ) = pl_module._handle_batch_format(batch, batch_idx, is_training=True)
        if should_skip or input_ids is None:
            return

        # Downsample the batch hard: a few sequences (rows) and a short prefix
        # (cols). The grid is G^2 full forwards, so the probe's cost is
        # G^2 * probe_seqs * probe_len - truncating the sequence is the single
        # biggest lever for a long-context model.
        k = max(1, int(self.cfg["probe_seqs"]))
        full_len = int(input_ids.size(-1))
        plen = int(self.cfg.get("probe_len", 0) or 0)
        cols = min(plen, full_len) if plen else full_len

        def _rows(t):  # sub-batch by sequence
            return t[:k] if torch.is_tensor(t) and t.dim() >= 1 else t

        def _cols(t):  # truncate per-token tensors (B, T, ...) to the prefix
            if cols < full_len and torch.is_tensor(t) and t.dim() >= 2:
                return t[:, :cols].contiguous()
            return t

        input_ids = _cols(_rows(input_ids))
        rewards = _rows(rewards)  # per-sequence, no token axis
        token_weights = _cols(_rows(token_weights))
        task_type_ids = _cols(_rows(task_type_ids))
        assistant_mask = _cols(_rows(assistant_mask))

        aligned = getattr(pl_module, "outputs_are_aligned", False)
        if aligned:
            labels = input_ids.contiguous()
        else:
            labels = input_ids[..., 1:].contiguous()

        fwd_kwargs = dict(
            input_ids=input_ids,
            labels=labels,
            rewards=rewards,
            token_weights=token_weights,
            task_type_ids=task_type_ids,
            assistant_mask=assistant_mask,
        )

        def loss_only():
            with torch.no_grad():
                out = core(**fwd_kwargs)
            return float(out.loss.detach())

        def loss_with_grad():
            with torch.enable_grad():
                out = core(**fwd_kwargs)
            return out.loss

        n_tokens = int(input_ids.numel())

        snapshots = {}
        metrics = {}

        # Parameter manifold: PCA terrain of a structured weight's rows. Pure
        # weight analysis (no forward passes), so it runs even when the
        # landscape's size guard skips the expensive slice.
        try:
            manifold = compute_param_manifold(
                core,
                grid=int(self.cfg["manifold_grid"]),
                max_rows=int(self.cfg["manifold_max_rows"]),
            )
        except Exception as e:
            manifold = None
            print(f"[RLCT] manifold failed at step {step}: {e}")
        if manifold is not None:
            manifold["status"] = "ok"
            manifold["step"] = step
            snapshots["param_manifold"] = manifold
            metrics["rlct_manifold_var"] = manifold["var_explained"]

        # Whole-model weight geometry (all params chunked + PCA + smoothed) -
        # also weight-only, cheap, runs even when the landscape is skipped.
        try:
            field = compute_param_field(
                core,
                grid=int(self.cfg["field_grid"]),
                chunk=int(self.cfg["field_chunk"]),
                max_points=int(self.cfg["field_max_points"]),
            )
        except Exception as e:
            field = None
            print(f"[RLCT] weight geometry failed at step {step}: {e}")
        if field is not None:
            field["status"] = "ok"
            field["step"] = step
            snapshots["param_field"] = field

        # Loss landscape: the expensive 2D loss slice + SGLD lambda-hat.
        t0 = time.monotonic()
        payload, lmetrics = probe_landscape(
            core,
            loss_only,
            loss_with_grad,
            n_tokens=n_tokens,
            step=step,
            cfg=self.cfg,
        )
        if payload is not None:
            payload["status"] = "ok"
            payload["elapsed"] = time.monotonic() - t0
            snapshots["rlct_landscape"] = payload
            if lmetrics:
                metrics.update(lmetrics)
        elif not self._announced:
            total = sum(p.numel() for p in core.parameters())
            print(
                f"[RLCT] landscape skipped: {total/1e6:.0f}M params "
                f"(> max_params={self.cfg['max_params']/1e6:.0f}M); manifold only."
            )

        if snapshots:
            core._rlct_landscape = snapshots
        if metrics:
            core._rlct_metrics = metrics

        if not self._announced and snapshots:
            lam = (payload or {}).get("lambda_hat") if payload else None
            lam_s = f"{lam:.3f}" if isinstance(lam, float) else "n/a"
            mname = manifold["weight_name"] if manifold else "n/a"
            took = (payload or {}).get("elapsed", 0.0)
            print(
                f"[RLCT] probe live in {took:.2f}s: "
                f"{int(self.cfg['grid'])}^2 grid x {k}seq x {cols}tok, "
                f"λ̂={lam_s}, manifold='{mname}' (every "
                f"{int(self.cfg['period'])} steps)."
            )
            self._announced = True
