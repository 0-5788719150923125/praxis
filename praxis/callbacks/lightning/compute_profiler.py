"""Lightning callback that samples per-module compute time during training.

Arms :class:`~praxis.metrics.compute.ComputeProfiler` around ONE microbatch
every ``interval`` steps - entering at ``on_train_batch_start`` and leaving at
whichever of ``on_train_batch_end`` / ``on_before_optimizer_step`` comes first,
which under automatic optimization is a complete forward *and* backward.

Deliberately one microbatch, not the whole accumulation cycle: every microbatch
runs the same graph, so the extra ones add no information while multiplying the
profiled step's cost by the accumulation factor.

Results are stashed on the uncompiled model as ``_compute_metrics`` (scalars,
drained by DynamicsLoggerCallback) and ``_compute_profile`` (the treemap
snapshot, served by ``/api/head_snapshots``), matching the RLCT probe's
stash-and-drain pattern: the profiler runs on a slow cadence while the dashboard
re-reads the standing value every tick.

Requires eager execution. Under ``torch.compile`` Dynamo traces the hook bodies
into the graph, where the timing side effects are elided and the graph fragments
around every module boundary - the numbers would describe a model nobody runs.
The callback detects that case and disables itself with an actionable message
rather than reporting something untrue.
"""

from lightning.pytorch.callbacks import Callback

from praxis.metrics.compute import COMPUTE_DEFAULTS, ComputeProfiler


def _log_quietly(message: str) -> None:
    """Emit a diagnostic without ever being able to raise.

    Every message this callback produces is optional telemetry commentary, and
    ``print`` is not safe for it: another thread can swap the process-global
    ``sys.stdout`` for a buffer of its own and close it (see
    praxis/web/spec_data.py), after which ``print`` raises ``ValueError: I/O
    operation on closed file``. Inside an ``except`` block that re-raises out of
    the handler and kills the run - which is precisely what happened on
    abstractinator-m, where a profiler that had already failed safely was made
    fatal by its own error message.
    """
    try:
        print(message)
    except Exception:
        pass

# Retained as a tripwire only: the window now closes after one microbatch, so
# this should never be reached. If it ever is, something stopped delivering
# on_train_batch_end and the profiler would otherwise record unboundedly.
_MAX_WINDOW_BATCHES = 64

# Scope depth used under torch.compile: 0 = the model's direct children only
# (encoder / decoder / head / embeddings / criterion). Measured on the real
# model: 8 hooks cost nothing (-6.5%, i.e. noise) and the forward shares track
# eager to within 2.5%. Depth 1 keeps the right ORDER but distorts magnitudes
# badly (decoder 91.7% -> 44.1%), and every module costs +103%/step. So coarse
# is not a compromise here - it is the only depth that is both free and true.
COMPILED_DEPTH = 0


class ComputeProfilerCallback(Callback):
    """Sample per-module GPU time on a slow cadence and stash the rollup.

    Args:
        cfg: Overrides merged over :data:`~praxis.metrics.compute.COMPUTE_DEFAULTS`.
            Exists for tests, not per-run tuning.
    """

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.cfg = {**COMPUTE_DEFAULTS, **(cfg or {})}
        self.profiler = ComputeProfiler(self.cfg)
        self._installed = False
        self._disabled = False
        self._active = None
        self._window_batches = 0
        self._last_sample_step = -1
        self._announced = False

    # ------------------------------------------------------------------ setup
    def on_train_start(self, trainer, pl_module):
        """Install hooks before the first forward, or disable with a reason."""
        if self._installed or self._disabled:
            return
        model = getattr(pl_module, "model", pl_module)
        compiled = getattr(model, "_orig_mod", None) is not None
        # Compiled models get a COARSE, forward-only profile; see COMPILED_DEPTH.
        depth = COMPILED_DEPTH if compiled else None
        self.profiler.forward_only = compiled

        device = str(getattr(getattr(pl_module, "device", None), "type", "cuda"))
        try:
            ok = self.profiler.install(model, device=device, max_depth=depth)
        except Exception as e:
            self._disabled = True
            _log_quietly(f"[ComputeProfiler] disabled: hook install failed: {e}")
            return
        if not ok:
            self._disabled = True
            _log_quietly("[ComputeProfiler] disabled: no submodules to instrument.")
            return
        self._installed = True
        scope = (
            f"top-level components only, forward only (torch.compile: measured "
            f"+2.5% max share error vs eager at this depth; per-module scopes "
            f"there cost +120%/step and misattribute)"
            if compiled
            else "every module, forward+backward"
        )
        _log_quietly(
            f"[ComputeProfiler] {scope}; sampling one step every "
            f"{self.cfg['interval']} after step {self.cfg['warmup_steps']} "
            f"(EMA alpha={self.cfg['ema_alpha']})"
        )

    # ------------------------------------------------------------- the window
    def _due(self, trainer) -> bool:
        step = int(trainer.global_step)
        if step < int(self.cfg["warmup_steps"]):
            return False
        if self._last_sample_step < 0:
            return True
        # Anchoring to the last sample (not step % interval) also de-dupes the
        # several on_train_batch_start calls per optimizer step under gradient
        # accumulation, where global_step does not advance between microbatches.
        return step - self._last_sample_step >= int(self.cfg["interval"])

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._disabled or not self._installed or self._active is not None:
            return
        if not trainer.is_global_zero or not self._due(trainer):
            return
        try:
            self._active = self.profiler.start()
            self._window_batches = 0
            self._last_sample_step = int(trainer.global_step)
        except Exception as e:
            self._active = None
            self._disabled = True
            # Report through logging, not print. This handler exists to make a
            # profiler failure survivable, and the failure it most needs to
            # survive is stdout being unusable - in which case ``print`` raises
            # the SAME exception from inside the handler, escapes the ``except``
            # entirely, and takes the training run down. That is exactly how
            # optional telemetry killed abstractinator-m.
            _log_quietly(f"[ComputeProfiler] disabled: could not start profiler: {e}")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Fires only on the accumulation boundary. Without accumulation this is
        # the natural close (backward done, optimizer not yet stepped); with it,
        # on_train_batch_end below has already closed after the first
        # microbatch.
        self._close(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._active is None:
            return
        # Close after ONE microbatch, which is a complete forward+backward under
        # automatic optimization. Spanning the whole accumulation cycle instead
        # made the window N times larger for no extra information - the graph is
        # the same every microbatch, so the shares are identical - and N is 16 on
        # abstractinator-g (target_batch_size 1024 / batch_size 64). That turned
        # a ~4x profiled step into a ~35x one, which showed up as 16-55s stalls
        # every 100 steps in a real run.
        self._window_batches += 1
        self._close(pl_module)

    def on_train_end(self, trainer, pl_module):
        self._close(pl_module)
        self.profiler.remove()

    def _close(self, pl_module) -> None:
        prof, self._active = self._active, None
        if prof is None:
            return
        try:
            ok = self.profiler.stop(prof)
        except Exception as e:
            _log_quietly(f"[ComputeProfiler] sample failed: {e}")
            return
        if not ok:
            if self.profiler.hooks_fired() == 0 and not self._disabled:
                self._disabled = True
                _log_quietly(
                    "[ComputeProfiler] disabled: hooks never fired. They must "
                    "be attached before the first forward - Dynamo installs no "
                    "guard on _forward_hooks, so anything added later is "
                    "silently ignored."
                )
            return
        self._stash(pl_module)

    def _stash(self, pl_module) -> None:
        """Publish the smoothed rollup where the dashboard reads it."""
        model = getattr(pl_module, "model", pl_module)
        core = getattr(model, "_orig_mod", model)
        try:
            metrics = self.profiler.metrics()
            snapshot = self.profiler.snapshot()
        except Exception as e:
            _log_quietly(f"[ComputeProfiler] rollup failed: {e}")
            return
        if metrics:
            core._compute_metrics = metrics
        if snapshot:
            core._compute_profile = snapshot
        if not self._announced and snapshot:
            self._announced = True
            _log_quietly(self.profiler.summary())
