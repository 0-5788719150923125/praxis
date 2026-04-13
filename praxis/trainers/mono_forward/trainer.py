"""MonoForwardTrainer - pipelined Mono-Forward training.

The trainer runs one ``LayerActor`` per ``LocalLayer``: each actor
owns its own copy of the layer + the shared output head + a local
optimizer, and trains against the same next-token objective as the
standard backprop path. Activation memory is O(1) in depth because
gradients never cross a layer boundary, and the driver pipelines
batches across layers so steady-state throughput is close to the
fastest layer, not the sum of them.

A brief tour of the internals:

- **Manual pipeline driver** (``_run_manual_pipeline``): keeps a dict
  of in-flight Ray futures keyed by ObjectRef, refills layer 0 from
  the dataloader whenever the pipeline has slack, and uses
  ``ray.wait`` to find whichever actor finished next. Completed
  layer-``i`` outputs are forwarded into layer ``i+1`` immediately;
  completed final-layer outputs finalize the batch, emit metrics,
  and free a slot for the next refill. In steady state with
  ``num_layers`` layers, up to ``num_layers`` batches are in flight
  simultaneously - each layer processes a different batch per "tick".

- **Compiled pipeline driver** (``_run_compiled_pipeline``): deferred.
  Ray's ``experimental_compile`` API needs more validation before we
  commit to it as a default; until then ``--ray-pipeline-api compiled``
  raises ``NotImplementedError`` with a clear pointer at the manual
  variant.

- **Head synchronization** (``_sync_head``): every
  ``ray_head_sync_every`` batches the driver drains the pipeline,
  gathers each actor's head state dict, averages them elementwise, and
  broadcasts the result back into every actor. The drain barrier is
  load-bearing - without it, an in-flight batch would be using an old
  head copy at some layer and the new copy at another, producing an
  inconsistent forward pass.

- **MetricsLogger emission**: ``avg_step_time``, ``loss``,
  ``num_tokens``, and ``learning_rate`` land in the corresponding
  native columns (the existing web dashboard already knows how to
  read these). ``layer_{i}_loss``, ``pipeline_in_flight``, and
  ``head_syncs_completed`` go into the ``extra_metrics`` JSON blob.

- **CLI flag plumbing** (``--ray-address``, ``--ray-num-replicas-per-layer``,
  ``--ray-head-sync-every``, ``--ray-pipeline-api``): main.py reads
  these from ``processed_args`` and threads them into
  ``create_trainer_with_module`` as keyword arguments; the factory's
  ``mono_forward`` branch pulls them out of ``kwargs`` and hands them
  to this class's ``__init__``.

The Ray worker backend is an implementation detail; the ``--ray-*``
flags live under the training group because they're the only knobs
Ray exposes that we care about today, but the trainer surface itself
is framework-agnostic (a future Lightning / native torch.distributed /
Hivemind backend would slot behind the same ``MonoForwardTrainer``).
"""

from __future__ import annotations

import copy
import math
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from praxis.utils import create_block_ids


def _force_cpu(module: torch.nn.Module) -> torch.nn.Module:
    """Move a module to CPU, including non-registered tensor attributes.

    ``nn.Module.cpu()`` only touches parameters and buffers registered
    via ``register_parameter`` / ``register_buffer``. Some submodules
    store tensors as plain instance attributes (common examples: RoPE
    frequency caches, pre-computed causal masks, cached cos/sin
    tables). Those survive ``.cpu()`` on whatever device they were
    created on. When Ray pickles the module for transport to a worker
    raylet, those CUDA tensors are serialized with their original
    device metadata and the worker's ``torch.load`` fails if it
    doesn't have CUDA.

    This helper does a belt-and-braces sweep: call ``.cpu()`` first
    (cheap, covers the common case), then walk every sub-module's
    ``__dict__`` looking for stray ``torch.Tensor`` instances on
    non-CPU devices and force them over. Modifies the module in
    place and returns it for chaining convenience.
    """
    module = module.cpu()
    for submodule in module.modules():
        for attr_name, attr_value in list(vars(submodule).items()):
            if isinstance(attr_value, torch.Tensor) and attr_value.device.type != "cpu":
                setattr(submodule, attr_name, attr_value.cpu())
    return module


_RAY_MISSING_MSG = (
    "mono_forward requires the optional 'ray' extra, which is not "
    "installed in this environment.\n\n"
    "Ray publishes no wheels for Python >= 3.14. The typical path is to "
    "run inside the Docker compose environment (Ubuntu 24.04 + Python 3.12, "
    "where Ray installs cleanly):\n"
    "    ./launch compose --mike --trainer-type mono_forward ...\n\n"
    "To install Ray in a local venv instead (only works on Python 3.10-3.13):\n"
    "    pip install -e '.[ray]'\n\n"
)


class MonoForwardTrainer:
    """Framework-agnostic trainer that drives distributed Mono-Forward.

    Unlike :class:`BackpropagationTrainer` (a Lightning module), this
    trainer is itself the trainer - it exposes a ``.fit`` method matching
    the call contract in ``main.py`` (``trainer.fit(model, datamodule,
    ckpt_path=..., weights_only=...)``) and owns the full training loop.

    The factory in ``praxis/trainers/factory.py`` constructs this class
    directly and returns ``(trainer, model)``, bypassing Lightning's
    ``Trainer`` wrapper entirely. Lightning-specific kwargs passed in via
    ``trainer_params`` (``accelerator``, ``callbacks``, ``precision``, ...)
    are accepted but largely ignored - we only read the handful that map
    to concepts we actually support.

    **Feature parity with BackpropagationTrainer** (keep this list
    current - every new backprop metric needs a matching MF entry or
    a documented reason to skip it, so drift is caught at review
    time):

    Supported:
    - ``loss`` (per-batch, per-layer loss averaged across actors)
    - ``batch`` / ``step`` counters (step = batch // accumulate_grad_batches)
    - ``learning_rate`` (from the last-hop actor's optimizer)
    - ``num_tokens`` (billions, same unit convention as backprop)
    - ``avg_step_time`` (EMA-smoothed, same alpha as backprop)
    - ``softmax_collapse`` (final-layer actor computes from projected logits)
    - ``val_loss`` / ``val_perplexity`` / ``val_bits_per_byte``
      (periodic validation sweep at ``val_check_interval`` batches)
    - per-layer gradient dynamics (``layer_{i}_grad_norm`` etc.,
      written to dynamics.db at ``dynamics_log_freq`` cadence)
    - per-dataset metrics via ``data_metrics.db`` (written by the
      dataset manager directly, framework-agnostic - MF gets this
      for free as long as the dataloader runs)
    - live inference via ``trainer.generate()`` and the API
      ``MonoForwardGenerator`` adapter (Phase 5 / Phase 6)

    Not supported, hard-errors at trainer init:
    - ``depth != num_layers`` (recurrent depth; decision D1)
    - ``router_type in {smear, distance, prismatic, scatter}``
      (shared LocalLayer instances; decision D4)
    - ``tie_word_embeddings=True`` (unregistered parameter in the
      TiedWeights head; deepcopy orphans it)
    - ``rl_type`` set (GRPO / REINFORCE / CoT; needs
      ``model.generate`` + reward rollouts on the driver, which
      under MF would run against untrained weights)

    Not supported, blocked by the above:
    - per-expert gradient dynamics (requires Prismatic/SMEAR)
    - router convergence metrics from ``model.get_metrics()``
      (requires Prismatic/SMEAR)
    - Prismatic ``modify_expert_gradients`` hook
      (fires inside the backprop ``on_after_backward`` Lightning
      hook which MF never runs)

    When you add a new metric to ``BackpropagationTrainer``, land
    it here too by (1) emitting it from ``LayerActor.train_batch``
    in the return dict, and (2) passing it to ``metrics_logger.log``
    in ``_handle_completion``'s metrics block. The dict-return
    shape from ``train_batch`` is intentional: new fields are
    non-breaking.
    """

    def __init__(self, **trainer_params: Any) -> None:
        # Verify Ray is importable *before* we do any other setup, and
        # raise a clean actionable error if not. The ``./launch`` bash
        # guard already handles this for most users; this branch catches
        # the ``python main.py`` direct-invocation case.
        try:
            import ray  # noqa: F401
        except ImportError as exc:
            raise ImportError(_RAY_MISSING_MSG) from exc

        # Ingest the Lightning-flavoured params dict. ``-1`` / ``None``
        # mean "unbounded" in Lightning conventions; we normalize to a
        # non-negative integer or ``None`` for our own loop.
        raw_max_steps = trainer_params.get("max_steps")
        if raw_max_steps is None or raw_max_steps < 0:
            self.max_steps: Optional[int] = None
        else:
            self.max_steps = int(raw_max_steps)
        self.log_every_n_steps: int = int(trainer_params.get("log_every_n_steps", 10))

        # Ray-specific knobs. The factory reads these from main.py's
        # processed argv via kwargs; callers that construct the trainer
        # directly (tests, scripted use) can pass them here. All have
        # sensible Phase 2/3 defaults.
        #
        # ``ray_address=None`` is the critical value: it tells Ray to
        # bootstrap a fresh in-process cluster instead of looking for a
        # pre-existing one. The string "auto" means the OPPOSITE ("find
        # an existing cluster, or error"). If ``RAY_ADDRESS`` is set in
        # the environment (Phase 4 compose test), ``ray.init`` honours
        # it regardless of what we pass here.
        self.ray_address: Optional[str] = trainer_params.get("ray_address", None)
        self.ray_num_replicas_per_layer: int = int(
            trainer_params.get("ray_num_replicas_per_layer", 1) or 1
        )
        self.ray_head_sync_every: int = int(
            trainer_params.get("ray_head_sync_every", 50) or 50
        )
        self.ray_pipeline_api: str = str(
            trainer_params.get("ray_pipeline_api", "manual") or "manual"
        )

        if self.ray_num_replicas_per_layer != 1:
            raise RuntimeError(
                f"--ray-num-replicas-per-layer={self.ray_num_replicas_per_layer} is "
                "not supported in Phase 3. Multi-replica data parallelism is a "
                "Phase 5+ concern; set --ray-num-replicas-per-layer 1 (the default) "
                "until then."
            )

        if self.ray_pipeline_api not in ("manual", "compiled"):
            raise RuntimeError(
                f"--ray-pipeline-api={self.ray_pipeline_api!r} is not a valid choice. "
                "Allowed values: 'manual' (default) or 'compiled' (Phase 3+)."
            )

        # ``cache_dir`` is injected by the factory (see factory.py). It's
        # where the trainer writes metrics.db and the final monolithic
        # checkpoint.
        self.cache_dir: Optional[str] = trainer_params.get("cache_dir")

        # Phase 6: bridge training state into the LiveMetrics singleton
        # so the web dashboard's Terminal tab (and anyone else reading
        # the /metrics-live WebSocket) sees live batch/loss/status
        # updates. Under backprop this is populated by the Lightning
        # TerminalInterface callback; MF bypasses Lightning entirely,
        # so we mirror the same state pushes ourselves inside the
        # pipeline loop. ``model_info`` is the static training config
        # main.py already builds to hand to TerminalInterface; we
        # accept it via kwargs so we can build the info panel without
        # duplicating config parsing.
        self.model_info: Dict[str, Any] = dict(trainer_params.get("model_info") or {})
        self.dashboard_url: Optional[str] = trainer_params.get("dashboard_url")
        self.device: str = str(trainer_params.get("device") or "cpu")
        # Display-only accumulation factor. Under MF every raw batch is
        # a real optimizer step (there's no gradient accumulation under
        # the hood), but the dashboard UX treats ``step`` as an
        # effective-update counter - ``batch // accumulate_grad_batches``
        # - so the user sees the same "N batches per step" ratio the
        # Lightning backprop path shows. Defaults to 1 (no scaling).
        self.accumulate_grad_batches: int = max(
            int(trainer_params.get("accumulate_grad_batches") or 1), 1
        )

        # Optimizer + scheduler config for each actor. main.py builds
        # these the same way the backprop path does (via
        # ``get_optimizer_profile`` + CLI flags), and we forward them
        # into every LayerActor's constructor so each actor rebuilds
        # the full ``get_optimizer(...) / get_scheduler_func(...)``
        # pipeline against its own (layer, head) params. ``None``
        # defaults mean the actor falls back to vanilla Adam(lr=1e-3),
        # which is the Phase 2/3 behaviour and what unit tests that
        # instantiate the trainer directly (without main.py) rely on.
        self.optimizer_config: Optional[Dict[str, Any]] = trainer_params.get(
            "optimizer_config"
        )
        self.optimizer_wrappers: Dict[str, bool] = dict(
            trainer_params.get("optimizer_wrappers") or {}
        )
        self.warmup_steps: int = int(
            trainer_params.get("warmup_steps") or 0
        )
        self.disable_schedule: bool = bool(
            trainer_params.get("disable_schedule") or False
        )

        # Validation and dynamics logging cadences - mirroring the
        # backprop path so a dashboard client sees the same columns
        # populated at the same step rhythm. ``val_check_interval``
        # defaults to None (disabled); main.py passes the Lightning
        # default ``1024 * target_batch_size // batch_size``.
        # ``dynamics_log_freq`` defaults to 10 - same as
        # DynamicsLoggerCallback - which logs every 10th completed
        # batch. ``num_experts`` is only non-zero if the blocked
        # router types ever get unblocked; until then the dynamics
        # DB just carries universal per-layer metrics.
        raw_val_every = trainer_params.get("val_every")
        self.val_check_interval: Optional[int] = (
            int(raw_val_every) if raw_val_every else None
        )
        self.dynamics_log_freq: int = int(
            trainer_params.get("dynamics_log_freq") or 10
        )
        self.byte_latent: bool = bool(
            trainer_params.get("byte_latent") or False
        )
        self._live_metrics: Optional[Any] = None
        self._live_metrics_ema_loss: Optional[float] = None

        # Phase 5 Task 2: live-inference-during-training demo hook. When
        # ``inference_prompt`` is set, the driver runs the prompt through
        # ``self.generate`` every ``inference_every_seconds`` of wall
        # clock (matching the ``--infer-every`` semantics the backprop
        # TerminalInterface already uses, so one CLI flag drives both
        # trainers' periodic-inference cadences). ``None`` disables the
        # hook entirely; ``0.0`` fires at every final-layer completion
        # boundary (useful for tests).
        #
        # ``tokenizer`` is optional; when present, the hook decodes
        # generated ids back into text for user-friendly logging. When
        # absent (tests that construct the trainer directly without a
        # tokenizer), the hook falls back to printing raw id lists.
        self.tokenizer: Optional[Any] = trainer_params.get("tokenizer")
        self.inference_prompt: Optional[Any] = trainer_params.get("inference_prompt")
        raw_every = trainer_params.get("inference_every_seconds")
        self.inference_every_seconds: Optional[float] = (
            float(raw_every) if raw_every is not None else None
        )
        # Maximum new tokens per inference hook fire. Matches the
        # backprop ``TerminalInterface._generate_text`` default: start
        # at 1, with a small geometric chance of drawing extras, so
        # the passage grows one token at a time (ish) rather than
        # landing a whole N-token chunk per fire. ``inference_max_new_tokens``
        # caps the geometric draw so no single fire can produce more
        # than this many tokens even on a lucky streak.
        self.inference_max_new_tokens: int = int(
            trainer_params.get("inference_max_new_tokens", 16) or 16
        )
        self.inference_max_context_chars: int = int(
            trainer_params.get("inference_max_context_chars", 512) or 512
        )
        # Last wall-clock time the inference hook fired; initialised
        # lazily in the pipeline loop so the first firing is gated on
        # "at least ``inference_every_seconds`` after the very first
        # batch completes", not on process start.
        self._last_inference_time: Optional[float] = None
        # Growing text buffer across hook fires, mirroring how the
        # backprop ``TerminalInterface`` accumulates text between
        # generations. Initialised lazily on the first fire once we
        # know the prompt as decoded text. ``StreamingContext`` handles
        # the reset-on-degeneracy path.
        self._inference_context: Optional[Any] = None
        # Internal cached reference to the live actor set for
        # ``generate()`` calls that happen *inside* ``fit()``. Set by
        # ``fit`` before the pipeline loop, cleared in the finally
        # block. Concurrent external ``generate()`` calls (from another
        # thread) can consume this reference safely because Ray
        # serializes method calls on an actor.
        self._actors: Optional[List[Any]] = None
        self._embeds: Optional[torch.nn.Module] = None
        self._config: Optional[Any] = None

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def _validate_model(self, model: Any) -> None:
        """Enforce the decoder constraints MF requires (D1 and D4).

        Recurrent depth and shared-layer routing modes both share
        ``LocalLayer`` instances across positions in the decoder, which
        is fundamentally incompatible with the per-layer optimizers MF
        needs. Fail loud at trainer init with a clear message pointing at
        the decision in ``PLAN.md`` rather than producing a silently
        wrong training run.
        """
        config = model.config

        if config.depth != config.num_layers:
            raise RuntimeError(
                f"mono_forward requires depth == num_layers "
                f"(got depth={config.depth}, num_layers={config.num_layers}). "
                "Recurrent depth is out of scope for Mono-Forward "
                "(decision D1 in PLAN.md). Set --depth == --num-layers, "
                "or use --trainer-type backpropagation."
            )

        blocked_routers = {"smear", "distance", "prismatic", "scatter"}
        if config.router_type in blocked_routers:
            raise RuntimeError(
                f"mono_forward does not support router_type={config.router_type!r}. "
                f"Routing modes in {sorted(blocked_routers)} share LocalLayer "
                "instances across depths, which is incompatible with per-layer "
                "MF optimizers (decision D4 in PLAN.md). Restoring support for "
                "these modes is gated on a future --materialize-shared-layers "
                "flag."
            )

        # RL training paths (GRPO, REINFORCE, CoT, etc.) need the
        # driver to call ``model.generate()`` mid-training to produce
        # response rollouts, then feed rewards back into the loss.
        # Under MF the trained weights live on Ray actors and the
        # driver's CPU copy of the model is pristine - ``model.generate``
        # on it would run against untrained weights. Phase 5's
        # ``trainer.generate()`` routes through the actor chain but
        # doesn't implement the RL reward plumbing (GRPO group
        # generation, reward evaluation, loss injection). Rather than
        # silently produce wrong training runs, hard-error here with
        # a clear pointer at backprop.
        rl_type = getattr(config, "rl_type", None)
        if rl_type:
            raise RuntimeError(
                f"mono_forward does not support rl_type={rl_type!r}. "
                "RL training paths (GRPO, REINFORCE, CoT) require the "
                "driver to call model.generate() mid-training for reward "
                "rollouts; under MF the driver's model is an untrained "
                "copy while the trained weights live on Ray actors. RL "
                "under MF is a future-phase concern; use "
                "--trainer-type backpropagation for any RL run."
            )

        # Tied weights issue, discovered during Phase 2 end-to-end run:
        # the TiedWeights head stores ``self.embedding_weight`` as a plain
        # instance attribute (not via ``register_parameter`` or ``nn.Parameter``
        # registration), so when we ``copy.deepcopy(head)`` into each Ray
        # actor the resulting tensor is not in ``head.parameters()`` and
        # therefore never trains. The driver's embedding and the actors'
        # replicated heads then drift apart silently, producing
        # absurdly-large initial losses and non-monotonic training
        # dynamics. Proper tied-head support needs driver<->actor head
        # synchronization, which is Phase 4+ work. Until then, hard-error
        # with a clear pointer at the fix: flip ``tie_weights: false``.
        if getattr(config, "tie_word_embeddings", False):
            raise RuntimeError(
                "mono_forward does not support tied output embeddings "
                "in Phase 3 (config.tie_word_embeddings is True). The tied "
                "head stores its projection weight as an unregistered "
                "instance attribute, which becomes an orphan tensor in the "
                "actor's replicated head copy and never trains. Set "
                "'tie_weights: false' in your experiment YAML (or pass "
                "--no-tie-weights on the CLI). Proper tied-weight support "
                "is deferred to Phase 4+, which will wire driver<->actor "
                "head synchronization."
            )

        # Sanity check the decoder surface we are about to drive.
        if not hasattr(model, "decoder") or not hasattr(model.decoder, "locals"):
            raise RuntimeError(
                "mono_forward expected model.decoder.locals to be a ModuleList "
                "of LocalLayers, but the model does not expose that attribute."
            )
        if not hasattr(model, "embeds") or model.embeds is None:
            raise RuntimeError(
                "mono_forward expected model.embeds to be set "
                "(encoder-based models are not yet supported under MF)."
            )
        if not hasattr(model, "head") or model.head is None:
            raise RuntimeError(
                "mono_forward expected model.head to be set "
                "(shared-head projection D2b needs a real head module)."
            )

    # ------------------------------------------------------------------
    # fit entry point
    # ------------------------------------------------------------------

    def fit(
        self,
        model: Any,
        datamodule: Any,
        ckpt_path: Optional[str] = None,
        weights_only: bool = False,
    ) -> Dict[str, Any]:
        """Train ``model`` against ``datamodule`` using pipelined MF via Ray.

        ``ckpt_path`` and ``weights_only`` are accepted for signature
        compatibility with the backpropagation path in ``main.py`` but
        are not used in Phase 3 (checkpoint resume is a Phase 4+ concern).
        Returns a small summary dict for callers that want to introspect
        the run.
        """
        import ray

        from praxis.trainers.mono_forward.actor import LayerActor

        self._validate_model(model)

        # Force EVERYTHING to CPU before Ray touches it. ``model.cpu()``
        # moves registered parameters and buffers, but some modules
        # store tensors as plain instance attributes (RoPE frequency
        # caches, attention masks, etc.) that ``.cpu()`` silently skips.
        # When the driver has CUDA and a worker raylet doesn't, those
        # orphaned CUDA tensors cause ``torch.load`` to raise
        # "Attempting to deserialize object on a CUDA device but
        # torch.cuda.is_available() is False" on the receiving side.
        # ``_force_cpu`` walks every attribute after ``.cpu()`` to catch
        # the stragglers.
        model_host = _force_cpu(model)
        # Release the CUDA memory that ``initialize_lazy_modules``
        # reserved when it moved the full model to GPU for the dummy
        # forward pass. ``_force_cpu`` moved the tensors back to CPU,
        # but CUDA's caching allocator keeps the reserved blocks
        # until explicitly told to release them. Without this, the
        # driver-side reservation sits alongside the actors' own GPU
        # allocations, roughly doubling VRAM usage for no benefit.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        embeds = model_host.embeds
        layers: List[Any] = list(model_host.decoder.locals)
        head = model_host.head
        criterion = model_host.criterion
        strategy = getattr(model_host, "strategy", None)
        num_layers = len(layers)

        # Phase 6: bridge into the LiveMetrics singleton FIRST, so the
        # subsequent startup banners and initialization logs get
        # mirrored into the web Terminal tab via ``self._log``.
        self._init_live_metrics(num_layers=num_layers)

        self._log(
            f"[MF] Pipelined training: num_layers={num_layers}, "
            f"max_steps={self.max_steps}, api={self.ray_pipeline_api}, "
            f"head_sync_every={self.ray_head_sync_every}, "
            f"val_check_interval={self.val_check_interval}"
        )

        # Initialise Ray. ``ignore_reinit_error=True`` lets the same
        # Python process drive multiple fit() calls (used by the tests)
        # without tripping on "Ray already initialised".
        ray.init(
            address=self.ray_address,
            ignore_reinit_error=True,
            log_to_driver=True,
        )

        # MetricsLogger is opt-in on cache_dir - tests that don't need
        # persistent metrics can construct the trainer with cache_dir=None.
        metrics_logger = None
        dynamics_logger = None
        if self.cache_dir:
            from praxis.logging.dynamics_logger import DynamicsLogger
            from praxis.logging.metrics_logger import MetricsLogger

            metrics_logger = MetricsLogger(run_dir=self.cache_dir)
            # Per-layer gradient dynamics into a separate SQLite so
            # the dashboard's Dynamics tab populates under MF. Matches
            # the DB file the backprop DynamicsLoggerCallback writes.
            try:
                dynamics_logger = DynamicsLogger(run_dir=self.cache_dir)
                self._log(
                    f"[MF] DynamicsLogger writing to {self.cache_dir}/dynamics.db "
                    f"(log_freq={self.dynamics_log_freq})"
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._log(f"[MF] DynamicsLogger init failed: {exc}")
                dynamics_logger = None

        actors: List[Any] = []
        try:
            # Determine CPU and GPU resources per actor from the
            # cluster's actual capacity and the user's ``--device``
            # setting. GPU is only requested when ``device`` starts
            # with "cuda" - each Ray actor runs in its own process,
            # and each process creates a CUDA context (~300-500MB),
            # so GPU is only worth the overhead for models large
            # enough that per-layer GPU compute dominates the fixed
            # context cost. CPU actors pipeline concurrently via Ray
            # and are fast enough for small-to-medium models.
            try:
                cluster_resources = ray.cluster_resources()
                cluster_cpus = float(cluster_resources.get("CPU", 1))
                cluster_gpus = float(cluster_resources.get("GPU", 0))
            except Exception:
                cluster_cpus = 1.0
                cluster_gpus = 0.0

            use_gpu = self.device.startswith("cuda") and cluster_gpus > 0
            num_cpus_per_actor = cluster_cpus / num_layers
            num_gpus_per_actor = cluster_gpus / num_layers if use_gpu else 0

            resource_parts = [f"{cluster_cpus} CPU(s) -> {num_cpus_per_actor:.2f}/actor"]
            if use_gpu:
                resource_parts.append(
                    f"{cluster_gpus} GPU(s) -> {num_gpus_per_actor:.2f}/actor"
                )
            else:
                resource_parts.append("GPU: off (device=cpu)")
            self._log(
                f"[MF] Cluster resources for {num_layers} actors: "
                + ", ".join(resource_parts)
            )

            actors = [
                LayerActor.options(
                    num_cpus=num_cpus_per_actor,
                    num_gpus=num_gpus_per_actor,
                ).remote(
                    layer_idx=i,
                    layer=copy.deepcopy(layers[i]),
                    head=copy.deepcopy(head),
                    criterion=copy.deepcopy(criterion),
                    strategy=copy.deepcopy(strategy) if strategy is not None else None,
                    optimizer_config=self.optimizer_config,
                    optimizer_wrappers=self.optimizer_wrappers,
                    warmup_steps=self.warmup_steps,
                    disable_schedule=self.disable_schedule,
                    num_layers=num_layers,
                )
                for i in range(num_layers)
            ]
            # Block until every actor is constructed and reachable. This
            # surfaces any deepcopy / pickling errors before the training
            # loop, not in the middle of it.
            ray.get([actor.ping.remote() for actor in actors])
            self._log(f"[MF] Spawned and pinged {num_layers} LayerActor(s)")

            # Expose actor set for concurrent ``generate()`` callers and
            # for the periodic-inference hook inside the pipeline loop.
            self._actors = actors
            self._embeds = embeds
            self._config = model_host.config

            if self.ray_pipeline_api == "manual":
                result = self._run_manual_pipeline(
                    ray=ray,
                    actors=actors,
                    embeds=embeds,
                    config=model_host.config,
                    datamodule=datamodule,
                    num_layers=num_layers,
                    metrics_logger=metrics_logger,
                    dynamics_logger=dynamics_logger,
                )
            else:  # "compiled"
                result = self._run_compiled_pipeline(
                    ray=ray,
                    actors=actors,
                    embeds=embeds,
                    datamodule=datamodule,
                    num_layers=num_layers,
                    metrics_logger=metrics_logger,
                )

            if result["completed_batches"] > 0:
                self._save_checkpoint(model_host, actors)

            return result

        finally:
            self._actors = None
            self._embeds = None
            self._config = None
            if metrics_logger is not None:
                try:
                    metrics_logger.close()
                except Exception:
                    pass
            if dynamics_logger is not None:
                try:
                    dynamics_logger.close()
                except Exception:
                    pass
            for actor in actors:
                try:
                    ray.kill(actor)
                except Exception:
                    pass
            ray.shutdown()
            self._log("[MF] Ray shutdown complete")

    # ------------------------------------------------------------------
    # manual pipeline driver
    # ------------------------------------------------------------------

    def _run_manual_pipeline(
        self,
        ray: Any,
        actors: List[Any],
        embeds: torch.nn.Module,
        config: Any,
        datamodule: Any,
        num_layers: int,
        metrics_logger: Optional[Any],
        dynamics_logger: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Drive the pipeline via ``ray.wait`` on in-flight futures.

        At most ``num_layers`` batches are in flight at any time; a new
        batch is pulled from the dataloader each time a slot opens up (a
        final-layer completion). Batches flow through the layer chain
        one hop at a time, with the driver forwarding each layer's
        output into the next layer as soon as the future resolves.

        Loss bookkeeping is per-batch: we collect the per-layer losses
        into ``loss_accumulator[batch_idx][layer_idx]`` and emit the
        averaged metric when the final layer reports its loss for that
        batch.

        The completion handler is factored into a closure so the head-
        sync drain path (which walks partially-complete batches all the
        way through the pipeline before the head average happens) uses
        the same "batch done" bookkeeping - otherwise a drain-time
        final-layer hop would silently skip the finalize path and drop
        a completed batch from ``completed_batches`` / metrics.
        """
        dataloader = datamodule.train_dataloader()
        dataloader_iter = iter(dataloader)

        # ObjectRef -> dict with {layer_idx, batch_idx, labels, input_ids,
        # block_ids, start_time}. Per-batch state (input_ids, block_ids)
        # stays pinned on the driver so every hop passes the same tensor
        # into the next actor - re-computing block_ids per hop would be
        # wasteful and risks drift if the input ever changed.
        in_flight: Dict[Any, Dict[str, Any]] = {}
        # batch_idx -> {layer_idx: loss_value}
        loss_accumulator: Dict[int, Dict[int, float]] = {}
        # batch_idx -> {layer_idx: {grad_norm, grad_var, update_ratio}}
        # Populated only on batches where ``_should_capture_dynamics``
        # returned True; flushed into dynamics.db when the final-layer
        # hop completes for the batch.
        dynamics_accumulator: Dict[int, Dict[int, Dict[str, float]]] = {}

        # Mutable state captured by the nested helpers. Kept as a single
        # dict so the closures don't need ~10 ``nonlocal`` declarations.
        state: Dict[str, Any] = {
            "first_loss": None,
            "last_loss": None,
            "loss_history": [],
            "per_layer_loss_history": {i: [] for i in range(num_layers)},
            "pipeline_in_flight_max": 0,
            "batches_started": 0,
            "completed_batches": 0,
            "head_syncs_done": 0,
            "dataloader_exhausted": False,
            "num_tokens_total": 0,
            "avg_step_time_ema": None,
            "last_softmax_collapse": None,
            "last_val_step": 0,
        }
        max_in_flight = num_layers  # pipeline steady-state capacity
        wall_clock_start = time.monotonic()
        ema_alpha = 0.1  # smoothing for avg_step_time emission

        def _should_capture_dynamics(batch_idx: int) -> bool:
            """Decide whether to ask actors to capture dynamics this batch.

            Gates on ``dynamics_logger`` being live (i.e. cache_dir was
            set) and on the driver's log_freq cadence. Amortizes the
            per-parameter walk across ``dynamics_log_freq`` batches so
            steady-state training isn't paying for metric capture on
            every single step.
            """
            if dynamics_logger is None:
                return False
            return batch_idx % max(self.dynamics_log_freq, 1) == 0

        def _refill_layer0() -> None:
            """Pull batches from the dataloader into layer 0 until the
            pipeline is full or we hit the step budget or the dataloader
            is exhausted.
            """
            while (
                len(in_flight) < max_in_flight
                and (
                    self.max_steps is None
                    or state["batches_started"] < self.max_steps
                )
                and not state["dataloader_exhausted"]
            ):
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    state["dataloader_exhausted"] = True
                    break

                if isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                else:
                    input_ids = batch

                with torch.no_grad():
                    activations = embeds(input_ids)
                labels = input_ids[..., 1:].contiguous()
                # block_ids masks attention across EOS boundaries so a
                # batch of packed sequences doesn't leak attention
                # between unrelated samples. The same tensor is reused
                # for every layer hop and threaded through the in-flight
                # state dict.
                block_ids = create_block_ids(input_ids, config.eos_token_id)

                batch_idx = state["batches_started"]
                start_time = time.monotonic()
                future = actors[0].train_batch.remote(
                    activations,
                    labels,
                    batch_idx,
                    input_ids,
                    block_ids,
                    _should_capture_dynamics(batch_idx),
                )
                in_flight[future] = {
                    "layer_idx": 0,
                    "batch_idx": batch_idx,
                    "labels": labels,
                    "input_ids": input_ids,
                    "block_ids": block_ids,
                    "start_time": start_time,
                }
                loss_accumulator[batch_idx] = {}
                state["num_tokens_total"] += int(input_ids.numel())
                state["batches_started"] += 1

        def _handle_completion(
            layer_idx: int,
            batch_idx: int,
            labels: torch.Tensor,
            input_ids: torch.Tensor,
            block_ids: torch.Tensor,
            start_time: float,
            result: Dict[str, Any],
        ) -> bool:
            """Process one completed future.

            Returns ``True`` if the completion was a final-layer hop
            (i.e. the batch finished and a pipeline slot is now free);
            ``False`` if the completion was an interior hop and a new
            future has been submitted to the next layer. Shared between
            the main driver loop and the head-sync drain path so the
            "batch done" finalize logic fires consistently.

            ``result`` is the dict returned by ``LayerActor.train_batch``.
            It carries the hidden-states tensor plus every per-batch
            metric we want to land in metrics.db / dynamics.db:
            ``loss``, ``lr``, ``softmax_collapse``, and ``dynamics``.
            When a new metric is added to the backprop trainer,
            adding it to ``train_batch``'s return dict AND to the
            ``metrics_logger.log(...)`` call below is all it takes -
            no tuple re-unpacking, no position tracking.
            """
            activations = result["hidden_states"]
            loss_value = result["loss"]
            current_lr = result["lr"]
            softmax_collapse = result.get("softmax_collapse")
            layer_dynamics = result.get("dynamics")

            loss_accumulator[batch_idx][layer_idx] = loss_value
            state["per_layer_loss_history"][layer_idx].append(loss_value)
            # Track the last-layer softmax_collapse we saw so it
            # can be logged to metrics.db when the batch finalizes.
            if softmax_collapse is not None:
                state["last_softmax_collapse"] = softmax_collapse
            # Stash any layer-scoped dynamics into a batch-keyed
            # accumulator so ``finalize`` can write one merged row
            # per batch to dynamics.db.
            if layer_dynamics is not None:
                dynamics_accumulator.setdefault(batch_idx, {})[layer_idx] = layer_dynamics

            if layer_idx + 1 < num_layers:
                # Interior hop: forward to the next layer with the
                # same labels (every layer supervises against the same
                # next-token target per decision D3). input_ids and
                # block_ids are the per-batch constants the driver
                # computed in ``_refill_layer0`` and reuses unchanged
                # across every hop. ``capture_dynamics`` propagates so
                # the next-layer actor uses the same cadence decision
                # this layer did.
                next_future = actors[layer_idx + 1].train_batch.remote(
                    activations,
                    labels,
                    batch_idx,
                    input_ids,
                    block_ids,
                    _should_capture_dynamics(batch_idx),
                )
                in_flight[next_future] = {
                    "layer_idx": layer_idx + 1,
                    "batch_idx": batch_idx,
                    "labels": labels,
                    "input_ids": input_ids,
                    "block_ids": block_ids,
                    "start_time": start_time,
                }
                return False

            # Final-layer hop: compute averaged loss, log metrics,
            # bump counters, free the slot.
            state["completed_batches"] += 1
            layer_losses = loss_accumulator.pop(batch_idx)
            avg_loss = sum(layer_losses.values()) / len(layer_losses)
            if state["first_loss"] is None:
                state["first_loss"] = avg_loss
            state["last_loss"] = avg_loss
            state["loss_history"].append(avg_loss)

            elapsed = time.monotonic() - start_time
            if state["avg_step_time_ema"] is None:
                state["avg_step_time_ema"] = elapsed
            else:
                state["avg_step_time_ema"] = (
                    ema_alpha * elapsed + (1 - ema_alpha) * state["avg_step_time_ema"]
                )

            if (
                state["completed_batches"] % self.log_every_n_steps == 0
                or state["completed_batches"] == 1
            ):
                per_layer_str = " ".join(
                    f"{layer_losses[i]:.4f}" for i in sorted(layer_losses)
                )
                self._log(
                    f"[MF] batch {batch_idx:4d}  avg={avg_loss:.4f}  "
                    f"pipe_in_flight={state['pipeline_in_flight_max']:2d}  "
                    f"layers=[{per_layer_str}]"
                )

            # Tokens are reported in BILLIONS, matching the convention
            # in praxis/trainers/backpropagation.py (which divides its
            # running token count by 1e9 before logging). The Research
            # tab, the Terminal tab's tokens field, and any external
            # chart consumer all expect the billions scale. We divide
            # here once and reuse the value for both metrics.db and
            # the LiveMetrics push below.
            num_tokens_billions = state["num_tokens_total"] / 1_000_000_000

            if metrics_logger is not None:
                extras: Dict[str, Any] = {
                    "pipeline_in_flight": len(in_flight),
                    "head_syncs_completed": state["head_syncs_done"],
                }
                for i, li_loss in layer_losses.items():
                    extras[f"layer_{i}_loss"] = float(li_loss)
                # Final-layer softmax collapse (same metric the
                # backprop trainer writes). Only meaningful when the
                # last-layer actor computed it during this batch.
                batch_collapse = state.get("last_softmax_collapse")
                collapse_kwarg: Dict[str, Any] = {}
                if batch_collapse is not None:
                    collapse_kwarg["softmax_collapse"] = float(batch_collapse)
                metrics_logger.log(
                    step=batch_idx,
                    loss=float(avg_loss),
                    num_tokens=float(num_tokens_billions),
                    avg_step_time=float(state["avg_step_time_ema"]),
                    learning_rate=float(current_lr),
                    batch=int(state["completed_batches"]),
                    **collapse_kwarg,
                    **extras,
                )

            # Flush aggregated per-layer dynamics for this batch to
            # dynamics.db. Only fires on batches where the driver
            # requested capture (``_should_capture_dynamics``), so
            # the cost of walking every parameter is amortized across
            # ``dynamics_log_freq`` batches, same as the backprop
            # ``DynamicsLoggerCallback``'s ``log_freq`` cadence.
            batch_layer_dynamics = dynamics_accumulator.pop(batch_idx, None)
            if (
                dynamics_logger is not None
                and batch_layer_dynamics
                and len(batch_layer_dynamics) == num_layers
            ):
                flat_dynamics: Dict[str, float] = {}
                for li, dyn in batch_layer_dynamics.items():
                    for key, value in dyn.items():
                        flat_dynamics[f"layer_{li}_{key}"] = float(value)
                try:
                    dynamics_logger.log(step=batch_idx, dynamics=flat_dynamics)
                except Exception as exc:  # pragma: no cover - defensive
                    self._log(f"[MF] dynamics logger failed: {exc}")

            # Phase 6: also push into LiveMetrics so the web Terminal
            # tab updates. Mirrors ``TerminalInterface._update_live_metrics``
            # under backprop - same fields, same singleton.
            self._push_live_metrics_batch(
                batch_idx=batch_idx,
                avg_loss=avg_loss,
                avg_step_time=state["avg_step_time_ema"],
                num_tokens_billions=num_tokens_billions,
                num_layers=num_layers,
                completed_batches=state["completed_batches"],
            )

            return True

        def _drain_pipeline() -> None:
            """Wait for every in-flight future to resolve, advancing
            each one through the remaining layers. Uses the same
            ``_handle_completion`` path as the main loop so
            ``completed_batches`` and metrics stay consistent - without
            this, the final-layer hops that happen during drain would
            silently consume batches.
            """
            while in_flight:
                ready, _ = ray.wait(list(in_flight.keys()), num_returns=1)
                future = ready[0]
                meta = in_flight.pop(future)
                result = self._safe_ray_get(future, ray)
                if result is None:
                    in_flight.clear()
                    return
                _handle_completion(
                    layer_idx=meta["layer_idx"],
                    batch_idx=meta["batch_idx"],
                    labels=meta["labels"],
                    input_ids=meta["input_ids"],
                    block_ids=meta["block_ids"],
                    start_time=meta["start_time"],
                    result=result,
                )

        # Kick the pipeline: fill layer 0 with as many batches as it can
        # accept for the first refill. Subsequent refills happen after
        # each final-layer completion.
        _refill_layer0()

        while in_flight:
            state["pipeline_in_flight_max"] = max(
                state["pipeline_in_flight_max"], len(in_flight)
            )

            # Block until any one future in the pipeline has a result.
            # ``_safe_ray_get`` catches shutdown errors (actor death,
            # cluster teardown, Ctrl+C) and returns None so we exit
            # the loop cleanly instead of crashing.
            ready, _ = ray.wait(list(in_flight.keys()), num_returns=1)
            future = ready[0]
            meta = in_flight.pop(future)
            result = self._safe_ray_get(future, ray)
            if result is None:
                in_flight.clear()
                break

            is_final_hop = _handle_completion(
                layer_idx=meta["layer_idx"],
                batch_idx=meta["batch_idx"],
                labels=meta["labels"],
                input_ids=meta["input_ids"],
                block_ids=meta["block_ids"],
                start_time=meta["start_time"],
                result=result,
            )

            if is_final_hop:
                # Head sync at every ``ray_head_sync_every`` completed
                # batch boundary. We do it here (not on layer-0 refill)
                # so we always have a well-defined "batches completed"
                # count - easier to reason about than "batches started".
                # The sync fires at the very last batch boundary too,
                # which is correct: it averages every actor's head into
                # a single copy that the subsequent checkpoint save can
                # use as a canonical "shared head" value.
                if (
                    self.ray_head_sync_every > 0
                    and state["completed_batches"] % self.ray_head_sync_every == 0
                ):
                    self._log(
                        f"[MF] Head sync #{state['head_syncs_done'] + 1}: "
                        f"draining pipeline ({len(in_flight)} futures in flight)"
                    )
                    _drain_pipeline()
                    self._sync_head(ray=ray, actors=actors)
                    state["head_syncs_done"] += 1

                # Periodic validation sweep. Uses a threshold check
                # (``>=``) rather than modulo (``%``) because the
                # head-sync drain above can process multiple batches
                # in one go, jumping ``completed_batches`` over a
                # validation boundary. A modulo check would miss that
                # boundary entirely; the threshold catches any
                # crossing regardless of how many batches the drain
                # consumed.
                current_step = state["completed_batches"] // self.accumulate_grad_batches
                next_val_at = state["last_val_step"] + (self.val_check_interval or 0)
                if (
                    self.val_check_interval is not None
                    and self.val_check_interval > 0
                    and current_step > 0
                    and current_step >= next_val_at
                ):
                    # Snap to the exact boundary so the step logged
                    # to metrics.db is a clean multiple of val_every
                    # (e.g. 1024, 2048, 3072) rather than wherever
                    # completed_batches happened to land.
                    val_step = (current_step // self.val_check_interval) * self.val_check_interval
                    state["last_val_step"] = val_step
                    _drain_pipeline()
                    self._run_validation(
                        ray=ray,
                        actors=actors,
                        embeds=embeds,
                        datamodule=datamodule,
                        config=config,
                        num_layers=num_layers,
                        current_step=val_step,
                        metrics_logger=metrics_logger,
                    )

                # Periodic-inference demo hook. Fire after any head sync
                # so the generated sample reflects the canonical averaged
                # head rather than an arbitrary actor's drifted copy.
                self._maybe_run_inference_hook(
                    completed_batches=state["completed_batches"],
                    config=config,
                )

                # Slot freed up - try to refill layer 0.
                _refill_layer0()

        total_wall = time.monotonic() - wall_clock_start
        self._log(
            f"[MF] Pipeline finished: {state['completed_batches']} batches in "
            f"{total_wall:.1f}s, start={state['first_loss']}, "
            f"end={state['last_loss']}, "
            f"max_pipeline_in_flight={state['pipeline_in_flight_max']}, "
            f"head_syncs={state['head_syncs_done']}"
        )

        return {
            "steps": state["completed_batches"],
            "completed_batches": state["completed_batches"],
            "first_loss": state["first_loss"],
            "final_loss": state["last_loss"],
            "loss_history": state["loss_history"],
            "per_layer_loss_history": state["per_layer_loss_history"],
            "pipeline_in_flight_max": state["pipeline_in_flight_max"],
            "head_syncs_done": state["head_syncs_done"],
        }

    # ------------------------------------------------------------------
    # compiled pipeline driver (Phase 3+ stub)
    # ------------------------------------------------------------------

    def _run_compiled_pipeline(
        self,
        ray: Any,
        actors: List[Any],
        embeds: torch.nn.Module,
        datamodule: Any,
        num_layers: int,
        metrics_logger: Optional[Any],
    ) -> Dict[str, Any]:
        """Ray ``experimental_compile`` DAG variant.

        Not implemented in Phase 3: Ray's DAG / compiled-graphs API is
        still labeled experimental (the name says so), and sinking time
        into validating it is not the highest-value move before Phase 4
        lands. ``--ray-pipeline-api manual`` is the only supported
        value until this stub is replaced. See PLAN.md "Approach 1b"
        for the fallback rationale.
        """
        raise NotImplementedError(
            "mono_forward does not yet support --ray-pipeline-api compiled. "
            "The compiled-graphs API is still experimental and needs validation "
            "before we commit to it as a supported default. Use "
            "--ray-pipeline-api manual (the default) for now; a Phase 4+ pass "
            "will implement the compiled variant on top of the same actor set."
        )

    # ------------------------------------------------------------------
    # Ray shutdown safety
    # ------------------------------------------------------------------

    _RAY_SHUTDOWN_ERRORS: tuple = ()  # populated lazily below

    def _safe_ray_get(self, refs, ray):
        """``ray.get`` wrapper that returns ``None`` on shutdown errors.

        Every ``ray.get`` in the trainer goes through this so a Ctrl+C
        or worker death during any phase (head sync, validation,
        checkpoint save, inference) exits cleanly instead of crashing
        with a C++ stack trace.
        """
        if not self._RAY_SHUTDOWN_ERRORS:
            # Populate once; avoids import at module level.
            MonoForwardTrainer._RAY_SHUTDOWN_ERRORS = (
                ray.exceptions.RayActorError,
                ray.exceptions.ActorUnavailableError,
                ray.exceptions.RayError,
                KeyboardInterrupt,
            )
        try:
            return ray.get(refs)
        except self._RAY_SHUTDOWN_ERRORS:
            self._log("[MF] Shutting down (actor lost or interrupted)")
            return None

    # ------------------------------------------------------------------
    # head synchronization
    # ------------------------------------------------------------------

    def _sync_head(self, ray: Any, actors: List[Any]) -> None:
        """Average head weights across actors and broadcast back.

        The caller must have already drained the pipeline via
        ``_drain_pipeline`` before invoking this method - without a
        clean drain, a batch could be halfway through the pipeline
        when the head mutation happens, producing an inconsistent
        forward pass (earlier layers would see the old head; later
        layers would see the new one).

        This method itself is pure gather + average + broadcast; it
        does not touch ``in_flight`` or any loss bookkeeping.
        """
        head_states = self._safe_ray_get(
            [actor.get_head_state.remote() for actor in actors], ray
        )
        if head_states is None:
            return

        averaged: Dict[str, torch.Tensor] = {}
        keys = list(head_states[0].keys())
        num_actors = len(head_states)
        for key in keys:
            stacked = torch.stack([hs[key] for hs in head_states], dim=0)
            averaged[key] = stacked.mean(dim=0)

        self._log(
            f"[MF] Head sync: broadcasting averaged head ({len(keys)} tensors) "
            f"to {num_actors} actor(s)"
        )
        self._safe_ray_get(
            [actor.set_head_state.remote(averaged) for actor in actors], ray
        )

    # ------------------------------------------------------------------
    # validation sweep
    # ------------------------------------------------------------------

    def _run_validation(
        self,
        ray: Any,
        actors: List[Any],
        embeds: torch.nn.Module,
        datamodule: Any,
        config: Any,
        num_layers: int,
        current_step: int,
        metrics_logger: Optional[Any],
    ) -> None:
        """Run a validation sweep through the active actor set.

        Mirrors :meth:`BackpropagationTrainer.validation_step` on the
        backprop side: iterates ``datamodule.val_dataloader()``, hops
        each val batch through every actor via
        :meth:`LayerActor.val_batch` (no-grad, no backward, no
        optimizer step), aggregates the last-layer loss into
        ``val_loss``, computes ``val_perplexity = exp(val_loss)``
        (or ``val_bits_per_byte`` when the trainer was constructed
        with ``byte_latent=True``), and writes both to metrics.db
        at ``current_step``.

        The caller is responsible for draining the training pipeline
        before invoking this method, so val batches run against a
        stable actor state with no in-flight training work.
        Validation is sequential on purpose (one batch fully
        through the actor chain, then the next) - pipelining it
        would add plumbing complexity for a metric that's gathered
        every few thousand training batches.
        """
        # Switch the Terminal tab mode indicator to "validation",
        # matching what TerminalInterface.on_validation_start does
        # under backprop. Restored to "train" at the end of this
        # method regardless of outcome.
        if self._live_metrics is not None:
            try:
                self._live_metrics.state.set_mode("validation")
                self._live_metrics._update_count += 1
            except Exception:
                pass

        val_loader_fn = getattr(datamodule, "val_dataloader", None)
        if val_loader_fn is None:
            self._restore_train_mode()
            return
        try:
            val_loader = val_loader_fn()
        except Exception as exc:  # pragma: no cover - defensive
            self._log(f"[MF] validation: val_dataloader() failed: {exc}")
            self._restore_train_mode()
            return
        if val_loader is None or (isinstance(val_loader, list) and not val_loader):
            # No validation set configured for this run. Silently skip.
            self._restore_train_mode()
            return

        max_val_batches = 64  # match Lightning's ``limit_val_batches`` default scale
        losses: List[float] = []

        self._log(
            f"[MF] Running validation sweep at step {current_step} "
            f"(max_val_batches={max_val_batches})"
        )
        val_start = time.monotonic()

        try:
            val_iter = iter(val_loader)
        except TypeError:
            self._restore_train_mode()
            return

        for _ in range(max_val_batches):
            try:
                batch = next(val_iter)
            except StopIteration:
                break
            except Exception as exc:  # pragma: no cover - defensive
                self._log(f"[MF] validation: dataloader error: {exc}")
                break

            if isinstance(batch, dict):
                input_ids = batch.get("input_ids")
            else:
                input_ids = batch
            if input_ids is None:
                continue

            with torch.no_grad():
                activations = embeds(input_ids)
            labels = input_ids[..., 1:].contiguous()
            block_ids = create_block_ids(input_ids, config.eos_token_id)

            # Hop through every actor sequentially. Each val_batch
            # call is a single ``ray.get`` - this is intentionally
            # un-pipelined, see docstring.
            last_loss: Optional[float] = None
            for layer_idx in range(num_layers):
                try:
                    result = self._safe_ray_get(
                        actors[layer_idx].val_batch.remote(
                            activations,
                            labels,
                            input_ids,
                            block_ids,
                        ),
                        ray,
                    )
                    if result is None:
                        self._restore_train_mode()
                        return
                except Exception as exc:  # pragma: no cover - defensive
                    self._log(
                        f"[MF] validation: actor {layer_idx} failed: {exc}"
                    )
                    last_loss = None
                    break
                activations = result["hidden_states"]
                last_loss = result["loss"]
            if last_loss is not None and math.isfinite(last_loss):
                losses.append(last_loss)

        if not losses:
            self._log("[MF] validation: no usable val batches, skipping log")
            self._restore_train_mode()
            return

        val_loss = sum(losses) / len(losses)
        # Perplexity = exp(val_loss) unless byte_latent mode, in which
        # case we emit bits_per_byte instead (matching
        # BackpropagationTrainer.validation_step).
        extra_val: Dict[str, Any] = {}
        if self.byte_latent:
            extra_val["val_bits_per_byte"] = val_loss / math.log(2.0)
        else:
            try:
                extra_val["val_perplexity"] = math.exp(val_loss)
            except OverflowError:
                extra_val["val_perplexity"] = float("inf")

        elapsed = time.monotonic() - val_start
        self._log(
            f"[MF] Validation done: val_loss={val_loss:.4f} "
            f"over {len(losses)} batches in {elapsed:.1f}s"
        )

        if metrics_logger is not None:
            try:
                metrics_logger.log(
                    step=int(current_step),
                    val_loss=float(val_loss),
                    **extra_val,
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._log(f"[MF] validation: metrics_logger.log failed: {exc}")

        # Push into LiveMetrics so the Terminal tab's val-loss chip
        # (if present) updates. The backprop path feeds this via
        # MetricsState.update_val().
        if self._live_metrics is not None:
            try:
                self._live_metrics.state.update_val(val_loss)
                self._live_metrics._update_count += 1
            except Exception:  # pragma: no cover - defensive
                pass

        self._restore_train_mode()

    def _restore_train_mode(self) -> None:
        """Switch the Terminal tab mode indicator back to "train".

        Called at every exit point of ``_run_validation`` so the
        dashboard never gets stuck showing "validation" after the
        sweep ends. Mirrors ``TerminalInterface.on_validation_end``
        under backprop.
        """
        if self._live_metrics is not None:
            try:
                self._live_metrics.state.set_mode("train")
                self._live_metrics._update_count += 1
            except Exception:
                pass

    # ------------------------------------------------------------------
    # web dashboard bridge (Phase 6)
    # ------------------------------------------------------------------

    def _init_live_metrics(self, num_layers: int) -> None:
        """Populate the :class:`LiveMetrics` singleton with static run state.

        Called once at ``fit()`` start. Mirrors what
        ``TerminalInterface.on_fit_start`` does on the backprop path:
        seeds the singleton with the run's identity (seed, hash, url,
        total params), the static info panel dict (optimizer name,
        vocab size, hidden size, etc.), and the start time for elapsed
        wall clock. Downstream updates are incremental via
        :meth:`_push_live_metrics_batch` and
        :meth:`_push_live_metrics_status`.

        Everything here is best-effort - a missing LiveMetrics
        import, a borked singleton, or any other failure must not
        interrupt training, so we wrap in a broad try/except and log
        the failure instead of raising.
        """
        try:
            from praxis.interface.state.live_metrics import LiveMetrics
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[MF] LiveMetrics bridge unavailable: {exc}")
            return

        try:
            lm = LiveMetrics()
            self._live_metrics = lm

            info = self.model_info or {}

            # Run identity + timing.
            if info.get("seed") is not None:
                lm.state.update_seed(info["seed"])
            if info.get("truncated_hash"):
                lm.state.arg_hash = info["truncated_hash"]
            if self.dashboard_url:
                lm.state.update_url(self.dashboard_url)
            if info.get("total_params"):
                lm.state.update_params(info["total_params"])
            from datetime import datetime

            lm.state.set_start_time(datetime.now())
            lm.state.set_mode("train")
            # Layer counts: under MF every layer is "local" to its own
            # actor, so we report the full count under ``local_layers``
            # and leave remote_layers at 0. A future multi-replica
            # deployment could populate remote_layers differently.
            lm.state.update_layer_count(num_layers, 0)

            # Static info-panel dict - the shape the frontend expects
            # matches what TerminalInterface._update_dashboard builds.
            info_dict: Dict[str, Any] = {
                "device": self.device,
                "optimizer": (info.get("optimizer_config") or {}).get(
                    "optimizer_name", "Unknown"
                ),
                "strategy": info.get("strategy"),
                "policy": info.get("rl_type"),
                "vocab_size": info.get("vocab_size"),
                # ``batch_size`` is the raw per-step batch; ``target_batch``
                # is the effective batch after gradient accumulation.
                # Under Mike these are 4 and 256 respectively. They must
                # come from different keys - pointing both at
                # ``target_batch_size`` is the bug that made the info
                # panel show ``Batch Size`` = ``Target Batch`` = 256.
                "batch_size": info.get("batch_size"),
                "target_batch": info.get("target_batch_size"),
                "depth": info.get("depth"),
                "local_layers": num_layers,
                "remote_layers": 0,
                "hidden_size": info.get("hidden_size"),
                "embed_size": info.get("embed_size"),
                "dropout": info.get("dropout"),
                "trainer": "mono_forward",
            }
            lm.state.update_info(info_dict)
            lm.info_dict = info_dict
            lm._update_count += 1
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[MF] LiveMetrics init failed: {exc}")
            self._live_metrics = None

    def _push_live_metrics_batch(
        self,
        batch_idx: int,
        avg_loss: float,
        avg_step_time: float,
        num_tokens_billions: float,
        num_layers: int,
        completed_batches: int,
    ) -> None:
        """Push a per-batch update to LiveMetrics.

        Runs after every final-layer hop (see ``_handle_completion``).
        Pushes the same fields TerminalInterface does under backprop,
        plus an EMA-smoothed loss for the ``loss`` scalar so the
        dashboard chart doesn't bounce between per-batch spikes.

        ``num_tokens_billions`` is in the same unit the backprop
        trainer reports: total processed tokens divided by 1e9. The
        frontend renders it as e.g. ``2.312B``.
        """
        if self._live_metrics is None:
            return
        try:
            alpha = 0.01
            if self._live_metrics_ema_loss is None:
                self._live_metrics_ema_loss = float(avg_loss)
            else:
                self._live_metrics_ema_loss = (
                    alpha * float(avg_loss)
                    + (1 - alpha) * self._live_metrics_ema_loss
                )
            state = self._live_metrics.state
            state.update_loss(self._live_metrics_ema_loss)
            # Dashboard UX convention: ``batch`` is the raw counter,
            # ``step`` is the effective-update counter. Under MF every
            # batch is a real per-layer optimizer step, but we report
            # step = completed_batches // accumulate_grad_batches to
            # match the Lightning TerminalInterface math (see
            # praxis/callbacks/lightning/terminal.py). The divisor is
            # 1 unless main.py passed a real ratio from --target-batch-size
            # / --batch-size.
            state.update_batch(completed_batches)
            state.update_step(completed_batches // self.accumulate_grad_batches)
            state.update_rate(float(avg_step_time or 0.0))
            state.update_tokens(float(num_tokens_billions))

            # Refresh RAM / VRAM in the info dict. The backprop
            # TerminalInterface does this on every batch via
            # ``self.get_memory_info(self.device)``; we match that
            # here. ``get_memory_info`` is a lightweight counter
            # read (psutil for RAM, torch.cuda for VRAM), so calling
            # it per final-layer-hop is fine. Reports the driver
            # node's memory - under multi-raylet MF that's the head
            # container, which is where both the model host copy and
            # the API server live.
            try:
                from praxis.utils import get_memory_info

                memory_info = get_memory_info(self.device)
                mem_update = {
                    "ram": f"{memory_info.get('ram_used', 'N/A')}/{memory_info.get('ram_total', 'N/A')}",
                }
                if self.device and self.device.startswith("cuda:"):
                    gpu_idx = int(self.device.split(":")[1])
                    gpu_reserved = f"gpu{gpu_idx}_reserved"
                    gpu_total = f"gpu{gpu_idx}_total"
                    if gpu_reserved in memory_info and gpu_total in memory_info:
                        mem_update["vram"] = (
                            f"{memory_info[gpu_reserved]}/{memory_info[gpu_total]}"
                        )
                    elif "gpu_status" in memory_info:
                        mem_update["vram"] = memory_info["gpu_status"]
                if mem_update:
                    state.update_info(mem_update)
                    self._live_metrics.info_dict.update(mem_update)
            except Exception:
                pass  # memory reporting is best-effort

            self._live_metrics._update_count += 1
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[MF] LiveMetrics batch push failed: {exc}")

    def _push_live_metrics_status(self, text: str) -> None:
        """Push the growing-text inference buffer to LiveMetrics.

        Feeds the Terminal tab's inference panel. Called from
        ``_maybe_run_inference_hook`` after the ``StreamingContext``
        update.
        """
        if self._live_metrics is None:
            return
        try:
            self._live_metrics.state.update_status(text)
            self._live_metrics.status_text = text
            self._live_metrics._update_count += 1
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[MF] LiveMetrics status push failed: {exc}")

    def _log(self, message: str) -> None:
        """Print + mirror into LiveMetrics log lines.

        Preserves stdout for operators running the compose stack in a
        foreground terminal, and also feeds the web Terminal tab's
        log stream. Best-effort on the LiveMetrics side: a push
        failure still prints normally.
        """
        print(message)
        if self._live_metrics is None:
            return
        try:
            self._live_metrics.add_log(message)
        except Exception:  # pragma: no cover - defensive
            pass

    # ------------------------------------------------------------------
    # live inference routed through the actor set
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        eos_token_id: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Iterator[torch.Tensor]:
        """Autoregressive generation routed through the active actors.

        Per Phase 5 Task 2, inference reuses the same LayerActor set
        that training is running against. Ray serializes method calls
        per actor, so an ``infer_batch`` submitted while a
        ``train_batch`` is in flight queues behind it and sees a
        consistent snapshot of actor state (either pre-step or
        post-step weights, never mid-update). This is the affordance
        that makes concurrent train+infer safe.

        Phase 5 explicitly accepts prefill-every-step cost: we
        recompute the entire prefix on every new token rather than
        managing a driver-owned KV cache. The KV-cache ownership
        question in PHASE_5.md is resolved by punting it to a later
        phase - the user flagged that "we don't care about the KV
        cache or recurrent state for now, really" for this phase.
        ``block_ids`` is the one LocalLayer kwarg that *is* load
        bearing, so we recompute it each step from the current prefix.

        Yields one-token tensors as they are produced (the streaming
        form Phase 6 HTTP plumbing will consume). Callers that want a
        batched return can ``torch.cat(list(trainer.generate(...)))``.

        Args:
            input_ids: Prompt token ids, shape ``[batch, seq_len]``.
                Must live on CPU (actors run CPU-only in Phase 5).
            max_new_tokens: Number of tokens to generate past the end
                of the prompt. Generation stops early if every row in
                the batch has emitted ``eos_token_id``.
            eos_token_id: Optional EOS id for early-stop. Defaults to
                ``self._config.eos_token_id`` if present. ``None`` means
                "never stop early; always produce ``max_new_tokens``".

        Yields:
            1-D (shape ``[batch]``) long tensors, one per decoded
            step.
        """
        if self._actors is None or self._embeds is None or self._config is None:
            raise RuntimeError(
                "MonoForwardTrainer.generate requires an active actor set. "
                "Call generate() while fit() is running (same process, same "
                "trainer instance)."
            )

        actors = self._actors
        embeds = self._embeds
        config = self._config
        num_layers = len(actors)

        if eos_token_id is None:
            eos_token_id = getattr(config, "eos_token_id", None)

        # Normalise eos_token_id to a set of ints for membership tests.
        eos_ids: Optional[List[int]] = None
        if eos_token_id is not None:
            if isinstance(eos_token_id, (list, tuple)):
                eos_ids = [int(x) for x in eos_token_id]
            else:
                eos_ids = [int(eos_token_id)]

        prefix = input_ids.detach().cpu().long()
        batch_size = prefix.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool)

        for _ in range(max_new_tokens):
            # Rebuild block_ids against the current prefix every step.
            # EOS-aware attention masking needs to reflect the prefix
            # we're about to run the forward on, not the original
            # prompt.
            block_ids = create_block_ids(prefix, config.eos_token_id)
            with torch.no_grad():
                activations = embeds(prefix)

            # Hop activations through the actor chain one layer at a
            # time, shoving block_ids into every hop. ``ray.get`` on
            # each hop serializes the pipeline for this single
            # inference request - pipelining multiple inference
            # requests is a Phase 6 concern.
            import ray  # local import to avoid hard dep at module load

            hidden = activations
            for layer_idx, actor in enumerate(actors):
                future = actor.infer_batch.remote(
                    hidden, layer_idx, block_ids, None
                )
                hidden, _kv = ray.get(future)

            # Project the final layer's hidden state through its head.
            # We use the final actor's head for now (the "use as-is"
            # branch of the open question in PHASE_5.md); measuring
            # consistency vs. a forced head-sync barrier is Phase 6+.
            logits = ray.get(actors[-1].project_logits.remote(hidden))

            # Decode the *last* position - we do a prefill each step so
            # the "next-token" logits live at index -1. Greedy by
            # default (deterministic for tests); the demo hook turns on
            # ``do_sample`` to avoid the "5 5 5 5 5 5" degenerate
            # greedy-decode pathology undertrained models exhibit.
            step_logits = logits[:, -1, :]
            if do_sample:
                if temperature != 1.0:
                    step_logits = step_logits / max(temperature, 1e-5)
                if top_k is not None and top_k > 0:
                    topk_vals, topk_idx = torch.topk(step_logits, top_k, dim=-1)
                    mask = torch.full_like(step_logits, float("-inf"))
                    mask.scatter_(-1, topk_idx, topk_vals)
                    step_logits = mask
                probs = torch.softmax(step_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = step_logits.argmax(dim=-1)

            # Early-stop: once a row has emitted EOS, freeze it.
            if eos_ids is not None:
                freshly_done = torch.zeros_like(finished)
                for tid in eos_ids:
                    freshly_done = freshly_done | (next_token == tid)
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, eos_ids[0]),
                    next_token,
                )
                finished = finished | freshly_done

            yield next_token.detach().clone()

            prefix = torch.cat([prefix, next_token.unsqueeze(-1)], dim=-1)
            if eos_ids is not None and bool(finished.all()):
                break

    def _maybe_run_inference_hook(
        self,
        completed_batches: int,
        config: Any,
    ) -> None:
        """Fire ``generate()`` periodically during training.

        Wired into the pipeline loop at every final-layer completion
        boundary. Cadence is gated on wall-clock seconds since the
        last firing, matching ``--infer-every`` semantics for the
        backprop TerminalInterface. ``inference_every_seconds=None``
        disables the hook entirely; ``0.0`` fires at every final-hop
        boundary.

        Mirrors the backprop ``TerminalInterface`` streaming pattern:
        the hook does NOT start fresh each fire. Instead it maintains
        a ``StreamingContext`` buffer seeded from the prompt, extends
        it by a few tokens per fire, and prints the growing passage
        so the user sees the model's output evolve. The context
        auto-resets to the seed when it detects stuck output, n-gram
        repetition, sequential repetition, or bracket-pipe
        degeneration.
        """
        if self.inference_prompt is None:
            return
        if self.inference_every_seconds is None or self.inference_every_seconds < 0:
            return

        now = time.monotonic()
        if self._last_inference_time is not None:
            if now - self._last_inference_time < self.inference_every_seconds:
                return
        self._last_inference_time = now

        context = self._ensure_inference_context()
        if context is None:
            return

        # Encode the growing text buffer into ids as the prompt for
        # this step. We intentionally re-encode every time rather than
        # appending ids directly, so the buffer's character-level
        # truncation and degeneracy resets all flow through cleanly.
        prompt_ids = self._encode_context_text(context.text)
        if prompt_ids is None:
            return

        # Push prompt token count into LiveMetrics so the Terminal
        # tab's "CONTEXT: X tokens" chip updates. Mirrors what
        # TerminalInterface._generate_text does at line 495-499.
        if self._live_metrics is not None and prompt_ids is not None:
            try:
                prompt_tokens = prompt_ids.shape[1]
                self._live_metrics.state.update_context_tokens(prompt_tokens)
            except Exception:
                pass

        # Match TerminalInterface._generate_text's token-count
        # distribution: default 1 token per fire, with a 10% chance
        # of drawing an extra (and compounding via the while loop),
        # capped at ``inference_max_new_tokens`` so a lucky streak
        # can't blow out the Terminal panel. Most fires emit exactly
        # one new token, occasional fires emit 2-3, very rarely 4+.
        import random

        draw = 1
        while random.random() < 0.1 and draw < self.inference_max_new_tokens:
            draw += 1

        tokens: List[torch.Tensor] = []
        # Sampling by default so an undertrained model doesn't collapse
        # into a single repeated token for the entire sample (the
        # pathological "5 5 5 5 5..." output under greedy decoding).
        for tok in self.generate(
            prompt_ids,
            max_new_tokens=draw,
            eos_token_id=getattr(config, "eos_token_id", None),
            do_sample=True,
            temperature=0.4,
            top_k=50,
        ):
            tokens.append(tok)
        if not tokens:
            return

        new_ids = torch.stack(tokens, dim=-1)
        full_ids = torch.cat([prompt_ids, new_ids], dim=-1)

        # Decode the full (prompt + continuation) sequence and feed
        # it back into the streaming buffer. The context handles
        # reset-on-degeneracy internally and returns True when a
        # reset fired (for optional logging).
        try:
            decoded = self.tokenizer.decode(
                full_ids[0].tolist(), skip_special_tokens=True
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(
                f"[MF] Inference hook @ batch {completed_batches}: "
                f"<decode-failed: {exc!r}>"
            )
            return

        did_reset = context.update(decoded)
        if did_reset:
            message = (
                f"[MF] Inference hook @ batch {completed_batches}: "
                f"(context reset) {context.text!r}"
            )
        else:
            message = (
                f"[MF] Inference hook @ batch {completed_batches}: "
                f"{context.text!r}"
            )
        self._log(message)

        # Phase 6: push the growing-text buffer into LiveMetrics so
        # the web Terminal tab's inference panel renders it. The
        # backprop TerminalInterface pushes its own ``self.text``
        # into ``live_metrics.status_text`` - same field here.
        self._push_live_metrics_status(context.text)

    def _ensure_inference_context(self):
        """Lazily construct the ``StreamingContext`` on first fire.

        Needs the tokenizer to be available (otherwise there's no way
        to encode the growing text buffer back into prompt ids each
        step). Returns ``None`` when the tokenizer is missing or when
        the prompt can't be resolved to any initial seed text.
        """
        if self._inference_context is not None:
            return self._inference_context
        if self.tokenizer is None:
            return None

        from praxis.generation import StreamingContext

        # Resolve the seed text from whatever shape the prompt came in
        # as. Strings pass through; id tensors get decoded.
        seed = self._resolve_prompt_as_text(self.inference_prompt)
        if seed is None:
            return None

        # Build the list of ignored n-grams the same way the backprop
        # TerminalInterface does - special tokens shouldn't count
        # against repetition detection since they appear legitimately
        # at sequence boundaries.
        ignored_n_grams = []
        for attr in ("bos_token", "eos_token", "pad_token", "sep_token"):
            tok = getattr(self.tokenizer, attr, None)
            if tok:
                ignored_n_grams.append(tok)

        self._inference_context = StreamingContext(
            initial_text=seed,
            max_length=self.inference_max_context_chars,
            ignored_n_grams=ignored_n_grams,
        )
        return self._inference_context

    def _resolve_prompt_as_text(self, prompt: Any) -> Optional[str]:
        """Convert any of the accepted prompt shapes to a plain string.

        Used to seed the ``StreamingContext``. Accepts strings, ``[seq]``
        / ``[batch, seq]`` id tensors, or plain lists of ids.
        """
        if isinstance(prompt, str):
            return prompt
        if self.tokenizer is None:
            return None
        if not isinstance(prompt, torch.Tensor):
            prompt = torch.as_tensor(prompt, dtype=torch.long)
        if prompt.dim() == 1:
            ids = prompt.tolist()
        else:
            ids = prompt[0].tolist()
        try:
            return self.tokenizer.decode(ids, skip_special_tokens=False)
        except Exception:  # pragma: no cover - defensive
            return None

    def _encode_context_text(self, text: str) -> Optional[torch.Tensor]:
        """Encode the growing text buffer into a ``[1, seq]`` id tensor."""
        if self.tokenizer is None:
            return None
        if not text:
            # Fallback: bos token id if available, otherwise bail.
            bos_id = getattr(self.tokenizer, "bos_token_id", None)
            if bos_id is None:
                return None
            return torch.tensor([[int(bos_id)]], dtype=torch.long)
        try:
            ids = self.tokenizer.encode(text)
        except Exception:  # pragma: no cover - defensive
            return None
        if not ids:
            return None
        return torch.as_tensor([ids], dtype=torch.long)

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, model_host: Any, actors: List[Any]) -> None:
        """Gather actor state and save a monolithic torch state_dict.

        Phase 2 wrote actor 0's head as the canonical head copy. Phase 3
        uses the averaged head produced by the last ``_sync_head`` call
        implicitly - every actor's head is identical right after a head
        sync. For checkpoints taken between sync boundaries, actor 0's
        head is still the right choice because the drift between actors
        over ``ray_head_sync_every`` batches is bounded by design.
        """
        import ray

        if self.cache_dir is None:
            self._log("[MF] No cache_dir provided; skipping checkpoint save")
            return

        self._log("[MF] Gathering actor state for checkpoint")
        layer_states = self._safe_ray_get(
            [actor.get_layer_state.remote() for actor in actors], ray
        )
        if layer_states is None:
            self._log("[MF] Checkpoint save aborted (actors unavailable)")
            return
        head_state = self._safe_ray_get(actors[0].get_head_state.remote(), ray)
        if head_state is None:
            self._log("[MF] Checkpoint save aborted (actors unavailable)")
            return

        for i, layer_state in enumerate(layer_states):
            model_host.decoder.locals[i].load_state_dict(layer_state)
        model_host.head.load_state_dict(head_state)

        os.makedirs(self.cache_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.cache_dir, "mono_forward.pt")
        torch.save(model_host.state_dict(), checkpoint_path)
        self._log(f"[MF] Saved checkpoint to {checkpoint_path}")

    # ------------------------------------------------------------------
    # unused BaseTrainer-ish methods (kept for future parity; explicit
    # NotImplementedError so callers crash loud if they rely on them).
    # ------------------------------------------------------------------

    def validate(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "mono_forward does not run validation in Phase 3. Use the "
            "standard backpropagation trainer to validate MF-trained "
            "checkpoints (they round-trip cleanly per D7)."
        )

    def test(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("mono_forward.test is not implemented")

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "mono_forward.predict is not implemented - inference uses "
            "the standard PraxisForCausalLM.forward path (decision D6)."
        )
