"""InProcessMonoForwardTrainer - single-process, single-CUDA-context MF trainer.

Functional twin of :class:`MonoForwardTrainer` that runs every
``LocalLayer`` inside the driver process rather than in a separate
Ray actor per layer. Eliminates the ~300-500MB-per-actor CUDA
context tax that makes the Ray backend infeasible for very deep
models on a single GPU.

Inherits from :class:`MonoForwardTrainer` so all the framework-
agnostic helpers (model validation, live-metrics bridge, log sink,
inference-context plumbing, prompt encoding) stay shared. Overrides
are scoped to the Ray-specific surface: ``fit``, the pipeline
driver, validation sweep, ``generate``, and checkpoint save/load.

Tradeoffs relative to the Ray backend:

- No multi-host / multi-raylet fan-out. This backend is single-host
  only. Multi-node training stays with the Ray trainer.
- No Python-level overlap between layer forwards. The driver is a
  single event loop that walks each batch through the depth chain
  synchronously. The throughput win comes from (a) eliminating the
  per-actor CUDA context, (b) direct on-device tensor handoff (no
  CPU round-trip), and (c) no Ray serialization / dispatch overhead.
- No ``ray.kill`` / ``ray.shutdown`` surface. Shutdown is just
  Python teardown; signal handlers still install cleanly.

``generate`` and periodic validation do walk the full depth chain,
so both remain functional. Checkpointing goes through the same
structured format the Ray trainer emits, so checkpoints round-trip
between backends cleanly.
"""

from __future__ import annotations

import copy
import math
import os
import time
from typing import Any, Dict, Iterator, List, Optional

import torch

from praxis.metrics.ema import STEP_TIME_EMA_ALPHA
from praxis.trainers.mono_forward.inprocess_worker import LocalLayerWorker
from praxis.trainers.mono_forward.trainer import MonoForwardTrainer
from praxis.utils import create_block_ids


def _patch_level_labels(
    input_ids: torch.Tensor, patch_lengths: torch.Tensor
) -> torch.Tensor:
    """Aggregate token-level ids into patch-level next-token labels.

    Byte-latent encoders produce activations at *patch* granularity:
    one hidden vector per patch, where each patch covers a variable
    number of bytes from ``input_ids``. The MF per-layer loss
    therefore needs labels at the same granularity. We aggregate by
    indexing the **first byte of each patch** out of ``input_ids``,
    then shift by one so each patch's hidden state predicts "the
    first byte of the next patch" - the natural next-step target at
    patch granularity.

    This is intentionally a lossy summary: the true backprop pipeline
    routes patch hidden states through ``encoder.decode``'s
    cross-attention to recover full byte-level logits. MF skips that
    decode (it'd be O(layers * decode-cost)) and trains each layer
    against the cheaper patch-level signal. Final inference still
    uses the real ``encoder.decode`` path; the patch-level loss is
    only what shapes the layer representations during MF training.

    Args:
        input_ids: ``[batch, seq_len]`` token ids.
        patch_lengths: ``[batch, num_patches]`` byte-count per patch
            (sums along ``dim=1`` to seq_len, modulo BOE padding the
            encoder may have added internally).

    Returns:
        ``[batch, num_patches - 1]`` patch-aligned token ids.
    """
    bsz = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    # Cumulative byte position at the *end* of each patch.
    patch_ends = patch_lengths.cumsum(dim=1)
    # Start of patch p is end of patch p-1; patch 0 starts at 0.
    patch_starts = torch.cat(
        [torch.zeros_like(patch_lengths[:, :1]), patch_ends[:, :-1]],
        dim=1,
    )
    # Encoders may add BOE/padding tokens beyond seq_len; clamp so
    # gather() never indexes past input_ids.
    patch_starts = patch_starts.clamp_max(seq_len - 1)
    first_byte_of_patch = torch.gather(input_ids, 1, patch_starts)
    # Predict the first byte of the *next* patch.
    return first_byte_of_patch[:, 1:].contiguous()


class InProcessMonoForwardTrainer(MonoForwardTrainer):
    """Single-process variant of :class:`MonoForwardTrainer`.

    Shares its ``__init__`` surface with the parent (same kwargs,
    same defaults). The ``ray_*`` knobs are accepted and ignored so
    experiment YAMLs and compose configs don't need per-backend
    editing; the only knob that actually changes behavior here is
    ``--mono-forward-backend`` (handled in the trainer factory).
    """

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
        """Train ``model`` against ``datamodule`` using in-process MF."""
        import signal

        self._shutdown_requested = False
        prev_sigint = signal.getsignal(signal.SIGINT)
        prev_sigterm = signal.getsignal(signal.SIGTERM)

        def _shutdown_handler(signum, frame):
            self._shutdown_requested = True
            self._log(
                f"[MF] Received signal {signum}, requesting graceful shutdown"
            )

        signal.signal(signal.SIGINT, _shutdown_handler)
        signal.signal(signal.SIGTERM, _shutdown_handler)

        self._validate_model(model)

        # Resolve the target device. Unlike the Ray backend which
        # forces everything to CPU before serializing out to actors,
        # the in-process backend keeps the full model on the target
        # device for the whole run - one CUDA context, zero IPC.
        target_device = torch.device(self.device if self.device else "cpu")
        try:
            model.to(target_device)
        except Exception:
            # Fall back to CPU if the requested device is unavailable
            # (common in CI). Match the Ray path's silent CPU fallback.
            target_device = torch.device("cpu")
            model.to(target_device)

        # Either ``embeds`` (token-id -> hidden lookup) or ``encoder``
        # (id -> patch hidden via a learned encode() that may also
        # downsample) feeds layer 0. ``_validate_model`` guarantees one
        # of the two is set; the pipeline's input-prep call below
        # picks the right one. Encoder mode is a "frozen-encoder" MF
        # path: encoder weights don't get gradient updates because each
        # MF layer's loss only flows back through that one layer.
        embeds = getattr(model, "embeds", None)
        encoder = model.encoder if getattr(model, "encoder", False) else None
        layers: List[Any] = list(model.decoder.locals)
        criterion = model.criterion
        strategy = getattr(model, "strategy", None)
        num_layers = len(layers)
        depth = model.config.depth

        if encoder is not None:
            # Encoder is co-trained with layer 0: layer 0's worker no
            # longer detaches its input when ``activations.requires_grad``
            # is set, so loss.backward() inside the worker propagates
            # straight through into encoder leaves. We then run a single
            # encoder-optimizer step per accumulation boundary, mirroring
            # the worker's accumulation logic. Keep encoder in train mode
            # (dropout etc. active) and grads on.
            encoder.train()
            for p in encoder.parameters():
                p.requires_grad_(True)

        # Depth->worker routing table. Identical to the Ray path so
        # recurrent-depth (depth > num_layers) runs route through the
        # worker set in the same cycle pattern.
        self._route_table = [i % num_layers for i in range(depth)]

        self._init_live_metrics(num_layers=num_layers)

        input_path = "encoder" if encoder is not None else "embeds"
        self._log(
            f"[MF] In-process training: num_layers={num_layers}, depth={depth}, "
            f"device={target_device}, input_path={input_path}, "
            f"max_steps={self.max_steps}, "
            f"val_check_interval={self.val_check_interval}"
        )
        if encoder is not None:
            self._log(
                "[MF] Encoder is co-trained with layer 0: gradients from "
                "layer 0's local loss flow back through the encoder, and "
                "a dedicated encoder optimizer steps once per accumulation "
                "boundary. Encoder aux_loss (if any) is added to the "
                "first-step gradient signal."
            )

        metrics_logger = None
        dynamics_logger = None
        if self.cache_dir:
            from praxis.logging.dynamics_logger import DynamicsLogger
            from praxis.logging.metrics_logger import MetricsLogger

            metrics_logger = MetricsLogger(run_dir=self.cache_dir)
            try:
                dynamics_logger = DynamicsLogger(run_dir=self.cache_dir)
                self._log(
                    f"[MF] DynamicsLogger writing to {self.cache_dir}/dynamics.db "
                    f"(log_freq={self.dynamics_log_freq})"
                )
            except Exception as exc:
                self._log(f"[MF] DynamicsLogger init failed: {exc}")
                dynamics_logger = None

        # Per-layer projection M_i has to predict in the *same* vocab
        # space the model's input adapter consumes - otherwise the
        # generated tokens fed back through the encoder hit an
        # out-of-range embedding lookup. For the embeds path, that's
        # ``config.vocab_size`` (the full tokenizer vocab). For the
        # encoder path, the encoder's ``tok_emb`` is sized for the
        # internal byte vocab (256 bytes + 4 special tokens = 260),
        # which is what ``encoder.byte_config.vocab_size`` exposes -
        # use that here so M_i's argmax can never produce an id
        # outside what ``encoder.encode`` can re-consume on the next
        # inference-hook fire.
        if encoder is not None:
            byte_config = getattr(encoder, "byte_config", None)
            projection_vocab_size = (
                int(byte_config.vocab_size)
                if byte_config is not None
                and getattr(byte_config, "vocab_size", None) is not None
                else int(model.config.vocab_size)
            )
            self._log(
                f"[MF] Encoder mode: per-layer projection M_i sized for "
                f"byte vocab = {projection_vocab_size} "
                f"(model.config.vocab_size = {model.config.vocab_size} "
                "is the *external* vocab and is not used by M_i in "
                "encoder mode)."
            )
        else:
            projection_vocab_size = int(model.config.vocab_size)

        # Build in-process workers. Each worker takes ownership of one
        # ``LocalLayer`` plus a fresh per-layer projection matrix; no
        # deepcopy is needed because ``model.decoder.locals`` already
        # holds distinct instances at distinct positions in the stack.
        workers: List[LocalLayerWorker] = []
        try:
            workers = [
                LocalLayerWorker(
                    layer_idx=i,
                    layer=layers[i],
                    criterion=copy.deepcopy(criterion),
                    hidden_size=model.config.hidden_size,
                    vocab_size=projection_vocab_size,
                    device=target_device,
                    strategy=copy.deepcopy(strategy) if strategy is not None else None,
                    optimizer_config=self.optimizer_config,
                    optimizer_wrappers=self.optimizer_wrappers,
                    warmup_steps=self.warmup_steps,
                    disable_schedule=self.disable_schedule,
                    num_layers=num_layers,
                    accumulate_grad_batches=self.accumulate_grad_batches,
                )
                for i in range(num_layers)
            ]
            self._log(f"[MF] Spawned {num_layers} LocalLayerWorker(s) in-process")

            self._actors = workers
            self._embeds = embeds
            self._encoder = encoder
            self._config = model.config

            # Build an optimizer + scheduler for the encoder when it's
            # in the picture. Mirrors the per-worker config so encoder
            # LR/wrappers/schedule track whatever the user asked for at
            # the trainer level - one source of truth, no extra knobs.
            self._encoder_optimizer = None
            self._encoder_scheduler = None
            if encoder is not None:
                from praxis.trainers.mono_forward._worker_common import (
                    build_optimizer, build_scheduler)

                class _EncoderShim(torch.nn.Module):
                    def __init__(self, enc: torch.nn.Module) -> None:
                        super().__init__()
                        self.encoder = enc

                encoder_shim = _EncoderShim(encoder)
                self._encoder_optimizer = build_optimizer(
                    shim=encoder_shim,
                    optimizer_config=self.optimizer_config,
                    wrappers=self.optimizer_wrappers or {},
                    fallback_lr=1e-3,
                    criterion=torch.nn.Identity(),
                    strategy=None,
                )
                self._encoder_scheduler = build_scheduler(
                    optimizer=self._encoder_optimizer,
                    optimizer_config=self.optimizer_config,
                    warmup_steps=self.warmup_steps,
                    disable_schedule=self.disable_schedule,
                )
                self._encoder_optimizer.zero_grad()
            # Stash a reference to the decoder's halting strategy (e.g.
            # ``halting_type: kl`` builds a ``KLDivergenceHalting`` with
            # randomized loop count during training and KL-based early
            # halting at inference). The base "none" halting always returns
            # ``config.depth`` so MF behavior is unchanged when
            # ``halting_type`` is not set.
            self._halting_strategy = getattr(
                model.decoder, "halting_strategy", None
            )
            if self._halting_strategy is not None:
                self._log(
                    f"[MF] Halting strategy in use: "
                    f"{type(self._halting_strategy).__name__}"
                )

            restored = {"completed_batches": 0, "num_tokens_total": 0}
            if ckpt_path is not None:
                restored = self._load_checkpoint_inprocess(
                    ckpt_path=ckpt_path,
                    workers=workers,
                    num_layers=num_layers,
                    datamodule=datamodule,
                )

            result = self._run_inprocess_pipeline(
                workers=workers,
                embeds=embeds,
                encoder=encoder,
                config=model.config,
                datamodule=datamodule,
                num_layers=num_layers,
                depth=depth,
                metrics_logger=metrics_logger,
                dynamics_logger=dynamics_logger,
                model_host=model,
                target_device=target_device,
                restored_batches=restored["completed_batches"],
                restored_tokens=restored["num_tokens_total"],
            )

            if result["completed_batches"] > 0 and not self._shutdown_requested:
                self._save_checkpoint_inprocess(
                    model_host=model,
                    workers=workers,
                    completed_batches=result["completed_batches"],
                    num_tokens_total=result.get("num_tokens_total", 0),
                    datamodule=datamodule,
                )

            return result

        finally:
            self._actors = None
            self._embeds = None
            self._encoder = None
            self._encoder_optimizer = None
            self._encoder_scheduler = None
            self._halting_strategy = None
            self._config = None
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)
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
            if target_device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # input preparation (token vs patch path)
    # ------------------------------------------------------------------

    def _prepare_inputs(
        self,
        input_ids_dev: torch.Tensor,
        embeds: Optional[torch.nn.Module],
        encoder: Optional[torch.nn.Module],
        config: Any,
        train_encoder: bool = False,
    ) -> tuple:
        """Build inputs for one batch.

        Returns ``(activations, labels, block_ids, aux_loss, decode_aux)``
        where ``decode_aux`` is ``None`` for the embeds path or a dict
        ``{h_encoder, patch_lengths, local_decoder_tokens}`` for the
        encoder path (so the trainer can run a separate decode-pass to
        train the encoder's output head).
        """
        if encoder is not None:
            if train_encoder:
                (
                    activations,
                    h_encoder,
                    patch_lengths,
                    block_ids,
                    aux_loss,
                    local_decoder_tokens,
                ) = encoder.encode(input_ids_dev)
            else:
                with torch.no_grad():
                    (
                        activations,
                        h_encoder,
                        patch_lengths,
                        block_ids,
                        aux_loss,
                        local_decoder_tokens,
                    ) = encoder.encode(input_ids_dev)
            labels = _patch_level_labels(input_ids_dev, patch_lengths)
            if not isinstance(aux_loss, torch.Tensor) or not aux_loss.requires_grad:
                aux_loss = None
            decode_aux = {
                "h_encoder": h_encoder,
                "patch_lengths": patch_lengths,
                "local_decoder_tokens": local_decoder_tokens,
            }
            return activations, labels, block_ids, aux_loss, decode_aux

        with torch.no_grad():
            activations = embeds(input_ids_dev)
        labels = input_ids_dev[..., 1:].contiguous()
        block_ids = create_block_ids(input_ids_dev, config.eos_token_id)
        return activations, labels, block_ids, None, None

    # ------------------------------------------------------------------
    # pipeline driver (in-process)
    # ------------------------------------------------------------------

    def _run_inprocess_pipeline(
        self,
        workers: List[LocalLayerWorker],
        embeds: Optional[torch.nn.Module],
        encoder: Optional[torch.nn.Module],
        config: Any,
        datamodule: Any,
        num_layers: int,
        depth: int,
        metrics_logger: Optional[Any],
        dynamics_logger: Optional[Any],
        model_host: Any,
        target_device: torch.device,
        restored_batches: int = 0,
        restored_tokens: int = 0,
    ) -> Dict[str, Any]:
        """Walk every batch through the depth chain.

        Without a multi-process actor pool we cannot overlap layer
        forwards across batches, so the steady-state shape here is
        "one batch at a time, all depth hops back-to-back". The
        bookkeeping still mirrors the Ray pipeline's
        ``loss_accumulator`` / ``dynamics_accumulator`` shape so the
        shared metrics emission paths stay unchanged.
        """
        route_table = self._route_table
        dataloader = datamodule.train_dataloader()
        dataloader_iter = iter(dataloader)

        state: Dict[str, Any] = {
            "first_loss": None,
            "last_loss": None,
            "loss_history": [],
            "per_layer_loss_history": {i: [] for i in range(depth)},
            "batches_started": restored_batches,
            "completed_batches": restored_batches,
            "dataloader_exhausted": False,
            "num_tokens_total": restored_tokens,
            "avg_step_time_ema": None,
            "last_softmax_collapse": None,
            "last_val_step": 0,
            "last_checkpoint_step": restored_batches,
        }
        wall_clock_start = time.monotonic()
        ema_alpha = STEP_TIME_EMA_ALPHA

        # Put the halting strategy in train mode so KL-style strategies
        # randomize their loop count per batch (vs. always-full-depth
        # eval mode). Restored to eval before validation / generate
        # below.
        halting_strategy = self._halting_strategy
        if halting_strategy is not None:
            halting_strategy.train()

        def _should_capture_dynamics(batch_idx: int) -> bool:
            if dynamics_logger is None:
                return False
            return batch_idx % max(self.dynamics_log_freq, 1) == 0

        while True:
            if self._shutdown_requested:
                self._log("[MF] Shutdown requested, exiting pipeline")
                break
            if (
                self.max_steps is not None
                and state["batches_started"] >= self.max_steps
            ):
                break
            if state["dataloader_exhausted"]:
                break

            try:
                batch = next(dataloader_iter)
            except StopIteration:
                state["dataloader_exhausted"] = True
                break

            if isinstance(batch, dict):
                input_ids = batch["input_ids"]
            else:
                input_ids = batch

            input_ids_dev = input_ids.to(target_device, non_blocking=True)
            activations, labels, block_ids, aux_loss, decode_aux = (
                self._prepare_inputs(
                    input_ids_dev,
                    embeds,
                    encoder,
                    config,
                    train_encoder=encoder is not None,
                )
            )

            # Bring encoder aux_loss into the gradient stream that
            # layer 0's worker is about to spin up. ``retain_graph=True``
            # keeps the shared encoder graph alive so the worker's own
            # backward (driven by layer 0's local loss flowing through
            # ``activations``) still has the nodes it needs. Encoder
            # grads from this call accumulate into the same param
            # buffers the worker will later add to.
            if aux_loss is not None:
                aux_loss.backward(retain_graph=True)

            batch_idx = state["batches_started"]
            state["batches_started"] += 1
            state["num_tokens_total"] += int(input_ids.numel())
            start_time = time.monotonic()
            capture_dyn = _should_capture_dynamics(batch_idx)

            layer_losses: Dict[int, float] = {}
            layer_dynamics: Dict[int, Dict[str, float]] = {}
            current_activations = activations
            current_lr = 0.0
            last_softmax_collapse: Optional[float] = None

            # Per-batch effective depth. With ``halting_type: kl`` this
            # samples a random loop count from a log-normal Poisson
            # (paper eqs. 1-2), so each batch may visit only a subset
            # of the full ``depth`` chain. With ``halting_type: none``
            # (the default) ``get_depth`` returns ``config.depth`` and
            # behavior is unchanged.
            effective_depth = (
                halting_strategy.get_depth() if halting_strategy is not None else depth
            )

            for step_idx in range(effective_depth):
                is_last_step = (step_idx + 1 == effective_depth)
                worker = workers[route_table[step_idx]]
                result = worker.train_batch(
                    current_activations,
                    labels,
                    batch_idx,
                    input_ids_dev,
                    block_ids,
                    capture_dyn,
                    is_last_step,
                )
                current_activations = result["hidden_states"]
                layer_losses[step_idx] = result["loss"]
                state["per_layer_loss_history"][step_idx].append(result["loss"])
                current_lr = result["lr"]
                if result.get("dynamics") is not None:
                    layer_dynamics[step_idx] = result["dynamics"]
                if result.get("softmax_collapse") is not None:
                    last_softmax_collapse = result["softmax_collapse"]

            if last_softmax_collapse is not None:
                state["last_softmax_collapse"] = last_softmax_collapse

            # Train the encoder's decode head + classifier. The MF
            # depth chain only updates per-layer projections M_i, so
            # encoder.decoder (the byte-level output transformer) and
            # encoder.classifier never see a gradient signal otherwise.
            # We feed the last MF layer's hidden state (already detached
            # by the worker, preserving MF isolation) into the encoder's
            # decode pipeline, take the byte-level next-token CE loss,
            # and backward. h_encoder is detached so the decode pass
            # only updates encoder.decoder + classifier - the local
            # encoder transformer is already getting grads via layer 0.
            decode_loss_value: Optional[float] = None
            if encoder is not None and decode_aux is not None:
                from torch.nn import functional as F

                final_hidden = current_activations.detach()
                h_encoder_detached = decode_aux["h_encoder"].detach()
                decode_logits, _ = encoder.decode(
                    final_hidden,
                    h_encoder_detached,
                    input_ids_dev,
                    decode_aux["patch_lengths"],
                    decode_aux["local_decoder_tokens"],
                )
                decode_targets = input_ids_dev[..., 1:].reshape(-1)
                decode_logits_flat = (
                    decode_logits[..., :-1, :]
                    .contiguous()
                    .view(-1, decode_logits.size(-1))
                )
                decode_loss = F.cross_entropy(
                    decode_logits_flat, decode_targets
                )
                (decode_loss / self.accumulate_grad_batches).backward()
                decode_loss_value = float(decode_loss.detach().item())

            # Encoder optimizer step at accum boundary. Layer 0's
            # worker has already accumulated encoder grads via the
            # input-grad path; aux_loss (when present) added more;
            # the decode pass above contributes grads to the decode
            # head + classifier. Mirror the worker's accumulation
            # cadence so encoder updates land in lockstep.
            if self._encoder_optimizer is not None:
                is_accum_boundary = (
                    state["completed_batches"] + 1
                ) % self.accumulate_grad_batches == 0
                if is_accum_boundary:
                    self._encoder_optimizer.step()
                    self._encoder_optimizer.zero_grad()
                    if self._encoder_scheduler is not None:
                        try:
                            self._encoder_scheduler.step()
                        except Exception:
                            pass

            # ---- finalize the batch ----
            state["completed_batches"] += 1
            avg_loss = sum(layer_losses.values()) / max(len(layer_losses), 1)
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
                decode_str = (
                    f"  decode={decode_loss_value:.4f}"
                    if decode_loss_value is not None
                    else ""
                )
                self._log(
                    f"[MF] batch {batch_idx:4d}  avg={avg_loss:.4f}{decode_str}  "
                    f"layers=[{per_layer_str}]"
                )

            num_tokens_billions = state["num_tokens_total"] / 1_000_000_000
            effective_step = state["completed_batches"] // self.accumulate_grad_batches
            is_accum_boundary = (
                state["completed_batches"] % self.accumulate_grad_batches == 0
            )

            if metrics_logger is not None and is_accum_boundary:
                extras: Dict[str, Any] = {"pipeline_in_flight": 1}
                for i, li_loss in layer_losses.items():
                    extras[f"layer_{i}_loss"] = float(li_loss)
                collapse_kwarg: Dict[str, Any] = {}
                if state["last_softmax_collapse"] is not None:
                    collapse_kwarg["softmax_collapse"] = float(
                        state["last_softmax_collapse"]
                    )
                metrics_logger.log(
                    step=effective_step,
                    loss=float(avg_loss),
                    num_tokens=float(num_tokens_billions),
                    avg_step_time=float(state["avg_step_time_ema"]),
                    learning_rate=float(current_lr),
                    batch=int(state["completed_batches"]),
                    **collapse_kwarg,
                    **extras,
                )

            if (
                dynamics_logger is not None
                and layer_dynamics
                and len(layer_dynamics) == effective_depth
            ):
                flat_dynamics: Dict[str, float] = {}
                for li, dyn in layer_dynamics.items():
                    for key, value in dyn.items():
                        flat_dynamics[f"layer_{li}_{key}"] = float(value)
                try:
                    dynamics_logger.log(step=effective_step, dynamics=flat_dynamics)
                except Exception as exc:
                    self._log(f"[MF] dynamics logger failed: {exc}")

            self._push_live_metrics_batch(
                batch_idx=batch_idx,
                avg_loss=avg_loss,
                avg_step_time=state["avg_step_time_ema"],
                num_tokens_billions=num_tokens_billions,
                num_layers=num_layers,
                completed_batches=state["completed_batches"],
            )

            # periodic checkpoint (step-based)
            batches_since = state["completed_batches"] - state["last_checkpoint_step"]
            if batches_since >= self.save_every:
                state["last_checkpoint_step"] = state["completed_batches"]
                self._save_checkpoint_inprocess(
                    model_host=model_host,
                    workers=workers,
                    completed_batches=state["completed_batches"],
                    num_tokens_total=state["num_tokens_total"],
                    datamodule=datamodule,
                )

            # periodic validation sweep
            current_step = state["completed_batches"] // self.accumulate_grad_batches
            next_val_at = state["last_val_step"] + (self.val_check_interval or 0)
            if (
                self.val_check_interval is not None
                and self.val_check_interval > 0
                and current_step > 0
                and current_step >= next_val_at
            ):
                val_step = (
                    current_step // self.val_check_interval
                ) * self.val_check_interval
                state["last_val_step"] = val_step
                self._run_validation_inprocess(
                    workers=workers,
                    embeds=embeds,
                    datamodule=datamodule,
                    config=config,
                    num_layers=num_layers,
                    current_step=val_step,
                    metrics_logger=metrics_logger,
                    target_device=target_device,
                )

            # periodic inference hook (wall-clock gated)
            self._maybe_run_inference_hook(
                completed_batches=state["completed_batches"],
                config=config,
            )

        total_wall = time.monotonic() - wall_clock_start
        self._log(
            f"[MF] Pipeline finished: {state['completed_batches']} batches in "
            f"{total_wall:.1f}s, start={state['first_loss']}, "
            f"end={state['last_loss']}"
        )

        return {
            "steps": state["completed_batches"],
            "completed_batches": state["completed_batches"],
            "num_tokens_total": state["num_tokens_total"],
            "first_loss": state["first_loss"],
            "final_loss": state["last_loss"],
            "loss_history": state["loss_history"],
            "per_layer_loss_history": state["per_layer_loss_history"],
            "pipeline_in_flight_max": 1,
        }

    # ------------------------------------------------------------------
    # validation sweep (in-process)
    # ------------------------------------------------------------------

    def _run_validation_inprocess(
        self,
        workers: List[LocalLayerWorker],
        embeds: torch.nn.Module,
        datamodule: Any,
        config: Any,
        num_layers: int,
        current_step: int,
        metrics_logger: Optional[Any],
        target_device: torch.device,
    ) -> None:
        """Sequential no-grad sweep through the worker chain."""
        if self._live_metrics is not None:
            try:
                self._live_metrics.state.set_mode("validation")
                self._live_metrics._update_count += 1
            except Exception:
                pass

        for worker in workers:
            worker.set_optimizer_eval()
        # KL-style halting strategies switch behavior between train (random
        # loop count) and eval (KL halting + full-depth ceiling).
        # Validation runs at full depth; flip to eval mode here and
        # back to train when validation finishes.
        halting_strategy = getattr(self, "_halting_strategy", None)
        if halting_strategy is not None:
            halting_strategy.eval()
        # Encoder shares dropout/noise plumbing with the rest of the
        # model; flip to eval for the validation sweep so the loss is
        # measured against deterministic encoder output.
        encoder_for_val = getattr(self, "_encoder", None)
        if encoder_for_val is not None:
            encoder_for_val.eval()

        val_loader_fn = getattr(datamodule, "val_dataloader", None)
        if val_loader_fn is None:
            self._restore_train_mode_inprocess(workers)
            return
        try:
            val_loader = val_loader_fn()
        except Exception as exc:
            self._log(f"[MF] validation: val_dataloader() failed: {exc}")
            self._restore_train_mode_inprocess(workers)
            return
        if val_loader is None or (isinstance(val_loader, list) and not val_loader):
            self._restore_train_mode_inprocess(workers)
            return

        max_val_batches = int(self._limit_val_batches)
        losses: List[float] = []
        self._log(
            f"[MF] Running validation sweep at step {current_step} "
            f"(max_val_batches={max_val_batches})"
        )
        val_start = time.monotonic()

        try:
            val_iter = iter(val_loader)
        except TypeError:
            self._restore_train_mode_inprocess(workers)
            return

        route = self._route_table
        for _ in range(max_val_batches):
            try:
                batch = next(val_iter)
            except StopIteration:
                break
            except Exception as exc:
                self._log(f"[MF] validation: dataloader error: {exc}")
                break

            if isinstance(batch, dict):
                input_ids = batch.get("input_ids")
            else:
                input_ids = batch
            if input_ids is None:
                continue

            input_ids_dev = input_ids.to(target_device, non_blocking=True)
            activations, labels, block_ids, _aux, _dec = self._prepare_inputs(
                input_ids_dev,
                getattr(self, "_embeds", None),
                getattr(self, "_encoder", None),
                config,
                train_encoder=False,
            )

            last_loss: Optional[float] = None
            for step_idx in range(len(route)):
                worker = workers[route[step_idx]]
                try:
                    result = worker.val_batch(
                        activations, labels, input_ids_dev, block_ids
                    )
                except Exception as exc:
                    self._log(
                        f"[MF] validation: step {step_idx} failed: {exc}"
                    )
                    last_loss = None
                    break
                activations = result["hidden_states"]
                last_loss = result["loss"]
            if last_loss is not None and math.isfinite(last_loss):
                losses.append(last_loss)

        if not losses:
            self._log("[MF] validation: no usable val batches, skipping log")
            self._restore_train_mode_inprocess(workers)
            return

        val_loss = sum(losses) / len(losses)
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
            except Exception as exc:
                self._log(f"[MF] validation: metrics_logger.log failed: {exc}")

        if self._live_metrics is not None:
            try:
                self._live_metrics.state.update_val(val_loss)
                self._live_metrics._update_count += 1
            except Exception:
                pass

        self._restore_train_mode_inprocess(workers)

    def _restore_train_mode_inprocess(
        self, workers: List[LocalLayerWorker]
    ) -> None:
        """Flip workers back to training mode after validation."""
        for worker in workers:
            try:
                worker.set_optimizer_train()
            except Exception:
                pass
        halting_strategy = getattr(self, "_halting_strategy", None)
        if halting_strategy is not None:
            halting_strategy.train()
        encoder = getattr(self, "_encoder", None)
        if encoder is not None:
            encoder.train()
        if self._live_metrics is not None:
            try:
                self._live_metrics.state.set_mode("train")
                self._live_metrics._update_count += 1
            except Exception:
                pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # live inference routed through the worker chain
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
        """Autoregressive generation routed through the in-process workers."""
        if self._actors is None or self._config is None:
            raise RuntimeError(
                "InProcessMonoForwardTrainer.generate requires an active "
                "worker set. Call generate() while fit() is running."
            )
        # ``_embeds`` is None for encoder-based models; ``_encoder`` is
        # None otherwise. Exactly one is set when fit() is active.
        if self._embeds is None and getattr(self, "_encoder", None) is None:
            raise RuntimeError(
                "InProcessMonoForwardTrainer.generate cannot find an "
                "input adapter (neither model.embeds nor model.encoder "
                "is registered on this trainer)."
            )

        workers = self._actors  # list[LocalLayerWorker] here
        embeds = self._embeds
        encoder = getattr(self, "_encoder", None)
        config = self._config

        if eos_token_id is None:
            eos_token_id = getattr(config, "eos_token_id", None)

        eos_ids: Optional[List[int]] = None
        if eos_token_id is not None:
            if isinstance(eos_token_id, (list, tuple)):
                eos_ids = [int(x) for x in eos_token_id]
            else:
                eos_ids = [int(eos_token_id)]

        # Infer the target device from the first worker.
        target_device = workers[0].device

        prefix = input_ids.detach().long().to(target_device)
        batch_size = prefix.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=target_device)

        # Encoder participates in training (dropout etc. on); flip to
        # eval for the duration of generation so the prefix is encoded
        # deterministically. Restored via try/finally so the iterator
        # always lands back in train mode even on early break / error.
        encoder_was_training = encoder is not None and encoder.training
        if encoder is not None:
            encoder.eval()

        try:
            for _ in range(max_new_tokens):
                # Re-prepare inputs every step. For the embeds path that's
                # cheap; for the encoder path that's a fresh patching pass
                # over the growing prefix (the patcher's boundaries shift
                # as new bytes arrive). Per-layer M_i then projects the
                # last patch's hidden state to "first byte of the next
                # patch" logits, which is exactly the next-byte prediction
                # we want at generation time.
                if encoder is not None:
                    with torch.no_grad():
                        (
                            activations,
                            _h_encoder,
                            _patch_lengths,
                            block_ids,
                            _aux_loss,
                            _local_decoder_tokens,
                        ) = encoder.encode(prefix)
                else:
                    block_ids = create_block_ids(prefix, config.eos_token_id)
                    with torch.no_grad():
                        activations = embeds(prefix)

                # KL-style halting at inference: ``seed`` captures the
                # baseline distribution from the pre-loop activations,
                # ``check`` is consulted after each layer hop and may
                # short-circuit the chain when the per-loop KL drop falls
                # below threshold. The "head" used for output distribution
                # is the last worker's projection M_i - any consistent
                # vocab projection works since check() is just measuring
                # distribution stability across consecutive loop boundaries.
                halting_strategy = getattr(self, "_halting_strategy", None)
                if halting_strategy is not None:
                    halting_strategy.eval()

                route = self._route_table
                last_worker = workers[route[-1]]
                kl_head = last_worker.projection
                if halting_strategy is not None:
                    with torch.no_grad():
                        halting_strategy.seed(activations, kl_head)

                hidden = activations
                visited_route: List[int] = []
                for step_idx in range(len(route)):
                    worker = workers[route[step_idx]]
                    hidden, _kv = worker.infer_batch(
                        hidden, step_idx, block_ids, None
                    )
                    visited_route.append(route[step_idx])
                    if halting_strategy is not None and halting_strategy.check(
                        hidden, step_idx, kl_head
                    ):
                        break

                # Mirror BaseController.post_forward's debug print so users
                # can see which layers a token actually traversed - critical
                # for verifying that a halting strategy (e.g. KL) is firing.
                # Gated on batch_size == 1 to match the standard path.
                if (
                    getattr(config, "debug", False)
                    and prefix.size(0) == 1
                ):
                    route_str = " -> ".join(str(r) for r in visited_route)
                    print(f"DEBUG: inferencing through:  {route_str}")

                logits = last_worker.project_logits(hidden)

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

                # Return CPU tokens so downstream consumers (terminal
                # interface, StreamingContext) don't have to juggle devices.
                yield next_token.detach().cpu().clone()

                prefix = torch.cat([prefix, next_token.unsqueeze(-1)], dim=-1)
                if eos_ids is not None and bool(finished.all()):
                    break
        finally:
            if encoder is not None and encoder_was_training:
                encoder.train()

    # ------------------------------------------------------------------
    # checkpointing (in-process)
    # ------------------------------------------------------------------

    def _load_checkpoint_inprocess(
        self,
        ckpt_path: str,
        workers: List[LocalLayerWorker],
        num_layers: int,
        datamodule: Any = None,
    ) -> Dict[str, Any]:
        """Load a MF checkpoint (structured or legacy) into the workers."""
        self._log(f"[MF] Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        is_structured = (
            isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        )
        if is_structured:
            model_sd = checkpoint["model_state_dict"]
        else:
            model_sd = checkpoint
            self._log(
                "[MF] Legacy checkpoint format detected (model weights only)"
            )

        layer_prefix = "decoder.locals."
        layer_state_dicts: Dict[int, Dict[str, Any]] = {}
        for key, value in model_sd.items():
            if not key.startswith(layer_prefix):
                continue
            rest = key[len(layer_prefix) :]
            dot_pos = rest.index(".")
            layer_idx = int(rest[:dot_pos])
            inner_key = rest[dot_pos + 1 :]
            if layer_idx not in layer_state_dicts:
                layer_state_dicts[layer_idx] = {}
            layer_state_dicts[layer_idx][inner_key] = value

        loaded = 0
        for i in range(num_layers):
            if i in layer_state_dicts:
                workers[i].load_layer_state(layer_state_dicts[i])
                loaded += 1
        if loaded:
            self._log(f"[MF] Distributed layer weights to {loaded} worker(s)")

        del model_sd
        del layer_state_dicts

        if is_structured:
            proj_states = checkpoint.get("projection_states")
            if proj_states and len(proj_states) == num_layers:
                for i in range(num_layers):
                    workers[i].load_projection_state(proj_states[i])
                self._log("[MF] Restored per-worker projection states")

            opt_states = checkpoint.get("optimizer_states")
            if opt_states and len(opt_states) == num_layers:
                for i in range(num_layers):
                    workers[i].load_optimizer_state(opt_states[i])
                self._log("[MF] Restored per-worker optimizer states")

            sched_states = checkpoint.get("scheduler_states")
            if sched_states and len(sched_states) == num_layers:
                for i in range(num_layers):
                    workers[i].load_scheduler_state(sched_states[i])
                self._log("[MF] Restored per-worker scheduler states")

            enc_opt_state = checkpoint.get("encoder_optimizer_state")
            if enc_opt_state is not None and self._encoder_optimizer is not None:
                try:
                    self._encoder_optimizer.load_state_dict(enc_opt_state)
                    self._log("[MF] Restored encoder optimizer state")
                except Exception as exc:
                    self._log(
                        f"[MF] Warning: could not restore encoder optimizer: {exc}"
                    )
            enc_sched_state = checkpoint.get("encoder_scheduler_state")
            if (
                enc_sched_state is not None
                and self._encoder_scheduler is not None
            ):
                try:
                    self._encoder_scheduler.load_state_dict(enc_sched_state)
                except Exception:
                    pass

            dm_state = checkpoint.get("datamodule_state")
            if dm_state is not None and datamodule is not None:
                if hasattr(datamodule, "load_state_dict"):
                    try:
                        datamodule.load_state_dict(dm_state)
                        self._log(
                            "[MF] Restored datamodule state (dataset positions)"
                        )
                    except Exception as exc:
                        self._log(
                            f"[MF] Warning: could not restore datamodule state: {exc}"
                        )

        restored_batches = 0
        restored_tokens = 0
        if is_structured:
            restored_batches = checkpoint.get("completed_batches", 0)
            restored_tokens = checkpoint.get("num_tokens_total", 0)

        del checkpoint

        self._log(
            f"[MF] Checkpoint loaded (resuming from batch "
            f"{restored_batches}, {restored_tokens} tokens)"
        )
        return {
            "completed_batches": restored_batches,
            "num_tokens_total": restored_tokens,
        }

    def _save_checkpoint_inprocess(
        self,
        model_host: Any,
        workers: List[LocalLayerWorker],
        completed_batches: int = 0,
        num_tokens_total: int = 0,
        datamodule: Any = None,
    ) -> None:
        """Gather worker state and save a structured checkpoint."""
        if self.cache_dir is None:
            self._log("[MF] No cache_dir provided; skipping checkpoint save")
            return

        self._log("[MF] Gathering worker state for checkpoint")
        layer_states = [w.get_layer_state() for w in workers]
        projection_states = [w.get_projection_state() for w in workers]
        optimizer_states = [w.get_optimizer_state() for w in workers]
        scheduler_states = [w.get_scheduler_state() for w in workers]

        for i, layer_state in enumerate(layer_states):
            model_host.decoder.locals[i].load_state_dict(layer_state)

        last_proj = projection_states[-1]
        head_mapped = {"lm_head." + k: v for k, v in last_proj.items()}
        # Encoder-based models don't have a top-level ``model.head`` -
        # the encoder owns its own decode projection. In that case the
        # MF projection M_i is the source of truth on the per-layer
        # side; we just save it and skip the head overwrite.
        if getattr(model_host, "head", None) is not None:
            try:
                model_host.head.load_state_dict(head_mapped)
            except Exception:
                # Model heads that don't share the ``lm_head.`` key shape
                # (tied weights, custom projections) can't be reconstructed
                # from M_i. We skip the head overwrite instead of crashing
                # the save - the per-worker projection_states are still
                # the source of truth for MF resume.
                pass

        datamodule_state = None
        if datamodule is not None and hasattr(datamodule, "state_dict"):
            try:
                datamodule_state = datamodule.state_dict()
            except Exception as exc:
                self._log(
                    f"[MF] Warning: could not save datamodule state: {exc}"
                )

        encoder_optimizer_state = None
        encoder_scheduler_state = None
        if self._encoder_optimizer is not None:
            try:
                raw = self._encoder_optimizer.state_dict()
                encoder_optimizer_state = {}
                for k, v in raw.items():
                    if k == "state":
                        encoder_optimizer_state[k] = {
                            pid: {
                                sk: (sv.detach().cpu()
                                     if isinstance(sv, torch.Tensor) else sv)
                                for sk, sv in pstate.items()
                            }
                            for pid, pstate in v.items()
                        }
                    else:
                        encoder_optimizer_state[k] = v
            except Exception as exc:
                self._log(f"[MF] Warning: could not save encoder optimizer: {exc}")
        if self._encoder_scheduler is not None:
            try:
                encoder_scheduler_state = self._encoder_scheduler.state_dict()
            except Exception:
                pass

        checkpoint = {
            "model_state_dict": model_host.state_dict(),
            "projection_states": projection_states,
            "optimizer_states": optimizer_states,
            "scheduler_states": scheduler_states,
            "encoder_optimizer_state": encoder_optimizer_state,
            "encoder_scheduler_state": encoder_scheduler_state,
            "completed_batches": completed_batches,
            "num_tokens_total": num_tokens_total,
            "datamodule_state": datamodule_state,
        }

        os.makedirs(self.cache_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.cache_dir, "mono_forward.pt")
        torch.save(checkpoint, checkpoint_path)
        self._log(
            f"[MF] Saved checkpoint to {checkpoint_path} "
            f"(batch {completed_batches}, {num_tokens_total} tokens)"
        )
