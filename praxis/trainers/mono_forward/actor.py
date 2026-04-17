"""LayerActor - a stateful Ray worker that owns one Mono-Forward layer.

Each actor owns exactly one ``LocalLayer`` plus a replicated copy of the
shared output head, its own optimizer over (layer + head) params, and
entry points for training (``train_batch``) and inference
(``infer_batch``) over that layer.

The training step routes its loss through
:func:`praxis.losses.compute_layer_wise_loss`, which calls the model's
real ``criterion`` (so ``--loss-func cut_cross_entropy`` is honoured and
D5 aux-loss folding fires). The inference step is a no-grad forward
that shares actor state with training - Ray serializes method calls on
an actor by default, so an ``infer_batch`` submitted while a
``train_batch`` is in flight queues behind it and sees a consistent
snapshot of the layer weights.

Ray is the current distributed backend for hosting workers, but the
trainer surface and the math are framework-agnostic; a future port to
Lightning, native ``torch.distributed``, Hivemind, or monarch would
replace this module without touching the trainer. Ray is imported at
module load time here because this module is only imported when the
Ray backend is actually in use - the trainer's ``fit`` method does
``from praxis.trainers.mono_forward.actor import LayerActor`` lazily
inside its Ray-gated code path, not at ``praxis`` import time. If Ray
is missing, the trainer surfaces a clear ``ImportError`` before we
ever try to import this module.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import ray
import torch
import torch.nn as nn

from praxis.losses.layer_wise import compute_layer_wise_loss
from praxis.metrics import compute_softmax_collapse, extract_layer_dynamics
from praxis.trainers.mono_forward._worker_common import \
    ActorParamShim as _ActorParamShim
from praxis.trainers.mono_forward._worker_common import \
    build_optimizer as _build_optimizer
from praxis.trainers.mono_forward._worker_common import \
    build_scheduler as _build_scheduler
from praxis.trainers.mono_forward.device import deep_to as _deep_to


@ray.remote(num_cpus=1, max_restarts=0)
class LayerActor:
    """Stateful Ray actor owning one LocalLayer + replicated head copy.

    The driver constructs actors by deep-copying the corresponding
    ``LocalLayer`` and ``head`` instances from the host-side model, so
    every actor has its own independent params from initialisation. Ray
    pickles those objects when passing them into ``.remote(...)``, which
    is equivalent to a second deepcopy on the worker side - the host
    model is untouched by anything the actor does thereafter.

    Ray resource decoration:

    - ``num_cpus=1``: each actor reserves one logical CPU slot on its
      raylet. This matters for Phase 4's multi-raylet test - with two
      raylets of ``num_cpus=2`` and four MF layers, the scheduler is
      forced to spread actors across both raylets because a single
      raylet can only host two of them. On a single-raylet run the
      scheduler is free to pack all actors onto the driver's local
      raylet, which is fine (Ray's plasma store handles the local
      tensor transport zero-copy anyway).
    - ``max_restarts=0``: fail-loud on actor death, per the locked
      decision in PROJECT_PLAN.md. If a raylet dies or an actor
      crashes, subsequent ``.remote()`` calls raise ``RayActorError``
      and the trainer propagates the error to the driver, where it
      turns into a Python exception that terminates the process.
      Systemd (on Platformer) or the user (on localhost) restarts the
      whole training run from the last checkpoint.
    """

    def __init__(
        self,
        layer_idx: int,
        layer: nn.Module,
        criterion: nn.Module,
        hidden_size: int,
        vocab_size: int,
        strategy: Optional[Any] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        optimizer_wrappers: Optional[Dict[str, bool]] = None,
        warmup_steps: int = 0,
        disable_schedule: bool = False,
        lr: float = 1e-3,
        num_layers: int = 1,
        accumulate_grad_batches: int = 1,
    ) -> None:
        self.layer_idx = layer_idx

        # Decide compute device. Ray assigns fractional GPU resources
        # to actors when the driver passes ``num_gpus > 0`` via
        # ``.options()``. If this actor got a GPU slice, use it;
        # otherwise train on CPU. The driver always sends CPU tensors
        # (``_force_cpu`` in the trainer ensures this), so we
        # deep-copy to CPU first and then move to the target device
        # so non-registered tensor attributes (RoPE caches, etc.)
        # don't carry stale CUDA metadata from the driver.
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.layer = _deep_to(copy.deepcopy(layer), self.device)
        self.criterion = _deep_to(copy.deepcopy(criterion), self.device)
        if strategy is not None:
            s = copy.deepcopy(strategy)
            self.strategy = _deep_to(s, self.device) if isinstance(s, nn.Module) else s
        else:
            self.strategy = None

        # Per-layer projection matrix M_i (paper Section 3.1). Each
        # layer gets its own independent projection - there is no
        # weight sharing or synchronisation between layers. The
        # goodness score is G_i = a_i @ M_i^T and the layer loss is
        # L_i = CE(softmax(G_i), labels). This is a fresh random
        # init, not a copy of the model's output head.
        from praxis.trainers.mono_forward.projection import ProjectionMatrix

        self.projection = ProjectionMatrix(hidden_size, vocab_size).to(self.device)
        self.layer.train()
        self.projection.train()

        # Build the optimizer. The actor owns per-layer + projection
        # params via a thin ``nn.Module`` shim so that
        # :func:`praxis.optimizers.get_optimizer` sees the same
        # ``named_parameters`` / ``named_modules`` surface it uses for
        # weight-decay banning under the backprop path.
        self._param_shim = _ActorParamShim(self.layer, self.projection)
        self.optimizer = _build_optimizer(
            shim=self._param_shim,
            optimizer_config=optimizer_config,
            wrappers=optimizer_wrappers or {},
            fallback_lr=lr,
            criterion=self.criterion,
            strategy=self.strategy,
        )
        # Build the LR scheduler the same way main.py does for the
        # backprop path - reconstruct ``scheduler_func`` locally from
        # the raw config dict and apply it to this actor's optimizer.
        # ``None`` when no optimizer_config was supplied (tests) or
        # when schedule construction fails; ``train_batch`` no-ops
        # the ``scheduler.step()`` in that case.
        self.scheduler = _build_scheduler(
            optimizer=self.optimizer,
            optimizer_config=optimizer_config,
            warmup_steps=warmup_steps,
            disable_schedule=disable_schedule,
        )

        # Record whether this actor owns the final layer of the decoder.
        # Used by ``train_batch`` to decide whether to compute the
        # softmax_collapse metric (same as the backprop trainer's
        # ``outputs.logits``-based computation - only the final-layer
        # logits matter for the "Grokking at the Edge of Stability"
        # collapse signal).
        self.num_layers = int(num_layers)
        # Gradient accumulation. The paper doesn't discuss it
        # explicitly, but with batch_size=4 and target_batch_size=256,
        # stepping after every single batch produces a very noisy
        # training signal. Accumulating over ``accumulate_grad_batches``
        # mini-batches before stepping matches the effective batch size
        # the backprop trainer uses, giving each layer the same
        # gradient signal quality. Loss is scaled by 1/N during
        # backward so the accumulated gradient is the mean over the
        # accumulation window (same as a single batch of size N).
        self.accumulate_grad_batches = max(int(accumulate_grad_batches), 1)

        # Stats, mostly for operational visibility in ray dashboard logs.
        self.batches_processed = 0

    def train_batch(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        batch_idx: int,
        input_ids: Optional[torch.Tensor] = None,
        block_ids: Optional[torch.Tensor] = None,
        capture_dynamics: bool = False,
        is_last_step: bool = False,
    ) -> Dict[str, Any]:
        """Run one MF forward/local-update pass on this layer.

        Returns a dict of:
          - ``hidden_states``: detached post-layer activations for the
            next actor in the chain.
          - ``loss``: scalar loss value (float).
          - ``batch_idx``: echoed back for driver-side bookkeeping.
          - ``lr``: current learning rate (metrics.db logging).
          - ``softmax_collapse``: final-layer softmax-collapse metric
            from "Grokking at the Edge of Stability" - ``None`` unless
            this actor owns the final layer.
          - ``dynamics``: per-layer grad-norm / grad-var / update-ratio
            dict (same math as the backprop ``DynamicsLoggerCallback``)
            when ``capture_dynamics=True``. ``None`` otherwise. The
            driver only passes ``True`` at log_freq cadence to amortize
            the overhead.

        Returning a dict instead of a tuple makes future additions
        non-breaking: the driver reads fields by name, so adding a new
        per-batch metric is a one-field change at both ends. This
        replaces the Phase 2/3 tuple that kept growing every time a
        new column appeared.
        """
        # (1) Move incoming CPU tensors to this actor's device and
        # detach so no gradient crosses the MF boundary.
        h = activations.to(self.device).detach().requires_grad_(True)
        labels_dev = labels.to(self.device)
        block_ids_dev = block_ids.to(self.device) if block_ids is not None else None
        input_ids_dev = input_ids.to(self.device) if input_ids is not None else None

        # (2) Layer forward. LocalLayer.forward signature (see
        # praxis/layers/local.py): inputs, attention_mask, past_key_values,
        # current_state, current_depth, block_ids. Per PHASE_5.md the
        # training path does not use KV cache or recurrent state, but
        # block_ids *is* load-bearing for attention masking across
        # special token boundaries.
        h_out, _kv, _state, aux_loss, _exit = self.layer(
            h,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=self.layer_idx,
            block_ids=block_ids_dev,
        )

        # (3) Compute the local loss via the framework-agnostic helper.
        # This routes through self.criterion (so cut-CE fires correctly)
        # and folds aux losses via self.strategy (D5).
        loss = compute_layer_wise_loss(
            hidden_states=h_out,
            labels=labels_dev,
            head=self.projection,
            criterion=self.criterion,
            strategy=self.strategy,
            aux_losses=[aux_loss] if aux_loss is not None else None,
            input_ids=input_ids_dev,
        )

        # (4) Gradient accumulation. Scale the loss by 1/N so the
        # accumulated gradient over N mini-batches is the mean (same
        # as a single batch of size N). Backward accumulates into
        # the existing .grad tensors. Only zero_grad + step at the
        # accumulation boundary.
        scaled_loss = loss / self.accumulate_grad_batches
        scaled_loss.backward()

        # (4a) Capture dynamics metrics while gradients are populated
        # (before the potential zero_grad at the step boundary).
        dynamics = self._capture_dynamics() if capture_dynamics else None

        # (4b) Step only at accumulation boundaries.
        is_accum_boundary = (
            self.batches_processed + 1
        ) % self.accumulate_grad_batches == 0
        if is_accum_boundary:
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except Exception:
                    pass

        # (5) Final-layer softmax collapse. Only the last actor
        # computes this, matching the backprop trainer where
        # ``outputs.logits`` is the final layer's projection. For
        # non-cut-CE we re-project (one extra head() call on the last
        # layer only, amortized across the full pipeline). For cut-CE
        # the logits are never materialized by the loss helper so
        # this is the only place they exist.
        softmax_collapse: Optional[float] = None
        if is_last_step:
            softmax_collapse = self._compute_softmax_collapse(h_out)

        # (6) Hand the detached post-layer activations to the driver.
        # Always return CPU tensors - Ray serializes them for
        # transport and the next actor (which may be on a different
        # host) will ``.to(self.device)`` on receipt.
        self.batches_processed += 1
        return {
            "hidden_states": h_out.detach().cpu(),
            "loss": float(loss.detach().item()),
            "batch_idx": batch_idx,
            "lr": float(self._current_lr()),
            "softmax_collapse": softmax_collapse,
            "dynamics": dynamics,
        }

    def val_batch(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        block_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """No-grad forward + local loss, for the validation loop.

        Runs the same shape as :meth:`train_batch` but skips the
        backward pass, the optimizer step, and the scheduler tick, so
        no actor state changes. Returns the detached hidden states
        (so the driver can hop to the next actor) and the scalar
        local loss value.

        The driver aggregates the last-layer loss into ``val_loss``
        and computes ``val_perplexity = exp(val_loss)`` the same way
        the backprop trainer does in ``validation_step``.
        """
        h = activations.to(self.device)
        labels_dev = labels.to(self.device)
        block_ids_dev = block_ids.to(self.device) if block_ids is not None else None
        input_ids_dev = input_ids.to(self.device) if input_ids is not None else None

        with torch.no_grad():
            h_out, _kv, _state, aux_loss, _exit = self.layer(
                h,
                attention_mask=None,
                past_key_values=None,
                current_state=None,
                current_depth=self.layer_idx,
                block_ids=block_ids_dev,
            )
            h_detached = h_out.detach()

        with torch.no_grad():
            loss = compute_layer_wise_loss(
                hidden_states=h_detached,
                labels=labels_dev,
                head=self.projection,
                criterion=self.criterion,
                strategy=self.strategy,
                aux_losses=[aux_loss] if aux_loss is not None else None,
                input_ids=input_ids_dev,
            )

        return {
            "hidden_states": h_detached.cpu(),
            "loss": float(loss.detach().item()),
        }

    def _compute_softmax_collapse(self, hidden_states: torch.Tensor) -> Optional[float]:
        """Compute softmax collapse from final-layer hidden states.

        Projects to logits via ``self.projection`` then delegates to the
        shared :func:`~praxis.metrics.compute_softmax_collapse`. Returns
        ``None`` on any failure - metrics must never interrupt training.
        """
        try:
            with torch.no_grad():
                logits = self.projection(hidden_states.detach())
                return compute_softmax_collapse(logits)
        except Exception:
            return None

    def _capture_dynamics(self) -> Optional[Dict[str, float]]:
        """Capture grad_norm / grad_var / update_ratio for this layer.

        Delegates to :func:`~praxis.metrics.extract_layer_dynamics`.
        The driver aggregates per-layer dicts from every actor into a
        single ``{layer_{i}_grad_norm: ...}`` dict and writes it to
        ``dynamics.db`` via :class:`DynamicsLogger`.
        """
        try:
            return extract_layer_dynamics(self.layer, self._current_lr())
        except Exception:
            return None

    def _current_lr(self) -> float:
        """Return the first param group's current learning rate.

        Best-effort: if the optimizer was wrapped (TRAC, Lookahead,
        etc.) or somehow lost its ``param_groups`` attribute, fall
        back to 0.0 so the driver's metrics.db logging never raises
        mid-batch on a reporting-only concern.
        """
        try:
            groups = getattr(self.optimizer, "param_groups", None)
            if groups:
                return float(groups[0].get("lr", 0.0))
        except Exception:
            pass
        return 0.0

    def infer_batch(
        self,
        activations: torch.Tensor,
        current_depth: Optional[int] = None,
        block_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """No-grad forward for inference. Does not mutate any state.

        Shares actor state with :meth:`train_batch`: Ray serializes
        method calls per actor, so an ``infer_batch`` submitted mid-run
        queues behind any in-flight ``train_batch`` and sees a consistent
        snapshot of this actor's weights.

        Per PHASE_5.md we intentionally ignore the KV cache (option (a)
        in the design doc says the driver owns it but Phase 5 accepts
        prefill-every-step cost). The ``past_key_values`` parameter is
        accepted and threaded into the underlying ``LocalLayer`` for
        forward-compatibility with a future caching path; callers that
        don't need it can just pass ``None``.

        Args:
            activations: Current hidden states, shape
                ``[batch, seq_len, hidden_size]``. For the final layer
                only the last position's logits are typically used, but
                this method returns the full hidden state - the driver
                decides what to do with it.
            current_depth: Optional override for the depth index passed
                into the underlying layer; defaults to this actor's
                ``layer_idx`` if ``None``.
            block_ids: Optional block-id tensor for attention masking
                across special tokens (same shape conventions as the
                training path).
            past_key_values: Optional KV cache (currently forwarded
                through and ignored by the Phase 5 path).

        Returns:
            A ``(hidden_states, new_past_key_values)`` tuple. The cache
            entry is typically ``None`` under the Phase 5 option (a);
            future work may populate it.
        """
        depth = current_depth if current_depth is not None else self.layer_idx
        # Snapshot training mode so we can restore it if we ever run
        # against an actor that's been toggled to ``eval()`` externally.
        # Under Phase 5 training runs leave ``self.layer.train()`` set
        # so dropout-equipped layers still sample at inference time - we
        # intentionally do NOT flip to eval here because doing so would
        # mutate shared actor state and race with concurrent
        # ``train_batch`` calls. Instead we rely on ``torch.no_grad()``
        # to keep the compute cheap; any dropout noise is tolerated.
        h = activations.to(self.device)
        block_ids_dev = block_ids.to(self.device) if block_ids is not None else None
        with torch.no_grad():
            h_out, new_kv, _state, _aux, _exit = self.layer(
                h,
                attention_mask=None,
                past_key_values=past_key_values,
                current_state=None,
                current_depth=depth,
                block_ids=block_ids_dev,
            )
        return h_out.detach().cpu(), new_kv

    def project_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states through this actor's replicated head.

        Used by the driver's ``generate`` method to turn the final
        layer's post-forward hidden state into token logits. Lives on
        the actor rather than the driver because the head weights
        may have drifted between sync boundaries (per the D2b head
        replication decision) and we want to use *this actor's*
        canonical copy for consistency with what it trained on.
        """
        with torch.no_grad():
            return self.projection(hidden_states.to(self.device)).detach().cpu()

    def get_layer_state(self) -> Dict[str, Any]:
        """Return this actor's layer parameters (for checkpointing)."""
        return {k: v.detach().cpu() for k, v in self.layer.state_dict().items()}

    def get_projection_state(self) -> Dict[str, Any]:
        """Return this actor's projection matrix M_i state (for checkpointing)."""
        return {k: v.detach().cpu() for k, v in self.projection.state_dict().items()}

    def load_layer_state(self, state_dict: Dict[str, Any]) -> None:
        """Reload layer params from a state dict."""
        mapped = {k: v.to(self.device) for k, v in state_dict.items()}
        self.layer.load_state_dict(mapped)

    def load_projection_state(self, state_dict: Dict[str, Any]) -> None:
        """Reload projection matrix M_i from a state dict."""
        mapped = {k: v.to(self.device) for k, v in state_dict.items()}
        self.projection.load_state_dict(mapped)

    def get_optimizer_state(self) -> Dict[str, Any]:
        """Return this actor's optimizer state (for checkpointing).

        Moves all tensors to CPU to avoid device mismatch on restore.
        """
        raw = self.optimizer.state_dict()
        # Optimizer state dicts can contain nested tensors in the
        # "state" sub-dict. Walk them to CPU for safe serialization.
        cpu_state: Dict[str, Any] = {}
        for k, v in raw.items():
            if k == "state":
                cpu_state[k] = {
                    param_id: {
                        sk: sv.detach().cpu() if isinstance(sv, torch.Tensor) else sv
                        for sk, sv in param_state.items()
                    }
                    for param_id, param_state in v.items()
                }
            else:
                cpu_state[k] = v
        return cpu_state

    def load_optimizer_state(self, state_dict: Dict[str, Any]) -> None:
        """Restore optimizer state from a checkpoint."""
        self.optimizer.load_state_dict(state_dict)

    def get_scheduler_state(self) -> Optional[Dict[str, Any]]:
        """Return this actor's LR scheduler state, or None if no scheduler."""
        if self.scheduler is None:
            return None
        return self.scheduler.state_dict()

    def load_scheduler_state(self, state_dict: Dict[str, Any]) -> None:
        """Restore LR scheduler state from a checkpoint."""
        if self.scheduler is not None and state_dict is not None:
            self.scheduler.load_state_dict(state_dict)

    def set_optimizer_eval(self) -> None:
        """Switch optimizer to eval mode (ScheduleFreeWrapper).

        Exposes internally-averaged parameters for more stable
        validation / inference. No-op for optimizers without an
        ``eval`` method.
        """
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()

    def set_optimizer_train(self) -> None:
        """Restore optimizer to training mode after validation."""
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()

    def ping(self) -> int:
        """Cheap liveness / scheduling verification probe."""
        return self.layer_idx

    def num_batches_processed(self) -> int:
        return self.batches_processed
