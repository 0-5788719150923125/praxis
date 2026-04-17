"""LocalLayerWorker - in-process (non-Ray) worker for Mono-Forward.

Mirrors :class:`praxis.trainers.mono_forward.actor.LayerActor` but
lives in the driver process and shares its CUDA context. All the
training math is identical; the differences are:

- No ``@ray.remote`` decoration; this is a plain class.
- No deep copy of layer / criterion / strategy in the constructor -
  the in-process trainer owns the layers itself and hands them in
  directly (each ``LocalLayer`` is already unique to its position in
  the decoder stack, so no extra copy is needed).
- No CPU round-trip on activation handoff. The driver passes GPU
  tensors straight to the next worker in the chain; activations
  never leave the device during training. This is the main
  throughput and VRAM win over the Ray backend.
- No ``ping`` liveness probe and no ``_deep_to`` dance on inputs -
  tensors are assumed to already live on this worker's device.
  Checkpoint save/load still does a CPU round-trip so checkpoints
  remain portable across devices.

Every other method (train_batch, val_batch, infer_batch,
project_logits, get_/load_* state accessors, set_optimizer_eval /
_train) shares its semantics and return shape with the Ray actor, so
the in-process trainer can reuse the Ray trainer's bookkeeping
closures essentially verbatim.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from praxis.losses.layer_wise import compute_layer_wise_loss
from praxis.metrics import compute_softmax_collapse, extract_layer_dynamics
from praxis.trainers.mono_forward._worker_common import (ActorParamShim,
                                                         build_optimizer,
                                                         build_scheduler)
from praxis.trainers.mono_forward.device import deep_to
from praxis.trainers.mono_forward.projection import ProjectionMatrix


class LocalLayerWorker:
    """In-process worker owning one LocalLayer + per-layer projection.

    Constructor args mirror :class:`LayerActor` for interchangeability,
    but the worker trusts the caller to pass ``layer`` / ``criterion``
    / ``strategy`` already constructed on the correct device (the
    in-process trainer moves the full model before spawning workers).
    """

    def __init__(
        self,
        layer_idx: int,
        layer: nn.Module,
        criterion: nn.Module,
        hidden_size: int,
        vocab_size: int,
        device: torch.device,
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
        self.device = device

        self.layer = deep_to(layer, self.device)
        self.criterion = deep_to(criterion, self.device)
        if strategy is not None:
            self.strategy = (
                deep_to(strategy, self.device)
                if isinstance(strategy, nn.Module)
                else strategy
            )
        else:
            self.strategy = None

        self.projection = ProjectionMatrix(hidden_size, vocab_size).to(self.device)
        self.layer.train()
        self.projection.train()

        self._param_shim = ActorParamShim(self.layer, self.projection)
        self.optimizer = build_optimizer(
            shim=self._param_shim,
            optimizer_config=optimizer_config,
            wrappers=optimizer_wrappers or {},
            fallback_lr=lr,
            criterion=self.criterion,
            strategy=self.strategy,
        )
        self.scheduler = build_scheduler(
            optimizer=self.optimizer,
            optimizer_config=optimizer_config,
            warmup_steps=warmup_steps,
            disable_schedule=disable_schedule,
        )

        self.num_layers = int(num_layers)
        self.accumulate_grad_batches = max(int(accumulate_grad_batches), 1)
        self.batches_processed = 0

    # ------------------------------------------------------------------
    # train / val / inference
    # ------------------------------------------------------------------

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
        """One MF forward / local-update pass. Same return shape as LayerActor.

        If ``activations`` already requires grad, the gradient is allowed
        to flow back through the input tensor into whatever produced it
        (e.g. an upstream encoder that the trainer wants to co-train with
        layer 0's loss). Otherwise we detach and re-attach a fresh leaf,
        matching the per-layer-isolation contract that drives MF.
        """
        activations_dev = activations.to(self.device, non_blocking=True)
        if activations_dev.requires_grad:
            h = activations_dev
        else:
            h = activations_dev.detach().requires_grad_(True)
        labels_dev = labels.to(self.device, non_blocking=True)
        block_ids_dev = (
            block_ids.to(self.device, non_blocking=True)
            if block_ids is not None
            else None
        )
        input_ids_dev = (
            input_ids.to(self.device, non_blocking=True)
            if input_ids is not None
            else None
        )

        h_out, _kv, _state, aux_loss, _exit = self.layer(
            h,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=self.layer_idx,
            block_ids=block_ids_dev,
        )

        loss = compute_layer_wise_loss(
            hidden_states=h_out,
            labels=labels_dev,
            head=self.projection,
            criterion=self.criterion,
            strategy=self.strategy,
            aux_losses=[aux_loss] if aux_loss is not None else None,
            input_ids=input_ids_dev,
        )

        scaled_loss = loss / self.accumulate_grad_batches
        scaled_loss.backward()

        dynamics = self._capture_dynamics() if capture_dynamics else None

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

        softmax_collapse: Optional[float] = None
        if is_last_step:
            softmax_collapse = self._compute_softmax_collapse(h_out)

        self.batches_processed += 1
        # NB: ``hidden_states`` stays on device - the driver forwards
        # it directly to the next worker without any CPU round-trip.
        # ``loss`` is materialized to a Python float via ``.item()``,
        # which is the synchronisation point where the GPU finishes
        # this layer's work; that scalar is all the driver needs.
        return {
            "hidden_states": h_out.detach(),
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
        """No-grad forward + local loss for validation."""
        h = activations.to(self.device, non_blocking=True)
        labels_dev = labels.to(self.device, non_blocking=True)
        block_ids_dev = (
            block_ids.to(self.device, non_blocking=True)
            if block_ids is not None
            else None
        )
        input_ids_dev = (
            input_ids.to(self.device, non_blocking=True)
            if input_ids is not None
            else None
        )

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
            "hidden_states": h_detached,
            "loss": float(loss.detach().item()),
        }

    def infer_batch(
        self,
        activations: torch.Tensor,
        current_depth: Optional[int] = None,
        block_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """No-grad forward for inference. Does not mutate any state."""
        depth = current_depth if current_depth is not None else self.layer_idx
        h = activations.to(self.device, non_blocking=True)
        block_ids_dev = (
            block_ids.to(self.device, non_blocking=True)
            if block_ids is not None
            else None
        )
        with torch.no_grad():
            h_out, new_kv, _state, _aux, _exit = self.layer(
                h,
                attention_mask=None,
                past_key_values=past_key_values,
                current_state=None,
                current_depth=depth,
                block_ids=block_ids_dev,
            )
        return h_out.detach(), new_kv

    def project_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states through this worker's projection M_i."""
        with torch.no_grad():
            return self.projection(hidden_states.to(self.device, non_blocking=True)).detach()

    # ------------------------------------------------------------------
    # metrics helpers (shared with LayerActor semantics)
    # ------------------------------------------------------------------

    def _compute_softmax_collapse(self, hidden_states: torch.Tensor) -> Optional[float]:
        try:
            with torch.no_grad():
                logits = self.projection(hidden_states.detach())
                return compute_softmax_collapse(logits)
        except Exception:
            return None

    def _capture_dynamics(self) -> Optional[Dict[str, float]]:
        try:
            return extract_layer_dynamics(self.layer, self._current_lr())
        except Exception:
            return None

    def _current_lr(self) -> float:
        try:
            groups = getattr(self.optimizer, "param_groups", None)
            if groups:
                return float(groups[0].get("lr", 0.0))
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------
    # checkpoint hooks
    # ------------------------------------------------------------------

    def get_layer_state(self) -> Dict[str, Any]:
        return {k: v.detach().cpu() for k, v in self.layer.state_dict().items()}

    def get_projection_state(self) -> Dict[str, Any]:
        return {k: v.detach().cpu() for k, v in self.projection.state_dict().items()}

    def load_layer_state(self, state_dict: Dict[str, Any]) -> None:
        mapped = {k: v.to(self.device) for k, v in state_dict.items()}
        self.layer.load_state_dict(mapped)

    def load_projection_state(self, state_dict: Dict[str, Any]) -> None:
        mapped = {k: v.to(self.device) for k, v in state_dict.items()}
        self.projection.load_state_dict(mapped)

    def get_optimizer_state(self) -> Dict[str, Any]:
        raw = self.optimizer.state_dict()
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
        self.optimizer.load_state_dict(state_dict)

    def get_scheduler_state(self) -> Optional[Dict[str, Any]]:
        if self.scheduler is None:
            return None
        return self.scheduler.state_dict()

    def load_scheduler_state(self, state_dict: Dict[str, Any]) -> None:
        if self.scheduler is not None and state_dict is not None:
            self.scheduler.load_state_dict(state_dict)

    def set_optimizer_eval(self) -> None:
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()

    def set_optimizer_train(self) -> None:
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()

    def num_batches_processed(self) -> int:
        return self.batches_processed
