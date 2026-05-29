"""Drives the HarmonicWeightPolicy from the training loop (see calm-c).

Episode loop: every ``period`` steps, summarize the current dynamics
(state), sample a harmonic edit for one weight row (action), apply it, let
training run ``horizon`` steps, then reward the controller by the loss
improvement and either keep the edit (it helped) or roll it back. The
reward is intentionally delayed and confounded by the model's own
optimizer - the point is to watch REINFORCE cope (or not).

Metrics (rl_reward, rl_baseline, rl_action_*, rl_edit_kept, ...) are
written to ``trainer.callback_metrics`` so MetricsLogger drains them; this
callback must be ordered before MetricsLogger.
"""

import math
from typing import Optional

import torch
from lightning.pytorch.callbacks import Callback

from praxis.policies.harmonic_weight_rl import build_gate_mask


class HarmonicWeightRLCallback(Callback):
    def __init__(
        self,
        policy,
        period: int = 50,
        horizon: int = 20,
        warmup_steps: int = 200,
        keep_threshold: float = 0.0,
        loss_ema_decay: float = 0.9,
        edit_mode: str = "harmonic",
        selector: str = "sinusoidal",
    ) -> None:
        super().__init__()
        self.policy = policy
        self.period = int(period)
        self.horizon = int(horizon)
        self.warmup_steps = int(warmup_steps)
        self.keep_threshold = float(keep_threshold)
        self.loss_ema_decay = float(loss_ema_decay)
        # "harmonic": modulate a row with a sinusoid (default, unchanged).
        # "anchor_gate": gate-replace a hash-selected subset of a row with a
        # frozen anchor copy (see next/hash_gated_anchor.md).
        self.edit_mode = str(edit_mode)
        self.selector = str(selector)

        self._loss_ema: Optional[float] = None
        self._step = 0
        self._episode: Optional[dict] = None  # set while an edit is live
        self._metrics: dict = {}
        self._editable = None  # cached candidate params
        self._anchor: dict = {}  # name -> frozen weight snapshot (anchor_gate)

    # ------------------------------------------------------------------

    def _candidate_params(self, model):
        # Editable chunks = rows of 2D weight matrices (one neuron's fan-in).
        # A row is a coherent functional unit; a random flat slice is not.
        if self._editable is None:
            self._editable = [
                (n, p)
                for n, p in model.named_parameters()
                if p.dim() == 2 and p.requires_grad and min(p.shape) >= 2
            ]
        return self._editable

    @staticmethod
    def _extract_loss(outputs, trainer) -> Optional[float]:
        val = None
        if torch.is_tensor(outputs):
            val = outputs
        elif isinstance(outputs, dict):
            val = outputs.get("loss")
        if val is None:
            val = trainer.callback_metrics.get("loss")
        if val is None:
            return None
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    def _make_state(self, loss_val: float, row_grad_norm: float) -> torch.Tensor:
        slope = 0.0 if self._loss_ema is None else (loss_val - self._loss_ema)
        # Bounded, normalized summary so the tiny policy net sees stable inputs.
        return torch.tensor(
            [
                math.tanh(loss_val),
                math.tanh(50.0 * slope),
                math.tanh(row_grad_norm),
            ],
            dtype=torch.float32,
        )

    @torch.no_grad()
    def _apply_edit(self, param, row, alpha, omega, phi):
        n = param.shape[1]
        idx = torch.arange(n, device=param.device, dtype=param.dtype)
        mod = 1.0 + alpha * torch.sin(omega * idx + phi)
        original = param.data[row].clone()
        param.data[row].mul_(mod)
        return original

    @torch.no_grad()
    def _apply_anchor_gate(self, param, row, mask, anchor_row):
        original = param.data[row].clone()
        param.data[row][mask] = anchor_row[mask]
        return original

    @torch.no_grad()
    def _snapshot_anchor(self, model):
        # Frozen reference, never optimized ("we don't optimize the frozen
        # copy"). A single early snapshot goes stale - EMA/refresh is a known
        # knob (see next/hash_gated_anchor.md).
        for name, param in self._candidate_params(model):
            self._anchor[name] = param.detach().clone()

    # ------------------------------------------------------------------

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        model = getattr(pl_module, "model", pl_module)
        model = getattr(model, "_orig_mod", model)  # unwrap torch.compile

        loss_val = self._extract_loss(outputs, trainer)
        if loss_val is not None:
            self._loss_ema = (
                loss_val
                if self._loss_ema is None
                else self.loss_ema_decay * self._loss_ema
                + (1.0 - self.loss_ema_decay) * loss_val
            )
            self._step += 1

            # Capture the frozen anchor once, post-warmup, so the live weights
            # have something to drift away from before gating starts.
            if (
                self.edit_mode == "anchor_gate"
                and self._step == self.warmup_steps
                and not self._anchor
            ):
                self._snapshot_anchor(model)

            if self._episode is None:
                if self._step >= self.warmup_steps and self._step % self.period == 0:
                    self._start_episode(model)
            elif self._step >= self._episode["end_step"]:
                self._finish_episode()

        # Carry-forward the latest RL scalars every step so MetricsLogger has
        # them (episodes are sparse; this keeps the series continuous).
        for k, v in self._metrics.items():
            trainer.callback_metrics[k] = torch.tensor(float(v))

    @torch.no_grad()
    def _start_episode(self, model):
        cands = self._candidate_params(model)
        if not cands:
            return
        pidx = int(torch.randint(len(cands), ()).item())
        name, param = cands[pidx]
        row = int(torch.randint(param.shape[0], ()).item())
        grad = param.grad
        row_grad_norm = float(grad[row].norm()) if grad is not None else 0.0

        state = self._make_state(self._loss_ema or 0.0, row_grad_norm)
        raw, harmonic = self.policy.act(state)

        ep = {
            "state": state,
            "raw": raw,
            "param": param,
            "row": row,
            "L_before": self._loss_ema,
            "end_step": self._step + self.horizon,
        }

        if self.edit_mode == "anchor_gate":
            threshold, omega, phi = (
                float(x) for x in self.policy.map_gate_action(raw)
            )
            anchor = self._anchor.get(name)
            if anchor is None:  # fallback: snapshot now (gate is identity once)
                anchor = param.detach().clone()
                self._anchor[name] = anchor
            n = param.shape[1]
            seed = (self._step * 1_000_003) ^ (pidx * 9_176) ^ row
            mask = build_gate_mask(
                self.selector, n, threshold, omega, phi, seed, param.device
            )
            ep["original"] = self._apply_anchor_gate(param, row, mask, anchor[row])
            ep["meta"] = {"rl_gate_frac": float(mask.float().mean())}
        else:
            alpha, omega, phi = harmonic
            ep["original"] = self._apply_edit(param, row, alpha, omega, phi)
            ep["meta"] = {
                "rl_action_alpha": alpha,
                "rl_action_omega": omega,
                "rl_action_phi": phi,
            }

        self._episode = ep

    def _finish_episode(self):
        ep = self._episode
        self._episode = None
        reward = float((ep["L_before"] or 0.0) - (self._loss_ema or 0.0))

        # Pin only better weights: keep the edit if loss improved, else undo.
        if reward < self.keep_threshold:
            with torch.no_grad():
                ep["param"].data[ep["row"]] = ep["original"]
            kept = 0.0
        else:
            kept = 1.0

        metrics = self.policy.update(ep["state"], ep["raw"], reward)
        metrics.update(ep["meta"])
        metrics["rl_edit_kept"] = kept
        self._metrics = metrics
