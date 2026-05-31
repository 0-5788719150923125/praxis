"""Drives the HarmonicWeightPolicy from the training loop (see calm-c).

Episode loop: every ``period`` steps, summarize the current dynamics
(state), sample a harmonic edit for one weight row (action), apply it, let
training run ``horizon`` steps while integrating an EMA return from the loss
improvement, then reward the controller by that return and either keep the
edit (it helped) or roll it back. The reward is intentionally delayed and
confounded by the model's own optimizer - the point is to watch REINFORCE
cope (or not). The EMA return is a partial answer to the delay: it credits a
benefit that ramps in over the horizon, not just the endpoint.

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
        reward_decay: float = 0.9,
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
        # A weight edit's benefit ramps in over many steps (weights are slow to
        # adjust, slower to converge), so a one-step endpoint delta under-credits
        # it. Instead the reward is an EMA-integrated return over the horizon:
        # each post-edit step folds its improvement-vs-L_before into an EMA,
        # which - read at the horizon's end - weights the latest (most-manifested)
        # steps most while smoothing noise. This is a discounted return; the
        # decay sets its effective memory within the horizon.
        self.reward_decay = float(reward_decay)
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
        self._sf = None  # cached ScheduleFreeWrapper (see _schedulefree)

    # ------------------------------------------------------------------

    def _schedulefree(self, trainer):
        """The wave-bearing optimizer in the stack, or None (cached).

        Wave mode needs an optimizer exposing ``set_wave`` (WaveScheduleFree or
        HalfLion). For z-mirroring: under schedule-free the model weight
        ``p.data`` is reconstructed each step from the base iterate ``z`` and the
        running average ``x``, so an in-place edit to ``p.data`` alone is smeared
        - we must edit ``z`` too (HalfLion has no ``z``, so ``_sf_z`` no-ops).
        """
        if self._sf is not None:
            return self._sf
        # Cache only on success and retry otherwise: trainer.optimizers may not
        # be populated at the first episode, and caching None would silently
        # disable wave mode (and z-mirroring) for the whole run.
        from pytorch_optimizer.optimizer import ScheduleFreeWrapper

        for opt in getattr(trainer, "optimizers", None) or []:
            o, depth = opt, 0
            while o is not None and depth < 6:  # unwrap Lightning/other wrappers
                # isinstance catches GatedScheduleFree/WaveScheduleFree subclasses;
                # set_wave catches HalfLion (a frozen-anchor wave, not schedule-free).
                if isinstance(o, ScheduleFreeWrapper) or hasattr(o, "set_wave"):
                    self._sf = o
                    return self._sf
                o = getattr(o, "optimizer", None)
                depth += 1
        return None

    def _sf_z(self, trainer, param):
        """The base iterate ``z`` schedule-free carries for ``param``, or None.

        Editing both ``p.data`` and ``z`` by the same op scales the whole
        (y, x, z) triple consistently (x = (y - (1-m)z)/m is linear in both),
        so the edit is a clean, revertible weight change under schedule-free.
        """
        sf = self._schedulefree(trainer)
        if sf is None:
            return None
        return sf.state.get(param, {}).get("z")

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
    def _apply_edit(self, param, row, alpha, omega, phi, z=None):
        n = param.shape[1]
        idx = torch.arange(n, device=param.device, dtype=param.dtype)
        mod = 1.0 + alpha * torch.sin(omega * idx + phi)
        original = param.data[row].clone()
        param.data[row].mul_(mod)
        z_original = None
        if z is not None:  # mirror the edit onto the schedule-free iterate
            z_original = z[row].clone()
            z[row].mul_(mod)
        return original, z_original

    @torch.no_grad()
    def _apply_anchor_gate(self, param, row, mask, anchor_row, z=None):
        original = param.data[row].clone()
        param.data[row][mask] = anchor_row[mask]
        z_original = None
        if z is not None:
            z_original = z[row].clone()
            z[row][mask] = anchor_row[mask]
        return original, z_original

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

            # While an edit is live, integrate its post-edit improvement into an
            # EMA return (the edit took effect on the step after _start_episode,
            # so this only runs on post-edit steps).
            if self._episode is not None:
                instant = (self._episode["L_before"] or 0.0) - self._loss_ema
                prev = self._episode["reward_ema"]
                d = self.reward_decay
                self._episode["reward_ema"] = (
                    instant if prev is None else d * prev + (1.0 - d) * instant
                )

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
                    self._start_episode(trainer, model)
            elif self._step >= self._episode["end_step"]:
                self._finish_episode()

        # Carry-forward the latest RL scalars every step so MetricsLogger has
        # them (episodes are sparse; this keeps the series continuous).
        for k, v in self._metrics.items():
            trainer.callback_metrics[k] = torch.tensor(float(v))

    @torch.no_grad()
    def _start_wave_episode(self, trainer, model):
        """Wave mode: the action drives the WaveScheduleFree optimizer's
        standing-wave gate (amp, cycles, phase) instead of editing weights.
        Rollback just restores the three scalars - no weight surgery."""
        sf = self._schedulefree(trainer)
        if sf is None or not hasattr(sf, "set_wave"):
            return  # needs a WaveScheduleFree optimizer (optimizer_wrappers)

        # Global grad-norm summary feeds the state's third feature.
        gnorm, n = 0.0, 0
        for _, p in self._candidate_params(model):
            if p.grad is not None:
                gnorm += float(p.grad.norm())
                n += 1
        state = self._make_state(self._loss_ema or 0.0, gnorm / max(n, 1))
        raw, _ = self.policy.act(state)
        amp, cycles, phase = (float(x) for x in self.policy.map_wave_action(raw))

        prev = (sf.wave_amp, sf.wave_cycles, sf.wave_phase)
        sf.set_wave(amp=amp, cycles=cycles, phase=phase)
        self._episode = {
            "state": state,
            "raw": raw,
            "sf": sf,
            "wave_prev": prev,
            "L_before": self._loss_ema,
            "reward_ema": None,
            "end_step": self._step + self.horizon,
            # Reuse the registered rl_action_* keys (amp/cycles/phase).
            "meta": {
                "rl_action_alpha": amp,
                "rl_action_omega": cycles,
                "rl_action_phi": phase,
            },
        }

    @torch.no_grad()
    def _start_episode(self, trainer, model):
        if self.edit_mode == "wave":
            self._start_wave_episode(trainer, model)
            return
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

        # Under schedule-free, mirror the edit onto the carried iterate z too.
        z = self._sf_z(trainer, param)
        ep = {
            "state": state,
            "raw": raw,
            "param": param,
            "row": row,
            "z": z,  # schedule-free base iterate (None if not wrapped)
            "L_before": self._loss_ema,
            "reward_ema": None,  # EMA return, accumulated over the horizon
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
            ep["original"], ep["z_original"] = self._apply_anchor_gate(
                param, row, mask, anchor[row], z=z
            )
            ep["meta"] = {"rl_gate_frac": float(mask.float().mean())}
        else:
            alpha, omega, phi = harmonic
            ep["original"], ep["z_original"] = self._apply_edit(
                param, row, alpha, omega, phi, z=z
            )
            ep["meta"] = {
                "rl_action_alpha": alpha,
                "rl_action_omega": omega,
                "rl_action_phi": phi,
            }

        self._episode = ep

    def _finish_episode(self):
        ep = self._episode
        self._episode = None
        # The EMA return over the horizon is the reward; the endpoint delta is
        # kept as a diagnostic so the two can be watched diverge.
        reward = float(ep["reward_ema"] or 0.0)
        instant = float((ep["L_before"] or 0.0) - (self._loss_ema or 0.0))

        # Pin only better weights: keep the edit if the return is positive, else
        # undo. (Using the integrated return, not just the endpoint.) Restore
        # both p.data and the schedule-free iterate z so the rollback is clean.
        if reward < self.keep_threshold:
            with torch.no_grad():
                if "param" in ep:  # weight-edit modes: restore the row (and z)
                    ep["param"].data[ep["row"]] = ep["original"]
                    if ep.get("z") is not None:
                        ep["z"][ep["row"]] = ep["z_original"]
                elif "wave_prev" in ep:  # wave mode: restore the wave scalars
                    amp, cycles, phase = ep["wave_prev"]
                    ep["sf"].set_wave(amp=amp, cycles=cycles, phase=phase)
            kept = 0.0
        else:
            kept = 1.0

        metrics = self.policy.update(ep["state"], ep["raw"], reward)
        metrics.update(ep["meta"])
        metrics["rl_edit_kept"] = kept
        metrics["rl_reward_instant"] = instant
        self._metrics = metrics
