"""HalfLion: blend the live weights with their frozen initialization.

A wrapper (over any base optimizer, Lion by default) that keeps a frozen copy of
each parameter at init - the *prior* - and, during training, deploys a per-
coordinate blend of the current weights and that prior. The blend fraction is a
standing wave over the flattened coordinate index:

    mix_i = amp * (1 + sin(2*pi * cycles * i/numel + phase)) / 2   in [0, amp]

with ``cycles`` and ``amp`` frozen constants (so the reweighting is smooth and
needs no per-experiment tuning) and ``phase`` drifting a fixed amount per step -
a traveling wave that sweeps which coordinates lean toward the prior. ``amp<1``
guarantees a current-weight core at every coordinate (both sets are always
present). The gradient is evaluated at the blend but applied to the *current*
weights; eval deploys 100% current. ``set_wave`` lets the harmonic-weight RL
controller drive ``(amp, cycles, phase)``, so the gate can be learned.

Unlike the schedule-free family this anchors to a *fixed* init snapshot rather
than a running average, and it does not disable the LR schedule. It owns the
forward-deployed weights, so it cannot stack with ``wave_schedule_free`` (they
would fight over what the forward sees) - use one or the other.
"""

import math

import torch
from pytorch_optimizer.base.optimizer import BaseOptimizer

WAVE_CYCLES = math.pi  # frozen spatial frequency (~pi bands across each tensor)
WAVE_AMP = 0.5  # frozen blend depth; mix in [0, 0.5] keeps a current core
WAVE_DRIFT = 0.05  # rad/step: smooth temporal travel of the bands


class HalfLion(BaseOptimizer):
    """Blend live weights with their frozen init via a traveling index wave."""

    def __init__(
        self,
        optimizer,
        wave_cycles: float = WAVE_CYCLES,
        wave_amp: float = WAVE_AMP,
        wave_drift: float = WAVE_DRIFT,
        **kwargs,
    ):
        self.validate_non_negative(wave_amp, "wave_amp")
        self.wave_cycles = float(wave_cycles)
        self.wave_amp = float(wave_amp)
        self.wave_drift = float(wave_drift)
        self.wave_phase = 0.0
        self.train_mode = False
        self._last_gate_mean = 0.0

        self.optimizer = self.load_optimizer(optimizer, **kwargs)
        from collections import defaultdict

        self.state = defaultdict(dict)
        self.defaults = self.optimizer.defaults

    def __str__(self) -> str:
        return "HalfLion"

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def gate_mean(self) -> float:
        return self._last_gate_mean

    def add_param_group(self, param_group):
        return self.optimizer.add_param_group(param_group)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none)

    def state_dict(self):
        return {
            "half_lion_state": self.state,
            "wave_phase": self.wave_phase,
            "wave_amp": self.wave_amp,
            "wave_cycles": self.wave_cycles,
            "train_mode": self.train_mode,
            "base_optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state) -> None:
        self.state = state["half_lion_state"]
        self.wave_phase = state.get("wave_phase", self.wave_phase)
        self.wave_amp = state.get("wave_amp", self.wave_amp)
        self.wave_cycles = state.get("wave_cycles", self.wave_cycles)
        self.train_mode = state.get("train_mode", self.train_mode)
        self.optimizer.load_state_dict(state["base_optimizer"])

    def set_wave(self, amp=None, cycles=None, phase=None) -> None:
        """Hook for the harmonic-weight RL controller to drive the wave (the
        learned gate). Unset args keep their value."""
        if amp is not None:
            self.wave_amp = float(amp)
        if cycles is not None:
            self.wave_cycles = float(cycles)
        if phase is not None:
            self.wave_phase = float(phase)

    def _mix(self, p: torch.Tensor) -> torch.Tensor:
        """Per-coordinate fraction toward the frozen prior, in [0, amp]."""
        n = max(p.numel(), 1)
        pos = torch.arange(n, device=p.device, dtype=p.dtype) / n
        phase = 2.0 * math.pi * self.wave_cycles * pos + self.wave_phase
        mix = self.wave_amp * 0.5 * (1.0 + torch.sin(phase))
        return mix.reshape(p.shape).clamp_(0.0, 1.0)

    @torch.no_grad()
    def _deploy_blend(self) -> None:
        """Set each param to lerp(current, prior, mix). Init from p on first use."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "w" not in state:  # capture init weights as the frozen prior
                    state["w0"] = p.detach().clone()
                    state["w"] = p.detach().clone()
                p.data.copy_(state["w"]).lerp_(state["w0"], self._mix(p))

    @torch.no_grad()
    def eval(self):
        if not self.train_mode:
            return
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "w" in state:
                    p.data.copy_(state["w"])  # deploy 100% current at eval
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()
        self.train_mode = False

    @torch.no_grad()
    def train(self):
        if self.train_mode:
            return
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()
        self._deploy_blend()  # capture the prior on first call; redeploy thereafter
        self.train_mode = True

    @torch.no_grad()
    def reset(self) -> None:
        pass

    def init_group(self, *args, **kwargs) -> None:
        pass  # wrapper overrides step entirely; satisfies the BaseOptimizer ABC

    @torch.no_grad()
    def step(self, closure=None):
        if not self.train_mode:
            raise ValueError(
                "optimizer was not in train mode when step is called. "
                "call .train() before training"
            )
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()  # grad at the currently-deployed blend

        # Restore the current weights so the base optimizer steps from them, not
        # the blend (the grad was evaluated at the blend, applied to current).
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "w" not in state:
                    state["w0"] = p.detach().clone()
                    state["w"] = p.detach().clone()
                p.data.copy_(state["w"])

        self.optimizer.step()

        self.wave_phase += self.wave_drift  # travel the bands, then redeploy at it
        mixes = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state["w"].copy_(p.data)
                mix = self._mix(p)
                mixes.append(float(mix.mean()))
                p.data.lerp_(state["w0"], mix)  # redeploy the blend

        if mixes:
            self._last_gate_mean = sum(mixes) / len(mixes)
        return loss
