"""Wave Schedule-Free: a standing-wave gate over the parameter index.

Sibling of :class:`GatedScheduleFree`. Instead of a content-dependent gradient-
SNR gate, the per-coordinate schedule-free averaging weight is a *periodic*
function of the flattened coordinate index::

    gate_i = (1 + amp * sin(2*pi * cycles * i/numel + phase)) / 2   in [0, 1]

So the bias-variance allocation oscillates across each parameter in smooth
bands: at peaks (gate -> 1) the coordinate tracks the active iterate ``z``
(variance); at troughs (gate -> 0) it holds the running average ``x`` (bias).
The frequency (``cycles``) is a frozen constant by default - like the harmonic
field's frozen Weyl phases, no per-experiment tuning - and ``set_wave`` lets the
harmonic-weight RL controller drive ``(amp, cycles, phase)`` per episode (the
controller already proposes that exact sinusoid). The index is arbitrary, so
this is content-free structure (a baseline against the SNR gate); its one edge
over a random hash is neighbor correlation - adjacent coordinates share a phase.

Subclasses ``ScheduleFreeWrapper``, overriding only ``step`` (the averaging
weight ``checkpoint -> checkpoint * gate``); train/eval/serialization inherited.
Gate pinned to 1 (``amp=1, phase=pi/2`` at a peak) is the parent at that cell.
"""

import math

import torch
from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.optimizer import ScheduleFreeWrapper

# Frozen default: ~pi oscillations across each tensor (irrational, no tuning).
WAVE_CYCLES = math.pi


class WaveScheduleFree(ScheduleFreeWrapper):
    """Schedule-free averaging gated by a standing wave over the param index."""

    def __init__(
        self,
        optimizer,
        momentum: float = 0.98,
        wave_cycles: float = WAVE_CYCLES,
        wave_amp: float = 1.0,
        wave_phase: float = 0.0,
        weight_decay: float = 0.0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        **kwargs,
    ):
        super().__init__(
            optimizer,
            momentum=momentum,
            weight_decay=weight_decay,
            r=r,
            weight_lr_power=weight_lr_power,
            **kwargs,
        )
        self.wave_cycles = float(wave_cycles)
        self.wave_amp = float(wave_amp)
        self.wave_phase = float(wave_phase)
        self._last_gate_mean: float = 0.5  # mean of a [0,1] sine

    def __str__(self) -> str:
        return "WaveScheduleFree"

    @property
    def gate_mean(self) -> float:
        return self._last_gate_mean

    def set_wave(self, amp=None, cycles=None, phase=None) -> None:
        """Hook for the harmonic-weight RL controller to drive the wave's
        ``(amp, cycles, phase)`` per episode. Unset args keep their value."""
        if amp is not None:
            self.wave_amp = float(amp)
        if cycles is not None:
            self.wave_cycles = float(cycles)
        if phase is not None:
            self.wave_phase = float(phase)

    def _wave(self, p: torch.Tensor) -> torch.Tensor:
        """Per-coordinate gate in [0, 1] from a sine over the normalized,
        flattened coordinate index (scale-invariant: ``cycles`` oscillations
        across each tensor regardless of its size)."""
        n = max(p.numel(), 1)
        pos = torch.arange(n, device=p.device, dtype=p.dtype) / n
        phase = 2.0 * math.pi * self.wave_cycles * pos + self.wave_phase
        gate = 0.5 * (1.0 + self.wave_amp * torch.sin(phase))
        return gate.reshape(p.shape).clamp_(0.0, 1.0)

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
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise NoSparseGradientError(str(self))

                state = self.state[p]
                if "z" not in state:
                    state["z"] = p.clone()
                z = state["z"]
                self.apply_weight_decay(
                    z,
                    grad,
                    lr=group["lr"],
                    weight_decay=self.weight_decay,
                    weight_decouple=True,
                    fixed_decay=False,
                )
                self.apply_weight_decay(
                    p,
                    grad,
                    lr=group["lr"],
                    weight_decay=self.weight_decay,
                    weight_decouple=True,
                    fixed_decay=False,
                    ratio=1.0 - self.momentum,
                )
                p.lerp_(end=z, weight=1.0 - 1.0 / self.momentum)
                self.swap(z, p)

        self.optimizer.step()

        gates = []
        for group in self.param_groups:
            group["step"] = group["step"] + 1 if "step" in group else 1
            lr = group["lr"] * group.get("d", 1.0)
            lr_max = group["lr_max"] = max(lr, group.get("lr_max", 0))
            weight = (group["step"] ** group["lr"]) * (lr_max**self.weight_lr_power)
            weight_sum = group["weight_sum"] = group.get("weight_sum", 0.0) + weight
            checkpoint = weight / weight_sum if weight_sum != 0.0 else 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                gate = self._wave(p)
                gates.append(float(gate.mean()))
                z = state["z"]
                self.swap(z, p)
                # The only change from the parent: the standing-wave gate on the
                # averaging weight (scalar checkpoint -> checkpoint * gate).
                p.lerp_(end=z, weight=checkpoint * gate)
                p.lerp_(end=state["z"], weight=1.0 - self.momentum)

        if gates:
            self._last_gate_mean = sum(gates) / len(gates)
        return loss
