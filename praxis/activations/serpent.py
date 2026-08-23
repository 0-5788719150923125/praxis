from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions.exponential import Exponential
from torch.nn import Module
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

# --- live-energy signal (opt-in, shared by Servant and Ouroboros) ----------
# Floor inside the log-energy, so an all-zero token cannot log(0).
ENERGY_EPS: float = 1e-6
# Live-energy z-score that maps to tanh(1) ~ 0.76. Two standard deviations of
# the OBSERVED spread, so a typical token lands in the graded middle of the
# tanh rather than on its shoulder.
SIGNAL_SIGMAS: float = 2.0
# EMA on the running log-energy statistics.
STAT_DECAY: float = 0.99
# Smallest std the normalizer will divide by.
LOG_S_STD_FLOOR: float = 1e-3

# Floor for the rectified inverse `a / (a^2 + INV_FLOOR_EPS^2)`. Caps the
# effective `1/alpha` factor at `1/INV_FLOOR_EPS` so tiny alpha values can
# no longer produce outlier activations that trigger intermittent gradient
# spikes. See next/harmony.md for the diagnosis.
INV_FLOOR_EPS: float = 0.1


class Serpent(LazyModuleMixin, Module):
    """Praxis' extended Snake activation with a second oscillation term:

        y = x + sin^2(α·x) · α / (α^2 + ε^2) + γ·sin(βx)

    α controls the primary squared-sine frequency (original Snake term).
    β and γ add a secondary sine with its own frequency and amplitude.
    All three are per-feature learnable parameters.

    The `1/α` factor in the original Snake is replaced by the smooth-rectified
    `α / (α^2 + ε^2)`: matches `1/α` for `|α| >> ε`, bounded by `1/ε` for
    `|α| ~ 0`. Prevents the tiny-α feature explosion that produces
    intermittent gradient spikes during training.

    Subclassing hooks (used by Servant, and any future variant we layer on):
    `_declare_extra_parameters` adds lazily-materialized params/buffers,
    `_initialize_extra_parameters` materializes them on first forward, and
    `_effective_frequency` returns the (possibly modulated) primary frequency
    that drives the sin^2 term. The base implementations leave the activation
    exactly as written above.
    """

    def __init__(
        self,
        a: Optional[float] = None,
        b: Optional[float] = None,
        g: Optional[float] = None,
        trainable: bool = True,
        exp_rate: float = 1.0,
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.trainable = trainable
        self.a_value = a
        self.b_value = b
        self.g_value = g
        self.exp_rate = exp_rate
        self.gamma_init = gamma_init

        self._declare_parameter("a")
        self._declare_parameter("b")
        self._declare_parameter("g")
        self._declare_extra_parameters()

    # -- subclassing hooks -------------------------------------------------

    def _declare_extra_parameters(self) -> None:
        """Declare additional lazy params/buffers (base: none). Call
        ``self._declare_parameter(name)`` for each."""

    def _initialize_extra_parameters(self, x: Tensor) -> None:
        """Materialize the extra params on first forward (base: none). Build
        the initial tensors from ``x`` and call ``self._materialize(...)``."""

    def _effective_frequency(self, a: Tensor, x: Tensor) -> Tensor:
        """The primary frequency driving the sin^2 term. Base: the static,
        per-feature ``a`` (already broadcast against ``x``)."""
        return a

    # -- lazy-parameter plumbing (shared by all variants) ------------------

    def _declare_parameter(self, name: str) -> None:
        if self.trainable:
            setattr(self, name, UninitializedParameter())
        else:
            self.register_buffer(name, None)

    def _materialize(self, *named_inits: Tuple[str, Tensor]) -> None:
        if self.trainable:
            for name, init in named_inits:
                param = getattr(self, name)
                param.materialize(init.shape, device=init.device, dtype=init.dtype)
                with torch.no_grad():
                    param.copy_(init)
        else:
            for name, init in named_inits:
                self.register_buffer(name, init)

    def initialize_parameters(self, x: Tensor, *args: Any, **kwargs: Any) -> None:
        feature_shape = x.shape[-1:]
        device, dtype = x.device, x.dtype
        exp_dist = Exponential(torch.tensor(self.exp_rate, device=device))

        initial_a = (
            torch.full(feature_shape, self.a_value, dtype=dtype, device=device)
            if self.a_value is not None
            else exp_dist.sample(feature_shape).to(dtype=dtype)
        )
        initial_b = (
            torch.full(feature_shape, self.b_value, dtype=dtype, device=device)
            if self.b_value is not None
            else exp_dist.sample(feature_shape).to(dtype=dtype)
        )
        initial_g = (
            torch.full(feature_shape, self.g_value, dtype=dtype, device=device)
            if self.g_value is not None
            else torch.empty(feature_shape, dtype=dtype, device=device).uniform_(
                -self.gamma_init, self.gamma_init
            )
        )

        self._materialize(("a", initial_a), ("b", initial_b), ("g", initial_g))
        self._initialize_extra_parameters(x)

    # -- live-energy signal (opt-in) ---------------------------------------

    def _declare_energy_stats(self) -> None:
        """Declare the running log-energy statistics. Call from
        ``_declare_extra_parameters``; pair with ``_initialize_energy_stats``.

        BUFFERS, never parameters, and never a value frozen at init. Both
        Servant and Ouroboros previously centred the energy signal on a single
        learnable ``log_s_ref`` materialized from the FIRST batch, and that has
        one failure mode with no way back out of it. See ``_energy_signal``.
        """
        self.register_buffer("log_s_mean", UninitializedBuffer())
        self.register_buffer("log_s_var", UninitializedBuffer())

    def _initialize_energy_stats(self, x: Tensor) -> None:
        """Seed the running statistics from the first batch."""
        log_s = self._log_energy(x)
        for name, init in (
            ("log_s_mean", log_s.mean()),
            ("log_s_var", log_s.var(unbiased=False)),
        ):
            buffer = getattr(self, name)
            buffer.materialize((1,), device=x.device, dtype=torch.float32)
            with torch.no_grad():
                buffer.copy_(init.reshape(1))

    @staticmethod
    def _log_energy(x: Tensor) -> Tensor:
        """Per-token log RMS over the feature axis, detached, in float32.

        By Parseval this is the token's total spectral power, which is why it
        is the right quantity to drive a harmonic term with. Detached: a
        measurement that steers the activation, not a path trained through.
        """
        s = x.detach().square().mean(dim=-1, keepdim=True).clamp_min(ENERGY_EPS).sqrt()
        return s.log().float()

    def _energy_signal(self, x: Tensor, live: bool) -> Tensor:
        """Standardized live-energy signal in (-1, 1), as ``tanh(z / SIGMAS)``.

        WHY THE Z-SCORE, and it is not a refinement. The predecessor was
        ``tanh(log s - log_s_ref)`` with ``log_s_ref`` a learnable scalar set
        from the first batch's mean. A tanh has an implicit width of ONE NAT,
        and neither of the two scales involved respects that:

          * the per-token SPREAD of log-RMS is already ~1.1 nats at init, so
            the signal starts on the shoulder of the tanh, not in its middle;
          * the OFFSET between the live activation scale and a reference pinned
            at step 0 grows to several nats as the network's scale drifts.

        Together they saturate it. Measured on abstractinator-s, ``E|m|`` was
        0.80 by step 30 and 0.999 from step 4000 onward, and stayed there for
        20k steps. A saturated ``m`` is a CONSTANT, so ``a_eff = a * (1 + c)``
        is a static rescaling of the frequency - not a chirp, and exactly
        redundant with ``a`` itself, which is how the coupling then got decayed
        away along the degenerate direction.

        Nor could it recover: ``dm/d(log_s_ref) = -sech^2``, which at |arg| = 4
        is 0.0013. Saturation is a one-way door for a learned reference.

        Dividing by the running std fixes both at once. ``z`` is in units of the
        signal's own spread, so the modulation stays graded no matter where the
        activation scale drifts to, and the running mean keeps ``E[m] ~ 0`` over
        a batch - which is what stops ``v`` from being a second, decayable copy
        of ``a``.
        """
        log_s = self._log_energy(x)
        if live:
            with torch.no_grad():
                self.log_s_mean.mul_(STAT_DECAY).add_((1.0 - STAT_DECAY) * log_s.mean())
                self.log_s_var.mul_(STAT_DECAY).add_(
                    (1.0 - STAT_DECAY) * log_s.var(unbiased=False)
                )
        std = self.log_s_var.clamp_min(0.0).sqrt().clamp_min(LOG_S_STD_FLOOR)
        z = (log_s - self.log_s_mean) / std
        return torch.tanh(z / SIGNAL_SIGMAS).to(x.dtype)

    def _broadcast(self, t: Tensor, x: Tensor) -> Tensor:
        """View a per-feature tensor so it broadcasts across ``x``'s leading dims."""
        if t.dim() < x.dim():
            return t.view([1] * (x.dim() - t.dim()) + list(t.shape))
        return t

    def forward(self, x: Tensor) -> Tensor:
        a = self._broadcast(self.a, x)
        b = self._broadcast(self.b, x)
        g = self._broadcast(self.g, x)

        a_eff = self._effective_frequency(a, x)
        inv_a = a_eff / (a_eff * a_eff + INV_FLOOR_EPS * INV_FLOOR_EPS)
        snake = x + torch.sin(a_eff * x).square() * inv_a
        return snake + g * torch.sin(b * x)
