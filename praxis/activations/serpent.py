from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions.exponential import Exponential
from torch.nn import Module
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter

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
