from typing import Any, Optional

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

        if trainable:
            self.a = UninitializedParameter()
            self.b = UninitializedParameter()
            self.g = UninitializedParameter()
        else:
            self.register_buffer("a", None)
            self.register_buffer("b", None)
            self.register_buffer("g", None)

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

        if self.trainable:
            for name, init in (("a", initial_a), ("b", initial_b), ("g", initial_g)):
                param = getattr(self, name)
                param.materialize(init.shape, device=device, dtype=dtype)
                with torch.no_grad():
                    param.copy_(init)
        else:
            self.register_buffer("a", initial_a)
            self.register_buffer("b", initial_b)
            self.register_buffer("g", initial_g)

    def forward(self, x: Tensor) -> Tensor:
        a = self.a
        b = self.b
        g = self.g
        # Broadcast params across leading dims of x
        if a.dim() < x.dim():
            shape = [1] * (x.dim() - a.dim()) + list(a.shape)
            a = a.view(shape)
            b = b.view(shape)
            g = g.view(shape)

        inv_a = a / (a * a + INV_FLOOR_EPS * INV_FLOOR_EPS)
        snake = x + torch.sin(a * x).square() * inv_a
        return snake + g * torch.sin(b * x)
