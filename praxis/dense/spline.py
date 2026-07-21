"""Learned-knot spline layer: piecewise-linear basis with adaptive placement.

Same shape as the FastKAN layer (per-feature 1D basis -> mixing linear), with
the OPPOSITE stance on where resolution lives. FastKAN's grid is a fixed
buffer - deliberately, so a test-time memory net never moves it ("the
multi-scale basis stays put"). Here the knot positions AND widths are
nn.Parameters: used as a test-time memory net, they join the fast weights, so
the surprise update MOVES THE KNOTS online - resolution concentrates where the
sequence is complex and coarsens where it is smooth. Adaptive resolution
becomes a second test-time adaptation axis, the spline analogue of Serpent's
online-tuned frequencies.

The basis is compact-support hat functions ``max(0, 1 - |x - k| / h)`` -
genuine piecewise-linear splines. Compact support is the point of difference
from the KAN's Gaussian bumps: a knot only shapes its own neighbourhood, so a
test-time edit at one part of the axis cannot bleed through infinite tails
into every other region.
"""

import math
from typing import Any, TypeVar

import torch
import torch.nn as nn
from torch import Tensor

from praxis.activations import ACT2FN
from praxis.dense.base import BaseDense
from praxis.dense.kan import SplineLinear

ConfigType = TypeVar("ConfigType", bound="AutoConfig")

# Additive floor on the knot widths. exp(log_width) keeps widths positive; the
# additive floor keeps the basis from collapsing into delta spikes while still
# passing gradient to log_widths (a clamp would zero it at the floor).
_WIDTH_FLOOR: float = 1e-2


class SplineNetwork(BaseDense):
    """Per-feature learned-knot hat-spline basis with a linear mixing readout.

    Each input feature owns ``num_knots`` knots (position + width), so every
    feature channel carries its own resolution profile over the activation
    axis. Init is a uniform grid over ``[-knot_span, knot_span]`` with widths
    equal to the knot gap (adjacent hats tile the interval); everything after
    that is learned - or, as a memory net, surprise-updated at test time.
    """

    def __init__(
        self,
        config: ConfigType,
        num_knots: int = 8,
        knot_span: float = 2.0,
        spline_weight_init_scale: float = 0.1,
        use_base_update: bool = True,
    ) -> None:
        super().__init__()
        dim = config.hidden_size
        self.input_dim: int = dim
        self.output_dim: int = dim
        self.num_knots: int = num_knots

        init_knots = torch.linspace(-knot_span, knot_span, num_knots)
        gap = 2.0 * knot_span / max(num_knots - 1, 1)
        self.knots = nn.Parameter(init_knots.repeat(dim, 1))  # [dim, K]
        self.log_widths = nn.Parameter(torch.full((dim, num_knots), math.log(gap)))
        self.spline_linear = SplineLinear(
            dim * num_knots, dim, spline_weight_init_scale
        )
        self.use_base_update: bool = use_base_update
        if use_base_update:
            self.base_activation = ACT2FN[config.activation]
            self.base_linear = nn.Linear(dim, dim)

    def forward(self, inputs: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        widths = torch.exp(self.log_widths) + _WIDTH_FLOOR
        u = (inputs[..., None] - self.knots) / widths  # [..., dim, K]
        basis = (1.0 - u.abs()).clamp_min(0.0)  # hat: compact support
        ret = self.spline_linear(basis.flatten(-2))
        if self.use_base_update:
            ret = ret + self.base_linear(self.base_activation(inputs))
        return ret
