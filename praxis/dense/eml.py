"""EML-tree dense block: the ``e^x - Log(y)`` function class.

From "EML Trees Are Universal Approximators" (arXiv:2606.23179): a single
compositional primitive ``EML(x, y) = e^x - Log(y)`` that represents any
elementary function, universal in ``W^{k,inf}`` with explicit size/depth bounds.
As a ``dim -> dim`` block it is a sibling to the MLP/KAN entries in
``DENSE_REGISTRY`` (a swap-in for the Titans memory net via ``build_memory_model``)
whose curve shape is deliberately ORTHOGONAL to an MLP's - the "log-minus-
exponent" regime. Paired against the exponential energy memory (a SMEAR blend of
the two, see praxis/memory), the two opposing regimes pull toward a center rather
than warping outward.

Stability (the paper's own caveat - real-valued EML needs a softplus-stabilized
log near zero): the exp operand is clamped so it cannot overflow, and the log
term reads ``log(softplus(.) + eps)`` so its argument is strictly positive. Both
matter here because the memory net's weights are updated at test time, where an
unbounded exp would NaN the fast weights.
"""

from typing import Any, Optional, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.dense.base import BaseDense

ConfigType = TypeVar("ConfigType", bound="AutoConfig")

# Upper bound on the exp operand: exp(8) ~ 3e3, large but finite, so a test-time
# weight update cannot drive the exponential term to overflow. Fixed, model-
# agnostic (not a per-experiment knob).
_EXP_CLAMP: float = 8.0
# Floor inside the log so its argument stays strictly positive.
_LOG_EPS: float = 1e-6


class EMLTree(BaseDense):
    """``dim -> dim`` stack of EML layers, each ``e^{Ax} - log(softplus(Bx))``."""

    def __init__(
        self,
        config: ConfigType,
        activation: Optional[str] = None,  # unused: EML is its own nonlinearity
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        dim = input_dim or config.hidden_size
        hidden = hidden_dim or dim
        num_layers = max(1, num_layers)
        widths = [dim] + [hidden] * (num_layers - 1)
        outs = [hidden] * (num_layers - 1) + [dim]
        # Per-layer exp-operand and log-operand projections (the affine transforms
        # are the learnable per-atom parameters), then a mix back to the layer's
        # output width.
        self.exp_proj = nn.ModuleList(
            [nn.Linear(widths[i], hidden) for i in range(num_layers)]
        )
        self.log_proj = nn.ModuleList(
            [nn.Linear(widths[i], hidden) for i in range(num_layers)]
        )
        self.mix = nn.ModuleList(
            [nn.Linear(hidden, outs[i]) for i in range(num_layers)]
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        x = inputs
        for exp_p, log_p, mix in zip(self.exp_proj, self.log_proj, self.mix):
            e = torch.exp(exp_p(x).clamp(max=_EXP_CLAMP))  # exponential term
            l = torch.log(F.softplus(log_p(x)) + _LOG_EPS)  # softplus-stabilized log
            x = mix(self.dropout(e - l))  # the EML primitive: e^x - Log(y)
        return x
