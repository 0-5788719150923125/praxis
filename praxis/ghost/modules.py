"""Drop-in replacements whose weight is DERIVED rather than stored.

Each wrapper keeps the host module's forward semantics exactly and differs in
one respect: the weight is not a parameter, it is the return value of an
``Expansion`` that owns a smaller real one. Shapes seen by callers are
unchanged, so a wrapper can be swapped in anywhere the original sat without the
consumer knowing.

One deliberate divergence from ``MergedLinear`` (praxis/routers/smear.py), which
holds the ORIGINAL ``Parameter`` so old checkpoint keys keep resolving. Ghosting
changes the stored shape by construction, so there is no checkpoint to stay
compatible with and the base weight is discarded rather than kept. ``.weight``
is still exposed, as a read-only property, because introspection paths
(parameter stats, metric probes, ``__repr__``) reach for it.

The bias is left REAL and untouched. It is ``out`` numbers against a weight of
``out * in * k``, so tying it saves nothing and would be the note's own "combine
site" mistake in miniature.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.ghost.expansions import Expansion

ExpansionFactory = Callable[[Sequence[int], str], Expansion]


class GhostLinear(nn.Module):
    """``nn.Linear`` with a derived weight."""

    def __init__(
        self, base: nn.Linear, factory: ExpansionFactory, tag: str = ""
    ) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.expansion = factory(tuple(base.weight.shape), tag)
        self.bias = base.bias

    @property
    def weight(self) -> Tensor:
        return self.expansion()

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.expansion(), self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class GhostConv1d(nn.Module):
    """``nn.Conv1d`` with a derived weight."""

    def __init__(
        self, base: nn.Conv1d, factory: ExpansionFactory, tag: str = ""
    ) -> None:
        super().__init__()
        if base.groups != 1:
            raise ValueError("grouped convolutions are not a ghost target")
        self.in_channels = base.in_channels
        self.out_channels = base.out_channels
        self.kernel_size = base.kernel_size
        self.stride = base.stride
        self.padding = base.padding
        self.dilation = base.dilation
        self.groups = base.groups
        self.padding_mode = base.padding_mode
        self.expansion = factory(tuple(base.weight.shape), tag)
        self.bias = base.bias

    @property
    def weight(self) -> Tensor:
        return self.expansion()

    def forward(self, x: Tensor) -> Tensor:
        return F.conv1d(
            x,
            self.expansion(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}"
        )


# Module class -> wrapper. A type not listed here is never ghosted, which is
# what keeps a broad target profile from silently reaching something whose
# forward the wrapper does not reproduce.
WRAPPERS: dict[type, Callable[..., nn.Module]] = {
    nn.Linear: GhostLinear,
    nn.Conv1d: GhostConv1d,
}
