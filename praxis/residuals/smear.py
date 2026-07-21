"""SMEAR-style residual router: mix residual styles, gated per depth step.

Residual styles are usually picked once for the whole model, but different
depths have different jobs, and the styles on offer differ exactly in how
much force the branch applies: standard is a fixed gain of 1 (tight
coupling), ReZero is a learned zero-init gain per depth (the stack enters as
the identity), HyperConnections learn projections. This module holds N child
residual connections and blends their ``connect_depth`` outputs through a
learned per-depth softmax - SMEAR's soft-merge idea applied at the residual
seam, with the depth index as the router input, so the router is exactly the
trivial thing it should be: one weight vector per depth step, no activations
consulted, no extra instability surface.

For the standard+rezero pair the blend collapses to an interpretable form:
``out = x + (w_std[d] + w_rz[d] * alpha[d]) * branch`` - a learned per-depth
effective gain with a FLOOR of ``w_std[d]``. Note what that changes: pure
ReZero enters training as the identity (gain 0); the uniform-init mix enters
at gain ~1/N (0.5 for two styles). The router can learn its way to either
extreme, and where it settles per depth is the readout
(``residual/mix_*_d{k}`` on the dashboard).

HyperConnections are NOT yet mixable: they widen the residual state to
``rate`` streams (``connect_width``/``format_state`` change shape), so
blending them with single-stream styles needs a stream-unification pass
first. The constructor rejects them explicitly rather than silently
mis-blending; the seam to extend is ``_COMPATIBLE``.
"""

from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.residuals.base import ResidualConnection
from praxis.residuals.rezero import ReZeroConnection

# Styles the mixer can hold: those sharing the base single-stream state
# contract (identity connect_width/format_state).
_COMPATIBLE = {
    "standard": ResidualConnection,
    "rezero": ReZeroConnection,
}


class SmearResidual(ResidualConnection):
    """Per-depth soft mix over child residual connections."""

    def __init__(
        self,
        dim: int,
        num_depths: int = 1,
        styles: Tuple[str, ...] = ("standard", "rezero"),
        **kwargs: Any,
    ) -> None:
        super().__init__()
        unknown = [s for s in styles if s not in _COMPATIBLE]
        if unknown:
            raise ValueError(
                f"SmearResidual cannot mix {unknown}: only single-stream "
                f"styles {sorted(_COMPATIBLE)} share the state contract "
                "(hyper widens the residual to rate streams and needs a "
                "stream-unification pass before it can join the mix)."
            )
        self.styles = tuple(styles)
        self.mix = nn.ModuleList(
            _COMPATIBLE[s](dim, num_depths=num_depths) for s in styles
        )
        # The router: one logit vector per depth step, zero-init -> uniform.
        self.logits = nn.Parameter(torch.zeros(max(1, int(num_depths)), len(styles)))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(styles={list(self.styles)})"

    def weights(self, current_depth: int) -> Tensor:
        idx = min(int(current_depth), self.logits.size(0) - 1)
        return F.softmax(self.logits[idx], dim=-1)

    def connect_depth(
        self, mix_h: Tensor, h_o: Tensor, beta: Tensor, current_depth: int = 0
    ) -> Tensor:
        w = self.weights(current_depth)
        out = None
        for weight, child in zip(w, self.mix):
            contrib = weight * child.connect_depth(mix_h, h_o, beta, current_depth)
            out = contrib if out is None else out + contrib
        return out

    @torch.no_grad()
    def style_shares(self) -> dict:
        """Per-depth softmax share of each non-first style (the first is the
        reference; shares sum to 1). Deterministic in the parameters, so this
        is free to read at metric time."""
        shares = F.softmax(self.logits, dim=-1).cpu()  # [depth, styles], one sync
        out = {}
        for s, style in enumerate(self.styles):
            if s == 0:
                continue
            for d in range(shares.size(0)):
                out[f"residual/mix_{style}_d{d}"] = float(shares[d, s])
        return out
