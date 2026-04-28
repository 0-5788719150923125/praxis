"""Reduction helpers shared across loss functions.

Per-task loss weighting plugs in here: a loss computes its per-token
tensor as usual, then defers to :func:`weighted_reduce` for the final
mean/sum. Weights of 0 fully mask a position; positive scales pass
through linearly. Positions where ``labels == -100`` are excluded
from the denominator so masked targets don't dilute the average.
"""

from typing import Optional

import torch
from torch import Tensor


def weighted_reduce(
    per_token_loss: Tensor,
    labels: Optional[Tensor] = None,
    loss_weights: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    """Reduce a per-token loss tensor with optional per-token weights.

    Args:
        per_token_loss: Flat or shaped tensor of per-position losses.
        labels: Optional labels matching ``per_token_loss`` shape. When
            provided, positions equal to ``-100`` are excluded.
        loss_weights: Optional per-position weights matching
            ``per_token_loss`` shape. ``None`` reproduces the standard
            ``reduction``.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``. Ignored when
            ``loss_weights`` is provided (always weighted mean).

    Returns:
        Scalar tensor for ``mean``/``sum`` or ``per_token_loss`` shape
        for ``none``. When ``loss_weights`` is provided, returns
        ``sum(w * loss) / sum(w)`` (always a scalar).
    """
    if loss_weights is None:
        if reduction == "mean":
            return per_token_loss.mean()
        if reduction == "sum":
            return per_token_loss.sum()
        return per_token_loss

    flat_loss = per_token_loss.reshape(-1)
    flat_w = loss_weights.reshape(-1).to(flat_loss.dtype)

    if labels is not None:
        active = (labels.reshape(-1) != -100).to(flat_loss.dtype)
        flat_w = flat_w * active

    denom = flat_w.sum()
    if denom <= 0:
        # Preserve the autograd graph so the backward pass still runs.
        return (flat_loss * flat_w).sum()
    return (flat_loss * flat_w).sum() / denom
