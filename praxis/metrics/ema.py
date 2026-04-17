"""Exponential moving average constants and helper."""

from typing import Optional

LOSS_EMA_ALPHA = 0.01
"""Smoothing factor for loss EMA (both backprop and mono-forward trainers)."""

STEP_TIME_EMA_ALPHA = 0.1
"""Smoothing factor for per-step wall-clock time EMA."""


def compute_ema(current: float, previous: Optional[float], alpha: float) -> float:
    """Return ``alpha * current + (1 - alpha) * previous``.

    If *previous* is ``None`` (first observation), returns *current*
    directly so the EMA initialises without bias.
    """
    if previous is None:
        return current
    return alpha * current + (1 - alpha) * previous
