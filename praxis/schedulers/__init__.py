"""Learning rate scheduler utilities."""

from functools import partial
from typing import Any, Dict

import torch

from praxis.schedulers.cosine_warmup import PatchedCosineAnnealingWarmupRestarts
from praxis.schedulers.warmup import LinearWarmupScheduler


def get_scheduler_func(
    optimizer_config: Dict[str, Any],
    disable_schedule: bool = False,
    warmup_steps: int = 4096,
    stage_anchor=None,
):
    """
    Returns a partial function for creating schedulers.
    Exactly matches the implementation from main.py.

    Args:
        optimizer_config: Configuration dict with lr and other settings
        disable_schedule: If True, use simple warmup without cosine annealing
        warmup_steps: Number of warmup steps
        stage_anchor: Optional ``() -> int`` returning the optimizer step at
            which a new warmup should begin (a multi-stage boundary, e.g. CALM's
            codec freeze), or -1 for none. Re-warms over the same ``warmup_steps``
            horizon so a newly-activated, cold parameter set ramps up instead of
            taking the full LR. Read live each step, so it catches a
            convergence-driven boundary; the anchor is a persistent buffer, so
            this is resume-safe. Only wired for the disable_schedule path
            (schedule-free), the only scheduler in use.

    Returns:
        A partial function that creates a scheduler when called with an optimizer
    """
    if disable_schedule:

        def lr_lambda(step):
            # Stage-1 warmup (the original ramp), held at 1.0 after.
            factor = min(1.0, float(step) / float(max(1, warmup_steps)))
            # Stage-2+ re-warmup: past a reported stage boundary, multiply by a
            # fresh ramp from that step so the cold params ease in.
            if stage_anchor is not None:
                anchor = stage_anchor()
                if anchor >= 0 and step >= anchor:
                    factor *= min(
                        1.0, float(step - anchor) / float(max(1, warmup_steps))
                    )
            return factor

        return partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=lr_lambda)
    else:
        return partial(
            PatchedCosineAnnealingWarmupRestarts,
            first_cycle_steps=1024 * 256,
            max_lr=optimizer_config["lr"],
            min_lr=optimizer_config["lr"] * 1e-2,
            gamma=1.0,
            warmup_steps=warmup_steps,
        )


__all__ = [
    "get_scheduler_func",
    "LinearWarmupScheduler",
    "PatchedCosineAnnealingWarmupRestarts",
]
