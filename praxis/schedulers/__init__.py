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
):
    """
    Returns a partial function for creating schedulers.
    Exactly matches the implementation from main.py.
    
    Args:
        optimizer_config: Configuration dict with lr and other settings
        disable_schedule: If True, use simple warmup without cosine annealing  
        warmup_steps: Number of warmup steps
    
    Returns:
        A partial function that creates a scheduler when called with an optimizer
    """
    if disable_schedule:
        def lr_lambda_with_warmup(current_step, warmup_steps=1024):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        return partial(
            torch.optim.lr_scheduler.LambdaLR,
            lr_lambda=lambda step: lr_lambda_with_warmup(step, warmup_steps),
        )
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