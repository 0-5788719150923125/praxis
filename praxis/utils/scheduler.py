"""Learning rate scheduler utilities."""

from functools import partial
from typing import List

import torch
from pytorch_optimizer import CosineAnnealingWarmupRestarts


def get_scheduler(
    optimizer, optimizer_config, disable_schedule=False, warmup_steps=4096
):
    """
    Create a learning rate scheduler for the optimizer.

    Args:
        optimizer: The optimizer to schedule
        optimizer_config: Configuration dict with lr and other settings
        disable_schedule: If True, use simple warmup without cosine annealing
        warmup_steps: Number of warmup steps

    Returns:
        A scheduler function/class
    """
    if disable_schedule:

        def lr_lambda_with_warmup(current_step, warmup_steps=1024):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        scheduler_func = partial(
            torch.optim.lr_scheduler.LambdaLR,
            lr_lambda=lambda step: lr_lambda_with_warmup(step, warmup_steps),
        )
    else:

        class PatchedCosineAnnealingWarmupRestarts(CosineAnnealingWarmupRestarts):
            def step(self, *args, **kwargs):
                super().step(*args, **kwargs)
                self._last_lr: List[float] = [
                    group["lr"] for group in self.optimizer.param_groups
                ]

        scheduler_func = partial(
            PatchedCosineAnnealingWarmupRestarts,
            first_cycle_steps=1024 * 256,
            max_lr=optimizer_config["lr"],
            min_lr=optimizer_config["lr"] * 1e-2,
            gamma=1.0,
            warmup_steps=warmup_steps,
        )

    return scheduler_func
