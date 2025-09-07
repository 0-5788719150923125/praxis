"""Patched cosine annealing with warmup and restarts scheduler."""

from typing import List

from pytorch_optimizer import CosineAnnealingWarmupRestarts


class PatchedCosineAnnealingWarmupRestarts(CosineAnnealingWarmupRestarts):
    """
    Patched version of CosineAnnealingWarmupRestarts that properly tracks _last_lr.
    This matches the implementation from main.py exactly.
    """
    
    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)
        self._last_lr: List[float] = [
            group["lr"] for group in self.optimizer.param_groups
        ]