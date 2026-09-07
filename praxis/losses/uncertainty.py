"""Uncertainty weighting for auxiliary objectives.

Kendall, Gal & Cipolla, *Multi-Task Learning Using Uncertainty to Weigh Losses
for Scene Geometry and Semantics* (CVPR 2018). Each task carries a learned
log-variance ``s`` and enters the total as

    exp(-s) * L + s

The gradient in ``s`` is ``1 - exp(-s) * L``, so ``s`` settles at ``log L`` and
the effective weight settles at ``1 / L``. Two consequences, both of which are
the point:

  - AN OBJECTIVE THAT STAYS HARD DOWN-WEIGHTS ITSELF. A loss stuck high early
    contributes gradient scaled by ``1/L``, so it cannot drown a task that is
    already working. As it becomes learnable its weight comes back up on its
    own. No schedule, no warmup, nothing per-experiment.
  - THE ``+ s`` TERM IS WHAT STOPS THE TRIVIAL SOLUTION. Without it the optimum
    is ``s -> inf`` and every auxiliary is muted. With it, muting costs ``s``.

Note this is a BALANCE, not a filter: an objective that is large and useless
still gets weight ``1/L`` rather than zero. Whether a term deserves to be
connected at all is a separate question, and its own diagnostic has to answer
it.

Written in log-variance directly rather than through ``sigma``, which keeps the
parameter unconstrained and the exponential stable.
"""

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn


class UncertaintyWeighting(nn.Module):
    """Learned per-objective weights over a fixed set of named losses.

    Parameters are shape ``[1]`` rather than 0-dim: ``schedule_free``'s
    ``swap()`` views parameters as uint8 and 0-dim parameters break it.
    """

    def __init__(self, names: Iterable[str]) -> None:
        super().__init__()
        self.log_var = nn.ParameterDict(
            {n: nn.Parameter(torch.zeros(1)) for n in names}
        )

    def weight(self, name: str) -> torch.Tensor:
        """Current ``exp(-s)`` for one objective, detached."""
        return torch.exp(-self.log_var[name].detach()).squeeze(0)

    def forward(self, name: str, loss: torch.Tensor) -> torch.Tensor:
        """Weight one loss. Starts at weight 1, so a run begins unchanged."""
        s = self.log_var[name].squeeze(0).to(loss.dtype)
        return torch.exp(-s) * loss + s

    def weights(self) -> Dict[str, float]:
        """Every current weight, for the dashboard."""
        return {n: float(torch.exp(-p.detach()).squeeze(0)) for n, p in self.log_var.items()}
