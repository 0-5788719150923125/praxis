"""Per-task loss weighting.

Each weighter maps ``task_type_ids`` -> per-token scalar weights. Fixed
and learnable variants share an interface: ``forward`` returns weights,
and an optional ``anchor_loss`` is folded into the training objective
by the model so gradient-driven variants don't drift unboundedly.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from praxis.tasks.types import TASK_NAMES, TaskType, task_id


def _targets_to_tensor(targets: Optional[Dict]) -> torch.Tensor:
    """Coerce a ``{task_name_or_id: float}`` dict into a ``[num_tasks]`` tensor."""
    out = torch.ones(len(TaskType), dtype=torch.float32)
    if not targets:
        return out
    for key, val in targets.items():
        tid = key if isinstance(key, int) else task_id(key)
        out[int(tid)] = float(val)
    return out


class TaskLossWeighter(nn.Module):
    """Base class: maps task IDs to per-token scalar weights."""

    def forward(self, task_type_ids: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def anchor_loss(self) -> Optional[torch.Tensor]:
        """Optional auxiliary loss. Return ``None`` for non-learnable variants."""
        return None

    def effective_weights(self) -> torch.Tensor:
        """Current ``[num_tasks]`` weight vector (for logging)."""
        raise NotImplementedError


class FixedTaskLossWeighter(TaskLossWeighter):
    """Fixed per-task weights stored as a non-persistent buffer.

    ``targets`` is a ``{task_name_or_id: float}`` dict. Missing keys
    default to 1.0. This is the identity weighter when ``targets`` is
    empty or None.
    """

    def __init__(self, targets: Optional[Dict] = None):
        super().__init__()
        weights = _targets_to_tensor(targets)
        self.register_buffer("weights", weights, persistent=False)

    def forward(self, task_type_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[task_type_ids.long()]

    def effective_weights(self) -> torch.Tensor:
        return self.weights.detach()


class LearnableTaskLossWeighter(TaskLossWeighter):
    """Per-task learnable scalars, sigmoid-gated around a target.

    ``weight_t = 2 * target_t * sigmoid(raw_t)``

    At init ``raw_t = 0`` so ``weight_t == target_t`` exactly. Gradients
    flow through ``raw_t`` so training can bias weights up (toward
    ``2 * target_t``) or down (toward 0) from the starting value. An L2
    anchor on ``raw`` pulls it back to 0 so the optimizer can't collapse
    every weight to zero just to minimize the weighted-mean loss. Tune
    ``anchor_weight`` to control stiffness: smaller = more drift allowed.
    """

    def __init__(
        self,
        targets: Optional[Dict] = None,
        anchor_weight: float = 0.01,
        lr_multiplier: float = 25.0,
    ):
        super().__init__()
        self.register_buffer("targets", _targets_to_tensor(targets), persistent=False)
        self.raw = nn.Parameter(torch.zeros(len(TaskType)))
        self.anchor_weight = float(anchor_weight)
        # Task weights live under a weighted-mean reduction that cancels
        # most of the gradient signal, so ``raw`` needs a much larger LR
        # than the transformer's main params to produce visible drift.
        # The optimizer builder reads this and splits ``raw`` into its
        # own param group.
        self.lr_multiplier = float(lr_multiplier)

    def _effective(self) -> torch.Tensor:
        return 2.0 * self.targets * torch.sigmoid(self.raw)

    def forward(self, task_type_ids: torch.Tensor) -> torch.Tensor:
        return self._effective()[task_type_ids.long()]

    def anchor_loss(self) -> Optional[torch.Tensor]:
        if self.anchor_weight <= 0:
            return None
        return self.anchor_weight * self.raw.pow(2).mean()

    def effective_weights(self) -> torch.Tensor:
        with torch.no_grad():
            return self._effective().detach()
