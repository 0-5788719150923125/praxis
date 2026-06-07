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

    # Dynamic weighters drift over training and are worth charting.
    # Fixed weighters return a constant and are excluded from dynamics logs.
    is_dynamic: bool = False

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

    Caveat: gradient descent on a multiplier in front of a positive
    loss always wants to shrink that multiplier, which makes this
    variant *down*weight high-loss tasks. Use
    :class:`DifficultyTaskLossWeighter` if you want the opposite.
    """

    is_dynamic = True
    metric_description = (
        "Per-task scalar multipliers applied to the loss. Values are "
        "2 * target * sigmoid(raw); raw drifts under gradient pressure "
        "and an L2 anchor pulls it toward 0 so weights stay near their "
        "starting targets. Note this variant tends to downweight high-loss "
        "tasks - the multiplier in front of a positive loss always wants "
        "to shrink."
    )

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


class DifficultyTaskLossWeighter(TaskLossWeighter):
    """Upweight hard tasks via a stop-gradient EMA of per-task loss.

    No gradient flows through the weights, so the optimizer can't
    collapse high-loss tasks the way the multiplicative-learnable
    variants (this codebase's ``LearnableTaskLossWeighter`` and
    Kendall-style uncertainty weighting) do. Call
    :meth:`observe` after each forward to feed in detached per-token
    losses and their task IDs.

    Effective weight per task::

        ratio_t = ema_loss_t / mean(ema_loss)
        weight_t = target_t * clamp(ratio_t ** gamma, floor, ceiling)

    ``gamma=1`` is gentle, ``gamma=2`` is focal-style aggressive.
    Unobserved tasks fall back to ``target_t`` (ratio = 1).
    """

    is_dynamic = True
    metric_description = (
        "Per-task scalar multipliers driven by EMA of per-task loss. "
        "Hard tasks (above-mean EMA loss) get upweighted, easy tasks "
        "downweighted, clamped to [floor, ceiling]. No gradient flows "
        "through the weights - the optimizer can't game them by collapsing "
        "high-loss tasks the way the learnable variant tends to."
    )

    def __init__(
        self,
        targets: Optional[Dict] = None,
        gamma: float = 1.0,
        ema_alpha: float = 0.05,
        floor: float = 0.1,
        ceiling: float = 4.0,
    ):
        super().__init__()
        self.register_buffer("targets", _targets_to_tensor(targets), persistent=False)
        # NaN sentinel marks "unobserved" so the first observation seeds
        # the EMA without being pulled toward an arbitrary prior.
        self.register_buffer(
            "ema_loss",
            torch.full((len(TaskType),), float("nan")),
            persistent=True,
        )
        self.gamma = float(gamma)
        self.ema_alpha = float(ema_alpha)
        self.floor = float(floor)
        self.ceiling = float(ceiling)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Tolerate checkpoints from before a TaskType was appended: pad the
        # saved EMA with NaN (unobserved) for new tasks.
        key = prefix + "ema_loss"
        saved = state_dict.get(key)
        if saved is not None and saved.numel() < self.ema_loss.numel():
            padded = torch.full_like(self.ema_loss, float("nan"))
            padded[: saved.numel()] = saved
            state_dict[key] = padded
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def observe(
        self, task_type_ids: torch.Tensor, per_token_loss: torch.Tensor
    ) -> None:
        if task_type_ids.shape != per_token_loss.shape:
            return
        flat_ids = task_type_ids.reshape(-1).long()
        flat_loss = per_token_loss.reshape(-1).detach().float().to(self.ema_loss.device)
        for tid in flat_ids.unique().tolist():
            mask = flat_ids == tid
            if not mask.any():
                continue
            mean_t = flat_loss[mask].mean()
            prev = self.ema_loss[tid]
            if torch.isnan(prev):
                self.ema_loss[tid] = mean_t
            else:
                self.ema_loss[tid] = (
                    self.ema_alpha * mean_t + (1.0 - self.ema_alpha) * prev
                )

    def _effective(self) -> torch.Tensor:
        ema = torch.where(
            torch.isnan(self.ema_loss),
            torch.ones_like(self.ema_loss),
            self.ema_loss,
        )
        ratio = ema / ema.mean().clamp(min=1e-6)
        scale = ratio.pow(self.gamma).clamp(self.floor, self.ceiling)
        return self.targets * scale

    def forward(self, task_type_ids: torch.Tensor) -> torch.Tensor:
        return self._effective()[task_type_ids.long()]

    def effective_weights(self) -> torch.Tensor:
        with torch.no_grad():
            return self._effective().detach()
