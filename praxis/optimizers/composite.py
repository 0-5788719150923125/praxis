"""Run two optimizers over disjoint parameter sets as one.

Muon orthogonalizes interior >=2D matrices; a secondary optimizer (e.g.
Lion) drives the vocab-facing params - embeddings, the LM head, norms,
biases. This composite presents a single optimizer interface (param_groups,
state, step, state_dict) so the scheduler, the wrapper stack
(low_rank_moment/half_lion), and the optimizer-metrics dashboard all see
one optimizer. It subclasses ``Optimizer`` because the wrapper stack does
an ``isinstance`` check.

LR coupling: the project's cosine scheduler writes one scheduled lr across
*every* param group (max_lr = the configured lr), flattening per-group
rates. To keep the secondary at its intended rate we treat its lr as a
fixed ratio of the primary's scheduled lr and re-impose that ratio on each
step - the same trick Muon's internal AdamW uses (adamw_lr_ratio * lr).
"""

from __future__ import annotations

from collections import ChainMap
from typing import Any, Dict

from torch.optim import Optimizer


class CompositeOptimizer(Optimizer):
    """Two optimizers over disjoint params, behind one interface."""

    def __init__(
        self, primary: Optimizer, secondary: Optimizer, secondary_lr_ratio: float
    ):
        self.primary = primary
        self.secondary = secondary
        self.secondary_lr_ratio = float(secondary_lr_ratio)
        self._initialized = False

        # Initialise the Optimizer machinery (hook dicts, etc.) over the union
        # of params, then point param_groups/state at the real sub-optimizers
        # so scheduler/metrics edits propagate to them (shared dict objects).
        all_params = [
            p
            for g in (primary.param_groups + secondary.param_groups)
            for p in g["params"]
        ]
        super().__init__(all_params, primary.defaults)
        self.param_groups = primary.param_groups + secondary.param_groups
        self.state = ChainMap(primary.state, secondary.state)
        self._initialized = True

    def _primary_lr(self) -> float:
        return float(self.primary.param_groups[0]["lr"])

    def step(self, closure=None):
        # The scheduler flattened every group to the primary's scheduled lr;
        # re-impose the secondary's ratio before it steps.
        secondary_lr = self._primary_lr() * self.secondary_lr_ratio
        for g in self.secondary.param_groups:
            g["lr"] = secondary_lr
        loss = self.primary.step(closure)
        self.secondary.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.primary.zero_grad(set_to_none)
        self.secondary.zero_grad(set_to_none)

    def add_param_group(self, group: dict) -> None:
        # During super().__init__ the base Optimizer registers the union group
        # on us directly; only route to the secondary once we're live.
        if not getattr(self, "_initialized", False):
            return super().add_param_group(group)
        # New groups (e.g. a promoted tasker.raw) go to the secondary, the
        # AdamW-family optimizer the rest of the codebase assumes; keep the
        # unified view in sync.
        self.secondary.add_param_group(group)
        self.param_groups = self.primary.param_groups + self.secondary.param_groups

    def state_dict(self) -> Dict[str, Any]:
        return {
            "primary": self.primary.state_dict(),
            "secondary": self.secondary.state_dict(),
            "secondary_lr_ratio": self.secondary_lr_ratio,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.primary.load_state_dict(state["primary"])
        self.secondary.load_state_dict(state["secondary"])
        self.secondary_lr_ratio = float(
            state.get("secondary_lr_ratio", self.secondary_lr_ratio)
        )
        self.param_groups = self.primary.param_groups + self.secondary.param_groups
        self.state = ChainMap(self.primary.state, self.secondary.state)

    def __repr__(self) -> str:
        return (
            f"CompositeOptimizer(primary={type(self.primary).__name__}, "
            f"secondary={type(self.secondary).__name__}, "
            f"ratio={self.secondary_lr_ratio:g})"
        )
