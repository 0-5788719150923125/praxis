"""Wrapper-optimizer state that survives a checkpoint.

``pytorch_optimizer``'s ``ScheduleFreeWrapper`` and ``Lookahead`` save
``self.state`` keyed by the live ``Parameter`` objects. A reloaded checkpoint's
keys are copies that match no parameter of the resumed model, so every resume
restarts the state (schedule-free's ``z``, so the averaged weights snap to the
training iterate) and the checkpoint keeps the unreachable copy, one more per
resume. These helpers key it by each parameter's position in its groups, as
``torch.optim.Optimizer`` does, and the subclasses below use them.
"""

from collections import defaultdict

import torch
from pytorch_optimizer.optimizer import Lookahead, ScheduleFreeWrapper


def _params(optimizer):
    return [p for group in optimizer.param_groups for p in group["params"]]


def state_by_index(optimizer) -> dict:
    """``optimizer.state`` keyed by parameter position."""
    index = {id(p): i for i, p in enumerate(_params(optimizer))}
    return {
        index[id(p)]: dict(entry)
        for p, entry in optimizer.state.items()
        if id(p) in index
    }


def state_by_param(optimizer, saved: dict) -> defaultdict:
    """``state_by_index`` output back on the live parameters and their devices.
    Entries keyed any other way match no parameter and are dropped."""
    params = _params(optimizer)
    state = defaultdict(dict)
    for i, entry in saved.items():
        if isinstance(i, int) and 0 <= i < len(params):
            p = params[i]
            state[p] = {
                k: v.to(p.device) if torch.is_tensor(v) else v for k, v in entry.items()
            }
    return state


class IndexedScheduleFree(ScheduleFreeWrapper):
    """``ScheduleFreeWrapper`` whose ``z`` and mode survive a checkpoint.

    The mode says what the saved weights are (``y`` in train mode, ``x`` in
    eval), so a resume restores it and the trainer's next ``train()`` call moves
    the weights only if they were saved as ``x``.
    """

    def state_dict(self) -> dict:
        return {
            "schedulefree_state": state_by_index(self),
            "train_mode": self.train_mode,
            "base_optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.state = state_by_param(self, state["schedulefree_state"])
        self.train_mode = state.get("train_mode", self.train_mode)
        self.optimizer.load_state_dict(state["base_optimizer"])


class IndexedLookahead(Lookahead):
    """``Lookahead`` whose slow weights survive a checkpoint."""

    def state_dict(self) -> dict:
        return {
            "lookahead_state": state_by_index(self),
            "base_optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.state = state_by_param(self, state["lookahead_state"])
        self.optimizer.load_state_dict(state["base_optimizer"])
