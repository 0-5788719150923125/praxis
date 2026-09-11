"""Tests for praxis/tasks/weighter.py."""

import torch

from praxis.tasks import TaskType
from praxis.tasks.weighter import DifficultyTaskLossWeighter


def test_difficulty_weighter_pads_shorter_checkpoints():
    """An ema_loss saved before the newest TaskType existed loads with its
    missing tail padded with NaN."""

    w = DifficultyTaskLossWeighter()
    sd = w.state_dict()
    sd["ema_loss"] = torch.ones(len(TaskType) - 1)  # one task short
    w2 = DifficultyTaskLossWeighter()
    w2.load_state_dict(sd)
    assert w2.ema_loss.numel() == len(TaskType)
    assert torch.isnan(w2.ema_loss[-1])
    assert (w2.ema_loss[:-1] == 1).all()
