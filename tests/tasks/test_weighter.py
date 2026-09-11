"""Tests for the engagement-prediction reward (P2) and policy (P3)."""

import pytest
import torch


def test_difficulty_weighter_pads_old_checkpoints():
    """ema_loss saved before TaskType.JOKE existed loads with NaN padding."""
    import torch

    from praxis.tasks import TaskType
    from praxis.tasks.weighter import DifficultyTaskLossWeighter

    w = DifficultyTaskLossWeighter()
    sd = w.state_dict()
    sd["ema_loss"] = torch.ones(len(TaskType) - 1)  # pre-JOKE checkpoint
    w2 = DifficultyTaskLossWeighter()
    w2.load_state_dict(sd)
    assert w2.ema_loss.numel() == len(TaskType)
    assert torch.isnan(w2.ema_loss[-1])
    assert (w2.ema_loss[:-1] == 1).all()
