"""Regularizer registry: default selection, build, and the activation option."""

import pytest
import torch

from praxis.losses.regularizers import build_regularizers


def test_activation_regularizer_forward_and_metrics():
    reg = build_regularizers(["activation"])[0]
    h = torch.randn(2, 8, 16)
    ids = torch.randint(0, 32, (2, 8))
    loss = reg(h, ids)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss >= 0
    m = reg.training_metrics()
    assert set(m) == {"activation_ar", "activation_tar"}


def test_activation_regularizer_single_token_no_tar():
    reg = build_regularizers(["activation"])[0]
    h = torch.randn(2, 1, 16)
    loss = reg(h, torch.zeros(2, 1, dtype=torch.long))
    assert torch.isfinite(loss)
    assert reg.training_metrics()["activation_tar"] == 0.0
