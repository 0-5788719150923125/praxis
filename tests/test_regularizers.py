"""Regularizer registry: default selection, build, and the activation option."""

import pytest
import torch

from praxis.losses.regularizers import (
    REGULARIZER_REGISTRY,
    build_regularizers,
)


def test_default_is_contrastive_isotropy():
    reg = build_regularizers(None)
    assert len(reg) == 1
    assert reg[0].name == "contrastive"


def test_empty_list_disables_all():
    assert len(build_regularizers([])) == 0


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        build_regularizers(["does_not_exist"])


def test_multiple_regularizers_compose():
    reg = build_regularizers(list(REGULARIZER_REGISTRY.keys()))
    assert len(reg) == len(REGULARIZER_REGISTRY)
    names = {m.name for m in reg}
    assert "contrastive" in names and "activation_reg" in names


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
