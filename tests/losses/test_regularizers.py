"""Regularizer registry: default selection, build, and the activation option."""

import pytest

from praxis.losses.regularizers import build_regularizers


def test_default_is_contrastive_isotropy():
    reg = build_regularizers(None)
    assert len(reg) == 1
    assert reg[0].name == "contrastive"


def test_empty_list_disables_all():
    assert len(build_regularizers([])) == 0


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        build_regularizers(["does_not_exist"])
