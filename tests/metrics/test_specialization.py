"""Activation specialization metrics (praxis/metrics/specialization.py)."""

import torch

from praxis.activations.serpent import Serpent
from praxis.metrics.specialization import (
    collect_activation_descriptions,
    collect_activation_metrics,
)


def _built(cls, x, **kwargs):
    """Lazy modules materialize on first forward."""
    module = cls(**kwargs)
    module(x)
    return module


def test_plain_serpent_publishes_nothing():
    """The walk must not invent metrics for activations that never opted in."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)

    root = torch.nn.Sequential(_built(Serpent, x))
    assert collect_activation_metrics(root) == {}
    assert collect_activation_descriptions(root) == {}
