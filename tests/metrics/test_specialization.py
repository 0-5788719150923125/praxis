import pytest
import torch

from praxis.activations.serpent import Serpent
from praxis.configuration import PraxisConfig
from praxis.metrics.specialization import (
    collect_activation_descriptions,
    collect_activation_metrics,
)

# ------------------------------------------------------------------------------
# servant
# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------
# arc_mixture
# ------------------------------------------------------------------------------
# Test suite for the ArcMixture router (cyclic mixture-of-depths).
#
# ArcMixture keys capacity to the *physical layer* index (current_depth % num_layers),
# so a given layer is the routed one on every recurrent pass, and keys a low-rank router
# weight delta to the *recurrent pass* (current_depth // num_layers). This mirrors the
# ArcGLU idiom; see praxis/routers/arc.py.


def make_config(**overrides):
    # calm-d shape: depth 9, 3 physical layers -> 3 recurrent passes each.
    config = dict(hidden_size=64, depth=9, num_layers=3, debug=False)
    config.update(overrides)
    return PraxisConfig(**config)
