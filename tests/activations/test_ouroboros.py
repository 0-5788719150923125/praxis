"""Tests for praxis/activations/ouroboros.py."""

import torch

from praxis import registry
from praxis.activations.ouroboros import MAX_STEPS, Ouroboros, drain_step_counts
from praxis.activations.serpent import Serpent
from tests.stubs import materialize


def test_eval_identity_to_serpent_at_init():
    """The -l experiment is only a controlled comparison against -k if
    Ouroboros starts as Serpent. Both gates saturate at init, so the residue is
    float rounding from writing `x + 1*(y - x)` instead of `y`."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)

    ouroboros = materialize(Ouroboros, x, a=1.0, b=1.0, g=0.1).eval()
    serpent = materialize(Serpent, x, a=1.0, b=1.0, g=0.1).eval()

    with torch.no_grad():
        assert (ouroboros(x) - serpent(x)).abs().max().item() < 1e-6


def test_accounting_drains_and_skips_eval():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)

    regularizer = registry.lookup("regularizers", "ouroboros_budget")()
    activation = materialize(Ouroboros, x).train()
    drain_step_counts()

    activation(x)
    activation(x)
    recorded = drain_step_counts()
    assert len(recorded) == 2
    survival, spread = recorded[0]
    assert survival.shape == (MAX_STEPS,)
    assert spread.shape == (9,)
    assert not drain_step_counts(), "drain must clear the stack"

    # Eval must not retain graphs, and an empty stack must not blow up.
    activation.eval()
    activation(x)
    assert not drain_step_counts()
    assert float(regularizer(x, torch.zeros(2, 5, dtype=torch.long))) == 0.0
