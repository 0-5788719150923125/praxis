import torch

from praxis.activations.serpent import Serpent
from praxis.activations.servant import MOD_MAX, Servant
from praxis.metrics.specialization import (
    collect_activation_descriptions,
    collect_activation_metrics,
)


def _built(cls, x, **kwargs):
    """Lazy modules materialize on first forward."""
    module = cls(**kwargs)
    module(x)
    return module


def test_identity_to_serpent_at_init_in_value_and_gradient():
    """`v` is zero-init, so a_eff == a and -m starts as -k exactly. Unlike
    Ouroboros this is a plain differentiable forward, so the GRADIENT is exact
    too, not just the value."""
    torch.manual_seed(0)
    x0 = torch.randn(2, 5, 16)

    servant = _built(Servant, x0, a=1.0, b=1.0, g=0.1)
    serpent = _built(Serpent, x0, a=1.0, b=1.0, g=0.1)

    xa = x0.clone().requires_grad_(True)
    xb = x0.clone().requires_grad_(True)
    ya, yb = servant(xa), serpent(xb)
    ya.sum().backward()
    yb.sum().backward()

    assert (ya - yb).abs().max().item() == 0.0
    assert (xa.grad - xb.grad).abs().max().item() == 0.0


def test_coupling_moves_the_frequency():
    """With v away from zero the frequency must actually track token energy,
    and the swing must stay inside MOD_MAX."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)

    servant = _built(Servant, x, a=1.0, b=1.0, g=0.1)
    with torch.no_grad():
        servant.v.fill_(2.0)

    # Two batches at deliberately different energies must chirp differently.
    quiet = servant(x * 0.1)
    loud = servant(x * 10.0)
    assert not torch.allclose(quiet, loud * 0.0)  # sanity: outputs are live

    metrics = servant.training_metrics()
    assert metrics["servant_coupling"] > 0.9  # tanh(2.0) ~ 0.96
    assert 0.0 < metrics["servant_chirp"] <= MOD_MAX


def test_metrics_are_zero_at_init_and_reachable_by_the_walk():
    """servant_coupling is the run's gate metric, so it has to start at exactly
    zero and has to survive the module walk the dynamics logger uses."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)

    servant = _built(Servant, x)
    servant(x)  # populate the realized-swing stash

    metrics = servant.training_metrics()
    assert metrics["servant_coupling"] == 0.0
    assert metrics["servant_coupling_std"] == 0.0
    assert metrics["servant_chirp"] == 0.0

    root = torch.nn.Sequential(torch.nn.Linear(16, 16), servant)
    collected = collect_activation_metrics(root)
    assert collected["servant_coupling"] == 0.0
    assert "servant_coupling" in collect_activation_descriptions(root)


def test_plain_serpent_publishes_nothing():
    """The walk must not invent metrics for activations that never opted in."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)

    root = torch.nn.Sequential(_built(Serpent, x))
    assert collect_activation_metrics(root) == {}
    assert collect_activation_descriptions(root) == {}


def test_uninitialized_module_reports_nothing():
    """training_metrics runs on the logging cadence and may fire before the
    first forward has materialized the lazy parameters."""
    assert Servant().training_metrics() == {}
