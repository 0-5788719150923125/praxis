import torch

from praxis import registry
from praxis.activations.ouroboros import MAX_STEPS, Ouroboros, drain_step_counts
from praxis.activations.serpent import Serpent
from praxis.activations.servant import Servant


def _built(cls, x, **kwargs):
    """Lazy modules materialize on first forward."""
    module = cls(**kwargs)
    module(x)
    return module


def test_eval_identity_to_serpent_at_init():
    """The -l experiment is only a controlled comparison against -k if
    Ouroboros starts as Serpent. Both gates saturate at init, so the residue is
    float rounding from writing `x + 1*(y - x)` instead of `y`."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)

    ouroboros = _built(Ouroboros, x, a=1.0, b=1.0, g=0.1).eval()
    serpent = _built(Serpent, x, a=1.0, b=1.0, g=0.1).eval()

    with torch.no_grad():
        assert (ouroboros(x) - serpent(x)).abs().max().item() < 1e-6


def test_no_zero_dim_parameters():
    """The schedule_free wrapper swaps parameters with
    ``x.view(torch.uint8).bitwise_xor_(y.view(torch.uint8))``, which raises
    "self.dim() cannot be 0 to view Float as Byte" on a 0-dim tensor. Every
    parameter must therefore have at least one dimension. This reproduces the
    exact operation rather than just asserting on shape."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16)

    modules = {
        "Ouroboros": _built(Ouroboros, x),
        "OuroborosBudget": registry.lookup("regularizers", "ouroboros_budget")(),
        "Servant": _built(Servant, x),
    }
    for tag, module in modules.items():
        for name, param in module.named_parameters():
            assert param.dim() > 0, f"{tag}.{name} is 0-dim"
            # The swap itself, byte-for-byte.
            param.detach().clone().view(torch.uint8)


def test_accounting_drains_and_skips_eval():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)

    regularizer = registry.lookup("regularizers", "ouroboros_budget")()
    activation = _built(Ouroboros, x).train()
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
