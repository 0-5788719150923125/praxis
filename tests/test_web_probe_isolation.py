"""Web probes run on the API thread. They must not touch training state.

The dashboard samples the live model while training runs on another thread.
Anything a probe does to shared state is visible mid-step to the trainer, and
the module tree is shared state:

  * ``torch.func.functional_call`` swaps entries in a module's ``_parameters``
    dict. It is not thread-safe, and the activations it was briefly used on here
    live INSIDE ``memory_model`` (``NeuralMemory(model=..., activation=serpent)``).
    The training thread read those swapped, detached tensors and died with "One
    of the differentiated Tensors does not require grad" - and, when the swap
    landed during the memory's own vmap, "tensor escaped from inside a function
    being vmapped". Hundreds of steps in, only under the full launcher, which is
    why no single-process test run ever caught it.
  * Building an autograd graph through live parameters lets the optimizer bump a
    version counter between the probe's forward and its backward.

So a probe reads, and does nothing else: no parameter swap, no graph, no grads.
A torn read costs one wrong sample on one poll; the next poll fixes it.
"""

import torch
import torch.nn as nn


class Activation(nn.Module):
    """Feature-dim parameters, like Serpent/Servant."""

    def __init__(self, dim=8):
        super().__init__()
        self.a = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return torch.sin(self.a * x) + self.b


def _probe(module, points=33):
    from praxis.web.routes.dynamics import _sample_activation

    return _sample_activation(
        module, -3.0, 3.0, points, torch.device("cpu"), torch.float32
    )


def test_probe_does_not_mutate_parameters():
    module = Activation()
    before = {n: p.detach().clone() for n, p in module.named_parameters()}
    assert _probe(module) is not None
    for name, p in module.named_parameters():
        assert torch.equal(p, before[name]), f"{name} was mutated by the probe"
        assert isinstance(p, nn.Parameter), f"{name} was swapped out for a plain tensor"


def test_probe_creates_no_gradients():
    """No graph through live parameters - that is the version-counter race."""
    module = Activation()
    assert _probe(module) is not None
    assert all(p.grad is None for p in module.parameters())
    assert all(p.requires_grad for p in module.parameters())


def test_probe_leaves_parameters_usable_by_functorch():
    """The exact operations the memory performs on its own parameters after a
    probe has run: a batched tensor left installed breaks both."""
    module = Activation()
    assert _probe(module) is not None
    for p in module.parameters():
        p.unsqueeze(0).expand(4, *p.shape)  # _init_weights
        p.detach().cpu()  # Lightning teardown


def test_probe_derivative_is_right():
    """Read-only means a numeric derivative; it still has to be correct."""
    module = Activation()
    sample = _probe(module, points=201)
    x = torch.tensor(sample["x"])
    got = torch.tensor(sample["backward"])
    want = module.a[0] * torch.cos(module.a[0] * x)  # d/dx sin(a x)
    interior = slice(2, -2)
    torch.testing.assert_close(
        got[interior], want[interior].to(got.dtype), rtol=0.02, atol=0.02
    )


def test_probe_does_not_reparametrize(monkeypatch):
    """A hard guard: functional_call must never be reached from this path."""
    import torch.func

    def boom(*a, **k):
        raise AssertionError("probe reparametrized a live module")

    monkeypatch.setattr(torch.func, "functional_call", boom)
    assert _probe(Activation()) is not None
