"""Sweeps over every entry of the losses and regularizers registries."""

import pytest
import torch
from torch import nn

from praxis import registry
from praxis.activations.ouroboros import drain_step_counts

VOCAB, HIDDEN = 1024, 128


@pytest.mark.parametrize("name", list(registry.namespace("losses").keys()))
def test_every_loss_is_a_differentiable_scalar(name):
    """Built the way get_loss_function builds it, on pre-shifted inputs."""
    torch.manual_seed(0)
    loss_function = registry.lookup("losses", name)(vocab_size=VOCAB)
    classifier = nn.Linear(HIDDEN, VOCAB)
    embeddings = torch.randn(4, 16, HIDDEN)
    labels = torch.randint(0, VOCAB, (4, 16))

    loss = loss_function(
        logits=classifier(embeddings)[..., :-1, :].contiguous(),
        embeddings=embeddings[..., :-1, :].contiguous(),
        classifier=classifier,
        labels=labels[..., 1:].contiguous(),
        input_ids=labels,
    )

    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert classifier.weight.grad is not None
    assert torch.isfinite(classifier.weight.grad).all()


@pytest.mark.parametrize("name", list(registry.namespace("regularizers").keys()))
def test_every_regularizer_builds_and_tolerates_the_bare_call(name):
    """Every entry builds with ``pad_id``, reports through training_metrics(),
    and - handed no classifier, head or activation to act on - returns a finite
    scalar rather than raising. A ``*_probe`` entry is an instrument: observe
    only, and never a graph."""
    drain_step_counts()  # ouroboros_budget reads a module-global stack
    reg = registry.lookup("regularizers", name)(pad_id=0)
    h = torch.randn(2, 8, 16, requires_grad=True)

    out = reg(h, torch.randint(1, 32, (2, 8)))

    assert out.ndim == 0 and torch.isfinite(out)
    assert isinstance(reg.training_metrics(), dict)
    if name.endswith("_probe"):
        assert reg.observe_only is True
        assert out.item() == 0.0 and not out.requires_grad


@pytest.mark.parametrize("name", list(registry.namespace("regularizers").keys()))
def test_no_zero_dim_parameters(name):
    """schedule_free's swap() views every parameter as uint8, which a 0-dim
    tensor cannot do (ouroboros_budget's lambda_raw is shape [1] for this)."""
    reg = registry.lookup("regularizers", name)(pad_id=0)
    zero_dim = [n for n, p in reg.named_parameters() if p.dim() == 0]
    assert zero_dim == [], zero_dim
