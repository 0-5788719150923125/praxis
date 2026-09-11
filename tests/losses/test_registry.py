import math

import pytest
import torch
from torch import nn

from praxis import registry
from praxis.losses.regularizers import build_regularizers

# ------------------------------------------------------------------------------
# loss_functions
# ------------------------------------------------------------------------------


LOSS_FUNCTIONS = list(registry.namespace("losses").values())


@pytest.fixture(params=LOSS_FUNCTIONS)
def module_setup(request, config):
    hidden_size = 128
    vocab_size = 1024
    classifier = nn.Linear(hidden_size, vocab_size)
    loss_function = request.param()
    return loss_function, classifier, hidden_size, vocab_size


def test_forward_pass(module_setup):
    """Test using parametrized module and dimensions."""
    loss_function, classifier, hidden_size, vocab_size = module_setup
    batch_size = 4
    seq_len = 16

    # cut_cross_entropy requires GPU (uses Triton kernels)
    is_cut_ce = loss_function.__class__.__name__ == "CutCrossEntropyLoss"
    device = torch.device("cuda" if is_cut_ce and torch.cuda.is_available() else "cpu")

    # Move classifier to device
    classifier = classifier.to(device)

    embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device)
    logits = classifier(embeddings)
    labels = torch.randint(
        low=0, high=vocab_size, size=(batch_size, seq_len), device=device
    )

    # cut_cross_entropy uses UNSHIFTED embeddings with shift=1 internally
    # Other loss functions use pre-shifted embeddings
    if is_cut_ce:
        loss_embeddings = embeddings  # Full unshifted
        loss_labels = labels  # Full unshifted (passed as input_ids)
    else:
        loss_embeddings = embeddings[..., :-1, :].contiguous()
        loss_labels = labels[..., 1:].contiguous()

    loss = loss_function(
        logits=logits[..., :-1, :].contiguous(),
        embeddings=loss_embeddings,
        classifier=classifier,
        labels=loss_labels,
        input_ids=labels,
    )
    assert torch.is_tensor(loss)
    assert not math.isnan(loss)


# ------------------------------------------------------------------------------
# regularizers
# ------------------------------------------------------------------------------
# Regularizer registry: default selection, build, and the activation option.


def test_multiple_regularizers_compose():
    reg = build_regularizers(list(registry.namespace("regularizers").keys()))
    assert len(reg) == len(registry.namespace("regularizers"))
    names = {m.name for m in reg}
    assert "contrastive" in names and "activation_reg" in names
