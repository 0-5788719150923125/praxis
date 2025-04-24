import math

import pytest
import torch
from torch import nn

from praxis.losses import LOSS_REGISTRY

LOSS_FUNCTIONS = list(LOSS_REGISTRY.values())
LOSS_FUNCTIONS.pop(-1)  # remove "cut_cross_entropy"


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
    embeddings = torch.randn(batch_size, seq_len, hidden_size)
    logits = classifier(embeddings)
    labels = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    loss = loss_function(
        logits=logits[..., :-1, :].contiguous(),
        embeddings=embeddings,
        classifier=classifier,
        labels=labels[..., 1:].contiguous(),
        input_ids=labels,
    )
    assert torch.is_tensor(loss)
    assert not math.isnan(loss)
