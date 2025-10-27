import math

import pytest
import torch
from torch import nn

from praxis.losses import LOSS_REGISTRY

LOSS_FUNCTIONS = list(LOSS_REGISTRY.values())


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


def test_cut_cross_entropy_with_tied_weights():
    """Test cut_cross_entropy with tied weights (no bias)."""
    # Import from integration
    try:
        from integrations.cut_cross_entropy.main import CutCrossEntropyLoss
    except ImportError:
        pytest.skip("cut_cross_entropy integration not installed")

    hidden_size = 128
    vocab_size = 1024
    batch_size = 4
    seq_len = 16

    # Use GPU if available (required for cce implementation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a classifier without bias (like TiedClassifier)
    class MockTiedClassifier(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight

    embedding_weight = torch.randn(vocab_size, hidden_size, device=device)
    classifier = MockTiedClassifier(embedding_weight)

    loss_function = CutCrossEntropyLoss()
    # Use FULL UNSHIFTED embeddings - cut_cross_entropy handles shifting with shift=1
    embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device)
    labels = torch.randint(
        low=0, high=vocab_size, size=(batch_size, seq_len), device=device
    )

    # Should not raise AttributeError for missing bias
    # Pass full unshifted tensors - shift=1 handles it internally
    loss = loss_function(
        embeddings=embeddings,
        classifier=classifier,
        labels=labels,
        input_ids=labels,  # Unshifted targets
    )

    assert torch.is_tensor(loss)
    assert not math.isnan(loss)
    assert loss.item() > 0  # Cross-entropy should be positive
