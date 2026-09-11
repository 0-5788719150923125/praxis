import math

import pytest
import torch
from torch import nn


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
