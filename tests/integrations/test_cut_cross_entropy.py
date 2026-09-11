"""The cut_cross_entropy integration (integrations/cut_cross_entropy/main.py)."""

import pytest
import torch
from torch import nn

pytest.importorskip("cut_cross_entropy")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="impl='cce' is a CUDA/Triton kernel"
)
def test_cut_cross_entropy_with_tied_weights():
    """A tied classifier has a weight and no bias; the loss must not need one."""
    from integrations.cut_cross_entropy.main import CutCrossEntropyLoss

    hidden_size, vocab_size, batch_size, seq_len = 128, 1024, 4, 16
    device = torch.device("cuda")

    class TiedProjection(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight

    scorer = TiedProjection(torch.randn(vocab_size, hidden_size, device=device))
    embeddings = torch.randn(batch_size, seq_len, hidden_size, device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Full unshifted tensors: the kernel shifts internally (shift=1).
    loss = CutCrossEntropyLoss()(
        embeddings=embeddings, scorer=scorer, labels=labels, input_ids=labels
    )
    assert torch.is_tensor(loss)
    assert torch.isfinite(loss)
    assert loss.item() > 0
