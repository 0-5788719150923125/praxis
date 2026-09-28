"""The local decoder's byte/trunk merge, and the in-context copy probe that
says whether the trunk's long-range signal reaches the output at all."""

import pytest
import torch
import torch.nn as nn

from praxis.metrics.copy_probe import copy_gain
from praxis.utils.tensors import create_block_ids

# --- the copy probe ------------------------------------------------------------


class _Copier(nn.Module):
    """Predicts the token that followed the previous occurrence of the current
    one, when there is one - an ideal in-context copier - and uniform otherwise."""

    def __init__(self, vocab=64):
        super().__init__()
        self.vocab = vocab

    def forward(self, input_ids, labels=None):
        b, t = input_ids.shape
        logits = torch.zeros(b, t, self.vocab)
        for row in range(b):
            for i in range(t):
                seen = (input_ids[row, :i] == input_ids[row, i]).nonzero().flatten()
                if len(seen):
                    logits[row, i, input_ids[row, seen[-1] + 1]] = 30.0
        return type("Out", (), {"logits": logits})()


class _Amnesiac(_Copier):
    """Same interface, no use of context at all."""

    def forward(self, input_ids, labels=None):
        b, t = input_ids.shape
        return type("Out", (), {"logits": torch.zeros(b, t, self.vocab)})()


def _distinct_rows(rows=2, t=40, vocab=64):
    """Rows of distinct tokens, so the only way to predict a repeat is to copy."""
    torch.manual_seed(0)
    return torch.stack([torch.randperm(vocab)[:t] for _ in range(rows)])


def test_copy_gain_separates_a_copier_from_a_model_without_context():
    ids = _distinct_rows()
    assert float(copy_gain(_Amnesiac(), ids, length=20)) == pytest.approx(0.0, abs=1e-6)
    gain = float(copy_gain(_Copier(), ids, length=20))
    assert gain > 5.0  # log2(64) = 6 bits per token, all but free the second time


class _DocumentCopier(_Copier):
    """_Copier that, like PraxisModel, cannot see past a separator: attention is
    confined to the block ``create_block_ids`` assigns."""

    def __init__(self, vocab=64, sep=0):
        super().__init__(vocab)
        self.config = type("Config", (), {"eos_token_id": sep})()

    def forward(self, input_ids, labels=None):
        blocks = create_block_ids(input_ids, self.config.eos_token_id)
        b, t = input_ids.shape
        logits = torch.zeros(b, t, self.vocab)
        for row in range(b):
            for i in range(t):
                same = (input_ids[row, :i] == input_ids[row, i]) & (
                    blocks[row, :i] == blocks[row, i]
                )
                seen = same.nonzero().flatten()
                if len(seen):
                    logits[row, i, input_ids[row, seen[-1] + 1]] = 30.0
        return type("Out", (), {"logits": logits})()


def test_copy_gain_reads_through_document_separators():
    torch.manual_seed(0)
    ids = torch.stack([torch.randperm(63)[:40] + 1 for _ in range(2)])
    ids[:, 5] = 0  # separators inside the copied passage
    ids[:, 13] = 0
    # Left in, each separator would start a new block in both copies and hide
    # every first-copy twin from its second-copy position.
    assert float(copy_gain(_DocumentCopier(sep=0), ids, length=20)) > 5.0


def test_copy_gain_runs_the_uncompiled_module():
    wrapper = nn.Module()
    wrapper._orig_mod = _Copier()
    wrapper.forward = lambda **kwargs: pytest.fail("the compiled module was called")
    assert float(copy_gain(wrapper, _distinct_rows(), length=20)) > 5.0


def test_copy_gain_skips_what_it_cannot_read():
    assert copy_gain(_Copier(), torch.arange(6).view(1, -1)) is None  # too short
    codec = _Copier()
    codec.encoder = type("Encoder", (), {"handles_loss": True})()
    assert copy_gain(codec, _distinct_rows(), length=20) is None
