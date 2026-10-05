"""The in-context learning score read off a validation forward: late positions in
a document against early ones, counted within the document."""

import math

import pytest
import torch

from praxis.metrics.context_gain import context_gain

V, T = 16, 64  # early band [2, 4), late band [32, 64)


def _logits(ids, confident):
    """Logits that predict each next token with confidence ``confident[b, t]``
    (0 = uniform), so a position's loss is set by hand."""
    b, t = ids.shape
    logits = torch.zeros(b, t, V)
    nxt = ids[:, 1:]
    logits[:, :-1].scatter_(2, nxt.unsqueeze(-1), confident[:, :-1].unsqueeze(-1))
    return logits


def test_a_model_that_learns_from_context_scores_its_savings():
    ids = torch.randint(0, V, (2, T))
    confident = torch.zeros(2, T)
    confident[:, T // 2 - 1 :] = 30.0  # targets from position T/2 on are certain
    value, rows = context_gain(_logits(ids, confident), ids)
    assert rows == 2
    assert float(value) == pytest.approx(math.log2(V), abs=1e-3)


def test_a_model_that_ignores_context_scores_zero():
    ids = torch.randint(0, V, (3, T))
    value, rows = context_gain(torch.zeros(3, T, V), ids)
    assert rows == 3 and float(value) == pytest.approx(0.0, abs=1e-6)


def test_positions_count_from_the_document_not_the_row():
    """A long document that starts mid-row is scored on ITS early positions:
    the row's early positions belong to another document."""
    ids = torch.randint(0, V, (1, T))
    block_ids = torch.ones(1, T, dtype=torch.long)
    block_ids[:, 8:] = 2  # document 2 runs 56 tokens, reaching position 32
    confident = torch.zeros(1, T)
    confident[:, :7] = 30.0  # document 1 is easy; document 2's opening is not
    confident[:, 8 + T // 2 - 1 :] = 30.0  # document 2's late band is easy
    value, rows = context_gain(_logits(ids, confident), ids, block_ids)
    assert rows == 1
    assert float(value) == pytest.approx(math.log2(V), abs=1e-3)


def test_short_documents_are_skipped():
    ids = torch.randint(0, V, (2, T))
    block_ids = (torch.arange(T) // (T // 4) + 1).expand(2, T)  # four 16-token docs
    assert context_gain(torch.zeros(2, T, V), ids, block_ids) is None
    assert context_gain(None, ids) is None
