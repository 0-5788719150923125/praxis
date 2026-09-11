"""Smoke tests for praxis.orchestration: the remote-expert pooling layer.

Uses trivially small in-process experts (LocalExpert wrapping a plain Linear
block) so the pool's mechanics - capacity reporting, non-blocking detached
training, stochastic-sampled inference, and the mixing strategies - are
exercised without any transport or real model.
"""

import torch
from torch import nn

from praxis.orchestration import LocalExpert

HIDDEN = 14
VOCAB = 16


def _make_expert(uid: str) -> LocalExpert:
    block = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.SiLU())
    return LocalExpert(uid, block, hidden_size=HIDDEN, vocab_size=VOCAB, lr=1e-2)


def _batch():
    acts = torch.randn(2, 6, HIDDEN)
    labels = torch.randint(0, VOCAB, (2, 6))
    return acts, labels


def test_local_expert_trains():
    e = _make_expert("e0")
    acts, labels = _batch()
    first = e.train_step(acts, labels)
    for _ in range(30):
        loss = e.train_step(acts, labels)
    assert loss < first  # it actually learns its own local objective
    assert e.steps == 31 and e.rank() == HIDDEN
