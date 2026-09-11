"""ExpertPool (praxis/orchestration/pool.py): capacity reporting, training
dispatch, and sampled inference over small in-process LocalExperts.
"""

import pytest
import torch
from torch import nn

from praxis.orchestration import ExpertPool, LocalExpert, build_pool

HIDDEN = 14
VOCAB = 16


def _make_expert(uid: str) -> LocalExpert:
    block = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.SiLU())
    return LocalExpert(uid, block, hidden_size=HIDDEN, vocab_size=VOCAB, lr=1e-2)


def _batch():
    acts = torch.randn(2, 6, HIDDEN)
    labels = torch.randint(0, VOCAB, (2, 6))
    return acts, labels


pytestmark = pytest.mark.usefixtures("pools")


def test_pool_capacity_tracks_membership():
    pool = build_pool([_make_expert(f"e{i}") for i in range(4)], mixing="mean")
    cap = pool.capacity()
    assert cap["experts_total"] == 4
    assert cap["experts_alive"] == 4
    assert cap["total_rank"] == 4 * HIDDEN
    assert cap["mixing"] == "mean"

    pool.remove("e0")
    assert pool.capacity()["experts_total"] == 3


def test_pool_train_step_updates_every_live_expert():
    pool = build_pool([_make_expert(f"e{i}") for i in range(5)], mixing="mean")
    acts, labels = _batch()
    res = pool.train_step(acts, labels)
    assert res["dispatched"] == 5
    assert res["completed"] == 5
    assert res["mean_loss"] is not None
    # Each expert advanced its own counter - independent local updates.
    assert all(e.steps == 1 for e in pool.experts)


def test_pool_infer_mixes_to_model_shape():
    pool = build_pool([_make_expert(f"e{i}") for i in range(6)], mixing="mean")
    acts, _ = _batch()
    out = pool.infer(acts)
    assert out is not None
    # mean mixer reduces [E, B, T, H] -> [B, T, H]
    assert tuple(out.shape) == (2, 6, HIDDEN)


def test_stochastic_sampling_uses_subset():
    pool = ExpertPool(
        [_make_expert(f"e{i}") for i in range(8)], mixing="mean", sample_size=3
    )
    acts, _ = _batch()
    gen = torch.Generator().manual_seed(0)
    out = pool.infer(acts, generator=gen)
    assert out is not None
    # Only the sampled subset served a pass this step.
    served = sum(1 for e in pool.experts if e.passes > 0)
    assert served == 3


def test_empty_pool_infer_returns_none():
    assert ExpertPool([], mixing="mean").infer(torch.randn(1, 3, HIDDEN)) is None


def test_dead_expert_drops_out():
    pool = build_pool([_make_expert(f"e{i}") for i in range(3)], mixing="mean")
    pool.experts[1]._alive = False
    assert pool.capacity()["experts_alive"] == 2
    res = pool.train_step(*_batch())
    assert res["dispatched"] == 2  # the dead one is skipped
