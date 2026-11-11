"""Wrapper optimizers resume from a checkpoint exactly where they left off."""

import io

import pytest
import torch
import torch.nn as nn

from praxis.optimization.wrappers import SequentialWrapper

STATEFUL = [
    "schedule_free",
    "gated_schedule_free",
    "wave_schedule_free",
    "lookahead",
    "half_lion",
]


def _build(key, seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4))
    optimizer = SequentialWrapper([key])(torch.optim.AdamW(model.parameters(), lr=0.01))
    if hasattr(optimizer, "train"):
        optimizer.train()  # as create_optimizer does
    return model, optimizer


def _steps(model, optimizer, n, start):
    for i in range(start, start + n):
        g = torch.Generator().manual_seed(i)
        x = torch.randn(32, 8, generator=g)
        loss = (model(x) - x[:, :4].sin()).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _round_trip(obj):
    """A checkpoint through torch.save, so Parameter keys come back as copies."""
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)


@pytest.mark.parametrize("key", STATEFUL)
@pytest.mark.parametrize("saved_in_eval", [False, True])
def test_a_resumed_run_continues_as_an_uninterrupted_one(key, saved_in_eval):
    model, optimizer = _build(key, seed=0)
    _steps(model, optimizer, 7, start=0)
    if saved_in_eval and hasattr(optimizer, "eval"):
        optimizer.eval()  # a checkpoint written during validation
    checkpoint = _round_trip(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    )
    if hasattr(optimizer, "train"):
        optimizer.train()
    _steps(model, optimizer, 6, start=7)

    # Resume as Lightning does: modules, then optimizer, then on_train_start.
    resumed, again = _build(key, seed=1)
    resumed.load_state_dict(checkpoint["model"])
    again.load_state_dict(checkpoint["optimizer"])
    if hasattr(again, "train"):
        again.train()
    _steps(resumed, again, 6, start=7)

    for a, b in zip(model.parameters(), resumed.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
    # Nothing unreachable is carried forward into the next checkpoint.
    assert len(state_dict_entries(again)) == len(state_dict_entries(optimizer))


def state_dict_entries(optimizer):
    state = optimizer.state_dict()
    inner = next(v for k, v in state.items() if k.endswith("_state"))
    assert all(isinstance(k, int) for k in inner)
    return inner
