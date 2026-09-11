"""Shared fixtures for tests/data: class-level state resets and a stub sampler."""

import pytest

from praxis.data.batch_schedule import BatchSchedule
from praxis.data.datasets.manager import InterleaveDataManager


@pytest.fixture(autouse=True)
def _clean_schedule():
    """The batch schedule is class-level state a governor writes on
    construction; a leaked schedule reshapes every governed dataset."""
    BatchSchedule.reset()
    yield
    BatchSchedule.reset()


@pytest.fixture(autouse=True)
def _interleave_state(monkeypatch):
    """InterleaveDataManager shares weights, losses and task weights across
    instances through class attributes."""
    for attr, value in (
        ("shared_weights", None),
        ("shared_weights_initialized", False),
        ("shared_losses", None),
        ("shared_task_weights", None),
    ):
        monkeypatch.setattr(InterleaveDataManager, attr, value)


class _StubSampler:
    """A sampler with a fixed task that serves a canned (or supplied) document."""

    def __init__(self, name="ds", get_document=None, task_type=0):
        self.dataset_path = name
        self.task_type = task_type
        self.weight = 1.0
        if get_document is not None:
            self.get_document = get_document

    def get_document(self):
        return {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "metadata": {},
        }


@pytest.fixture
def make_sampler():
    return _StubSampler
