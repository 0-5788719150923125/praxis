"""GNS batch governor: estimator math, tier control, Lightning wiring."""

import pytest

from praxis.data.batch_schedule import BatchSchedule


@pytest.fixture(autouse=True)
def _clean_schedule():
    """The schedule is class-level state a governor writes on construction."""
    BatchSchedule.reset()
    yield
    BatchSchedule.reset()
