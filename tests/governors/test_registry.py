"""Sweeps over the ``governors`` registry."""

import pytest

from praxis import registry
from praxis.callbacks.lightning.governor import GNSBatchGovernor
from praxis.data.batch_schedule import BatchSchedule


@pytest.fixture(autouse=True)
def _clean_schedule():
    """The schedule is class-level state a governor writes on construction."""
    BatchSchedule.reset()
    yield
    BatchSchedule.reset()


def test_registry_builds_callback_with_ceiling():
    gov = registry.lookup("governors", "gns_batch")(
        batch_size=16, target_batch_size=512, val_every=1024
    )
    assert isinstance(gov, GNSBatchGovernor)
    assert gov.controller.max_rows == 512  # target_batch_size, in rows
    assert gov.controller.min_rows == 2  # NOT batch_size
    assert gov.row_ceiling == 16  # batch_size, a per-microbatch ceiling
    assert gov.val_every == 1024
