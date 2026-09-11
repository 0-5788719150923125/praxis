"""Fixtures for the orchestration tests: pools that shut down and a status
module that is left empty."""

import pytest

from praxis.orchestration import status
from praxis.orchestration.pool import ExpertPool


@pytest.fixture(autouse=True)
def _clean_status():
    """``ExpertPool.capacity()`` publishes into the process-global status
    module; leave it as a fresh process would find it."""
    status.clear()
    yield
    status.clear()


@pytest.fixture
def pools(monkeypatch):
    """Every ExpertPool built during the test, shut down afterwards so its
    worker threads do not outlive the test."""
    built = []
    original = ExpertPool.__init__

    def tracked(self, *args, **kwargs):
        original(self, *args, **kwargs)
        built.append(self)

    monkeypatch.setattr(ExpertPool, "__init__", tracked)
    yield built
    for pool in built:
        pool.shutdown()
