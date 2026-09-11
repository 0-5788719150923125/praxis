import pytest

from praxis.activations.ouroboros import drain_step_counts


@pytest.fixture(autouse=True)
def _drain_ouroboros_steps():
    """Ouroboros pushes each training forward's step counts onto a module-global
    stack; drain it around every test so no graph outlives the test that made it."""
    drain_step_counts()
    yield
    drain_step_counts()
