import pytest

from praxis import PraxisConfig


@pytest.fixture
def config():
    """Base configuration fixture."""
    return PraxisConfig()
