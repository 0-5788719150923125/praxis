import os
import sys

import pytest

from praxis import PraxisConfig

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def config():
    """Base configuration fixture."""
    return PraxisConfig()
