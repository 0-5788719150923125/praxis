# tests/conftest.py
# from dataclasses import dataclass

import pytest

from praxis import PraxisConfig

# @dataclass
# class MockAutoConfig:
#     """Base configuration class for testing."""

#     hidden_size: int = 64
#     activation: str = "gelu"
#     dropout: float = 0.1

#     def update(self, **kwargs):
#         """Utility method to update config values."""
#         for key, value in kwargs.items():
#             if hasattr(self, key):
#                 setattr(self, key, value)
#         return self


@pytest.fixture
def config():
    """Base configuration fixture."""
    return PraxisConfig()


# @pytest.fixture
# def large_config():
#     """Configuration fixture with larger dimensions."""
#     return MockAutoConfig(hidden_size=256, dropout=0.2)


# @pytest.fixture
# def minimal_config():
#     """Configuration fixture with minimal dimensions."""
#     return MockAutoConfig(hidden_size=32, dropout=0.0)
