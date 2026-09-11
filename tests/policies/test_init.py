"""Tests for the helpers in praxis/policies/__init__.py."""

import pytest

from praxis.policies import normalize_rl_types


@pytest.mark.parametrize(
    "rl_type,expected",
    [
        (None, []),
        ("engagement", ["engagement"]),
        ("engagement, harmonic_weight_wave", ["engagement", "harmonic_weight_wave"]),
        (["joke", " preference ", ""], ["joke", "preference"]),
    ],
)
def test_normalize_rl_types(rl_type, expected):
    assert normalize_rl_types(rl_type) == expected
