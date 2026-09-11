"""Tests for the engagement-prediction reward (P2) and policy (P3)."""

import pytest

from praxis.policies import needs_rl_datasets, normalize_rl_types


def test_engagement_needs_no_rl_datasets():
    assert normalize_rl_types("engagement") == ["engagement"]
    assert needs_rl_datasets("engagement") is False
    # Coexists with a weight controller; neither pulls the RL collection.
    assert needs_rl_datasets("harmonic_weight_wave") is False
