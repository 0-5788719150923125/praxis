"""Tests for praxis/spider/__init__.py: ``spider_settings`` resolution."""

import pytest

from praxis.spider import spider_settings


def test_disabled_when_flag_absent():
    assert spider_settings(None) is None


def test_bare_flag_uses_gentle_defaults():
    settings = spider_settings([])
    assert settings.profile == "gentle"
    assert settings.tick_seconds == 300


def test_key_value_overrides():
    settings = spider_settings(["profile=ghost", "max_sites=4"])
    assert settings.profile == "ghost"
    assert settings.max_sites == 4
    assert settings.tick_seconds == 15  # from ghost


def test_values_are_cast_to_their_field_types():
    # A dict from yml carries strings; the CLI list carries key=value text.
    assert spider_settings({"tick_seconds": "600"}).tick_seconds == 600
    settings = spider_settings(["tick_seconds=3600", "revisit_days=0.05"])
    assert settings.tick_seconds == 3600
    assert settings.revisit_days == 0.05  # fractional, not truncated


@pytest.mark.parametrize("bad", [["nope"], ["profile=missing"], ["bogus=1"]])
def test_invalid_entries_raise(bad):
    with pytest.raises(ValueError):
        spider_settings(bad)
