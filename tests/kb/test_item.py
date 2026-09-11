"""Tests for praxis/kb/item.py: ``with_provenance`` stamping."""

from praxis.kb import KBItem
from praxis.kb.item import with_provenance


def _item(**overrides):
    base = dict(
        id="doc:test",
        type="doc",
        label="Wiki",
        title="Test Doc",
        body="# Heading\nFirst real line of content.\nSecond line.",
        uri="docs/test.md",
        updated=100.0,
    )
    base.update(overrides)
    return KBItem(**base)


def test_with_provenance_stamps_source_and_summary():
    stamped = with_provenance(_item(), "docs")
    assert stamped.source == "docs"
    assert stamped.summary == "First real line of content."


def test_with_provenance_keeps_explicit_summary():
    stamped = with_provenance(_item(summary="custom"), "docs")
    assert stamped.summary == "custom"
