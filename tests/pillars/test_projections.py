import pytest

from praxis.pillars.projections import render_card, render_sheet_pdf

AUTHORS = ["Ryan J. Brooks"]
DONATE = "https://example.com/donate"


def test_render_card_deterministic():
    kwargs = dict(authors=AUTHORS, donations=DONATE, run_hash="abc123")
    a = render_card("front", 42, "light", 161, **kwargs)
    b = render_card("front", 42, "light", 161, **kwargs)
    assert a == b
    assert a != render_card("back", 42, "light", 161, **kwargs)
    assert a != render_card("front", 43, "light", 161, **kwargs)


def test_chaos_changes_field():
    kwargs = dict(authors=AUTHORS, donations=DONATE, run_hash="abc123")
    lo = render_card("front", 42, "light", 161, chaos=0.0, **kwargs)
    hi = render_card("front", 42, "light", 161, chaos=1.0, **kwargs)
    assert lo != hi
    assert lo == render_card("front", 42, "light", 161, chaos=0.0, **kwargs)


def test_sheet_is_pdf():
    out = render_sheet_pdf("back", 7, "dark", 200, AUTHORS, DONATE, "abc123")
    assert out[:4] == b"%PDF"
