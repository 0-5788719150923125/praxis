"""Business-card projections (praxis/pillars/projections.py): deterministic per
seed, and every modulation axis visible in the output."""

from praxis.pillars.projections import render_card, render_sheet_pdf

AUTHORS = ["Ryan J. Brooks"]
DONATE = "https://example.com/donate"


def test_render_card_is_a_function_of_its_arguments():
    """Same arguments, same bytes; side, seed and chaos each change them."""
    kwargs = dict(authors=AUTHORS, donations=DONATE, run_hash="abc123")
    a = render_card("front", 42, "light", 161, chaos=0.0, **kwargs)
    assert a == render_card("front", 42, "light", 161, chaos=0.0, **kwargs)
    assert a != render_card("back", 42, "light", 161, chaos=0.0, **kwargs)
    assert a != render_card("front", 43, "light", 161, chaos=0.0, **kwargs)
    assert a != render_card("front", 42, "light", 161, chaos=1.0, **kwargs)


def test_sheet_is_pdf():
    out = render_sheet_pdf("back", 7, "dark", 200, AUTHORS, DONATE, "abc123")
    assert out[:4] == b"%PDF"
