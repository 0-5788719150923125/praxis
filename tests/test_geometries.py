"""Center-PCA figure layout: the panel grid must fit inside the text block.

Regression guard for a failure that only appears at 5+ panels, which is rare
enough to ship unnoticed: a fixed 0.46\\linewidth panel width put three rows plus
a seven-line caption past \\textheight, and the caption printed over the page
number. The width now scales with the row count.
"""

from praxis.pillars.geometries import (
    _CAPTION_ALLOWANCE,
    _MAX_PANEL_WIDTH,
    _PANEL_ASPECT,
    _TEXT_HEIGHT,
    _panel_width,
    figure_tex,
)


def test_one_and_two_row_grids_keep_the_two_up_width():
    """The common case (up to 4 panels) must not shrink - only overflow does."""
    assert _panel_width(1) == _MAX_PANEL_WIDTH
    assert _panel_width(2) == _MAX_PANEL_WIDTH


def test_grid_plus_caption_fits_the_text_block():
    # Past two rows the width is derived to consume the budget exactly, so the
    # comparison lands on the boundary and needs a float tolerance rather than
    # a strict <=. The margin that matters is already inside _TEXT_HEIGHT.
    for rows in range(1, 9):
        height = rows * _panel_width(rows) * _PANEL_ASPECT + _CAPTION_ALLOWANCE
        assert height <= _TEXT_HEIGHT + 1e-9, f"{rows} rows overflow: {height:.3f}"


def test_width_never_exceeds_the_two_up_default():
    assert all(_panel_width(r) <= _MAX_PANEL_WIDTH for r in range(0, 9))


def test_five_panels_emit_a_narrowed_three_row_grid():
    tex = figure_tex([f"figures/geometry_{i}.png" for i in range(1, 6)], [])
    assert tex.count("includegraphics") == 5
    assert tex.count(r"\\[6pt]") == 2  # three rows -> two row breaks
    assert "0.460\\linewidth" not in tex
