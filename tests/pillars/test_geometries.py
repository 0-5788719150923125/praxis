"""Center-PCA figure layout: the panel grid must fit inside the text block.

Regression guard for a failure that only appears at 5+ panels, which is rare
enough to ship unnoticed: a fixed 0.46\\linewidth panel width put three rows plus
a seven-line caption past \\textheight, and the caption printed over the page
number. The width now scales with the row count.
"""

import inspect

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


def test_halo_prototypes_are_not_mistaken_for_crystal_geometry():
    """``HaloClassifier`` also names its parameter ``centers`` - exclude it.

    prismatic5/6 carry a HaloHead arm whose ``lm_head.centers`` is [vocab, dim]
    exactly like a crystal's, so the bare ``.centers`` suffix swept it into the
    figure as a fifth "crystal head". It is not one: HALO's prototypes live on a
    hypersphere, and its gate share is near zero, so the panel was a raw ``randn``
    init - a featureless blob printed under a caption claiming settled geometry.

    The discriminator is structural: HaloClassifier owns a learnable ``gamma``
    beside its centers, CrystalClassifier owns nothing beside them.
    """
    from praxis.pillars.geometries import is_crystal_centers

    crystal = "model.head.branches.0.bank.experts.2.centers"
    halo = "model.head.branches.2.lm_head.centers"
    sd = {
        crystal: object(),
        halo: object(),
        "model.head.branches.2.lm_head.gamma": object(),
    }
    assert is_crystal_centers(sd, crystal)
    assert not is_crystal_centers(sd, halo)


def test_caption_names_the_head_the_run_actually_used():
    """The bank clause hardcoded ``prismatic4`` and kept asserting it.

    abstractinator-n ran ``head_type: prismatic6``, and the figure still told the
    reader it was looking at prismatic4's bank. The head name now comes from the
    run's own spec, and falls back to naming no head at all rather than guessing.
    """
    from praxis.pillars.geometries import figure_tex

    def geo(head_type):
        return {
            "name": "abstractinator-n",
            "label": "branch 0 - expert 0",
            "intra_run": True,
            "head_type": head_type,
        }

    tex = figure_tex(["figures/geometry_1.png"], [geo("prismatic6")])
    assert "prismatic6's VEAR crystal bank" in tex
    assert "prismatic4" not in tex

    # No spec on disk: say nothing rather than assert a head name.
    tex = figure_tex(["figures/geometry_1.png"], [geo(None)])
    assert "a VEAR crystal bank" in tex
    assert "prismatic" not in tex


def test_intra_run_panels_respect_the_limit():
    """A bank wider than the budget used to return every expert regardless.

    The cross-run path honoured ``limit``; the multi-head early return did not,
    so the figure could overflow its float page (and did, at five panels).
    """
    import praxis.pillars.geometries as g

    src = inspect.getsource(g.collect_geometries)
    assert "run_geos[:limit]" in src, (
        "the intra-run early return must truncate to `limit`, or a wide bank "
        "emits an unbounded number of panels"
    )
