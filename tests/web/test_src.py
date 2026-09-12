"""Source guards over the frontend (praxis/web/src js and css).

Invariants a browser run would show but that span two files, pinned at the
source until a browser probe replaces them.

The run-selector swatch must follow the theme, and must not be dropped by CSS.

A hex baked into an inline ``style`` cannot follow a theme switch (charts
retint through a MutationObserver; plain HTML does not). And ``--run-hue`` must
stay unitless like ``--accent-hue``: ``calc()`` refuses <number> + <angle>, the
whole ``hsl()`` goes invalid, and the swatch renders with no background.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "praxis" / "web" / "src"
STATE_JS = SRC / "js" / "state.js"
CHARTS_JS = SRC / "js" / "charts.js"
COMPONENTS_CSS = SRC / "css" / "components.css"
VARIABLES_CSS = SRC / "css" / "variables.css"


def test_accent_hue_is_unitless():
    """Everything below depends on this. If it ever gains a unit, the calc()
    flips to being valid only WITH units and these tests must flip with it."""
    declarations = re.findall(r"--accent-hue:\s*([^;]+);", VARIABLES_CSS.read_text())
    assert declarations, "no --accent-hue defined"
    for value in declarations:
        assert re.fullmatch(r"-?[\d.]+", value.strip()), (
            f"--accent-hue: {value!r} carries a unit; "
            "chartLineColorVars emits a bare number to match"
        )


def test_run_hue_is_emitted_without_a_unit():
    """calc(<number> + <angle>) is invalid and silently drops the background."""
    source = STATE_JS.read_text()
    body = re.search(
        r"export function chartLineColorVars\(index\)\s*\{(.*?)\n\}", source, re.S
    )
    assert body, "chartLineColorVars not found"
    emitted = re.search(r"--run-hue:\$\{[^}]+\}([a-z]*)", body.group(1))
    assert emitted, "chartLineColorVars no longer emits --run-hue"
    assert emitted.group(1) == "", (
        f"--run-hue emitted with unit {emitted.group(1)!r}; "
        "--accent-hue is unitless, so calc() would reject the sum"
    )


def test_css_default_for_run_hue_is_unitless():
    """The fallback has to be unitless too, or an un-styled swatch drops out."""
    css = COMPONENTS_CSS.read_text()
    calc = re.search(
        r"calc\(var\(--accent-hue\)\s*\+\s*var\(--run-hue,\s*([^)]*)\)\)", css
    )
    assert calc, "the swatch no longer composes its hue from --accent-hue"
    assert re.fullmatch(
        r"-?[\d.]+", calc.group(1).strip()
    ), f"--run-hue fallback {calc.group(1)!r} carries a unit"


def test_swatch_has_no_baked_colour():
    """A hex written into inline style cannot follow a theme switch."""
    charts = CHARTS_JS.read_text()
    swatch = re.search(r'<span class="run-color-indicator[^"]*"[^>]*>', charts)
    assert swatch, "run-color-indicator markup not found"
    assert "background:" not in swatch.group(
        0
    ), "swatch bakes a colour inline; it must inherit from --accent-hue"


def test_swatch_uses_the_same_palette_slot_as_the_chart_line():
    """The dot has to identify the line. The selector used the raw loop index
    while the charts use runColorIndex(run), so they disagreed once any run was
    filtered or reordered."""
    charts = CHARTS_JS.read_text()
    assert re.search(
        r"slot\s*=\s*runColorIndex\(run\)", charts
    ), "selector swatch is not keyed by runColorIndex(run)"
    assert re.search(
        r"chartLineColorVars\(\s*slot\s*\)", charts
    ), "selector swatch no longer derives its hue from the palette slot"


def test_colours_are_assigned_by_selection_not_history():
    """The palette is ten hues spread around the wheel, so slots 1-3 are the ones
    maximally distinct from the accent. Indexing on the FULL run history handed
    the second selected run slot 7 ("leaf green"), which sits beside the accent
    green - and two green lines on nearly-coincident validation curves read as
    one. That is what "the older run does not show up" actually was: both series
    were present, plotted and visible, in indistinguishable colours."""
    charts = CHARTS_JS.read_text()
    body = re.search(r"function runColorIndex\(run\)\s*\{(.*?)\n\}", charts, re.S)
    assert body, "runColorIndex not found"
    assert "selectedHistoricalRuns" in body.group(1), (
        "runColorIndex indexes the whole history again; selected runs must take "
        "consecutive palette slots so they stay distinguishable"
    )


def test_cross_run_charts_do_not_match_points_by_array_index():
    """Run comparison must hover by x VALUE, not by position in the array.

    Chart.js's ``index`` mode returns the same array index from every dataset.
    Nothing keeps those indices aligned across runs - LTTB selects different
    rows per run, validation rows are sparse and deduped, and runs end at
    different lengths - so hovering compared one run's point against a point in
    another run at an unrelated x. Measured on the token axis with two real
    runs, a single hover matched 0.0044B tokens against 0.0166B, and a run
    shorter than the chosen index dropped out of the tooltip entirely.

    Source-level guard. The behaviour itself was verified in a browser against
    live run data; this only catches a revert to the built-in mode.
    """
    src = CHARTS_JS.read_text()

    assert "praxisNearestX" in src, "the nearest-by-x interaction mode is gone"
    assert src.count("crossRunInteraction()") >= 2, (
        "both the scalar run-comparison chart and the multi-series composite "
        "chart must use the x-matching interaction"
    )
    # One occurrence only: the fallback inside crossRunInteraction, taken when
    # Chart.js is somehow unavailable to register a custom mode. Any second one
    # is a chart wiring itself straight back to index-matching.
    index_modes = src.count("mode: 'index'")
    assert index_modes == 1, (
        "expected exactly one 'index' mode - crossRunInteraction's fallback - "
        f"but found {index_modes}"
    )


# ---------------------------------------------------------------------------
# localStorage keys
# ---------------------------------------------------------------------------

CONFIG_JS = SRC / "js" / "config.js"


def _registered_storage_keys() -> set:
    body = re.search(
        r"export const STORAGE_KEYS = \{(.*?)\n\};", CONFIG_JS.read_text(), re.S
    )
    assert body, "STORAGE_KEYS not found in config.js"
    return set(re.findall(r"^\s*'?([\w:]+)'?\s*:", body.group(1), re.M))


def test_every_storage_key_used_is_registered():
    """`storage.get`/`set` look the key up in STORAGE_KEYS and, on a miss,
    console.warn and return without touching localStorage. That is silent by
    design in the browser, and it is exactly how the Settings form stopped
    persisting: the generation-kwargs keys were read and written all run and
    never stored. A miss has to fail here instead."""
    registered = _registered_storage_keys()
    used = {}
    for path in sorted((SRC / "js").glob("*.js")):
        for key in re.findall(r"storage\.(?:get|set|remove)\(\s*[\'\"`]([\w:]+)", path.read_text()):
            used.setdefault(key, path.name)

    missing = sorted(f"{k} (used in {used[k]})" for k in used if k not in registered)
    assert not missing, (
        "localStorage keys used but not registered in STORAGE_KEYS, so every "
        f"read and write of them is a silent no-op: {missing}"
    )


def test_registered_storage_keys_map_to_distinct_slots():
    """Two names sharing one localStorage slot would overwrite each other."""
    body = re.search(
        r"export const STORAGE_KEYS = \{(.*?)\n\};", CONFIG_JS.read_text(), re.S
    ).group(1)
    slots = re.findall(r":\s*\'([^\']+)\'", body)
    assert len(slots) == len(set(slots)), f"duplicate localStorage slots: {slots}"


def test_a_run_default_has_a_seed_slot_beside_it():
    """`resolveDefault` tells an edit from an untouched default by comparing the
    stored value against the default it was SEEDED with, so every value seeded
    from the run needs its `:default` companion registered too."""
    registered = _registered_storage_keys()
    for key in ("developerPrompt", "generationKwargs"):
        assert key in registered
        assert f"{key}:default" in registered, f"{key} has no seed slot"
