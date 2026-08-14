"""The run-selector swatch must follow the theme, and must not be dropped by CSS.

Two bugs, one after the other:

  * The swatch baked a hex string into an inline ``style`` at render time.
    Charts survive a theme switch through the MutationObserver retint in
    charts.js; plain HTML does not, so the dots stayed green after switching to
    the blue hue.
  * The CSS fix then emitted ``--run-hue: 44deg`` while ``--accent-hue`` is a
    bare number. ``calc()`` refuses to add a <number> to an <angle>, the whole
    ``hsl()`` became invalid, and the swatch rendered with NO background at all.

So the invariant has two halves that have to agree, and they live in different
files: the JS emits the offset, the CSS adds it to the accent.
"""

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "praxis" / "web" / "src"
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
    calc = re.search(r"calc\(var\(--accent-hue\)\s*\+\s*var\(--run-hue,\s*([^)]*)\)\)", css)
    assert calc, "the swatch no longer composes its hue from --accent-hue"
    assert re.fullmatch(r"-?[\d.]+", calc.group(1).strip()), (
        f"--run-hue fallback {calc.group(1)!r} carries a unit"
    )


def test_swatch_has_no_baked_colour():
    """A hex written into inline style cannot follow a theme switch."""
    charts = CHARTS_JS.read_text()
    swatch = re.search(r'<span class="run-color-indicator[^"]*"[^>]*>', charts)
    assert swatch, "run-color-indicator markup not found"
    assert "background:" not in swatch.group(0), (
        "swatch bakes a colour inline; it must inherit from --accent-hue"
    )


def test_swatch_uses_the_same_palette_slot_as_the_chart_line():
    """The dot has to identify the line. The selector used the raw loop index
    while the charts use runColorIndex(run), so they disagreed once any run was
    filtered or reordered."""
    charts = CHARTS_JS.read_text()
    assert re.search(r"slot\s*=\s*runColorIndex\(run\)", charts), (
        "selector swatch is not keyed by runColorIndex(run)"
    )
    assert re.search(r"chartLineColorVars\(\s*slot\s*\)", charts), (
        "selector swatch no longer derives its hue from the palette slot"
    )


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


@pytest.mark.parametrize("accent", [161, 212, 16])
def test_css_and_js_agree_on_the_resulting_hue(accent):
    """CSS computes accent + (base - ref); JS computes base + (accent - ref).
    Same number, two files - pinned so a change to either is caught."""
    source = STATE_JS.read_text()
    ref = int(re.search(r"CHART_PALETTE_REF_HUE\s*=\s*(\d+)", source).group(1))
    palette = re.findall(r"\[\s*(\d+),\s*(\d+),\s*(\d+)\s*\]", source)
    assert palette, "palette not found"
    for h, _, _ in palette[:10]:
        css = (accent + (int(h) - ref)) % 360
        js = (int(h) + (accent - ref)) % 360
        assert css == js
