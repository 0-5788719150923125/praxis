"""Dashboard card invariants for the Research-tab metric registries.

The Research tab builds its deck with ``buildScalarConfigsFromRegistry`` and
``buildCompositeConfigsFromRegistry`` (praxis/web/src/js/charts.js), which
concatenate ALL scalars ahead of ALL composites and sort each half flat by
``order``. Neither honours ``group``, ``group_order`` or ``series_group`` - those
belong to the Dynamics tab's manifest builder. Two things went wrong because of
that and are pinned here:

  * four (since removed) density entries carried ``series_group`` expecting to
    merge into two cards, so the deck rendered four;
  * they also carried ``order: 10``, tying with ``loss``, and a stable sort puts
    the earlier-declared entry first - which put a research probe at deck
    position 1, ahead of training loss.

The information-density probe now emits only ``readout_*`` keys into
extra_metrics (no schema columns), claimed by the composite cards pinned below.
"""

import re

import pytest

from praxis.metrics.training_metrics import (
    COMPOSITE_METRIC_REGISTRY,
    TRAINING_METRIC_REGISTRY,
    X_AXIS_REGISTRY,
)

READOUT_BANDS = ("bag", "coarse", "mid")
READOUT_POSITIONS = ("head", "q1", "q2", "q3", "q4", "q5", "q6", "tip")
READOUT_KEYS = tuple(
    f"readout_{band}_{suffix}"
    for band in READOUT_BANDS
    for suffix in ("rim_gap", "depth_gain")
) + tuple(
    f"readout_cell_{pos}_{i}"
    for pos in READOUT_POSITIONS
    for i in range(len(READOUT_BANDS))
)


def scalar_cards():
    """Scalar cards in the order the deck renders them."""
    entries = [
        (v["chart"].get("order", 0), k, v["chart"])
        for k, v in TRAINING_METRIC_REGISTRY.items()
        if v.get("chart")
    ]
    return sorted(entries, key=lambda e: e[0])


def test_training_loss_leads_the_deck():
    """Position 1 is training loss, 2 is validation loss."""
    order = [key for _, key, _ in scalar_cards()]
    assert order[0] == "loss"
    assert order[1] == "val_loss"


def test_no_scalar_card_outranks_training_loss():
    """A research probe must not tie its way in front of the headline metric."""
    loss_order = TRAINING_METRIC_REGISTRY["loss"]["chart"]["order"]
    for order, key, _ in scalar_cards():
        if key == "loss":
            continue
        assert order > loss_order, f"{key} (order {order}) renders before loss"


def test_card_titles_are_unique():
    """Two cards with the same title are indistinguishable in the carousel."""
    scalar_titles = [c["title"] for _, _, c in scalar_cards()]
    composite_titles = [e["title"] for e in COMPOSITE_METRIC_REGISTRY]
    for label, titles in (("scalar", scalar_titles), ("composite", composite_titles)):
        dupes = {t for t in titles if titles.count(t) > 1}
        assert not dupes, f"duplicate {label} card titles: {sorted(dupes)}"
    overlap = set(scalar_titles) & set(composite_titles)
    assert not overlap, f"title collides across registries: {sorted(overlap)}"


def test_readout_keys_are_not_schema_columns():
    """The probe's keys ride extra_metrics; a registry column per key would
    add 40 columns to every run's schema for one research card family."""
    for key in READOUT_KEYS:
        assert key not in TRAINING_METRIC_REGISTRY, f"{key} became a column"


def test_readout_composites_cover_every_readout_key():
    """Every emitted key must be claimed by some composite, or it vanishes."""
    patterns = [
        re.compile(e["key_pattern"])
        for e in COMPOSITE_METRIC_REGISTRY
        if e.get("key_pattern")
    ]
    for key in READOUT_KEYS:
        assert any(p.match(key) for p in patterns), f"{key} is on no card"


def test_readout_profile_is_one_heatmap_over_position_and_band():
    """fig:density is a shaded strip over position; the card is a heatmap
    whose row group is the position label and column group the band index."""
    by_key = {e["key"]: e for e in COMPOSITE_METRIC_REGISTRY}
    card = by_key["readout_profile"]
    assert card["type"] == "expert_routing_heatmap"
    assert card.get("uniform_note") is False  # cells are R², not routing weights
    pattern = re.compile(card["key_pattern"])
    cells = {k for k in READOUT_KEYS if pattern.match(k)}
    assert cells == {k for k in READOUT_KEYS if k.startswith("readout_cell_")}
    for pos in READOUT_POSITIONS:
        m = pattern.match(f"readout_cell_{pos}_1")
        assert m and m.group(1) == pos and m.group(2) == "1"


def test_readout_summary_cards_carry_every_band():
    by_key = {e["key"]: e for e in COMPOSITE_METRIC_REGISTRY}
    for card, suffix in (
        ("readout_rim_gap", "rim_gap"),
        ("readout_depth_gain", "depth_gain"),
    ):
        pattern = re.compile(by_key[card]["key_pattern"])
        assert {k for k in READOUT_KEYS if pattern.match(k)} == {
            f"readout_{band}_{suffix}" for band in READOUT_BANDS
        }


def test_composite_orders_do_not_collide():
    """Ties fall back to declaration order, which is invisible in the source."""
    orders = [e.get("order", 0) for e in COMPOSITE_METRIC_REGISTRY]
    dupes = {o for o in orders if orders.count(o) > 1}
    # Pre-existing ties are grandfathered; the point is that new cards do not
    # silently join them. Assert only that the ones we placed are clean.
    for key in (
        "readout_profile",
        "readout_rim_gap",
        "readout_depth_gain",
        "smear_coefficients",
        "smear_target_dispersion",
        "smear_input_dependence",
        "smear_expert_utilization",
    ):
        entry = next(e for e in COMPOSITE_METRIC_REGISTRY if e["key"] == key)
        assert (
            entry["order"] not in dupes
        ), f"{key} at order {entry['order']} ties with another card"


# --- run comparison ----------------------------------------------------------


def _payload(run_hash, limit=1000):
    """The exact per-run payload /api/metrics builds for the Research tab."""
    import pathlib

    from praxis.web.routes.metrics import (
        _downsample_metrics,
        _read_metrics_file,
        _transform_metrics,
    )

    db = pathlib.Path("build/runs") / run_hash / "metrics.db"
    if not db.exists():
        return None
    rows = _read_metrics_file(db, 0, max_rows=limit * 3)
    if not rows:
        return None
    if len(rows) > limit:
        rows = _downsample_metrics(rows, limit, "lttb")
    return _transform_metrics(rows)


def _has(payload, key):
    values = payload.get(key)
    return bool(values) and any(v is not None for v in values)


def test_no_run_loses_a_series_it_actually_recorded():
    """A run must surface every charted metric it genuinely has data for.

    Older runs used to vanish from the Research tab entirely: the SELECT named
    every registry column, so any run predating one raised ``no such column``,
    ``_read_metrics_file`` swallowed it, and the caller dropped the run - taking
    its loss curve with it. The projection is per-database now
    (``_projection_for``), and this pins the property that guarantees: what the
    database holds is what the payload serves.

    Deliberately NOT "every run draws every card". A run that stopped before its
    first validation step has no val_loss, and that is correct, not a defect.
    """
    import pathlib
    import sqlite3

    checked = 0
    for path in sorted(pathlib.Path("build/runs").iterdir())[:6]:
        db = path / "metrics.db"
        if not db.exists():
            continue
        payload = _payload(path.name, limit=200)
        if payload is None:
            continue
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
        try:
            columns = {r[1] for r in conn.execute("PRAGMA table_info(metrics)")}
            for key, entry in TRAINING_METRIC_REGISTRY.items():
                if not entry.get("chart") or key not in columns:
                    continue
                recorded = conn.execute(
                    f"SELECT COUNT(*) FROM metrics WHERE {key} IS NOT NULL"
                ).fetchone()[0]
                if recorded:
                    assert _has(payload, key), (
                        f"{path.name}: {key!r} has {recorded} recorded values "
                        f"but was dropped from the payload"
                    )
        finally:
            conn.close()
        checked += 1
    if checked == 0:
        pytest.skip("no runs on disk")


def test_core_metrics_survive_in_every_run_on_disk():
    """loss / val_loss are the comparison baseline - they must never drop out."""
    import pathlib

    checked = 0
    for path in sorted(pathlib.Path("build/runs").iterdir()):
        if not (path / "metrics.db").exists():
            continue
        payload = _payload(path.name, limit=200)
        if payload is None:
            continue
        checked += 1
        assert "steps" in payload, f"{path.name}: no step axis"
        assert _has(payload, "loss"), f"{path.name}: lost its loss series"
    if checked == 0:
        pytest.skip("no runs on disk")


# --- x axes ------------------------------------------------------------------


def test_every_x_axis_declares_what_the_frontend_reads():
    """The picker is registry-driven, so a malformed entry is a blank axis."""
    seen = set()
    for axis in X_AXIS_REGISTRY:
        for field in ("key", "label", "axis_title", "source", "order"):
            assert field in axis, f"x axis {axis.get('key')!r} is missing {field}"
        assert axis["key"] not in seen, f"duplicate x axis key {axis['key']!r}"
        seen.add(axis["key"])
    assert "step" in seen, "step must stay available as the fallback axis"


def test_validation_points_carry_every_x_coordinate():
    """A val point must know exactly where it sits on EVERY axis.

    This is the property the whole feature rests on. Validation is computed
    every ``val_check_interval`` steps, and the worry was that plotting it
    against tokens would silently lag by up to one interval. It does not:
    ``MetricsLoggerCallback.on_validation_end`` drains callback_metrics at
    ``trainer.global_step``, and ``MetricsLogger.log`` upserts on the step
    primary key, so the val write MERGES into the row the training step already
    wrote - carrying that step's exact num_tokens and ts.

    If a refactor ever splits validation onto its own row, the coordinates go
    null and this fails. That failure is the point: a val curve plotted against
    a carried-forward token count is wrong in a way nobody would see.
    """
    import pathlib

    sources = [axis["source"] for axis in X_AXIS_REGISTRY]

    checked = 0
    for path in sorted(pathlib.Path("build/runs").iterdir()):
        if not (path / "metrics.db").exists():
            continue
        payload = _payload(path.name, limit=200)
        if payload is None or not _has(payload, "val_loss"):
            continue
        checked += 1
        val_rows = [i for i, v in enumerate(payload["val_loss"]) if v is not None]
        for source in sources:
            column = payload.get(source)
            assert column is not None, f"{path.name}: no {source} column"
            assert len(column) == len(
                payload["val_loss"]
            ), f"{path.name}: {source} is not index-aligned with val_loss"
            missing = [i for i in val_rows if column[i] is None]
            assert not missing, (
                f"{path.name}: {len(missing)} validation points have no {source} "
                f"coordinate - they cannot be plotted on that axis"
            )
    if checked == 0:
        pytest.skip("no run on disk has validation data")


def test_elapsed_discounts_a_suspension():
    """A stopped-and-resumed run must not be billed for the hours it sat idle."""
    from praxis.web.routes.metrics import _TS_RAW, _elapsed_seconds

    rows = lambda stamps: [{_TS_RAW: t} for t in stamps]

    # Uninterrupted: elapsed is exactly the span.
    assert _elapsed_seconds(rows([0, 10, 20, 30]))[-1] == 30

    # An 8-hour pause between 10-second rows contributes one typical interval,
    # not 8 hours.
    paused = _elapsed_seconds(rows([0, 10, 20, 28800, 28810]))
    assert paused[-1] < 100, f"pause was billed as training time: {paused[-1]}s"

    # Never runs backwards, whatever the clock did.
    wobbly = _elapsed_seconds(rows([0, 10, 5, 15]))
    assert all(b >= a for a, b in zip(wobbly, wobbly[1:]))

    # Degenerate inputs must not raise - they reach here on brand-new runs.
    assert _elapsed_seconds([]) == []
    assert _elapsed_seconds(rows([1234.5])) == [0.0]
    assert _elapsed_seconds([{}, {}]) == [0.0, 0.0]


def test_transform_strips_the_raw_timestamp():
    """``_ts_epoch`` is scratch for the wall-clock maths, never payload."""
    from praxis.web.routes.metrics import _TS_RAW, _transform_metrics

    out = _transform_metrics([{"step": 0, "ts": "x", _TS_RAW: 100.0, "loss": 1.0}])
    assert _TS_RAW not in out and "ts" not in out
    assert out["elapsed_s"] == [0.0]


def test_heatmap_patterns_expose_both_grid_coordinates():
    """A heatmap key_pattern must capture (row, column) or it draws nothing.

    ``createExpertRoutingChart`` builds its grid from capture groups 1 and 2 of
    the registry's pattern. It used to hardcode
    ``layer_(\\d+)_expert_(\\d+)_routing_weight`` instead, while the gate that
    decides whether a card is SHOWN used key_pattern - so SMEAR Merge
    Coefficients was displayed whenever its data existed and then rendered an
    empty canvas, on every run, because the two matched different key families.

    A pattern with fewer than two groups now yields no cells at all, silently.
    """
    import re

    heatmaps = [
        e
        for e in COMPOSITE_METRIC_REGISTRY
        if e.get("type") == "expert_routing_heatmap"
    ]
    assert heatmaps, "no heatmap cards declared"
    for entry in heatmaps:
        pattern = entry.get("key_pattern")
        assert pattern, f"{entry['key']}: heatmap needs a key_pattern"
        groups = re.compile(pattern).groups
        assert groups >= 2, (
            f"{entry['key']}: key_pattern {pattern!r} has {groups} capture "
            f"group(s); the heatmap needs two - group 1 the row, group 2 the "
            f"column index"
        )


def test_smear_merge_coefficients_pattern_parses_real_keys():
    """The card's pattern must match what praxis/routers/smear.py emits.

    Target labels contain underscores (``attn_depth_bias``), so the row group
    has to be greedy enough to swallow them while still leaving the trailing
    ``_<index>`` for the column.
    """
    import re

    entry = next(
        e for e in COMPOSITE_METRIC_REGISTRY if e["key"] == "smear_coefficients"
    )
    pattern = re.compile(entry["key_pattern"])

    cases = {
        "smear_coeff_attn_0": ("attn", 0),
        "smear_coeff_attn_depth_bias_3": ("attn_depth_bias", 3),
        "smear_coeff_ffn_res_mix_1_2": ("ffn_res_mix_1", 2),
    }
    for key, (row, col) in cases.items():
        match = pattern.match(key)
        assert match, f"{key!r} does not match the card's pattern"
        assert match.group(1) == row, f"{key!r} row: {match.group(1)!r} != {row!r}"
        assert int(match.group(2)) == col

    assert not pattern.match("smear_coeff_attn"), "a key with no index is not a cell"


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
    import pathlib

    src = pathlib.Path("praxis/web/src/js/charts.js").read_text()

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


# --- readability hints -------------------------------------------------------


def test_clip_and_smooth_hints_are_well_formed():
    """Both hints change what the reader sees, so a typo must not pass silently."""
    for key, entry in TRAINING_METRIC_REGISTRY.items():
        chart = entry.get("chart")
        if not chart:
            continue
        if "y_clip_percentile" in chart:
            pct = chart["y_clip_percentile"]
            assert isinstance(
                pct, (int, float)
            ), f"{key}: y_clip_percentile not numeric"
            assert 90 <= pct < 100, (
                f"{key}: y_clip_percentile={pct}; below 90 discards real signal "
                f"and 100 is a no-op"
            )
        if "smooth" in chart:
            assert isinstance(chart["smooth"], bool), f"{key}: smooth must be a bool"


def test_readability_hints_stay_opt_in():
    """Clipping and smoothing are per-metric, never blanket defaults.

    Both are wrong for most cards: clipping hides real maxima where the tail IS
    the signal (softmax_collapse, gradient spikes), and smoothing a sparse
    validation series would draw a trend through a handful of points.
    """
    charts = [
        (k, v["chart"]) for k, v in TRAINING_METRIC_REGISTRY.items() if v.get("chart")
    ]
    clipped = [k for k, c in charts if c.get("y_clip_percentile")]
    smoothed = [k for k, c in charts if c.get("smooth")]

    assert clipped, "expected at least the training-loss card to clip"
    assert len(clipped) < len(charts) / 2, "clipping has become a de-facto default"
    assert len(smoothed) < len(charts) / 2, "smoothing has become a de-facto default"

    # A sparse series has too few points for a rolling window to mean anything.
    for key in smoothed:
        assert not TRAINING_METRIC_REGISTRY[key]["chart"].get("is_validation"), (
            f"{key}: smoothing a validation series - its points are one per "
            f"val_check_interval, far too sparse for a rolling window"
        )


def test_training_loss_declares_both_hints():
    """The card this was built for: a startup transient plus per-step noise."""
    chart = TRAINING_METRIC_REGISTRY["loss"]["chart"]
    assert chart.get("y_clip_percentile"), "training loss lost its robust y bound"
    assert chart.get("smooth") is True, "training loss lost its trend line"


def test_raw_trace_is_excluded_from_legend_and_tooltip():
    """The faint raw series must not double every run in the chart chrome.

    Smoothing draws two datasets per run sharing one name and colour. Without
    the flag check in both the legend filter and the interaction mode, a
    two-run comparison lists four legend entries and four tooltip lines.
    """
    import pathlib

    src = pathlib.Path("praxis/web/src/js/charts.js").read_text()
    assert "RAW_TRACE_FLAG" in src, "the raw-trace marker is gone"
    # One definition, one legend filter, one interaction skip, one tag site.
    assert src.count("RAW_TRACE_FLAG") >= 4, (
        "raw traces must be flagged, filtered from the legend, AND skipped by "
        "the hover interaction"
    )


# ── Description length ──────────────────────────────────────────────────────
# A card's description renders as the subtitle directly under its title, above
# a chart a few hundred pixels tall. Descriptions kept growing - design notes,
# falsifiers, and accounts of what the metric USED to measure - until several
# were longer than the plots they introduced. Say what the number is and how to
# read it; the reasoning belongs in a code comment or in next/.
MAX_DESCRIPTION_CHARS = 180

_DESC_HOLDERS = re.compile(
    r"^(metric_descriptions|field_metric_descriptions|all_metric_descriptions"
    r"|[A-Z_]*METRIC_DESCRIPTIONS|[A-Z_]*METRIC_REGISTRY|[A-Z_]*CHART_REGISTRY)$"
)


def _described_dicts(tree):
    """Every dict literal that lives under a metric-description holder."""
    import ast

    out = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = [node.name]
        matched = [n for n in names if _DESC_HOLDERS.match(n)]
        if matched:
            # Registries nest their descriptions under a "description" key;
            # metric_descriptions may instead map a metric name straight to
            # its text, so every string value there is a description.
            plain_form = not any(n.endswith("REGISTRY") for n in matched)
            out.append((node, plain_form))
    return out


def _descriptions_in(node, plain_form):
    """(lineno, text) for every description string under ``node``.

    ``metric_descriptions`` accepts two shapes (see praxis/metrics/descriptions):
    the rich ``{"description": ..., "chart": ...}`` dict and a bare string.
    """
    import ast

    hint_keys = {"title", "y_label", "renderer", "type", "key_pattern"}
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Dict):
            continue
        literal = {k.value for k in sub.keys if isinstance(k, ast.Constant)}
        # A chart/snapshot hint dict, not a mapping of metric name -> text.
        bare = plain_form and "description" not in literal and not (literal & hint_keys)
        for key, value in zip(sub.keys, sub.values):
            if not isinstance(key, ast.Constant):
                continue
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            if key.value == "description" or (bare and key.value != "caller"):
                yield value.lineno, value.value


def metric_descriptions_on_disk():
    """Every metric-card description declared anywhere under ``praxis/``."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "praxis"
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for holder, plain_form in _described_dicts(tree):
            for lineno, text in _descriptions_in(holder, plain_form):
                yield f"{path.relative_to(root.parent)}:{lineno}", text


def test_no_metric_description_becomes_an_essay():
    """Card subtitles stay readable at a glance, everywhere they are declared."""
    found = list(metric_descriptions_on_disk())
    assert len(found) > 100, "the description scan stopped finding declarations"

    too_long = [
        (where, len(text)) for where, text in found if len(text) > MAX_DESCRIPTION_CHARS
    ]
    assert not too_long, "metric descriptions over %d chars: %s" % (
        MAX_DESCRIPTION_CHARS,
        ", ".join(f"{w} ({n})" for w, n in too_long),
    )
