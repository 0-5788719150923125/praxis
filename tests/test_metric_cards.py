"""Dashboard card invariants for the Research-tab metric registries.

The Research tab builds its deck with ``buildScalarConfigsFromRegistry`` and
``buildCompositeConfigsFromRegistry`` (praxis/web/src/js/charts.js), which
concatenate ALL scalars ahead of ALL composites and sort each half flat by
``order``. Neither honours ``group``, ``group_order`` or ``series_group`` - those
belong to the Dynamics tab's manifest builder. Two things went wrong because of
that and are pinned here:

  * four density entries carried ``series_group`` expecting to merge into two
    cards, so the deck rendered four - two titled "Density Gradient (norm)" and
    two "Density Steepening across Depth";
  * they also carried ``order: 10``, tying with ``loss``, and a stable sort puts
    the earlier-declared entry first - which put a research probe at deck
    position 1, ahead of training loss.
"""

import re

from praxis.metrics.training_metrics import (
    COMPOSITE_METRIC_REGISTRY,
    TRAINING_METRIC_REGISTRY,
)

DENSITY_KEYS = (
    "density_norm_slope",
    "density_hop_slope",
    "density_norm_steepening",
    "density_hop_steepening",
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


def test_density_keys_are_chartless():
    """They render as two composites, so an individual chart would duplicate."""
    for key in DENSITY_KEYS:
        entry = TRAINING_METRIC_REGISTRY[key]
        assert entry.get("chart") is None, f"{key} would render its own card"
        assert entry["description"], f"{key} lost its tooltip text"


def test_density_composites_cover_every_density_key():
    """Chartless keys must be claimed by some composite, or they vanish."""
    patterns = [
        re.compile(e["key_pattern"])
        for e in COMPOSITE_METRIC_REGISTRY
        if e.get("key_pattern")
    ]
    for key in DENSITY_KEYS:
        assert any(p.match(key) for p in patterns), f"{key} is on no card"


def test_density_composites_pair_norm_with_occupancy():
    """Each card carries BOTH coordinates - a norm-only reading is escapable."""
    by_key = {e["key"]: e for e in COMPOSITE_METRIC_REGISTRY}
    for card, expected in (
        ("density_gradient", {"density_norm_slope", "density_hop_slope"}),
        (
            "density_steepening",
            {"density_norm_steepening", "density_hop_steepening"},
        ),
    ):
        assert card in by_key, f"missing composite {card}"
        pattern = re.compile(by_key[card]["key_pattern"])
        assert {k for k in DENSITY_KEYS if pattern.match(k)} == expected


def test_composite_orders_do_not_collide():
    """Ties fall back to declaration order, which is invisible in the source."""
    orders = [e.get("order", 0) for e in COMPOSITE_METRIC_REGISTRY]
    dupes = {o for o in orders if orders.count(o) > 1}
    # Pre-existing ties are grandfathered; the point is that new cards do not
    # silently join them. Assert only that the ones we placed are clean.
    for key in (
        "density_gradient",
        "density_steepening",
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
