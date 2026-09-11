import re

import pytest

from praxis import registry
from praxis.metrics import COMPOSITE_METRIC_REGISTRY
from praxis.metrics.training_metrics import (
    COMPOSITE_METRIC_REGISTRY,
    TRAINING_METRIC_REGISTRY,
    X_AXIS_REGISTRY,
)
from praxis.routers.bank import ExpertBank as SMEAR

# ------------------------------------------------------------------------------
# metric_cards
# ------------------------------------------------------------------------------
# Dashboard card invariants for the Research-tab metric registries.
#
# The Research tab builds its deck with ``buildScalarConfigsFromRegistry`` and
# ``buildCompositeConfigsFromRegistry`` (praxis/web/src/js/charts.js), which concatenate
# ALL scalars ahead of ALL composites and sort each half flat by ``order``. Neither
# honours ``group``, ``group_order`` or ``series_group`` - those belong to the Dynamics
# tab's manifest builder. Two things went wrong because of that and are pinned here:
#
# * four (since removed) density entries carried ``series_group`` expecting to merge
# into two cards, so the deck rendered four; * they also carried ``order: 10``, tying
# with ``loss``, and a stable sort puts the earlier-declared entry first - which put a
# research probe at deck position 1, ahead of training loss.
#
# The information-density probe now emits only ``readout_*`` keys into extra_metrics (no
# schema columns), claimed by the composite cards pinned below.


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
    ties = {
        o: [e["key"] for e in COMPOSITE_METRIC_REGISTRY if e.get("order", 0) == o]
        for o in orders
        if orders.count(o) > 1
    }
    assert not ties, f"composite cards share an order: {ties}"


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


# ------------------------------------------------------------------------------
# routing_metrics
# ------------------------------------------------------------------------------
# Routing diagnostics must describe the ROUTER, not a subclass's transform.
#
# The defect these pin: VEAR sharpens routing probabilities by ``p**4`` before the
# batch-mean merge, and the metrics were computed on the sharpened values. That drove
# the logged entropy to float-exact one-hot, where it stopped responding to the weights
# entirely - measured bit-identical across four models whose losses differed by 5%,
# which reads as a finding rather than as an absent metric.


def test_new_metrics_are_charted():
    """Registry-driven charts: a metric with no key_pattern never renders."""
    patterns = {c.get("key_pattern") for c in COMPOSITE_METRIC_REGISTRY}
    for name in (
        "routing_entropy",
        "routing_entropy_seq",
        "routing_input_dependence",
        "routing_merge_entropy",
    ):
        assert rf"^layer_\d+_{name}$" in patterns, name
