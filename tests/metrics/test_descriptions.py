import re

import pytest
import torch
import torch.nn as nn

from praxis import registry
from praxis.data.seq_probe import SequenceProbe
from praxis.metrics.compute import COMPUTE_METRIC_DESCRIPTIONS
from praxis.metrics.descriptions import get_metric_descriptions
from praxis.metrics.optimizer import OPTIMIZER_METRIC_DESCRIPTIONS
from praxis.metrics.rlct import RLCT_METRIC_DESCRIPTIONS

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

    root = pathlib.Path(__file__).resolve().parents[2] / "praxis"
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


def test_dynamically_built_descriptions_respect_the_length_cap():
    """The static scan reads `metric_descriptions` class attrs, so cards built
    at runtime (ParallelHead's per-arm set, ObjectiveConflict's per-term set)
    slip past it entirely - and every one of them was 190-612 chars when first
    written. Same 180-char cap, checked where the scan cannot reach."""
    import torch

    from praxis.losses.conflict import conflict_metric_descriptions
    from tests.stubs import Cfg, Enc

    torch.manual_seed(0)
    built = dict(
        registry.lookup("heads", "prismatic9")(Cfg(), encoder=Enc())._arm_descriptions()
    )
    built.update(
        conflict_metric_descriptions(
            ["conflict_mtp", "conflict_mag_mtp", "conflict_min"]
        )
    )
    assert built, "no dynamic descriptions found - the builders moved"
    too_long = {
        k: len(v["description"])
        for k, v in built.items()
        if len(v["description"]) > 180
    }
    assert not too_long, f"dynamic metric descriptions over 180 chars: {too_long}"


# ------------------------------------------------------------------------------
# rlct_landscape
# ------------------------------------------------------------------------------
# RLCT loss-landscape probe (praxis.metrics.rlct + snapshot wiring).


def test_descriptions_exposed_with_snapshot_hint():
    desc = get_metric_descriptions(nn.Linear(4, 4))
    for key in RLCT_METRIC_DESCRIPTIONS:
        assert key in desc
    snap = desc["rlct_landscape"]["snapshot"]
    assert snap["renderer"] == "rlct_mesh"
    assert desc["rlct_lambda"]["caller"] == "RLCT"
    # The LLC trio shares one chart via series_group.
    assert desc["rlct_llc_mean"]["chart"]["series_group"] == "rlct_llc"


def test_field_descriptions_exposed():
    from praxis.metrics.descriptions import get_metric_descriptions

    desc = get_metric_descriptions(nn.Linear(8, 8))
    assert desc["param_field"]["snapshot"]["renderer"] == "param_field"


# ------------------------------------------------------------------------------
# optimizer_metrics
# ------------------------------------------------------------------------------
# Optimizer-state telemetry suite (praxis.metrics.optimizer).


def test_descriptions_in_dynamics_manifest():
    from praxis.metrics.descriptions import get_metric_descriptions

    descs = get_metric_descriptions(nn.Linear(2, 2))
    for k in OPTIMIZER_METRIC_DESCRIPTIONS:
        assert k in descs and descs[k]["chart"]["group"] == "optimizer"


def test_descriptions_stamp_producing_caller():
    """Each entry carries the class name of the module that raised it."""
    from praxis.metrics.descriptions import get_metric_descriptions

    class Field(nn.Module):
        metric_descriptions = {"field_amp": "amplitude"}

    class Head(nn.Module):
        metric_descriptions = {"head_loss": "aux loss"}

        def __init__(self):
            super().__init__()
            self.field = Field()

        def all_metric_descriptions(self):
            out = {}
            for mod in self.modules():
                descs = getattr(type(mod), "metric_descriptions", None)
                if isinstance(descs, dict):
                    out.update(descs)
            return out

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = Head()

    descs = get_metric_descriptions(Model())
    assert descs["field_amp"]["caller"] == "Field"
    assert descs["head_loss"]["caller"] == "Head"
    # Optimizer telemetry is universal and attributed generically.
    for k in OPTIMIZER_METRIC_DESCRIPTIONS:
        assert descs[k]["caller"] == "Optimizer"


# ------------------------------------------------------------------------------
# compute_profiler
# ------------------------------------------------------------------------------
# Per-module compute-time attribution: scoping, EMA smoothing, payload shape.


def test_metric_descriptions_declare_a_renderer():
    entry = COMPUTE_METRIC_DESCRIPTIONS["compute_profile"]
    assert entry["snapshot"]["renderer"] == "compute_treemap"
    for key, spec in COMPUTE_METRIC_DESCRIPTIONS.items():
        assert spec.get("description"), f"{key} needs a description"


# ------------------------------------------------------------------------------
# seq_probe
# ------------------------------------------------------------------------------
# Probe-attribution sequence curriculum (praxis/data/seq_probe.py).
#
# The invariants that matter are the ones the previous controller failed:
#
# - an arm's coefficient must recover its true value from the regression, - an arm with
# no measurable edge must not be handed a confident share, - the fit must track a change
# in which arm is best rather than average over all of history, - the fixed per-tier
# roll must remain the cold-start path.
#
# The controller this replaced (a learning-progress bandit scoring each arm by the loss
# drop between two visits to it) is gone rather than deprecated: that drop measures how
# much the WHOLE model improved in the interval, so it carried no information about arm
# quality - a worthless arm still earned a full share, and sampling an arm more often
# shortened its own interval, making the mechanism negative feedback on visit rate that
# drove the mix to uniform.


def test_metric_descriptions_fold_in_when_armed():
    from praxis.metrics.descriptions import get_metric_descriptions

    class _Bare:
        pass

    plain = _Bare()
    assert "seq_tstat_x2" not in get_metric_descriptions(plain)

    armed = _Bare()
    armed._seq_probe_metrics = {"seq_prob_x1": 0.5}
    descs = get_metric_descriptions(armed)
    assert descs["seq_tstat_x2"]["caller"] == "SequenceProbe"
    assert descs["seq_value_x2"]["chart"]["series_group"] == "seq_value"
