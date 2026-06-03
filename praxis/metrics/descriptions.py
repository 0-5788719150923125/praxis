"""Metric descriptions, discovered from the live model.

Each component declares its own descriptions via a class attribute
``metric_descriptions``. Values may be either:

* a plain string - the description text only (no auto-rendering hints), or
* a dict ``{"description": str, "chart": {...} | None}`` - the rich form
  that opts a scalar metric into the dashboard's data-driven chart
  manifest (see ``renderScalarMetricsFromManifest`` in the frontend).

The ``chart`` sub-dict may contain:

* ``title``: chart title text
* ``y_label``: y-axis label
* ``y_scale``: ``"linear"`` (default) or ``"logarithmic"``
* ``group``: a stable section key (e.g., ``"harmonic_head"``) used to
  cluster related metrics into a single dashboard section
* ``order``: integer ordering within the group (default 0)
* ``series_group``: optional key; metrics sharing it render as multiple
  lines on ONE chart (the lowest-``order`` member supplies the title,
  axis and subtitle). Use for same-scale companions (e.g. min/mean/max).
* ``series_label``: this metric's legend label within its ``series_group``

The walker normalizes both forms into the rich shape on the wire, so
the frontend always sees ``{description, chart}`` per entry.
"""

from typing import Any, Dict, Iterable, Optional

from praxis.memory.surfacings import MemoryBase
from praxis.metrics.specialization import collect_arc_descriptions


def _normalize(value: Any) -> Optional[Dict[str, Any]]:
    """Coerce a raw description value into ``{description, chart, snapshot}``.

    Returns None for malformed entries. Both ``chart`` and ``snapshot``
    hints are optional; either may be present, both may be present, or
    neither (legacy string form is description-only).
    """
    if isinstance(value, str):
        return {"description": value, "chart": None, "snapshot": None}
    if isinstance(value, dict):
        desc = value.get("description")
        if not isinstance(desc, str):
            return None
        chart = value.get("chart")
        snapshot = value.get("snapshot")
        entry: Dict[str, Any] = {
            "description": desc,
            "chart": chart if isinstance(chart, dict) else None,
            "snapshot": snapshot if isinstance(snapshot, dict) else None,
        }
        # A producer may pin its own caller (e.g. ParallelHead's namespaced
        # per-branch keys, which the model walk in _stamp_callers can't reach).
        caller = value.get("caller")
        if isinstance(caller, str):
            entry["caller"] = caller
        return entry
    return None


def _collect_from(descriptions: Any) -> Dict[str, Dict[str, Any]]:
    """Normalize a raw ``metric_descriptions`` dict."""
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(descriptions, dict):
        return out
    for key, value in descriptions.items():
        entry = _normalize(value)
        if entry is not None:
            out[str(key)] = entry
    return out


def _candidates(model: Any) -> Iterable[Dict[str, Any]]:
    """Raw description dicts contributed by live-model components."""
    head = getattr(model, "head", None)
    if head is not None and hasattr(head, "all_metric_descriptions"):
        yield head.all_metric_descriptions()

    weighter = getattr(model, "tasker", None)
    if weighter is not None and getattr(weighter, "is_dynamic", False):
        single = getattr(weighter, "metric_description", None)
        if isinstance(single, str):
            yield {"task_weights": single}

    if hasattr(model, "modules"):
        memory_descs = MemoryBase.collect_metric_descriptions(model)
        if memory_descs:
            yield memory_descs

        arc_descs = collect_arc_descriptions(model)
        if arc_descs:
            yield arc_descs

    iso = getattr(model, "contrastive_isotropy", None)
    if iso is not None:
        descs = getattr(type(iso), "metric_descriptions", None)
        if isinstance(descs, dict):
            yield descs

    # Loss-owning encoders (e.g. CALM) declare chart hints as a class attr;
    # guard against ``model.encoder = False`` (the no-encoder sentinel).
    encoder = getattr(model, "encoder", None)
    if encoder:
        descs = getattr(type(encoder), "metric_descriptions", None)
        if isinstance(descs, dict):
            yield descs

    # Loss functions (e.g. HALO) declare chart/snapshot hints as a class attr.
    criterion = getattr(model, "criterion", None)
    if criterion is not None:
        descs = getattr(type(criterion), "metric_descriptions", None)
        if isinstance(descs, dict):
            yield descs


def resolve_callers(root: Any) -> Dict[str, str]:
    """Map each metric key to the class name of the module that declares it.

    Walks ``root.modules()`` parents-first, first declarer wins. Used to stamp
    the live model and, by ``ParallelHead``, to attribute its per-branch keys
    to the owning leaf class (e.g. ``HarmonicField``, not its head wrapper).
    """
    out: Dict[str, str] = {}
    if not hasattr(root, "modules"):
        return out
    for mod in root.modules():
        descs = getattr(type(mod), "metric_descriptions", None)
        if isinstance(descs, dict):
            for key in descs:
                out.setdefault(str(key), type(mod).__name__)
    return out


def _stamp_callers(out: Dict[str, Dict[str, Any]], model: Any) -> None:
    """Annotate each entry with the class name of the module that raised it.

    Lets the dashboard show which component owns a metric. We walk the live
    model (parents before children, first declarer wins) then fill in the
    non-module sources. Entries that already carry a pinned ``caller`` (e.g.
    ParallelHead's namespaced keys) are left untouched; unowned keys stay bare.
    """

    def claim(key: str, caller: str) -> None:
        entry = out.get(str(key))
        if entry is not None and "caller" not in entry:
            entry["caller"] = caller

    for key, caller in resolve_callers(model).items():
        claim(key, caller)

    iso = getattr(model, "contrastive_isotropy", None)
    if iso is not None:
        descs = getattr(type(iso), "metric_descriptions", None)
        if isinstance(descs, dict):
            for key in descs:
                claim(key, type(iso).__name__)

    weighter = getattr(model, "tasker", None)
    if weighter is not None and getattr(weighter, "is_dynamic", False):
        claim("task_weights", type(weighter).__name__)

    from praxis.metrics.optimizer import OPTIMIZER_METRIC_DESCRIPTIONS

    for key in OPTIMIZER_METRIC_DESCRIPTIONS:
        claim(key, "Optimizer")


def get_metric_descriptions(model: Any) -> Dict[str, Dict[str, Any]]:
    """Return ``{key: {description, chart, caller}}`` for the live model."""
    if model is None:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for raw in _candidates(model):
        out.update(_collect_from(raw))
    # Optimizer telemetry is universal (not model-attached); always include it.
    from praxis.metrics.optimizer import OPTIMIZER_METRIC_DESCRIPTIONS

    out.update(_collect_from(OPTIMIZER_METRIC_DESCRIPTIONS))
    _stamp_callers(out, model)
    return out
