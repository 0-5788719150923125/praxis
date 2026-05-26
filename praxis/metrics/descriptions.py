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

The walker normalizes both forms into the rich shape on the wire, so
the frontend always sees ``{description, chart}`` per entry.
"""

from typing import Any, Dict, Iterable, Optional

from praxis.memory.surfacings import MemoryBase


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
        return {
            "description": desc,
            "chart": chart if isinstance(chart, dict) else None,
            "snapshot": snapshot if isinstance(snapshot, dict) else None,
        }
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


def get_metric_descriptions(model: Any) -> Dict[str, Dict[str, Any]]:
    """Return ``{key: {description, chart}}`` for the live model."""
    if model is None:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for raw in _candidates(model):
        out.update(_collect_from(raw))
    return out
