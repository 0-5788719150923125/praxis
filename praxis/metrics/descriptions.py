"""Metric descriptions, discovered from the live model.

Each component declares its own description text via a class attribute
(``metric_description`` for a single metric/chart, ``metric_descriptions``
for a dict of keys -> strings). The registry walks the model and collects
whatever the *currently active* components have to say, so the frontend
never holds a stale copy of what a metric means.
"""

from typing import Any, Dict, Iterable


def _collect_from(obj: Any) -> Dict[str, str]:
    """Pull description fields off a single object, if present."""
    if obj is None:
        return {}
    out: Dict[str, str] = {}
    descriptions = getattr(obj, "metric_descriptions", None)
    if isinstance(descriptions, dict):
        for key, value in descriptions.items():
            if isinstance(value, str):
                out[str(key)] = value
    return out


def _candidates(model: Any) -> Iterable[Any]:
    """Components that may carry metric descriptions on the live model."""
    yield getattr(model, "harmonic_field", None)

    head = getattr(model, "head", None)
    yield getattr(head, "field", None) if head is not None else None

    weighter = getattr(model, "taskmaster", None)
    if weighter is not None and getattr(weighter, "is_dynamic", False):
        single = getattr(weighter, "metric_description", None)
        if isinstance(single, str):
            yield {"metric_descriptions": {"task_weights": single}}

    # Future component contributors slot in here.


def get_metric_descriptions(model: Any) -> Dict[str, str]:
    """Return ``{metric_or_chart_key: description}`` for the live model."""
    if model is None:
        return {}
    out: Dict[str, str] = {}
    for candidate in _candidates(model):
        if isinstance(candidate, dict):
            # Inline ``{"metric_descriptions": {...}}`` dicts let callers
            # contribute without subclassing nn.Module.
            inner = candidate.get("metric_descriptions", {})
            for key, value in inner.items():
                if isinstance(value, str):
                    out[str(key)] = value
            continue
        out.update(_collect_from(candidate))
    return out
