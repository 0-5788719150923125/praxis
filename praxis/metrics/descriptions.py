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
* ``group``: a stable section key (e.g., ``"harmonic_classifier"``) used to
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
from praxis.metrics.specialization import (
    collect_activation_descriptions,
    collect_arc_descriptions,
    collect_attention_descriptions,
)


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
        # A producer may pin its own caller (e.g. ParallelClassifier's namespaced
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
    classifier = getattr(model, "classifier", None)
    if classifier is not None and hasattr(classifier, "all_metric_descriptions"):
        yield classifier.all_metric_descriptions()

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

        # Activations that publish diagnostics (e.g. Servant's chirp).
        activation_descs = collect_activation_descriptions(model)
        if activation_descs:
            yield activation_descs

        # Attention mechanisms that publish diagnostics (SSOG's field).
        attention_descs = collect_attention_descriptions(model)
        if attention_descs:
            yield attention_descs

    # The decoder's sorting slot, when it holds a learnable positional field
    # (decay_bias / amplitude_field); the stateless sorts declare nothing.
    decoder = getattr(model, "decoder", None)
    order = getattr(decoder, "order", None) if decoder is not None else None
    if order is not None:
        descs = getattr(type(order), "metric_descriptions", None)
        if isinstance(descs, dict):
            yield descs

    # MTP harmonic-field diagnostics (vear bank's Serpent spectrum).
    mtp = getattr(model, "mtp", None)
    if mtp is not None and hasattr(mtp, "field_metric_descriptions"):
        field_descs = mtp.field_metric_descriptions()
        if field_descs:
            yield field_descs

    # Loss-owning encoders (e.g. CALM) declare chart hints as a class attr;
    # guard against ``model.encoder = False`` (the no-encoder sentinel).
    encoder = getattr(model, "encoder", None)
    if encoder:
        descs = getattr(type(encoder), "metric_descriptions", None)
        if isinstance(descs, dict):
            yield descs
        # A loss-owning encoder may run a non-CE reconstruction loss (HALO),
        # or carry a geometric aux loss (CALM's halo mode), whose
        # chart/snapshot hints live on the loss class.
        for attr in ("recon_loss_fn", "geo_loss_fn"):
            fn = getattr(encoder, attr, None)
            if fn is not None:
                loss_descs = getattr(type(fn), "metric_descriptions", None)
                if isinstance(loss_descs, dict):
                    yield loss_descs

    # Every registered objective (the criterion, the regularizers, the terms
    # other modules compute) declares chart/snapshot hints as a class attr.
    criterion = getattr(model, "criterion", None)
    if criterion is not None and hasattr(criterion, "metric_descriptions"):
        yield from criterion.metric_descriptions()


def resolve_callers(root: Any) -> Dict[str, str]:
    """Map each metric key to the class name of the module that declares it.

    Walks ``root.modules()`` parents-first, first declarer wins. Used to stamp
    the live model and, by ``ParallelClassifier``, to attribute its per-branch keys
    to the owning leaf class (e.g. ``HarmonicField``, not its classifier wrapper).
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
    ParallelClassifier's namespaced keys) are left untouched; unowned keys stay bare.
    """

    def claim(key: str, caller: str) -> None:
        entry = out.get(str(key))
        if entry is not None and "caller" not in entry:
            entry["caller"] = caller

    for key, caller in resolve_callers(model).items():
        claim(key, caller)

    weighter = getattr(model, "tasker", None)
    if weighter is not None and getattr(weighter, "is_dynamic", False):
        claim("task_weights", type(weighter).__name__)

    from praxis.metrics.optimizer import OPTIMIZER_METRIC_DESCRIPTIONS

    for key in OPTIMIZER_METRIC_DESCRIPTIONS:
        claim(key, "Optimizer")

    from praxis.metrics.rlct import RLCT_METRIC_DESCRIPTIONS

    for key in RLCT_METRIC_DESCRIPTIONS:
        claim(key, "RLCT")


def get_metric_descriptions(model: Any) -> Dict[str, Dict[str, Any]]:
    """Return ``{key: {description, chart, caller}}`` for the live model."""
    if model is None:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for raw in _candidates(model):
        out.update(_collect_from(raw))
    # Optimizer + RLCT telemetry are universal (not model-attached): always in.
    from praxis.metrics.optimizer import OPTIMIZER_METRIC_DESCRIPTIONS
    from praxis.metrics.rlct import RLCT_METRIC_DESCRIPTIONS

    out.update(_collect_from(OPTIMIZER_METRIC_DESCRIPTIONS))
    out.update(_collect_from(RLCT_METRIC_DESCRIPTIONS))
    # Governor telemetry is opt-in: only present when a governor callback has
    # stashed metrics on the live model (see GNSBatchGovernor._stash).
    _core = getattr(model, "_orig_mod", model)
    if getattr(_core, "_governor_metrics", None) is not None:
        from praxis.governors import GOVERNOR_METRIC_DESCRIPTIONS

        out.update(_collect_from(GOVERNOR_METRIC_DESCRIPTIONS))
        for key in GOVERNOR_METRIC_DESCRIPTIONS:
            entry = out.get(key)
            if entry is not None and "caller" not in entry:
                entry["caller"] = "GNSBatchGovernor"
    # Same pattern for the compute profiler: only present once a profiled step
    # has landed, so a run that never profiles (torch.compile) shows no empty card.
    if getattr(_core, "_compute_profile", None) is not None:
        from praxis.metrics.compute import COMPUTE_METRIC_DESCRIPTIONS

        out.update(_collect_from(COMPUTE_METRIC_DESCRIPTIONS))
        for key in COMPUTE_METRIC_DESCRIPTIONS:
            entry = out.get(key)
            if entry is not None and "caller" not in entry:
                entry["caller"] = "ComputeProfiler"
    # Same pattern for objective conflict, with one difference: the series are
    # named after whichever loss terms this config actually carries, so the
    # cards are built from the live stash's keys rather than a fixed dict.
    _conflict = getattr(_core, "_conflict_metrics", None)
    if _conflict:
        from praxis.losses.conflict import conflict_metric_descriptions

        descs = conflict_metric_descriptions(_conflict.keys())
        out.update(_collect_from(descs))
        for key in descs:
            entry = out.get(key)
            if entry is not None and "caller" not in entry:
                entry["caller"] = "ObjectiveConflict"
    # Same pattern for the loss-combination strategy: series named after
    # whichever terms this config carries, so cards come from the live stash.
    _blend = getattr(_core, "_strategy_metrics", None)
    if _blend:
        from praxis.strategies.anchor_capped import blend_metric_descriptions

        descs = blend_metric_descriptions(_blend.keys())
        out.update(_collect_from(descs))
        strategy_name = type(getattr(_core, "strategy", None)).__name__
        for key in descs:
            entry = out.get(key)
            if entry is not None and "caller" not in entry:
                entry["caller"] = strategy_name
    # Same pattern for the probe-attribution sequence curriculum.
    if getattr(_core, "_seq_probe_metrics", None) is not None:
        from praxis.data.seq_probe import SEQ_PROBE_METRIC_DESCRIPTIONS

        out.update(_collect_from(SEQ_PROBE_METRIC_DESCRIPTIONS))
        for key in SEQ_PROBE_METRIC_DESCRIPTIONS:
            entry = out.get(key)
            if entry is not None and "caller" not in entry:
                entry["caller"] = "SequenceProbe"
    _stamp_callers(out, model)
    return out
