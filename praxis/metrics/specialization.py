"""Module-walk diagnostics: Arc depth specialization, attention, activations.

ArcAttention and ArcGLU give each recurrent-depth pass its own learned
parameters (per-depth Q/K/V/O biases, per-pass activations). The risk is
that those copies collapse to identical values, erasing any benefit over a
single shared parameter. ``depth_dispersion`` measures how far a stack of
per-depth vectors has diverged; the collectors walk a live model and average
each Arc module's report so the dashboard sees one number per metric.

The walks exist because these modules have no loss hook: an activation is
called as a bare ``act(x)`` and an attention block returns a tensor, so neither
can attach anything to the training step. Each opts in by defining
``training_metrics`` and is reached here. A mechanism that publishes
diagnostics but is not covered by a walk logs nothing at all, silently.

Kept free of praxis imports at module load - the Arc classes are imported
lazily inside the collectors so this stays clear of the
``memory.models -> praxis.dense`` import cycle.
"""

from typing import Dict, Iterator, Optional

import torch
import torch.nn.functional as F

from praxis import registry

_EPS = 1e-12


def depth_dispersion(w: torch.Tensor) -> Optional[Dict[str, float]]:
    """Specialization stats for a ``[D, dim]`` stack of per-depth vectors.

    Returns ``specialization`` = between-depth variance over total energy
    ``(mean||row||^2 - ||mean||^2) / mean||row||^2`` in [0, 1] (0 = every depth
    holds identical values, i.e. collapsed, and also the zero-init case;
    higher = rows diverge from their shared mean) and ``similarity`` = mean
    pairwise cosine between rows (~1 = depths point the same way; lower =
    directions diverging). Returns None when there's nothing to measure
    (< 2 depths).
    """
    if w.dim() != 2 or w.shape[0] < 2:
        return None
    w = w.detach().float()
    depth = w.shape[0]

    energy = w.pow(2).sum(dim=1).mean()
    mean_sq = w.mean(dim=0).pow(2).sum()
    specialization = ((energy - mean_sq) / (energy + _EPS)).clamp(0.0, 1.0)

    # Self-cosine of an all-zero row is 0, so subtracting the diagonal (not D)
    # keeps the average well-defined at zero-init.
    wn = F.normalize(w, dim=1)
    sims = wn @ wn.t()
    similarity = (sims.sum() - sims.diagonal().sum()) / (depth * (depth - 1))

    return {"specialization": float(specialization), "similarity": float(similarity)}


def _arc_modules(root) -> Iterator:
    """Yield the ArcAttention/ArcGLU modules under ``root``.

    Imports the Arc classes lazily (called only at train/request time) to
    avoid the metrics<->dense import cycle.
    """
    from praxis.attention.arc import ArcAttention
    from praxis.dense.arc import ArcGLU
    from praxis.routers.arc import ArcMixture

    arc_types = (ArcAttention, ArcGLU, ArcMixture)
    for module in root.modules():
        if isinstance(module, arc_types):
            yield module


def collect_arc_metrics(root) -> Dict[str, float]:
    """Average each Arc depth-specialization metric across the Arc modules
    under ``root`` (empty when none are present)."""
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for module in _arc_modules(root):
        for key, value in module.training_metrics().items():
            if value is None:
                continue
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums}


def _attention_modules(root) -> Iterator:
    """Yield attention modules under ``root`` that publish diagnostics.

    Arc is collected separately by :func:`collect_arc_metrics`, so the Arc
    classes are excluded here and nothing is counted twice. Everything else in
    the ``attention`` registry opts in the same way an activation does - by defining
    ``training_metrics`` - and, like activations, has no loss hook, so a module
    walk is the only way to reach it. Without this walk a mechanism can publish
    a full diagnostic suite that never reaches the log, which is exactly what
    happened to SSOG's field metrics.
    """
    from praxis.attention.arc import ArcAttention

    classes = tuple(
        {getattr(v, "func", v) for v in registry.namespace("attention").values()}
    )
    for module in root.modules():
        if (
            isinstance(module, classes)
            and not isinstance(module, ArcAttention)
            and hasattr(module, "training_metrics")
        ):
            yield module


def _layered_attention(root) -> Iterator:
    """Yield ``(layer, module)`` for the attention modules under ``root``.

    ``layer`` counts only the modules that declare ``per_layer_metrics``, in
    module order, and is None for the rest. Metrics and descriptions both walk
    through here, so a key and its declaration always agree on the layer.
    """
    layer = 0
    for module in _attention_modules(root):
        if getattr(module, "per_layer_metrics", None):
            yield layer, module
            layer += 1
        else:
            yield None, module


def layer_key(key: str, layer: int) -> str:
    """The logged name of one layer's value of a per-layer metric."""
    return f"{key}_layer{layer}"


def collect_attention_metrics(root) -> Dict[str, float]:
    """Average each attention diagnostic across the attention modules under
    ``root`` (empty when none opt in).

    Averaging across modules is the same convention Arc and the activations
    use. A module can opt keys out of it with ``per_layer_metrics``: those are
    logged once per layer instead (:func:`layer_key`), for diagnostics whose
    depth profile is the signal.
    """
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    out: Dict[str, float] = {}
    for layer, module in _layered_attention(root):
        per_layer = getattr(module, "per_layer_metrics", ())
        for key, value in module.training_metrics().items():
            if value is None:
                continue
            if key in per_layer:
                out[layer_key(key, layer)] = value
                continue
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    out.update({key: sums[key] / counts[key] for key in sums})
    return out


def _activation_modules(root) -> Iterator:
    """Yield activation modules under ``root`` that publish diagnostics.

    Activations are called as a bare ``act(x)`` and have no loss hook, so a
    module walk is the only way to reach them. ``nn.Module`` defines no
    ``training_metrics``, so the attribute check alone selects the opted-in
    ones (currently Servant and ActivationMixture); the rest of the registry is
    skipped. ``DepthActivation`` is included explicitly: it is a wrapper rather
    than a registry entry, so it is not in ``activation_classes()``, but it is
    the module that knows whether the passes are specializing.
    """
    from praxis.activations import activation_classes
    from praxis.activations.depth import DepthActivation

    classes = tuple(activation_classes()) + (DepthActivation,)
    for module in root.modules():
        if isinstance(module, classes) and hasattr(module, "training_metrics"):
            yield module


def collect_activation_metrics(root) -> Dict[str, float]:
    """Average each activation diagnostic across the activation modules under
    ``root`` (empty when none opt in)."""
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for module in _activation_modules(root):
        for key, value in module.training_metrics().items():
            if value is None:
                continue
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums}


def collect_activation_descriptions(root) -> Dict[str, dict]:
    """Gather ``metric_descriptions`` from the activation modules under ``root``."""
    out: Dict[str, dict] = {}
    for module in _activation_modules(root):
        descs = getattr(type(module), "metric_descriptions", None)
        if isinstance(descs, dict):
            out.update(descs)
    return out


def collect_attention_descriptions(root) -> Dict[str, dict]:
    """Gather ``metric_descriptions`` from the attention modules under ``root``.

    Without this the Dynamics tab has no declaration for an attention key, and
    a metric with no declaration is logged to the database and then dropped on
    the floor - the manifest is built from descriptions, not from columns.

    Read off the instance, which falls back to the class: a module composed
    from stages declares what its stages publish. Per-layer keys are expanded
    to one series per layer on a shared chart.
    """
    out: Dict[str, dict] = {}
    for layer, module in _layered_attention(root):
        descs = getattr(module, "metric_descriptions", None)
        if not isinstance(descs, dict):
            continue
        per_layer = getattr(module, "per_layer_metrics", ())
        for key, value in descs.items():
            if key not in per_layer:
                out[key] = value
            elif isinstance(value, dict) and isinstance(value.get("chart"), dict):
                out[layer_key(key, layer)] = {
                    **value,
                    "chart": {
                        **value["chart"],
                        "series_group": key,
                        "series_label": f"layer {layer}",
                    },
                    "caller": type(module).__name__,
                }
    return out


def collect_attention_snapshots(root) -> Dict[str, dict]:
    """Gather live ``dashboard_snapshots()`` from attention modules under ``root``."""
    out: Dict[str, dict] = {}
    for module in _attention_modules(root):
        if hasattr(module, "dashboard_snapshots"):
            out.update(module.dashboard_snapshots() or {})
    return out


def collect_arc_descriptions(root) -> Dict[str, dict]:
    """Gather ``metric_descriptions`` from the Arc modules under ``root``."""
    out: Dict[str, dict] = {}
    for module in _arc_modules(root):
        descs = getattr(type(module), "metric_descriptions", None)
        if isinstance(descs, dict):
            out.update(descs)
    return out
