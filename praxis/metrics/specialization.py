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
    ATTENTION_REGISTRY opts in the same way an activation does - by defining
    ``training_metrics`` - and, like activations, has no loss hook, so a module
    walk is the only way to reach it. Without this walk a mechanism can publish
    a full diagnostic suite that never reaches the log, which is exactly what
    happened to SSOG's field metrics.
    """
    from praxis.attention import ATTENTION_REGISTRY
    from praxis.attention.arc import ArcAttention

    classes = tuple(
        {getattr(v, "func", v) for v in ATTENTION_REGISTRY.values()}
    )
    for module in root.modules():
        if (
            isinstance(module, classes)
            and not isinstance(module, ArcAttention)
            and hasattr(module, "training_metrics")
        ):
            yield module


def collect_attention_metrics(root) -> Dict[str, float]:
    """Average each attention diagnostic across the attention modules under
    ``root`` (empty when none opt in).

    Averaging across modules is the same convention Arc and the activations
    use. At ``num_layers: 1`` there is one module and the average is the value;
    with several distinct attention blocks the grid cards below become a mean
    field, which is a real limitation and the reason the per-module story would
    need a layer prefix if it is ever wanted.
    """
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for module in _attention_modules(root):
        for key, value in module.training_metrics().items():
            if value is None:
                continue
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums}


def _activation_modules(root) -> Iterator:
    """Yield activation modules under ``root`` that publish diagnostics.

    Activations are called as a bare ``act(x)`` and have no loss hook, so a
    module walk is the only way to reach them. ``nn.Module`` defines no
    ``training_metrics``, so the attribute check alone selects the opted-in
    ones (currently Servant); the rest of the registry is skipped.
    """
    from praxis.activations import ACT2CLS

    classes = tuple({v[0] if isinstance(v, tuple) else v for v in ACT2CLS.values()})
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
    """
    out: Dict[str, dict] = {}
    for module in _attention_modules(root):
        descs = getattr(type(module), "metric_descriptions", None)
        if isinstance(descs, dict):
            out.update(descs)
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
