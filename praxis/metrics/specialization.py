"""Depth-specialization diagnostics for Arc modules.

ArcAttention and ArcGLU give each recurrent-depth pass its own learned
parameters (per-depth Q/K/V/O biases, per-pass activations). The risk is
that those copies collapse to identical values, erasing any benefit over a
single shared parameter. ``depth_dispersion`` measures how far a stack of
per-depth vectors has diverged; the collectors walk a live model and average
each Arc module's report so the dashboard sees one number per metric.

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


def collect_arc_descriptions(root) -> Dict[str, dict]:
    """Gather ``metric_descriptions`` from the Arc modules under ``root``."""
    out: Dict[str, dict] = {}
    for module in _arc_modules(root):
        descs = getattr(type(module), "metric_descriptions", None)
        if isinstance(descs, dict):
            out.update(descs)
    return out
