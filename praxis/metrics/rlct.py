"""RLCT landscape: the first loss-geometry probe in Praxis.

Singular Learning Theory's Local Learning Coefficient (LLC, the estimable form
of the Real Log Canonical Threshold) measures how *degenerate* the current
minimum is. Low LLC = a flat, degenerate basin = high bias ("water pools" settle
in the minima); high LLC = a sharp, high-effective-dimension region = high
variance (the "red ridges" where the geometry stretches). See ``../RLCT.md`` and
``next/rlct_landscape.md``.

This module is the producer's compute core; it owns both the numbers and their
dashboard cards (the co-locate-metrics convention). It exposes:

* :func:`probe_landscape` - the one entry point a callback calls each probe. It
  evaluates a 2D loss slice through the live weights (the terrain) and, best
  effort, an SGLD estimate of the scalar coefficient lambda-hat (the headline
  number). Pure of any Lightning/batch specifics: the caller hands in two loss
  closures and the module drives the weight perturbation + exact restore.
* :data:`RLCT_METRIC_DESCRIPTIONS` - the universal (not model-attached) chart +
  snapshot hints, folded into ``get_metric_descriptions`` like the optimizer
  suite.

Design notes (deliberate, model-agnostic, no per-run tuning - see
``feedback_no_hyperparameter_tuning``):

* **Directions** are two filter-normalized random directions (Li et al. 2018,
  "Visualizing the Loss Landscape"), drawn from a *fixed* seed so the slice
  plane's orientation is stable across the whole run - you watch the same
  cross-section deepen and sharpen rather than a plane that swims every probe.
  Each direction is scaled per parameter tensor to that tensor's own norm, so
  ``extent`` reads as a fraction of the weight scale.
* **lambda-hat** uses the WBIC/SGLD estimator nb*(E_w[L] - L(w*)) with the
  calibrated temperature b = 1/log n and a fixed localization gamma. SGLD is
  notoriously step-size sensitive, so the chain auto-backs-off epsilon on
  divergence and returns ``None`` rather than a blown-up number. Read lambda-hat
  *relatively* - its trajectory and phase-transition drops, not its absolute
  value as a calibrated dimension count.
"""

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch

# Model-agnostic constants. Not CLI flags, not per-experiment knobs: a registry
# of fixed defaults baked into the probe (see feedback_registry_over_cli_args).
RLCT_DEFAULTS: Dict[str, float] = {
    "period": 250,  # recompute every N optimizer steps
    "warmup_steps": 200,  # let the initial chaos settle before probing
    "grid": 17,  # G x G landscape resolution (G^2 forward passes / probe)
    "extent": 0.7,  # slice spans +/- this fraction of the weight scale
    "probe_seqs": 2,  # sub-batch sequences used for the probe loss (cost control)
    "probe_len": 256,  # truncate the probe sequence to this many tokens. The
    #   single biggest cost lever: a long-context model (block_size 4096) does
    #   G^2 full forwards otherwise. CALM pads to a multiple of K internally and
    #   plain models accept any length, so truncation is safe.
    "direction_seed": 1729,  # fixed -> stable slice-plane orientation over a run
    "chain_steps": 6,  # SGLD chain length for lambda-hat
    "sgld_epsilon": 1e-5,  # base SGLD step (auto-backs-off on divergence)
    "sgld_gamma": 1.0,  # localization elasticity to w*
    "sgld_backoff": 1,  # times to halve epsilon before giving up on lambda-hat
    "max_params": 150_000_000,  # skip the probe above this (the 3x param clone)
    "manifold_grid": 28,  # G x G bins for the parameter-manifold terrain
    "manifold_max_rows": 20000,  # subsample rows above this before PCA
    "field_max_cells": 128,  # max grid dim for the literal parameter-field terrain
}

# Substrings that flag a structured-head weight worth projecting (harmonic
# spectra, crystal/geometry centers, fields) - preferred over a plain FFN matrix
# when picking the manifold's target, since that is where geometry lives.
_STRUCTURED_HINTS = ("amplitud", "harmonic", "crystal", "center", "field", "spectr")


def _probe_params(core) -> List[torch.nn.Parameter]:
    """Trainable floating-point parameters of the model, in a stable order."""
    return [
        p
        for p in core.parameters()
        if p.requires_grad and p.is_floating_point()
    ]


def _filter_normalized_directions(
    params: List[torch.nn.Parameter], seed: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Two random directions, each tensor scaled to its own weight norm.

    Filter (here, per-tensor) normalization keeps the perturbation commensurate
    with each layer's scale, so a single ``extent`` is meaningful across the
    whole model instead of being swamped by the largest-norm tensor.
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def one() -> List[torch.Tensor]:
        out = []
        for p in params:
            # Draw on CPU for a device-independent, reproducible plane, then
            # move to the parameter's device/dtype.
            d = torch.randn(p.shape, generator=gen, dtype=torch.float32)
            d = d.to(device=p.device, dtype=p.dtype)
            wnorm = p.detach().norm()
            dnorm = d.norm()
            if dnorm > 0:
                d = d * (wnorm / (dnorm + 1e-12))
            out.append(d)
        return out

    return one(), one()


@torch.no_grad()
def _set_offset(
    params: List[torch.nn.Parameter],
    base: List[torch.Tensor],
    d1: List[torch.Tensor],
    d2: List[torch.Tensor],
    a: float,
    b: float,
) -> None:
    """Set every param to ``base + a*d1 + b*d2`` in place."""
    for p, w0, e1, e2 in zip(params, base, d1, d2):
        p.data.copy_(w0)
        if a != 0.0:
            p.data.add_(e1, alpha=a)
        if b != 0.0:
            p.data.add_(e2, alpha=b)


@torch.no_grad()
def _restore(params: List[torch.nn.Parameter], base: List[torch.Tensor]) -> None:
    for p, w0 in zip(params, base):
        p.data.copy_(w0)


def _sgld_lambda(
    params: List[torch.nn.Parameter],
    base: List[torch.Tensor],
    loss_with_grad: Callable[[], torch.Tensor],
    l0: float,
    n_tokens: int,
    cfg: Dict[str, float],
) -> Optional[float]:
    """WBIC/SGLD estimate of the local learning coefficient lambda-hat.

    lambda-hat = n*beta * (mean_t L(w_t) - L(w*)), with the chain localized to
    w* by an elastic term. Auto-backs-off the step size on divergence; returns
    ``None`` if it can't produce a finite estimate (the landscape still ships).
    """
    n = max(int(n_tokens), 3)
    beta = 1.0 / math.log(n)
    gamma = float(cfg["sgld_gamma"])
    eps = float(cfg["sgld_epsilon"])
    steps = int(cfg["chain_steps"])
    gen = torch.Generator(device="cpu").manual_seed(int(cfg["direction_seed"]) + 7)

    for _ in range(int(cfg["sgld_backoff"]) + 1):
        _restore(params, base)
        losses: List[float] = []
        diverged = False
        for _t in range(steps):
            for p in params:
                p.grad = None
            loss = loss_with_grad()
            if not torch.isfinite(loss):
                diverged = True
                break
            lval = float(loss.detach())
            # A chain that climbs far past the clean loss has blown up; the
            # estimate from such a run is meaningless. Back off the step.
            if lval > l0 + 10.0 * (abs(l0) + 1.0):
                diverged = True
                break
            losses.append(lval)
            loss.backward()
            with torch.no_grad():
                for p, w0 in zip(params, base):
                    g = p.grad
                    drift = -gamma * (p.data - w0)
                    if g is not None:
                        drift = drift - beta * n * g
                    noise = torch.randn(
                        p.shape, generator=gen, dtype=torch.float32
                    ).to(device=p.device, dtype=p.dtype)
                    p.data.add_(drift, alpha=eps * 0.5).add_(
                        noise, alpha=math.sqrt(eps)
                    )
        if not diverged and losses:
            lam = n * beta * (sum(losses) / len(losses) - l0)
            if math.isfinite(lam):
                return float(lam)
        eps *= 0.5  # backoff and retry

    return None


def probe_landscape(
    core,
    loss_only: Callable[[], float],
    loss_with_grad: Optional[Callable[[], torch.Tensor]] = None,
    *,
    n_tokens: int = 1,
    step: int = 0,
    cfg: Dict[str, float] = RLCT_DEFAULTS,
) -> Tuple[Optional[dict], Optional[dict]]:
    """Evaluate the loss landscape (and lambda-hat) around the live weights.

    Args:
        core: the (uncompiled) model whose parameters are perturbed in place.
        loss_only: ``() -> float``, a no-grad forward returning the scalar loss
            at the *current* weights. Called once per grid cell.
        loss_with_grad: ``() -> Tensor``, a grad-bearing forward returning the
            scalar loss tensor (for SGLD). ``None`` skips lambda-hat.
        n_tokens: effective sample count for the SGLD temperature b = 1/log n.
        step: current training step, stamped into the payload.
        cfg: constants (see :data:`RLCT_DEFAULTS`).

    Returns ``(landscape_payload, scalar_metrics)``; ``(None, None)`` if the
    model is too large to probe safely. The caller is responsible for having a
    consistent model state on entry; this restores params and buffers to exactly
    their entry values before returning.
    """
    params = _probe_params(core)
    if not params:
        return None, None
    total = sum(p.numel() for p in params)
    if total > int(cfg["max_params"]):
        return None, None

    G = int(cfg["grid"])
    extent = float(cfg["extent"])

    # Snapshot params and buffers so the model is bit-restored afterwards. The
    # extra grad-bearing forwards mutate train-mode buffers (memory state, etc.);
    # restoring them keeps the probe side-effect-free for the optimizer.
    base = [p.detach().clone() for p in params]
    buffers = [(b, b.detach().clone()) for b in core.buffers()]
    d1, d2 = _filter_normalized_directions(params, int(cfg["direction_seed"]))

    try:
        # Clean loss at the center (w*).
        l0 = float(loss_only())

        coords = [(-extent + 2.0 * extent * k / (G - 1)) if G > 1 else 0.0
                  for k in range(G)]
        grid: List[List[float]] = []
        for a in coords:
            row: List[float] = []
            for b in coords:
                _set_offset(params, base, d1, d2, a, b)
                row.append(float(loss_only()))
            grid.append(row)
        _restore(params, base)

        lam = None
        if loss_with_grad is not None:
            try:
                lam = _sgld_lambda(
                    params, base, loss_with_grad, l0, n_tokens, cfg
                )
            except Exception:
                lam = None
            _restore(params, base)
    finally:
        # Always restore - params, buffers, and any probe gradients.
        _restore(params, base)
        with torch.no_grad():
            for b, b0 in buffers:
                if b.is_floating_point():
                    b.data.copy_(b0)
        for p in params:
            p.grad = None

    flat = [v for row in grid for v in row]
    z_min = min(flat)
    z_max = max(flat)
    # Per-cell LLC proxy: loss increase above the basin floor. Mean = corpus
    # average, max = peak-variance region, min ~ tightest minimum, std =
    # landscape roughness (the four card numbers from RLCT.md).
    llc = [v - z_min for v in flat]
    mean = sum(llc) / len(llc)
    var = sum((x - mean) ** 2 for x in llc) / len(llc)
    std = math.sqrt(var)

    payload = {
        "grid": grid,
        "rows": G,
        "cols": G,
        "z_min": z_min,
        "z_max": z_max,
        "l0": l0,
        "extent": extent,
        "lambda_hat": lam,
        "step": int(step),
    }
    metrics: Dict[str, float] = {
        "rlct_llc_mean": mean,
        "rlct_llc_max": max(llc),
        "rlct_llc_min": min(llc),
        "rlct_llc_std": std,
    }
    if lam is not None and math.isfinite(lam):
        metrics["rlct_lambda"] = lam

    return payload, metrics


def _pick_manifold_weight(core):
    """Choose a 2D weight whose rows carry geometry worth projecting.

    Structured-first: if any harmonic/crystal/field weight exists (with enough
    rows), pick the largest of those - that is where geometry lives, and we want
    it even when a plain FFN/embedding has more rows. Only when there are none
    does it fall back to the richest matrix overall. Returns ``(name, weight)``
    or ``None``.
    """
    structured = []  # (rows, name, param)
    general = []
    for name, p in core.named_parameters():
        if not p.requires_grad or not p.is_floating_point() or p.dim() != 2:
            continue
        rows, cols = p.shape
        if rows < 16 or cols < 2:
            continue
        general.append((rows, name, p))
        if any(h in name.lower() for h in _STRUCTURED_HINTS) and rows >= 32:
            structured.append((rows, name, p))
    pool = structured or general
    if not pool:
        return None
    _, name, p = max(pool, key=lambda t: t[0])  # largest by rows within the tier
    return name, p


@torch.no_grad()
def compute_param_manifold(
    core, *, grid: int = 28, max_rows: int = 20000
) -> Optional[dict]:
    """Project a weight tensor's rows to 2D (PCA) and bin into a terrain.

    Each row of the chosen weight is a point in feature space; PCA lays the
    cloud out along its two highest-variance axes, so the terrain's *shape* is
    the weight geometry (a Gaussian blob for an unstructured layer, rings/arms
    for a structured head). Height = row density (where parameters cluster);
    color tint = mean row amplitude (||row||). Pure weight analysis - no forward
    passes, no perturbation.
    """
    picked = _pick_manifold_weight(core)
    if picked is None:
        return None
    name, W = picked
    W = W.detach().float()
    if W.shape[0] > max_rows:
        idx = torch.linspace(0, W.shape[0] - 1, max_rows).long().to(W.device)
        W = W.index_select(0, idx)
    N, d = W.shape

    mean = W.mean(dim=0, keepdim=True)
    Wc = W - mean
    q = min(3, d, N)
    try:
        _, S, V = torch.pca_lowrank(Wc, q=q, center=False)
    except Exception:
        return None
    if V.shape[1] < 2:
        return None

    coords = Wc @ V[:, :2]  # (N, 2)
    amp = W.norm(dim=1)  # (N,)
    var = S * S
    var_explained = float((var[:2].sum() / (var.sum() + 1e-9)).item())

    def _robust(x):
        lo = torch.quantile(x, 0.01)
        hi = torch.quantile(x, 0.99)
        rng = (hi - lo).clamp(min=1e-9)
        return ((x - lo) / rng * 2 - 1).clamp(-1, 1)

    cx = _robust(coords[:, 0])
    cy = _robust(coords[:, 1])

    G = int(grid)
    ix = ((cx + 1) * 0.5 * (G - 1)).round().long().clamp_(0, G - 1)
    iy = ((cy + 1) * 0.5 * (G - 1)).round().long().clamp_(0, G - 1)
    flat = (ix * G + iy).cpu()
    amp_c = amp.cpu().float()

    count = torch.bincount(flat, minlength=G * G).float()
    ampsum = torch.zeros(G * G, dtype=torch.float32).scatter_add_(0, flat, amp_c)
    mean_amp = torch.where(count > 0, ampsum / count.clamp(min=1), torch.zeros_like(ampsum))
    amax = float(mean_amp.max().clamp(min=1e-9))

    density = count.view(G, G).tolist()
    tint = (mean_amp / amax).view(G, G).tolist()

    return {
        "density": density,
        "tint": tint,
        "rows": G,
        "cols": G,
        "max_count": int(count.max().item()),
        "n_points": int(N),
        "weight_name": name,
        "var_explained": var_explained,
    }


@torch.no_grad()
def compute_param_field(core, *, max_cells: int = 128) -> Optional[dict]:
    """Literal weight terrain: actual parameter values by index, height = |value|.

    Unlike the manifold (which PCA-projects rows into a density cloud), this
    renders the chosen weight tensor *as it is* - each cell is a real parameter,
    laid out at its native index, with amplitude as height. For a structured
    head this is the actual harmonic/crystal amplitude grid as terrain. Full
    fidelity when the tensor fits under ``max_cells`` per axis; above that it is
    max-pooled (peak-preserving) down to a renderable grid.
    """
    picked = _pick_manifold_weight(core)
    if picked is None:
        return None
    name, W = picked
    A = W.detach().abs().float()
    R, C = A.shape
    pooled = False
    if R > max_cells or C > max_cells:
        import torch.nn.functional as F

        out_r, out_c = min(R, max_cells), min(C, max_cells)
        A = F.adaptive_max_pool2d(A.view(1, 1, R, C), (out_r, out_c)).view(out_r, out_c)
        pooled = True
    amax = float(A.max().clamp(min=1e-9))
    amp = (A / amax).cpu().tolist()
    return {
        "amp": amp,
        "rows": A.shape[0],
        "cols": A.shape[1],
        "weight_name": name,
        "pooled": pooled,
        "native_shape": [int(R), int(C)],
        "n_params": int(W.numel()),
    }


# ─── Dashboard cards (universal, folded into get_metric_descriptions) ─────────

_RLCT_GROUP = "rlct"

RLCT_METRIC_DESCRIPTIONS: Dict[str, dict] = {
    "rlct_lambda": {
        "description": (
            "Local Learning Coefficient (SGLD/WBIC estimate of the RLCT). "
            "Low = a flat, degenerate basin (high bias); high = a sharp, "
            "high-effective-dimension minimum (high variance). Read it "
            "relatively: a sharp DROP marks a phase transition where the model "
            "collapses onto a simpler solution. Best-effort - blank when the "
            "SGLD chain can't produce a stable estimate."
        ),
        "chart": {
            "title": "Local Learning Coefficient (λ̂)",
            "y_label": "λ̂",
            "y_scale": "linear",
            "group": _RLCT_GROUP,
            "group_order": 35,
            "order": 10,
        },
    },
    "rlct_llc_mean": {
        "description": (
            "Mean LLC over the loss-landscape slice - the corpus-average "
            "geometry (the grey relief). Rises as the slice develops structure."
        ),
        "chart": {
            "title": "Landscape LLC Distribution",
            "y_label": "loss increase",
            "y_scale": "linear",
            "group": _RLCT_GROUP,
            "order": 20,
            "series_group": "rlct_llc",
            "series_label": "mean (corpus average)",
        },
    },
    "rlct_llc_max": {
        "description": (
            "Max LLC over the slice - the peak-variance region, where the "
            "geometry stretches most (the red ridges)."
        ),
        "chart": {
            "title": "Landscape LLC Distribution",
            "y_label": "loss increase",
            "y_scale": "linear",
            "group": _RLCT_GROUP,
            "order": 21,
            "series_group": "rlct_llc",
            "series_label": "max (peak variance)",
        },
    },
    "rlct_llc_min": {
        "description": (
            "Min LLC over the slice - the tightest minimum (the basin floor, "
            "where the water pools)."
        ),
        "chart": {
            "title": "Landscape LLC Distribution",
            "y_label": "loss increase",
            "y_scale": "linear",
            "group": _RLCT_GROUP,
            "order": 22,
            "series_group": "rlct_llc",
            "series_label": "min (tightest basin)",
        },
    },
    "rlct_llc_std": {
        "description": (
            "Std of LLC over the slice - landscape roughness. High = a jagged, "
            "high-variance neighborhood; low = a smooth, degenerate basin."
        ),
        "chart": {
            "title": "Landscape Roughness",
            "y_label": "LLC std",
            "y_scale": "linear",
            "group": _RLCT_GROUP,
            "order": 30,
        },
    },
    "rlct_manifold_var": {
        "description": (
            "Fraction of the manifold weight's variance captured by its top-2 "
            "PCA axes. High = the weight geometry is nearly planar (a flat, "
            "anisotropic sheet the terrain shows faithfully); low = the geometry "
            "is genuinely high-dimensional and the 2D projection is a shadow."
        ),
        "chart": {
            "title": "Manifold Planarity (top-2 PCA var)",
            "y_label": "variance explained",
            "y_scale": "linear",
            "group": _RLCT_GROUP,
            "order": 40,
        },
    },
    # The terrain itself: a bespoke 3D mesh snapshot, routed through the
    # rlct_mesh renderer (hillshade relief + cyan basins + red ridges).
    "rlct_landscape": {
        "description": (
            "Loss-landscape slice through the live weights along two fixed "
            "filter-normalized directions. Height is loss; cyan pools mark "
            "flat low-loss basins (high bias), red ridges mark high-curvature "
            "walls (high variance), grey is the corpus-average relief between. "
            "The terrain deepens and sharpens as training moves through this "
            "fixed cross-section."
        ),
        "snapshot": {
            "title": "RLCT Landscape",
            "renderer": "rlct_mesh",
            "group": _RLCT_GROUP,
            "order": 100,
        },
    },
    # Parameter manifold: the PCA terrain of a structured weight's rows. Where
    # the loss landscape is a smooth bowl, this shows the actual weight geometry.
    "param_manifold": {
        "description": (
            "PCA projection of a structured weight tensor's rows into a terrain. "
            "Each row is a point laid out along its two highest-variance axes, so "
            "the terrain SHAPE is the weight geometry - a Gaussian hill for an "
            "unstructured layer, rings or arms for a harmonic/crystal head. "
            "Height = where parameters cluster (density), color = mean row "
            "amplitude. The card names the weight it projected."
        ),
        "snapshot": {
            "title": "Parameter Manifold",
            "renderer": "param_manifold",
            "group": _RLCT_GROUP,
            "order": 110,
        },
    },
    # Literal weight terrain: actual parameter values by index, height = |value|.
    # Where the manifold abstracts via PCA, this is the raw structure of the
    # weights - the harmonic/crystal amplitude grid rendered as it is.
    "param_field": {
        "description": (
            "The chosen weight tensor rendered literally: each cell is a real "
            "parameter at its native index, height = its absolute value. No "
            "projection - this is the actual geometry of the weights (a "
            "harmonic spectrum as mountains, a crystal grid as a lattice). Full "
            "resolution when it fits; max-pooled above the render cap. The card "
            "names the weight and its native shape."
        ),
        "snapshot": {
            "title": "Parameter Field",
            "renderer": "param_field",
            "group": _RLCT_GROUP,
            "order": 120,
        },
    },
}
