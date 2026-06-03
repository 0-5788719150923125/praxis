"""Recent crystal-head ``Center PCA Density`` geometries, rendered for the paper.

The dashboard's Dynamics tab shows a live ``crystal_centers_pca`` snapshot: the
top-2 PCA projection of a crystal head's vocabulary centers, binned to a density
grid (see :func:`praxis.heads.crystal._pca_density_grid`). This module reproduces
that view offline for the most recent runs and tiles them into a figure for the
paper's ``Geometry that looks like nature`` section - an ablation of the
geometries different heads/runs converge to.

A geometry is detected straight from the checkpoint: any parameter whose key
ends in ``lm_head.centers`` is a set of crystal centers. A run contributes one
geometry per such tensor, so prismatic/stacked heads yield several and a run
with no crystal head yields none. Runs are scanned newest-first until ``limit``
geometries are collected.

Output (all generated, none committed):
- ``research/figures/geometry_N.png`` - one density heatmap per panel.
- ``research/geometries.tex`` - defines ``\\paperGeometryFigure``, the figure
  body. ``research/framing.tex``'s geometry fragment drops the macro in place,
  and ``main.tex`` falls back to an empty ``\\providecommand`` on a clean
  checkout, so the prose simply renders without the figure.

Entry point: :func:`export_geometries`, driven by :mod:`praxis.pillars.build`.
"""

import glob
import json
import os

from praxis.pillars.runs import experiment_name, experiment_stems

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
RUNS_DIR = os.path.join(REPO_ROOT, "build", "runs")
RESEARCH_DIR = os.path.join(REPO_ROOT, "research")
FIG_DIR = os.path.join(RESEARCH_DIR, "figures")
OUT_TEX = os.path.join(RESEARCH_DIR, "geometries.tex")
GRID_SIZE = 64
CENTERS_SUFFIX = "lm_head.centers"
# Shared with the web dashboard (praxis/web/src/js/colormaps.js is generated from
# the same file), so the printed figure matches the live Center PCA Density card.
COLORMAPS_JSON = os.path.join(REPO_ROOT, "praxis", "web", "src", "colormaps.json")
COLORMAP_NAME = "praxis_heat"


def runs_newest_first():
    """[(created_key, hash, name, run_dir)] sorted newest-first."""
    stems = experiment_stems()
    out = []
    for cfg_path in glob.glob(os.path.join(RUNS_DIR, "*", "config.json")):
        run_dir = os.path.dirname(cfg_path)
        try:
            cfg = json.load(open(cfg_path))
        except (OSError, ValueError):
            continue
        key = cfg.get("created") or str(os.path.getmtime(cfg_path))
        name = experiment_name(cfg.get("command", ""), stems)
        out.append((key, cfg.get("truncated_hash", os.path.basename(run_dir)), name, run_dir))
    return sorted(out, key=lambda r: r[0], reverse=True)


def latest_checkpoint(run_dir):
    """Newest .ckpt in a run's model/ dir (resolving last.ckpt), or None."""
    model_dir = os.path.join(run_dir, "model")
    cks = [c for c in glob.glob(os.path.join(model_dir, "*.ckpt"))
           if not os.path.islink(c)]
    if not cks:
        return None
    return max(cks, key=os.path.getmtime)


def pca_density_grid(W, grid_size=GRID_SIZE):
    """Top-2 PCA density grid of row vectors. Mirrors the crystal head's
    snapshot so the paper figure matches the dashboard card."""
    import torch

    W = W.detach().to(torch.float32)
    centered = W - W.mean(dim=0, keepdim=True)
    # Deterministic full SVD, not the randomized svd_lowrank: the latter draws
    # from the global RNG, which would make the figure non-reproducible (PDF
    # churn) and - since the paper build runs in-process during training -
    # perturb the training RNG stream. The matrices here are tiny.
    _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vh[:2].transpose(-2, -1)
    spans, bins = [], []
    for i in range(2):
        lo, hi = float(proj[:, i].min()), float(proj[:, i].max())
        span = max(hi - lo, 1e-12)
        bins.append(((proj[:, i] - lo) / span * (grid_size - 1)).long().clamp_(0, grid_size - 1))
        spans.append(span)
    flat = bins[1] * grid_size + bins[0]
    grid = torch.bincount(flat, minlength=grid_size * grid_size).view(grid_size, grid_size)
    n = max(centered.shape[0] - 1, 1)
    total_var = float(centered.pow(2).sum() / n)
    ve = [float(v) / total_var for v in (S[:2].pow(2) / n).tolist()] if total_var > 0 else [0.0, 0.0]
    return grid.cpu().numpy(), ve


def branch_label(key):
    """Short human tag for which head produced a centers tensor, e.g.
    ``model.head.branches.1.heads.1.lm_head.centers`` -> ``branch 1``."""
    parts = key.split(".")
    for marker in ("branches", "branch", "heads"):
        if marker in parts:
            i = parts.index(marker)
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                return f"branch {parts[i + 1]}"
    return ""


def collect_geometries(limit, scan):
    """Newest-first list of geometry dicts (up to ``limit``), scanning at most
    ``scan`` runs. Each: {name, hash, label, grid, var_explained, n_points}."""
    import torch

    geometries = []
    for _, run_hash, name, run_dir in runs_newest_first()[:scan]:
        if len(geometries) >= limit:
            break
        ckpt = latest_checkpoint(run_dir)
        if not ckpt:
            continue
        try:
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        except Exception:
            continue
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        keys = sorted(k for k in sd if k.endswith(CENTERS_SUFFIX))
        multi = len(keys) > 1
        for key in keys:
            if len(geometries) >= limit:
                break
            W = sd[key]
            if not hasattr(W, "dim") or W.dim() != 2 or W.shape[0] < 3:
                continue
            grid, ve = pca_density_grid(W)
            geometries.append({
                "name": name,
                "hash": run_hash,
                "label": branch_label(key) if multi else "",
                "grid": grid,
                "var_explained": ve,
                "n_points": int(W.shape[0]),
            })
    return geometries


def _shared_cmap():
    """matplotlib colormap built from the shared colormaps.json stops, so the
    paper figure and the dashboard render the same ramp. Empty (zero-density)
    cells fall through to the ramp's first color (black)."""
    from matplotlib.colors import LinearSegmentedColormap

    spec = json.load(open(COLORMAPS_JSON))[COLORMAP_NAME]
    colors = [(pos, [c / 255.0 for c in rgb]) for pos, rgb in spec["stops"]]
    cmap = LinearSegmentedColormap.from_list(COLORMAP_NAME, colors)
    cmap.set_bad(colors[0][1])  # zero-density cells = ramp floor (black)
    return cmap


def render_png(geo, index):
    """Write one density heatmap; return its repo-relative figure path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    grid = geo["grid"]
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3, 3))
    masked = np.where(grid > 0, grid, np.nan)
    ax.imshow(masked, origin="lower", cmap=_shared_cmap(),
              norm=LogNorm(vmin=1, vmax=max(int(grid.max()), 2)))
    title = geo["name"] + (f" {geo['label']}" if geo["label"] else "")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    out = os.path.join(FIG_DIR, f"geometry_{index}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return os.path.relpath(out, RESEARCH_DIR)


def figure_tex(paths, geometries):
    """The ``\\paperGeometryFigure`` macro: a 2-column grid of the panels.

    ``[tbp]`` lets it take a dedicated float page if the block is too tall to sit
    with text, which keeps it near its section instead of drifting to the end.
    """
    rows = []
    for i in range(0, len(paths), 2):
        cells = " \\hfill\n  ".join(
            f"\\includegraphics[width=0.46\\linewidth]{{{p}}}" for p in paths[i:i + 2]
        )
        rows.append(cells)
    body = " \\\\[6pt]\n  ".join(rows)
    names = ", ".join(
        g["name"] + (f" ({g['label']})" if g["label"] else "") for g in geometries
    )
    caption = (
        "Center PCA density for the "
        f"{len(geometries)} most recent crystal-head runs ({names}). Each panel "
        "is the top-2 PCA projection of a head's vocabulary centers, binned to a "
        f"{GRID_SIZE}$\\times${GRID_SIZE} density grid (log color) - the same "
        "snapshot the dashboard renders live. Different runs settle into "
        "structurally different geometries, not noise."
    )
    return (
        "\\newcommand{\\paperGeometryFigure}{%\n"
        "\\begin{figure}[tbp]\n  \\centering\n  "
        f"{body}\n"
        f"  \\caption{{{caption}}}\n"
        "  \\label{fig:geometry}\n"
        "\\end{figure}\n}\n"
    )


def export_geometries(limit: int = 4, scan: int = 40) -> dict:
    """Render up to ``limit`` recent geometries (scanning ``scan`` runs) into
    figures/ + geometries.tex. Returns a summary dict."""
    geometries = collect_geometries(limit, scan)
    if not geometries:
        # No crystal geometry anywhere in scan: emit an empty macro so the
        # paper still builds (fragment renders prose without a figure).
        with open(OUT_TEX, "w") as fh:
            fh.write("% Generated by praxis/pillars/geometries.py - no crystal geometries found.\n"
                     "\\newcommand{\\paperGeometryFigure}{}\n")
        return {"count": 0, "panels": []}

    paths = [render_png(g, i + 1) for i, g in enumerate(geometries)]
    with open(OUT_TEX, "w") as fh:
        fh.write("% Generated by praxis/pillars/geometries.py - do not edit by hand.\n")
        fh.write(figure_tex(paths, geometries))

    return {
        "count": len(geometries),
        "panels": [
            {"name": g["name"], "hash": g["hash"], "label": g["label"],
             "var_explained": [round(v, 4) for v in g["var_explained"]],
             "n_points": g["n_points"]}
            for g in geometries
        ],
    }
