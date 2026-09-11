"""Recent crystal-classifier ``Center PCA Density`` geometries, rendered for the paper.

The dashboard's Dynamics tab shows a live ``crystal_centers_pca`` snapshot: the
top-2 PCA projection of a crystal classifier's vocabulary centers, binned to a density
grid (see :func:`praxis.classifiers.crystal._pca_density_grid`). This module reproduces
that view offline for the most recent runs and tiles them into a figure for the
paper's ``Geometry that looks like nature`` section - an ablation of the
geometries different classifiers/runs converge to.

A geometry is detected straight from the checkpoint: any [V, D] parameter whose
key ends in ``.centers`` (and has no HALO ``gamma`` sibling) is a set of crystal
centers. A run contributes one geometry per such tensor, so prismatic/sequential
classifiers yield several and a run with no crystal classifier yields none. Runs
are scanned newest-first until ``limit`` geometries are collected. Checkpoints
written before the heads -> classifiers rename are read through
:func:`praxis.renames.rename_legacy_state_dict`, so their keys match too.

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
from praxis.renames import rename_legacy_config, rename_legacy_state_dict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(REPO_ROOT, "build", "runs")
RESEARCH_DIR = os.path.join(REPO_ROOT, "research")
FIG_DIR = os.path.join(RESEARCH_DIR, "figures")
OUT_TEX = os.path.join(RESEARCH_DIR, "geometries.tex")
GRID_SIZE = 64
# Match any CrystalGeometry centers tensor (its param is always `.centers`):
# CrystalClassifier exposes it as `...scorer.centers`, while a VEAR crystal bank
# (CrystalVearClassifier) exposes N of them as `...bank.experts.<i>.centers`. Matching
# the bare suffix catches both; the [V, D] shape guard in collect_geometries
# rejects anything else that happens to end in `.centers`.
CENTERS_SUFFIX = ".centers"

# ...except HaloGeometry (praxis/classifiers/halo.py:96) ALSO names its parameter
# `centers`, and prismatic5/6 carry a HaloClassifier arm, so the bare suffix silently
# swept HALO's hyperspherical prototypes in as though they were a crystal
# geometry. They are neither: they live on a sphere, not in the crystal's
# Euclidean center space, and HALO's gate share is near zero, so the panel is a
# raw `randn` init - a featureless blob captioned as a settled geometry.
#
# The two are told apart structurally, not by name: HaloGeometry owns a
# learnable `gamma` temperature beside its centers, CrystalGeometry owns
# nothing beside them.
HALO_SIBLING = "gamma"
# Shared with the web dashboard (praxis/web/src/js/colormaps.js is generated from
# the same file), so the printed figure matches the live Center PCA Density card.
COLORMAPS_JSON = os.path.join(REPO_ROOT, "praxis", "web", "src", "colormaps.json")
COLORMAP_NAME = "praxis_heat"


def is_crystal_centers(sd, key):
    """True if ``key`` is a CrystalGeometry's centers, not HALO's prototypes.

    Both parameters are called ``centers`` and both are [vocab, dim], so shape
    cannot separate them; the sibling ``gamma`` can.
    """
    prefix = key[: -len("centers")]
    return f"{prefix}{HALO_SIBLING}" not in sd


def classifier_type_of(run_dir):
    """The run's resolved ``classifier_type``, or None if it never wrote a spec.

    Read rather than assumed: the caption used to hardcode one classifier name,
    so it kept asserting ``prismatic4`` long after the runs had moved to
    ``prismatic6``. Specs written before the rename store ``head_type``.
    """
    try:
        with open(os.path.join(run_dir, "spec.json")) as fh:
            args = dict(json.load(fh).get("args", {}))
        return rename_legacy_config(args).get("classifier_type") or None
    except (OSError, ValueError, AttributeError, TypeError):
        return None


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
        out.append(
            (key, cfg.get("truncated_hash", os.path.basename(run_dir)), name, run_dir)
        )
    return sorted(out, key=lambda r: r[0], reverse=True)


def latest_checkpoint(run_dir):
    """Newest .ckpt in a run's model/ dir (resolving last.ckpt), or None."""
    model_dir = os.path.join(run_dir, "model")
    cks = [
        c for c in glob.glob(os.path.join(model_dir, "*.ckpt")) if not os.path.islink(c)
    ]
    if not cks:
        return None
    return max(cks, key=os.path.getmtime)


def pca_density_grid(W, grid_size=GRID_SIZE):
    """Top-2 PCA density grid of row vectors. Mirrors the crystal classifier's
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
        bins.append(
            ((proj[:, i] - lo) / span * (grid_size - 1)).long().clamp_(0, grid_size - 1)
        )
        spans.append(span)
    flat = bins[1] * grid_size + bins[0]
    grid = torch.bincount(flat, minlength=grid_size * grid_size).view(
        grid_size, grid_size
    )
    n = max(centered.shape[0] - 1, 1)
    total_var = float(centered.pow(2).sum() / n)
    ve = (
        [float(v) / total_var for v in (S[:2].pow(2) / n).tolist()]
        if total_var > 0
        else [0.0, 0.0]
    )
    return grid.cpu().numpy(), ve


def branch_label(key):
    """Short human tag for which arm produced a centers tensor, e.g.
    ``...branches.1.stages.1.scorer.centers`` -> ``branch 1``, and a prismatic4
    VEAR crystal bank ``...branches.1...bank.experts.2.centers`` ->
    ``branch 1 · expert 2`` (so the bank's N crystals get distinct panels)."""
    parts = key.split(".")
    branch = ""
    for marker in ("branches", "branch", "stages"):
        if marker in parts:
            i = parts.index(marker)
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                branch = f"branch {parts[i + 1]}"
                break
    if "experts" in parts:
        i = parts.index("experts")
        if i + 1 < len(parts) and parts[i + 1].isdigit():
            expert = f"expert {parts[i + 1]}"
            return f"{branch} · {expert}" if branch else expert
    return branch


def collect_geometries(limit, scan):
    """Geometry dicts for the figure. Two modes, decided by the newest
    crystal-bearing run:

    * INTRA-RUN - if that run carries SEVERAL geometries (e.g. prismatic4's
      VEAR crystal bank), render *its own* geometries and stop. The bank's
      experts are the natural comparison set; other runs carry a single,
      differently-shaped classifier, so a cross-run mix would be
      apples-to-oranges (and the bank is THIS model's story anyway).
    * CROSS-RUN - otherwise, one-or-few panels per run, newest-first up to
      ``limit`` (the original behaviour, for single-geometry runs).

    Each dict: {name, hash, label, grid, var_explained, n_points, intra_run}."""
    import torch

    def _panels(sd, name, run_hash, classifier_type, keys, multi):
        out = []
        for key in keys:
            grid, ve = pca_density_grid(sd[key])
            out.append(
                {
                    "name": name,
                    "hash": run_hash,
                    "classifier_type": classifier_type,
                    "label": branch_label(key) if multi else "",
                    "grid": grid,
                    "var_explained": ve,
                    "n_points": int(sd[key].shape[0]),
                    "intra_run": multi,
                }
            )
        return out

    cross_run = []
    for _, run_hash, name, run_dir in runs_newest_first()[:scan]:
        if len(cross_run) >= limit:
            break
        ckpt = latest_checkpoint(run_dir)
        if not ckpt:
            continue
        try:
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        except Exception:
            continue
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        if isinstance(sd, dict):
            rename_legacy_state_dict(sd)
        keys = [
            k
            for k in sorted(sd)
            if k.endswith(CENTERS_SUFFIX)
            and hasattr(sd[k], "dim")
            and sd[k].dim() == 2
            and sd[k].shape[0] >= 3
            and is_crystal_centers(sd, k)
        ]
        if not keys:
            continue
        multi = len(keys) > 1
        classifier_type = classifier_type_of(run_dir)
        run_geos = _panels(sd, name, run_hash, classifier_type, keys, multi)
        # Newest crystal run is a multi-geometry bank: render its own, done.
        # Truncated to `limit` like the cross-run path - a bank wider than the
        # budget used to return every expert and overflow the float page.
        if multi and not cross_run:
            return run_geos[:limit]
        for g in run_geos:
            if len(cross_run) >= limit:
                break
            cross_run.append(g)
    return cross_run


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

    grid = np.asarray(geo["grid"], dtype=float)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3, 3))
    # Mirror the dashboard's renderHeatmap2D exactly: normalize by
    # log1p(count) / log1p(peak) and map straight through the shared ramp with
    # nearest-neighbour upscaling. The old LogNorm(vmin=1) floored every
    # single-point cell to black - and in a sparse PCA grid most occupied cells
    # hold exactly one center, so the panel read as near-empty. log1p lifts those
    # to visible color, matching the live card's density.
    peak = max(float(grid.max()), 1.0)
    v = np.log1p(grid) / np.log1p(peak)
    ax.imshow(
        v,
        origin="lower",
        cmap=_shared_cmap(),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    title = geo["name"] + (f" {geo['label']}" if geo["label"] else "")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    out = os.path.join(FIG_DIR, f"geometry_{index}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return os.path.relpath(out, RESEARCH_DIR)


# Float-page height budget, all in \linewidth units so the arithmetic is
# resolution- and paper-independent. A fixed 0.46 panel width overflowed as soon
# as the bank produced 5 geometries: three rows plus a seven-line caption ran past
# \textheight and the caption printed over the page number. The grid now shrinks
# to fit instead, which leaves the common 1-2 row case untouched.
_PANEL_ASPECT = 1.08  # rendered panel height / width (3x3in axes + title strip)
_TEXT_HEIGHT = 1.55  # article \textheight / \linewidth (~1.59), with margin
_CAPTION_ALLOWANCE = 0.30  # caption + \captionsetup skip, ~8 lines
_MAX_PANEL_WIDTH = 0.46  # two-up width; never exceeded, only reduced
_ROW_GAP_PT = 6


def _panel_width(n_rows: int) -> float:
    """Panel width (in \\linewidth) that keeps ``n_rows`` plus the caption inside
    the text block. Returns the two-up default whenever the grid already fits."""
    if n_rows < 1:
        return _MAX_PANEL_WIDTH
    budget = _TEXT_HEIGHT - _CAPTION_ALLOWANCE
    return min(_MAX_PANEL_WIDTH, budget / (n_rows * _PANEL_ASPECT))


def figure_tex(paths, geometries):
    """The ``\\paperGeometryFigure`` macro: a 2-column grid of the panels.

    ``[tbp]`` lets it take a dedicated float page if the block is too tall to sit
    with text, which keeps it near its section instead of drifting to the end.
    Panel width scales down past two rows so the caption cannot overrun the
    footer - see ``_panel_width``.
    """
    width = _panel_width((len(paths) + 1) // 2)
    rows = []
    for i in range(0, len(paths), 2):
        cells = " \\hfill\n  ".join(
            f"\\includegraphics[width={width:.3f}\\linewidth]{{{p}}}"
            for p in paths[i : i + 2]
        )
        rows.append(cells)
    body = f" \\\\[{_ROW_GAP_PT}pt]\n  ".join(rows)
    grid_desc = (
        f"the top-2 PCA projection of a head's vocabulary centers, binned to a "
        f"{GRID_SIZE}$\\times${GRID_SIZE} density grid (log color) - the same "
        "snapshot the dashboard renders live."
    )
    if geometries and all(g.get("intra_run") for g in geometries):
        # Single multi-geometry model (a crystal bank): the panels are its OWN
        # geometries.
        run = geometries[0]["name"]
        labels = ", ".join(g["label"] for g in geometries if g["label"])
        # Never name a classifier the run did not use. This clause hardcoded
        # "prismatic4" and went on asserting it through every later one.
        classifier_type = geometries[0].get("classifier_type")
        bank = (
            f"{classifier_type}'s VEAR crystal bank"
            if classifier_type
            else "a VEAR crystal bank"
        )
        caption = (
            f"Center PCA density for the {len(geometries)} crystal heads of "
            f"{run} - a single multi-head model ({bank}), "
            f"so these are that run's own heads ({labels}), not a cross-run mix. "
            f"Each panel is {grid_desc} The bank's experts settle into structurally "
            "distinct geometries - the between-expert variance the router selects over."
        )
    else:
        names = ", ".join(
            g["name"] + (f" ({g['label']})" if g["label"] else "") for g in geometries
        )
        caption = (
            "Center PCA density for the "
            f"{len(geometries)} most recent crystal-head runs ({names}). Each panel "
            f"is {grid_desc} Different runs settle into structurally different "
            "geometries, not noise."
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
            fh.write(
                "% Generated by praxis/pillars/geometries.py - no crystal geometries found.\n"
                "\\newcommand{\\paperGeometryFigure}{}\n"
            )
        return {"count": 0, "panels": []}

    paths = [render_png(g, i + 1) for i, g in enumerate(geometries)]
    with open(OUT_TEX, "w") as fh:
        fh.write("% Generated by praxis/pillars/geometries.py - do not edit by hand.\n")
        fh.write(figure_tex(paths, geometries))

    return {
        "count": len(geometries),
        "panels": [
            {
                "name": g["name"],
                "hash": g["hash"],
                "label": g["label"],
                "var_explained": [round(v, 4) for v in g["var_explained"]],
                "n_points": g["n_points"],
            }
            for g in geometries
        ],
    }
