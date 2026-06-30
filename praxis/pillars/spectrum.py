"""The harmonic head's static amplitude spectrum, rendered for the paper.

The dashboard's Dynamics tab shows a live ``Harmonic Spectrum`` snapshot: the
``|amp[f_t, f_d]|`` magnitude grid of a harmonic field, routed through the
generic ``heatmap_2d`` renderer (see :meth:`HarmonicHead.dashboard_snapshots`).
This module reproduces that view offline for the newest harmonic-bearing run and
tiles its fields into a figure for the paper's ``Bias is the static spectrum``
paragraph (Section~\\ref{sec:manifold}) - the heatmap the prose promises, the
one that is "indistinguishable from noise" early and "develops a clean
low-frequency band" once the smoothness prior has shaped it.

The spectrum is read straight from the checkpoint: any parameter whose key ends
in ``field.amplitudes`` is a harmonic field's learned amplitude grid
(``[F_t, F_d]``). The pink ``1/f^alpha`` prior lives in the frozen basis, not in
these amplitudes, so ``|amplitudes|`` is exactly what the live card draws and
exactly what :meth:`HarmonicField.concentration` measures - the Hoyer sparsity
reported per panel is therefore the same number the dashboard logs, computed
from the same tensor. A prismatic (multi-branch) run yields one panel per arm.

Output (all generated, none committed):
- ``research/figures/spectrum_N.png`` - one magnitude heatmap per field.
- ``research/spectrum.tex`` - defines ``\\paperSpectrumFigure``, the figure
  body. ``research/body.tex`` drops the macro after the ``Bias is the static
  spectrum`` paragraph, and ``main.tex`` falls back to an empty
  ``\\providecommand`` on a clean checkout, so the prose renders without it.

Entry point: :func:`export_spectrum`, driven by :mod:`praxis.pillars.build`.
"""

import math
import os

from praxis.pillars.geometries import (
    FIG_DIR,
    RESEARCH_DIR,
    _shared_cmap,
    branch_label,
    latest_checkpoint,
    runs_newest_first,
)

OUT_TEX = os.path.join(RESEARCH_DIR, "spectrum.tex")
# A harmonic field's learned amplitude grid is always `...field.amplitudes`
# (HarmonicField.amplitudes, an [F_t, F_d] nn.Parameter). Prismatic/parallel
# heads expose one per branch (`...branches.<i>.heads.0.field.amplitudes`).
AMPLITUDES_SUFFIX = "field.amplitudes"


def hoyer(a):
    """Hoyer sparsity of an amplitude grid in [0, 1] - a 1:1 reimplementation
    of :meth:`HarmonicField.concentration` (which reads the same ``amplitudes``
    tensor), so the figure's number matches the live diagnostic exactly.
    1 = all energy in one cell, 0 = perfectly uniform."""
    import torch

    a = a.detach().abs().to(torch.float32).flatten()
    n = a.numel()
    if n < 2:
        return 0.0
    sqrt_n = math.sqrt(n)
    l1 = float(a.sum())
    l2 = math.sqrt(float((a * a).sum()) + 1e-12)
    return (sqrt_n - l1 / l2) / (sqrt_n - 1)


def collect_spectra(limit, scan):
    """Amplitude-grid dicts for the figure, all from the newest harmonic run.

    Scans runs newest-first up to ``scan``; the first run carrying any
    ``field.amplitudes`` tensor wins, and we render up to ``limit`` of *its own*
    fields (one per branch for a prismatic head). A single run keeps the panels
    comparable - different runs use different ``F_t``/``F_d`` and amplitude
    scales, so a cross-run mix would not share an axis.

    Each dict: {name, hash, label, grid (numpy [F_t, F_d]), hoyer, multi}."""
    import torch

    for _, run_hash, name, run_dir in runs_newest_first()[:scan]:
        ckpt = latest_checkpoint(run_dir)
        if not ckpt:
            continue
        try:
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        except Exception:
            continue
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        keys = [
            k
            for k in sorted(sd)
            if k.endswith(AMPLITUDES_SUFFIX)
            and hasattr(sd[k], "dim")
            and sd[k].dim() == 2
            and min(sd[k].shape) >= 2
        ]
        if not keys:
            continue
        keys = keys[:limit]
        multi = len(keys) > 1
        return [
            {
                "name": name,
                "hash": run_hash,
                "label": branch_label(k) if multi else "",
                "grid": sd[k].detach().abs().to(torch.float32).cpu().numpy(),
                "hoyer": hoyer(sd[k]),
                "multi": multi,
            }
            for k in keys
        ]
    return []


def render_png(spec, index):
    """Write one ``|amp[f_t, f_d]|`` heatmap; return its repo-relative path.

    Mirrors the dashboard's ``heatmap_2d`` renderer for ``harmonic_spectrum``:
    linear normalization by the grid peak, straight through the shared
    ``praxis_heat`` ramp, native-resolution (no smoothing). Rows are temporal
    frequency ``f_t`` (origin at bottom), columns feature frequency ``f_d``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    grid = np.asarray(spec["grid"], dtype=float)
    peak = max(float(grid.max()), 1e-12)
    v = grid / peak

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.imshow(
        v,
        origin="lower",
        cmap=_shared_cmap(),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )
    title = (spec["label"] or spec["name"]) + f"  (H={spec['hoyer']:.2f})"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("feature freq. $f_d$", fontsize=8)
    ax.set_ylabel("temporal freq. $f_t$", fontsize=8)
    ax.tick_params(labelsize=7)
    out = os.path.join(FIG_DIR, f"spectrum_{index}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return os.path.relpath(out, RESEARCH_DIR)


def figure_tex(paths, spectra):
    """The ``\\paperSpectrumFigure`` macro: a 1- or 2-column grid of panels."""
    rows = []
    for i in range(0, len(paths), 2):
        cells = " \\hfill\n  ".join(
            f"\\includegraphics[width=0.46\\linewidth]{{{p}}}" for p in paths[i : i + 2]
        )
        rows.append(cells)
    body = " \\\\[6pt]\n  ".join(rows)

    run = spectra[0]["name"]
    hoyers = ", ".join(f"{s['hoyer']:.2f}" for s in spectra)
    measure = (
        f"Hoyer concentration $H={spectra[0]['hoyer']:.2f}$"
        if len(spectra) == 1
        else f"Hoyer concentrations $H = {hoyers}$ (left to right)"
    )
    arms = (
        f" across the {len(spectra)} arms of {run}'s prismatic head"
        if spectra[0]["multi"]
        else f" for {run}"
    )
    caption = (
        "The static amplitude spectrum $|a[f_t, f_d]|$" + arms + " - the learned "
        "grid that, inverse-transformed through the frozen Weyl phases, becomes "
        "the bias field. This is the heatmap the text describes: energy that "
        "settles into a low-frequency band (small $f_t$, $f_d$) is the corpus "
        "rhythm crystallizing, while a grid that stays uniform is a field still "
        "indistinguishable from noise. " + measure + " quantifies it on the same "
        "$[0,1]$ scale the prose names ($1$ = a single cell, $0$ = uniform), "
        "computed from this very tensor by \\texttt{concentration()} - the same "
        "value the dashboard logs live. The pink $1/f^{\\alpha}$ prior lives in "
        "the frozen basis, not in these amplitudes, so the grid shown is exactly "
        "what the head exposes."
    )
    return (
        "\\newcommand{\\paperSpectrumFigure}{%\n"
        "\\begin{figure}[tbp]\n  \\centering\n  "
        f"{body}\n"
        f"  \\caption{{{caption}}}\n"
        "  \\label{fig:spectrum}\n"
        "\\end{figure}\n}\n"
        # The in-prose reference is gated with the figure: defined only when the
        # figure renders, so \\ref{fig:spectrum} can never dangle in a build
        # where the (gated) figure is absent.
        "\\newcommand{\\paperSpectrumRef}{ (Figure~\\ref{fig:spectrum})}\n"
    )


def export_spectrum(limit: int = 3, scan: int = 40) -> dict:
    """Render the newest harmonic run's amplitude spectra (scanning ``scan``
    runs, up to ``limit`` fields) into figures/ + spectrum.tex. Returns a
    summary dict. Emits an empty macro when no harmonic field is found, so the
    paper still builds and the prose renders without the figure."""
    spectra = collect_spectra(limit, scan)
    if not spectra:
        with open(OUT_TEX, "w") as fh:
            fh.write(
                "% Generated by praxis/pillars/spectrum.py - no harmonic spectrum found.\n"
                "\\newcommand{\\paperSpectrumFigure}{}\n"
                "\\newcommand{\\paperSpectrumRef}{}\n"
            )
        return {"count": 0, "panels": []}

    paths = [render_png(s, i + 1) for i, s in enumerate(spectra)]
    with open(OUT_TEX, "w") as fh:
        fh.write("% Generated by praxis/pillars/spectrum.py - do not edit by hand.\n")
        fh.write(figure_tex(paths, spectra))

    return {
        "count": len(spectra),
        "panels": [
            {
                "name": s["name"],
                "hash": s["hash"],
                "label": s["label"],
                "hoyer": round(s["hoyer"], 4),
                "shape": list(s["grid"].shape),
            }
            for s in spectra
        ],
    }
