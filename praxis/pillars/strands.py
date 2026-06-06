"""Render the prismatic head's bias/variance strand fields into the paper.

A 1-to-1 still of the dashboard's "Bias/Variance Strands" card, for the arms
of the prismatic (ParallelHead) head side by side: the bias arm (a static field,
its feature-hairs pure bias = blue), the variance arm (an input-conditional
field, its hairs spanning the blue-to-red bias/variance spectrum), and - with
prismatic3 - a pure-variance arm (no static spectrum, hairs grow pure red).

The data is exactly what the card uses - each arm's
``HarmonicField.field_strands()`` (angle, per-feature bias/variance energy). The
variance arm's spectrum only exists once a forward has populated its
input-conditional coefficients, so this reads them off the *live* training model
(passed in by the PaperBuildCallback), a read-only snapshot like the dashboard's.

Output: research/figures/strands.png + research/strands.tex (\\paperStrandsFigure).
Best-effort: no prismatic field -> an empty macro so the paper still builds.
Geometry + colormap mirror praxis/web/src/js/dynamics.js:renderHarmonicStrands.
"""

import json
import math
import os

from praxis.pillars.geometries import FIG_DIR, REPO_ROOT, RESEARCH_DIR

OUT_TEX = os.path.join(RESEARCH_DIR, "strands.tex")
COLORMAPS_JSON = os.path.join(REPO_ROOT, "praxis", "web", "src", "colormaps.json")

# Mirror the JS renderer's constants so the printed hairs match the card.
TILT = 26.0 * math.pi / 180.0
HEIGHT = 2.4
STEPS = 28
TWIST = 1.25 * math.pi
ROT = 0.6  # fixed 3/4 view (the card slowly rotates; we freeze one frame)


def _cmap(name: str):
    from matplotlib.colors import LinearSegmentedColormap

    spec = json.load(open(COLORMAPS_JSON))[name]
    colors = [(pos, [c / 255.0 for c in rgb]) for pos, rgb in spec["stops"]]
    return LinearSegmentedColormap.from_list(name, colors)


def _branch_fields(model):
    """(branch, harmonic field) per prismatic branch, in order (bias arm first)."""
    head = getattr(model, "head", None)
    branches = getattr(head, "branches", None)
    if branches is None:
        return []
    out = []
    for b in branches:
        fld = None
        heads = getattr(b, "heads", None)
        if heads is not None and len(heads):
            fld = getattr(heads[0], "field", None)
        if fld is None or not hasattr(fld, "field_strands"):
            for m in b.modules():  # fallback: first submodule that can produce strands
                cand = m if hasattr(m, "field_strands") else getattr(m, "field", None)
                if cand is not None and hasattr(cand, "field_strands"):
                    fld = cand
                    break
        if fld is not None and hasattr(fld, "field_strands"):
            out.append((b, fld))
    return out


def _stack_label(branch) -> str:
    """The branch's head stack as ``A -> B``, from its compose_repr
    (``Sequential(HarmonicField, CrystalClassifier)`` -> ``HarmonicField ->
    CrystalClassifier``)."""
    try:
        rep = branch.compose_repr()
    except Exception:
        return ""
    inner = rep
    if "(" in rep and rep.endswith(")"):
        inner = rep[rep.index("(") + 1 : -1]
    return " -> ".join(p.strip() for p in inner.split(",") if p.strip())


def _segments_and_colors(data):
    """Per-feature hair polylines + RGBA, reproducing the JS cylinder morph.

    Returns (segments, rgba, order) where segments[i] is the i-th hair's
    projected (x, y) points and order sorts back-to-front by mean depth.
    """
    import numpy as np

    angle = np.asarray(data.get("angle", []), dtype=np.float64)
    bias = np.maximum(np.asarray(data.get("bias_energy", []), dtype=np.float64), 0.0)
    var = np.maximum(np.asarray(data.get("var_energy", []), dtype=np.float64), 0.0)
    n = min(len(angle), len(bias), len(var))
    if n < 2:
        return None
    angle, bias, var = angle[:n], bias[:n], var[:n]

    # Scale by peak energy on either axis, so a bias-free ("pure") arm still
    # spans the unit geometry instead of collapsing to a point.
    bmax = max(float(bias.max()), float(var.max()), 1e-6)
    r = np.sqrt(bias / bmax)  # [n]
    px = r * 2.0 - 1.0
    py = np.sqrt(var / bmax) * 2.0 - 1.0
    t = np.where(bias + var > 1e-9, var / (bias + var), 0.0)  # variance fraction [n]

    # Color by variance share RELATIVE to the heaviest feature (mirrors the
    # card): tref = max per-feature share, floored at 0.04 so a near-zero
    # variance field stays blue. rel = t/tref reddens the heaviest feature
    # fully even when the absolute field fraction is small - this is the scaling
    # the flat raw-t coloring was missing.
    tref = max(float(t.max()), 0.04)
    rel = np.minimum(1.0, t / tref)  # [n]

    s = np.arange(STEPS + 1) / STEPS  # [S]
    wgt = 1.0 - s
    th = angle[:, None] + s[None, :] * TWIST  # [n, S]
    gx = r[:, None] * np.cos(th) * wgt[None, :] + px[:, None] * s[None, :]
    gy = r[:, None] * np.sin(th) * wgt[None, :] + py[:, None] * s[None, :]
    gz = (s - 0.5) * HEIGHT  # [S]

    cosA, sinA, cosE, sinE = (
        math.cos(ROT),
        math.sin(ROT),
        math.cos(TILT),
        math.sin(TILT),
    )
    Xs = gx * cosA - gy * sinA
    Ys = gx * sinA + gy * cosA
    sx = Xs
    sy = gz[None, :] * cosE - Ys * sinE  # y-up (no screen-flip)
    depth = (Ys * cosE + gz[None, :] * sinE).mean(axis=1)  # [n]

    # Each hair ramps base(blue)->tip via cmap(rel*z); split into per-segment
    # lines so the gradient renders. Segment s->s+1 takes z=(s+1)/STEPS, like
    # the card. Per-hair depth alpha (far hairs dimmer). A bias-free ("pure")
    # arm reverses the ramp - variance pushes back from the other end of the
    # corkscrew, so red enters at the ring.
    pure = float(bias.max()) <= 1e-12
    cmap = _cmap("bias_variance")
    zmid = (np.arange(STEPS) + 1) / STEPS  # [STEPS]
    zcol = (1.0 - zmid) if pure else zmid
    alpha = 0.2 + 0.5 * np.clip((depth + 1.5) / 3.0, 0.0, 1.0)  # [n]
    order = np.argsort(depth)  # back-to-front
    segs, cols = [], []
    for i in order:
        rgba = cmap(rel[i] * zcol)  # [STEPS, 4]
        rgba[:, 3] = alpha[i]
        for k in range(STEPS):
            segs.append([(sx[i, k], sy[i, k]), (sx[i, k + 1], sy[i, k + 1])])
        cols.append(rgba)
    return segs, np.vstack(cols), float(data.get("separated", 0.0) or 0.0)


def _render_panel(ax, data, title_main, title_sub):
    from matplotlib.collections import LineCollection

    built = _segments_and_colors(data)
    # Light title on the dark facecolor (matplotlib's default black title was
    # invisible here). Head tag on top, the branch's head stack beneath.
    ax.set_title(title_main, color="#e6e9f0", fontsize=10, fontweight="bold", pad=12)
    if title_sub:
        ax.text(
            0.5,
            1.02,
            title_sub,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            color="#9aa3b2",
            fontsize=7.5,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#0f1117")
    ax.set_aspect("equal")
    if built is None:
        ax.text(
            0.5,
            0.5,
            "no field",
            ha="center",
            va="center",
            color="#888",
            transform=ax.transAxes,
            fontsize=8,
        )
        return
    segs, cols, sep = built
    lc = LineCollection(segs, colors=cols, linewidths=0.9)
    ax.add_collection(lc)
    ax.autoscale()
    ax.text(
        0.03,
        0.97,
        f"variance {sep * 100:.0f}% of field energy",
        transform=ax.transAxes,
        color="#cfd3dc",
        fontsize=7,
        va="top",
    )


def export_strands(model=None) -> dict:
    """Render the prismatic head's strand fields. Needs a live model with a
    prismatic (>=2 branch) head; otherwise writes an empty macro."""
    pairs = _branch_fields(model) if model is not None else []
    if len(pairs) < 2:
        with open(OUT_TEX, "w") as fh:
            fh.write(
                "% Generated by praxis/pillars/strands.py - no prismatic field found.\n"
                "\\newcommand{\\paperStrandsFigure}{}\n"
            )
        return {"rendered": False, "branches": len(pairs)}

    datasets = [(b, f.field_strands()) for b, f in pairs[:3]]
    # Don't downgrade a good figure to all-blue. The variance arm's spectrum
    # needs populated forward state (_last_input_coeffs); it is non-persistent,
    # so right after a resume - before any forward - every hair reads as pure
    # bias. If no arm shows variance yet but a rendered figure already exists,
    # keep it and let a later build (with forward state) refresh it.
    max_sep = max(float(d.get("separated", 0.0) or 0.0) for _, d in datasets)
    out = os.path.join(FIG_DIR, "strands.png")
    if max_sep < 1e-4 and os.path.exists(out):
        return {"rendered": False, "reason": "no forward state yet; kept prior figure"}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    n_arms = len(datasets)
    fig, axes = plt.subplots(1, n_arms, figsize=(3 * n_arms, 3.3), facecolor="#0f1117")
    # branch 0 is the bias arm, branch 1 the variance arm (the prismatic
    # profile); prismatic3 adds a third, variance-only arm.
    roles = ["bias arm", "variance arm", "pure-variance arm"]
    for i, (ax, (branch, d)) in enumerate(zip(axes, datasets)):
        tag = chr(ord("A") + i)
        role = roles[i] if i < len(roles) else ""
        main = f"Head {tag}" + (f"  -  {role}" if role else "")
        _render_panel(ax, d, main, _stack_label(branch))
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "strands.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    rel = os.path.relpath(out, RESEARCH_DIR)

    third = (
        "Head C (right) is the pure-variance arm: it carries no static "
        "spectrum at all, so its hairs grow pure red from nothing - variance "
        "arriving from the opposite end of the decomposition. "
        if n_arms >= 3
        else ""
    )
    caption = (
        f"The prismatic head's {'three' if n_arms >= 3 else 'two'} arms, "
        "rendered exactly as the dashboard's "
        "bias/variance strand card. Each hair is one feature, morphing from a "
        "ring (its static-field phase) out to its (bias, variance) position with "
        "a helical twist; color runs blue (pure bias) to red (pure variance). "
        "Head A (left) is the bias arm: it reads a static field, so every hair "
        f"stays blue - a constant structural vote. Head B {'(center)' if n_arms >= 3 else '(right)'} "
        "is the variance "
        "arm, whose input-conditional envelope lets hairs swing toward red - the "
        f"data-dependent vote. {third}"
        "Each panel's subtitle lists that arm's head stack. "
        "Together they are the orthogonal bias/variance decomposition made "
        "architectural - convex and radial from the center, chaotic in the arms."
    )
    with open(OUT_TEX, "w") as fh:
        fh.write("% Generated by praxis/pillars/strands.py - do not edit by hand.\n")
        fh.write(
            "\\newcommand{\\paperStrandsFigure}{%\n"
            "\\begin{figure}[tbp]\n  \\centering\n  "
            f"\\includegraphics[width=0.92\\linewidth]{{{rel}}}\n"
            f"  \\caption{{{caption}}}\n"
            "  \\label{fig:strands}\n"
            "\\end{figure}\n}\n"
        )
    return {"rendered": True, "branches": len(pairs), "path": rel}
