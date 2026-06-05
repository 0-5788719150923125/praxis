"""Praxis modeling its own evolution: per-subsystem churn over git history.

A living-paper figure built from ``git log --numstat`` - how much each subsystem
changed over the project's history - with strength decayed by distance from HEAD
(the recency kernel the model itself uses, turned on the repo). Honest: real
churn over real commits, the decay shown as a fade of the distant past, not a
distortion of the history.

Output: research/figures/evolution.png + research/evolution.tex
(\\paperEvolutionFigure). Best-effort: no git / no history -> empty macro so the
paper still builds. See next/the_dial.md.
"""

import math
import os
import subprocess
from collections import defaultdict

from praxis.pillars.geometries import FIG_DIR, REPO_ROOT, RESEARCH_DIR

OUT_TEX = os.path.join(RESEARCH_DIR, "evolution.tex")
MAX_COMMITS = 4000  # cap for speed on long histories
N_BINS = 16  # time columns - chunky enough to read as isometric blocks

OTHER = "other"

# Subsystem label -> the path prefixes that belong to it. Each architectural
# feature lists BOTH its current directory and the historical praxis/modules/*.py
# file it grew out of, so a heavy hitter like 'dense' (which has existed since the
# flat-module days) is credited to itself instead of vanishing into 'core'. Dict
# order is the display order (depth + legend); matching is longest-prefix-first
# (see _subsystem), so a more specific prefix always wins regardless of order.
SUBSYSTEM_PREFIXES = {
    "attention": ["praxis/attention", "praxis/modules/attention"],
    "memory": [
        "praxis/memory",
        "praxis/modules/memory",
        "praxis/modules/attention_memory",
    ],
    "encoders": ["praxis/encoders", "praxis/modules/encoder"],
    "decoders": ["praxis/decoders", "praxis/modules/decoder"],
    "blocks": ["praxis/blocks", "praxis/modules/block"],
    "dense": ["praxis/dense", "praxis/modules/dense", "praxis/modules/kan"],
    "heads": ["praxis/heads", "praxis/modules/head"],
    "routers": [
        "praxis/routers",
        "praxis/modules/router",
        "praxis/modules/moe",
        "praxis/modules/switch_moe",
        "praxis/modules/experts",
        "praxis/modules/peer",
        "praxis/modules/smear",
    ],
    "controllers": ["praxis/controllers", "praxis/modules/controller"],
    "encoding": ["praxis/encoding", "praxis/modules/encoding"],
    "recurrent": ["praxis/recurrent", "praxis/modules/recurrent"],
    "embeddings": ["praxis/embeddings", "praxis/modules/embeddings"],
    "residuals": ["praxis/residuals", "praxis/modules/residual"],
    "normalization": ["praxis/normalization"],
    "activations": ["praxis/activations"],
    "compression": ["praxis/compression", "praxis/modules/compression"],
    "losses": ["praxis/losses"],
    "optimizers": ["praxis/optimizers"],
    "policies": ["praxis/policies"],
    "web": ["praxis/web"],
    "paper": ["praxis/pillars", "research"],
}

# (prefix, label) flattened and sorted longest-prefix-first, plus the catch-alls.
# Longest-first makes "praxis/modules/attention_memory" -> memory beat
# "praxis/modules/attention" -> attention, and any "praxis/<dir>" beat
# "praxis" -> core. Everything non-praxis falls through to OTHER.
SUBSYSTEMS = [(p, lbl) for lbl, ps in SUBSYSTEM_PREFIXES.items() for p in ps]
SUBSYSTEMS.append(("praxis", "core"))
SUBSYSTEMS.sort(key=lambda pl: len(pl[0]), reverse=True)

# Display order for depth/legend: architectural features (as declared above),
# then the catch-alls last.
LABEL_ORDER = list(SUBSYSTEM_PREFIXES) + ["core", OTHER]

# Stable color per subsystem (dark-theme palette, distinct hues).
COLORS = {
    "attention": "#e08a3c",
    "memory": "#46c2c8",
    "encoders": "#e8c84a",
    "decoders": "#b8922e",
    "blocks": "#5fd08a",
    "dense": "#e0566f",
    "heads": "#d06fd0",
    "routers": "#5a8cf0",
    "controllers": "#9a6ff0",
    "encoding": "#66c2e8",
    "recurrent": "#8ad06f",
    "embeddings": "#f0a868",
    "residuals": "#a99ad8",
    "normalization": "#5ec9a8",
    "activations": "#b6e04a",
    "compression": "#cf8f5a",
    "losses": "#3fae9e",
    "optimizers": "#6a78c8",
    "policies": "#d64f86",
    "web": "#7d7ae6",
    "paper": "#c25fb0",
    "core": "#8a93a6",
    OTHER: "#586072",
}


def _subsystem(path: str) -> str:
    """Longest matching prefix wins (SUBSYSTEMS is pre-sorted longest-first)."""
    for prefix, label in SUBSYSTEMS:
        if path.startswith(prefix):
            return label
    return OTHER


def _commits():
    """[(unix_ts, {subsystem: churn})] oldest-first, or [] if git is unusable."""
    try:
        out = subprocess.run(
            [
                "git",
                "log",
                f"-n{MAX_COMMITS}",
                "--no-merges",
                "--numstat",
                "--date=unix",
                "--pretty=format:__C__ %at",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        ).stdout
    except Exception:
        return []
    commits, ts, churn = [], None, defaultdict(float)
    for line in out.splitlines():
        if line.startswith("__C__ "):
            if ts is not None:
                commits.append((ts, churn))
            ts, churn = int(line[6:].strip() or 0), defaultdict(float)
        elif "\t" in line:
            ins, dele, path = (line.split("\t", 2) + ["", "", ""])[:3]
            # binary files report "-"; skip them (no line churn).
            if ins == "-" or not path:
                continue
            churn[_subsystem(path)] += float(ins or 0) + float(dele or 0)
    if ts is not None:
        commits.append((ts, churn))
    commits.reverse()  # oldest-first
    return commits


def evolution_data() -> dict:
    """Shared, JSON-serializable evolution signal: per-subsystem line churn over
    ``N_BINS`` time bins (oldest -> HEAD), the colors, and the recency-weighted
    focus ranking. The single source for BOTH output formats - the LaTeX figure
    (matplotlib, below) and the web card (/api/evolution -> canvas). Renderers
    own only presentation (recency fade, fonts); the numbers live here.
    Returns ``None`` when git history is unusable."""
    commits = _commits()
    if len(commits) < 3:
        return None

    t0, t1 = commits[0][0], commits[-1][0]
    span = max(t1 - t0, 1)
    labels = list(dict.fromkeys(LABEL_ORDER))
    series = {lbl: [0.0] * N_BINS for lbl in labels}
    for ts, churn in commits:
        b = min(int((ts - t0) / span * (N_BINS - 1)), N_BINS - 1)
        for lbl, v in churn.items():
            series.setdefault(lbl, [0.0] * N_BINS)[b] += v

    # Drop the OTHER junk-drawer entirely (non-Praxis paths: lockfiles,
    # staging/archive, root scripts). It isn't a subsystem, so rather than render
    # it as a block field it becomes the empty background - the black vacuum the
    # real subsystems sit in. This also frees the height normalization to span the
    # actual subsystems instead of being crushed by OTHER's large cells.
    present = [l for l in labels if l != OTHER and sum(series[l]) > 0]
    # Recency-weighted churn (linear decay: newest bin weight 1, oldest ~0) gives
    # the development "center of gravity".
    w = [i / (N_BINS - 1) for i in range(N_BINS)]
    focus = sorted(
        present, key=lambda l: sum(c * wi for c, wi in zip(series[l], w)), reverse=True
    )
    return {
        "bins": N_BINS,
        "subsystems": present,
        "series": {l: series[l] for l in present},
        "colors": {l: COLORS.get(l, "#586072") for l in present},
        "n_commits": len(commits),
        "focus": focus[:3],
        "x_labels": ["first commit", "now"],
    }


def export_evolution() -> dict:
    data = evolution_data()
    if data is None:
        with open(OUT_TEX, "w") as fh:
            fh.write(
                "% Generated by praxis/pillars/evolution.py - no git history.\n"
                "\\newcommand{\\paperEvolutionFigure}{}\n"
            )
        return {"rendered": False, "commits": 0}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    present = data["subsystems"]
    boxes = _iso_boxes(data)  # painter-ordered (back->front) (polygon, rgb) faces

    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    ax.add_collection(
        PolyCollection(
            [p for p, _ in boxes],
            facecolors=[c for _, c in boxes],
            edgecolors="#0f1117",
            linewidths=0.4,
        )
    )
    ax.autoscale()
    ax.margins(0.03)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Praxis evolving: recent peaks over the historical prior",
        color="#e6e9f0",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    ax.text(
        0.5,
        -0.04,
        "first commit → now    depth: subsystem    height: recency-weighted churn",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color="#9aa3b2",
        fontsize=7.5,
    )
    leg = ax.legend(
        handles=[Patch(facecolor=COLORS.get(s, "#586072"), label=s) for s in present],
        loc="upper left",
        fontsize=6.5,
        ncol=2,
        framealpha=0.0,
        labelcolor="#cfd3dc",
        handlelength=1.0,
        handleheight=1.0,
        borderpad=0.2,
    )
    for txt in leg.get_texts():
        txt.set_color("#cfd3dc")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "evolution.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    rel = os.path.relpath(out, RESEARCH_DIR)

    focus = data["focus"]
    caption = (
        f"Praxis's git history as an isometric terrain over the last "
        f"{data['n_commits']} commits. Each peak is one subsystem's line churn in one "
        "time window, weighted by recency (the kernel the model applies to a "
        "sequence, turned on the repository): recent windows tower into colored "
        "peaks while history settles toward the always-present prior - the low "
        "valley we roll into. Color fades from each subsystem's hue toward a neutral "
        "prior into the past, banded in phased strata over a non-linear timescale. "
        "It is the bias/variance geometry read along the frequency axis (see the "
        "project notes): loud recent corrections over the quiet, ever-present base. "
        f"By recency-weighted churn the current center of gravity is "
        f"{', '.join(focus[:3])}. The framework charts the hand that builds it."
    )
    with open(OUT_TEX, "w") as fh:
        fh.write("% Generated by praxis/pillars/evolution.py - do not edit by hand.\n")
        fh.write(
            "\\newcommand{\\paperEvolutionFigure}{%\n"
            "\\begin{figure}[tbp]\n  \\centering\n  "
            f"\\includegraphics[width=0.9\\linewidth]{{{rel}}}\n"
            f"  \\caption{{{caption}}}\n"
            "  \\label{fig:evolution}\n"
            "\\end{figure}\n}\n"
        )
    return {"rendered": True, "commits": data["n_commits"], "focus": focus, "path": rel}


# Isometric projection of the time x subsystem x churn terrain. Shared formula,
# mirrored in praxis/web/src/js/charts.js (createEvolutionChart):
#   sx = (x - y) * ISO_TX
#   sy = z * ISO_HMAX * ISO_TZ - (x + y) * ISO_TY      (y-up; the canvas flips)
#
# Deepening (the observer-frequency reading, next/observer_frequency.md): height
# is recency-weighted churn, so recent windows tower into colored peaks while
# history settles toward the always-present prior - the low valley we roll into.
# Peaks taper, and color fades from each subsystem's hue (recent, structured)
# toward a neutral prior (past), banded in phased strata over a non-linear
# timescale. Loud recent corrections over the quiet, ever-present base.
ISO_TX, ISO_TY, ISO_TZ, ISO_HMAX, ISO_GAP = 1.0, 0.5, 1.0, 4.4, 0.14
RECENCY_DECAY = 2.4  # exp falloff into the past; recent has far more impact
ISO_FLOOR = 0.05  # the always-present prior valley
ISO_TAPER = 0.6  # top-footprint shrink -> peaks
PHASE_AMP, PHASE_CYCLES = 0.24, 3.0  # color strata over (1-u)^1.3
PRIOR_RGB = (0.34, 0.38, 0.45)  # neutral the past fades into


def _hex_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _shade(rgb, f: float):
    return tuple(min(1.0, c * f) for c in rgb)


def _iso_boxes(data: dict):
    """Painter-ordered (back -> front) ``(polygon, rgb)`` faces for the iso
    terrain, in unit iso coordinates. Height is recency-weighted churn over a
    floor (recent towers, history settles to the prior valley); peaks taper;
    color fades from the subsystem hue toward a neutral prior into the past,
    banded in phased strata over a non-linear timescale."""
    subs, series, colors = data["subsystems"], data["series"], data["colors"]
    T = data["bins"]
    cmax = max((max(series[s]) for s in subs if series[s]), default=0.0) or 1.0

    def proj(x, y, z):
        return ((x - y) * ISO_TX, z * ISO_HMAX * ISO_TZ - (x + y) * ISO_TY)

    def mix(a, b, t):
        return tuple(a[k] + t * (b[k] - a[k]) for k in range(3))

    blocks = []  # (depth_key, [(poly, rgb), ...])
    for j, s in enumerate(subs):
        base = _hex_rgb(colors.get(s, "#586072"))
        for i in range(T):
            c = series[s][i] / cmax
            if c <= 0:
                continue  # subsystem untouched this window
            u = i / max(T - 1, 1)  # 0 oldest .. 1 now
            w = math.exp(-RECENCY_DECAY * (1.0 - u))  # recency weight
            h = ISO_FLOOR + (1.0 - ISO_FLOOR) * c * w  # recency-weighted height
            # Taper the top footprint as the block rises -> peaks.
            tp = ISO_TAPER * (h - ISO_FLOOR) / (1.0 - ISO_FLOOR)
            g = ISO_GAP / 2.0
            x0, x1, y0, y1 = i + g, i + 1 - g, j + g, j + 1 - g
            ix, iy = tp * (x1 - x0) / 2.0, tp * (y1 - y0) / 2.0
            tx0, tx1, ty0, ty1 = x0 + ix, x1 - ix, y0 + iy, y1 - iy
            # Color: subsystem hue fading to the prior into the past, * phased
            # strata over a non-linear timescale.
            col = mix(PRIOR_RGB, base, w)
            phase = 1.0 + PHASE_AMP * math.cos(
                2 * math.pi * PHASE_CYCLES * (1.0 - u) ** 1.3
            )
            col = tuple(min(1.0, col[k] * phase) for k in range(3))
            top = [
                proj(tx0, ty0, h),
                proj(tx1, ty0, h),
                proj(tx1, ty1, h),
                proj(tx0, ty1, h),
            ]
            east = [
                proj(x1, y0, 0),
                proj(x1, y1, 0),
                proj(tx1, ty1, h),
                proj(tx1, ty0, h),
            ]
            south = [
                proj(x0, y1, 0),
                proj(x1, y1, 0),
                proj(tx1, ty1, h),
                proj(tx0, ty1, h),
            ]
            blocks.append(
                (
                    i + j,
                    [
                        (south, _shade(col, 0.6)),
                        (east, _shade(col, 0.8)),
                        (top, _shade(col, 1.0)),
                    ],
                )
            )
    blocks.sort(key=lambda b: b[0])  # back to front
    return [face for _, faces in blocks for face in faces]
