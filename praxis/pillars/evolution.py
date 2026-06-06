"""Praxis modeling its own evolution: per-subsystem churn over git history.

A living-paper figure built from ``git log --numstat`` - how much each subsystem
changed over the project's history - with strength decayed by distance from HEAD
(the recency kernel the model itself uses, turned on the repo). Honest: real
churn over real commits, the decay shown as a fade of the distant past, not a
distortion of the history.

Output: research/figures/evolution.png + research/evolution_vars.tex (data
macros for the static evolution.tex). Best-effort: no git / no history -> no
vars file, and the paper's \\providecommand fallback omits the figure. See
next/the_dial.md.
"""

import math
import os
import subprocess
from collections import defaultdict

from praxis.pillars.geometries import FIG_DIR, REPO_ROOT, RESEARCH_DIR

OUT_VARS = os.path.join(RESEARCH_DIR, "evolution_vars.tex")
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
    """[(unix_ts, {subsystem: [churn, net]})] oldest-first, or [] if git is
    unusable. ``churn`` = insertions + deletions (activity); ``net`` =
    insertions - deletions (accumulates into total lines / codebase size)."""
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
    commits, ts, cell = [], None, defaultdict(lambda: [0.0, 0.0])
    for line in out.splitlines():
        if line.startswith("__C__ "):
            if ts is not None:
                commits.append((ts, cell))
            ts, cell = int(line[6:].strip() or 0), defaultdict(lambda: [0.0, 0.0])
        elif "\t" in line:
            ins, dele, path = (line.split("\t", 2) + ["", "", ""])[:3]
            # binary files report "-"; skip them (no line churn).
            if ins == "-" or dele == "-" or not path:
                continue
            i, d = float(ins or 0), float(dele or 0)
            sub = _subsystem(path)
            cell[sub][0] += i + d  # churn (activity)
            cell[sub][1] += i - d  # net (-> total lines)
    if ts is not None:
        commits.append((ts, cell))
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

    def _bin(ts):
        return min(int((ts - t0) / span * (N_BINS - 1)), N_BINS - 1)

    series = {lbl: [0.0] * N_BINS for lbl in labels}  # churn per bin (activity)
    totals = {lbl: [None] * N_BINS for lbl in labels}  # cumulative net (total lines)
    running = defaultdict(float)
    for ts, cell in commits:
        b = _bin(ts)
        for lbl, (churn, net) in cell.items():
            series.setdefault(lbl, [0.0] * N_BINS)[b] += churn
            running[lbl] += net
        for lbl, r in running.items():  # snapshot codebase size at this bin
            totals.setdefault(lbl, [None] * N_BINS)[b] = max(0.0, r)
    for arr in totals.values():  # forward-fill bins with no commit (size persists)
        last = 0.0
        for b in range(N_BINS):
            if arr[b] is None:
                arr[b] = last
            else:
                last = arr[b]

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
        "totals": {l: totals[l] for l in present},
        "colors": {l: COLORS.get(l, "#586072") for l in present},
        "n_commits": len(commits),
        "focus": focus[:3],
        "x_labels": ["first commit", "now"],
    }


def export_evolution() -> dict:
    data = evolution_data()
    if data is None:
        if os.path.exists(OUT_VARS):
            os.remove(OUT_VARS)  # stale data; let the paper's fallback omit the figure
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
        "Praxis evolving: towers of accumulated code, capped by recent churn",
        color="#e6e9f0",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    ax.text(
        0.5,
        -0.04,
        "first commit → now    depth: subsystem    stack: total lines    cap: recent churn",
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
    with open(OUT_VARS, "w") as fh:
        fh.write("% Generated by praxis/pillars/evolution.py - do not edit by hand.\n")
        fh.write(f"\\newcommand{{\\evoCommitCount}}{{{data['n_commits']}}}\n")
        fh.write(f"\\newcommand{{\\evoFocus}}{{{', '.join(focus[:3])}}}\n")
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
ISO_FLOOR = 0.0  # the base stack is the foundation now
ISO_TAPER = 0.6  # churn-cap top-footprint shrink -> peaks
PHASE_AMP, PHASE_CYCLES = 0.24, 3.0  # color strata over (1-u)^1.3
PRIOR_RGB = (0.34, 0.38, 0.45)  # neutral the past fades into
# Third dimension: total lines (codebase size) as a stack of base blocks under
# each churn cap. Recency-weighted, so recent windows stack tall (the big current
# codebase) and history thins toward the floor - more blocks near us, decaying
# with that value and recency over time.
MAX_STACK = 6  # max base blocks in a total-lines tower
BASE_UNIT = 0.12  # height of one base block
STACK_GAP = 0.28  # gap fraction between stacked base blocks
CAP_HMAX = 0.32  # max churn-cap height on top of the stack


def _hex_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _shade(rgb, f: float):
    return tuple(min(1.0, c * f) for c in rgb)


def _iso_boxes(data: dict):
    """Painter-ordered (back -> front) ``(polygon, rgb)`` faces. Each cell is a
    stack of base blocks (total lines / codebase size, recency-weighted) topped by
    a tapered churn cap (the activity in that window). Recent windows stack tall
    and cap vivid; history thins toward the floor and fades to the prior."""
    subs, series, totals, colors = (
        data["subsystems"],
        data["series"],
        data["totals"],
        data["colors"],
    )
    T = data["bins"]
    cmax = max((max(series[s]) for s in subs if series[s]), default=0.0) or 1.0
    tmax = max((max(totals[s]) for s in subs if totals[s]), default=0.0) or 1.0

    def proj(x, y, z):
        return ((x - y) * ISO_TX, z * ISO_HMAX * ISO_TZ - (x + y) * ISO_TY)

    def mix(a, b, t):
        return tuple(a[k] + t * (b[k] - a[k]) for k in range(3))

    def box(x0, x1, y0, y1, tx0, tx1, ty0, ty1, z0, z1, col):
        # Three visible faces; sides slant from the base footprint (z0) to the
        # (possibly inset) top footprint (z1).
        top = [
            proj(tx0, ty0, z1),
            proj(tx1, ty0, z1),
            proj(tx1, ty1, z1),
            proj(tx0, ty1, z1),
        ]
        east = [
            proj(x1, y0, z0),
            proj(x1, y1, z0),
            proj(tx1, ty1, z1),
            proj(tx1, ty0, z1),
        ]
        south = [
            proj(x0, y1, z0),
            proj(x1, y1, z0),
            proj(tx1, ty1, z1),
            proj(tx0, ty1, z1),
        ]
        return [
            (south, _shade(col, 0.6)),
            (east, _shade(col, 0.8)),
            (top, _shade(col, 1.0)),
        ]

    blocks = []  # (depth_key, [(poly, rgb), ...])
    for j, s in enumerate(subs):
        base_rgb = _hex_rgb(colors.get(s, "#586072"))
        for i in range(T):
            u = i / max(T - 1, 1)  # 0 oldest .. 1 now
            w = math.exp(-RECENCY_DECAY * (1.0 - u))  # recency weight
            g = ISO_GAP / 2.0
            x0, x1, y0, y1 = i + g, i + 1 - g, j + g, j + 1 - g
            phase = 1.0 + PHASE_AMP * math.cos(
                2 * math.pi * PHASE_CYCLES * (1.0 - u) ** 1.3
            )
            faces = []
            # Base stack: total lines (codebase size), recency-weighted -> N
            # rectangular blocks (sqrt so smaller subsystems still show some).
            tot = totals[s][i] / tmax
            n_stack = int(round((tot**0.5) * w * MAX_STACK))
            base_col = tuple(
                min(1.0, c * phase) for c in mix(PRIOR_RGB, base_rgb, min(1.0, w * 0.7))
            )
            for k in range(n_stack):
                z0 = ISO_FLOOR + k * BASE_UNIT
                z1 = z0 + BASE_UNIT * (1.0 - STACK_GAP)
                faces += box(x0, x1, y0, y1, x0, x1, y0, y1, z0, z1, base_col)
            base_top = ISO_FLOOR + n_stack * BASE_UNIT
            # Churn cap: the activity peak on top, vivid + tapered.
            c = series[s][i] / cmax
            if c > 0:
                cap_h = c * w * CAP_HMAX
                tp = ISO_TAPER * (cap_h / CAP_HMAX)
                ix, iy = tp * (x1 - x0) / 2.0, tp * (y1 - y0) / 2.0
                cap_col = tuple(
                    min(1.0, c2 * phase) for c2 in mix(PRIOR_RGB, base_rgb, w)
                )
                faces += box(
                    x0,
                    x1,
                    y0,
                    y1,
                    x0 + ix,
                    x1 - ix,
                    y0 + iy,
                    y1 - iy,
                    base_top,
                    base_top + cap_h,
                    cap_col,
                )
            if faces:
                blocks.append((i + j, faces))
    blocks.sort(key=lambda b: b[0])  # back to front
    return [face for _, faces in blocks for face in faces]
