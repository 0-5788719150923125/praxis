"""Procedural 2D projection fields rendered onto business cards.

A projection is a seeded function that draws gear-like geometry into a
matplotlib axes whose data coordinates are millimeters of the card. The
card is a small viewport into a larger field, so renders look cropped and
zoomed rather than centered.
"""

import io

import numpy as np

# US standard card (3.5 x 2 in) on US Letter, matching Avery 28371:
# 2 cols x 5 rows = 10 per sheet, 0.75 in side and 0.5 in top/bottom
# margins, no gutters. EU/A4 support deferred.
CARD_W, CARD_H = 88.9, 50.8
BORDER_MM = 3.0
PAGE_W, PAGE_H = 215.9, 279.4
SHEET_X0, SHEET_Y0 = 19.05, 12.7
SHEET_COLS, SHEET_ROWS = 2, 5

PRINT_NOTE = "Print at 100% scale (no fit-to-page) on US Letter."
PAPER_NOTE = (
    "Paper: Avery micro-perforated business card sheets, 10-up 2 x 3.5 in on "
    "US Letter, 80 lb / 216 gsm matte - this layout lands exactly on the "
    "perforations."
)
PAPER_LINK_NOTE = (
    "Buy: inkjet - Avery 28371, amazon.com/dp/B00004Z5DG. Laser - Avery 5371, "
    "same 10-up layout."
)
PAPER_URL = "https://www.amazon.com/dp/B00004Z5DG"


def _hsl(h, s, l):
    import colorsys

    return colorsys.hls_to_rgb((h % 360) / 360.0, l, s)


def _palette(theme, hue):
    dark = theme == "dark"
    return {
        "paper": "#1a1a1a" if dark else "#f9f9f9",
        "border": "#000000" if dark else "#ffffff",
        "ink": "#e8e8e8" if dark else "#1f1f1f",
        "stroke": _hsl(hue, 0.6, 0.62 if dark else 0.30),
        "faint": _hsl(hue, 0.35, 0.30 if dark else 0.80),
        "fill": _hsl(hue, 0.45, 0.16 if dark else 0.90),
    }


PHI = (1 + 5 ** 0.5) / 2
GOLDEN = 2 * np.pi * (1 - 1 / PHI)  # golden angle, ~137.5 deg


def _fbm1(rng, n, octaves=5):
    """1D fractal value noise along a strand, roughly [-1, 1]."""
    out = np.zeros(n)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        k = 2 ** (o + 2)
        g = rng.standard_normal(k + 1)
        xi = np.linspace(0, k - 1e-9, n)
        i0 = np.floor(xi).astype(int)
        f = xi - i0
        f = f * f * (3 - 2 * f)
        out += amp * (g[i0] * (1 - f) + g[i0 + 1] * f)
        total += amp
        amp *= 0.55
    return out / total


def _normals(x, y):
    dx, dy = np.gradient(x), np.gradient(y)
    length = np.hypot(dx, dy)
    length[length == 0] = 1
    return -dy / length, dx / length


def _displace(rng, x, y, chaos, tooth_h, period, noise_h):
    """The modulation axis: square teeth (discrete cuts) at chaos=0,
    fractal wander at chaos=1, applied along the strand normal."""
    n = len(x)
    s = np.arange(n, dtype=float)
    teeth = np.tanh(6 * np.sin(2 * np.pi * s / period + float(rng.uniform(0, 6))))
    d = (1 - chaos) * tooth_h * teeth + chaos * noise_h * _fbm1(rng, n)
    nx, ny = _normals(x, y)
    return x + d * nx, y + d * ny


def _spiral_strand(rng, chaos, cx, cy, r0, turns, facets=13, n=900):
    """Golden-ratio log spiral. The continuous curve and its chord-sampled
    polygon (a Fibonacci number of facets per turn) are the two ends of one
    axis: blending them morphs linear cuts into smooth coil."""
    th = np.linspace(0, 2 * np.pi * turns, n)
    r = r0 * PHI ** (th / (2 * np.pi))
    xs, ys = cx + r * np.cos(th), cy + r * np.sin(th)

    step = 2 * np.pi / facets
    verts = np.arange(0, th[-1] + step, step)
    rv = r0 * PHI ** (verts / (2 * np.pi))
    xp = np.interp(th, verts, cx + rv * np.cos(verts))
    yp = np.interp(th, verts, cy + rv * np.sin(verts))

    x = (1 - chaos) * xp + chaos * xs
    y = (1 - chaos) * yp + chaos * ys
    scale = 0.04 * r  # displacement grows with the coil
    return _displace(rng, x, y, chaos,
                     tooth_h=float(rng.uniform(0.3, 1.0)) * scale,
                     period=float(rng.uniform(30, 80)),
                     noise_h=float(rng.uniform(0.6, 1.6)) * scale)


def _line_strand(rng, chaos, arc=0.0, overshoot=0.0, n=700):
    """A cut across the field; teeth turn it into a gear rack, noise into
    a wandering ridge. arc bows it, overshoot stretches it well past the
    card."""
    a = GOLDEN * int(rng.integers(0, 13))
    px = float(rng.uniform(0, CARD_W))
    py = float(rng.uniform(0, CARD_H))
    half = 110.0 * (1 + 2.5 * overshoot * float(rng.random()))
    t = np.linspace(-half, half, n)
    x, y = px + t * np.cos(a), py + t * np.sin(a)
    if arc > 0:
        bow = arc * float(rng.uniform(0.05, 0.6))
        sag = bow * (t ** 2 - half ** 2) / half
        x, y = x - sag * np.sin(a), y + sag * np.cos(a)
    return _displace(rng, x, y, chaos,
                     tooth_h=float(rng.uniform(0.8, 2.2)),
                     period=float(rng.uniform(25, 70)),
                     noise_h=float(rng.uniform(2.0, 6.0)))


# The sampled modulation axes, each in [0, 1]. All endpoints accept them
# as query-param overrides; unset axes are drawn per seed.
MOD_AXES = ("chaos", "recurrence", "arc", "wobble", "overshoot")


def _make_stroke(ax, rng, mods):
    """Shared finishing pipeline: wobble drifts strands off-position, arc
    swirls the whole field, overshoot tags some strands to clip at the
    card edge (through the border frame) instead of the inner panel."""
    arc, wob, ov = mods["arc"], mods["wobble"], mods["overshoot"]
    swirl_cx = float(rng.uniform(0, CARD_W))
    swirl_cy = float(rng.uniform(0, CARD_H))
    swirl_k = float(rng.uniform(0.5, 1.4)) * arc
    falloff = float(rng.uniform(35, 90))

    def stroke(x, y, color, lw, z):
        if wob > 0:
            amp = wob * float(rng.uniform(2.0, 8.0))
            x = x + amp * _fbm1(rng, len(x), octaves=2)
            y = y + amp * _fbm1(rng, len(y), octaves=2)
        if swirl_k > 0:
            dx, dy = x - swirl_cx, y - swirl_cy
            ang = swirl_k * np.exp(-np.hypot(dx, dy) / falloff)
            ca, sa = np.cos(ang), np.sin(ang)
            x = swirl_cx + dx * ca - dy * sa
            y = swirl_cy + dx * sa + dy * ca
        line, = ax.plot(x, y, color=color, lw=lw, zorder=z)
        if ov > 0 and rng.random() < 0.6 * ov:
            line.set_gid("overshoot")

    return stroke


def strand_field(ax, rng, pal, mods):
    """Vib-Ribbon line art: every mark is one continuous stroke, pushed
    around by the modulation axes. chaos blends discrete golden cuts into
    fractal wander; recurrence spawns child coils."""
    chaos, rec, arc, wob, ov = (mods[k] for k in MOD_AXES)
    stroke = _make_stroke(ax, rng, mods)

    def coil(cx, cy, scale, depth, facets, z):
        n_coils = int(rng.integers(5, 10))
        for i in range(n_coils):
            r0 = scale * 1.2 * PHI ** (i / 2.2)
            turns = float(rng.uniform(2.0, 3.2)) * (1 + 0.6 * ov)
            x, y = _spiral_strand(rng, chaos, cx, cy, r0, turns, facets, n=700)
            stroke(x, y, pal["stroke"], 1.2 if i % 4 == 0 else 0.55, z)
        # Recurrence: children bud at golden-angle offsets, shrunken by phi.
        if depth > 0:
            for _ in range(int(round(rec * rng.integers(1, 4)))):
                a = GOLDEN * int(rng.integers(1, 9))
                rr = scale * 1.2 * PHI ** (n_coils / 2.2) * float(rng.uniform(0.5, 1.0))
                coil(cx + rr * np.cos(a), cy + rr * np.sin(a),
                     scale / PHI, depth - 1, facets, max(z - 1, 2))

    # Faint background cuts for depth.
    for _ in range(int(rng.integers(5, 9))):
        x, y = _line_strand(rng, chaos, arc, ov)
        stroke(x, y, pal["faint"], 0.5, 1)

    coil(float(rng.uniform(-5, CARD_W + 5)), float(rng.uniform(-5, CARD_H + 5)),
         1.0, 1 + int(rec > 0.4), int(rng.choice([8, 13, 21])), 3)

    # Foreground cuts.
    for _ in range(int(rng.integers(2, 5))):
        x, y = _line_strand(rng, chaos, arc, ov)
        stroke(x, y, pal["stroke"], 0.7, 2)


def shatter_field(ax, rng, pal, mods):
    """Broken glass: jagged rays from impact points, webbed together by
    chords between neighboring rays. chaos sets the jaggedness."""
    chaos = mods["chaos"]
    stroke = _make_stroke(ax, rng, mods)
    for _ in range(int(rng.integers(1, 3))):
        cx = float(rng.uniform(5, CARD_W - 5))
        cy = float(rng.uniform(5, CARD_H - 5))
        n_rays = int(rng.integers(7, 14))
        angles = np.sort(rng.uniform(0, 2 * np.pi, n_rays))
        tips, n = [], 140
        for a in angles:
            r = np.linspace(0, float(rng.uniform(40, 130)), n)
            jitter = (0.02 + 0.10 * chaos) * _fbm1(rng, n, octaves=4) * r
            aa = a + jitter / np.maximum(r, 1)
            x, y = cx + r * np.cos(aa), cy + r * np.sin(aa)
            tips.append((r[-1], x, y))
            stroke(x, y, pal["stroke"], float(rng.uniform(0.5, 1.2)), 3)
        # Web chords between adjacent rays at staggered radii.
        for ring in np.cumsum(rng.uniform(5, 16, 6)):
            for i in range(n_rays):
                r1, x1, y1 = tips[i]
                r2, x2, y2 = tips[(i + 1) % n_rays]
                if ring > min(r1, r2) or rng.random() < 0.15:
                    continue
                j1 = int(ring / r1 * (n - 1))
                j2 = int(ring / r2 * (n - 1))
                stroke(np.array([x1[j1], x2[j2]]), np.array([y1[j1], y2[j2]]),
                       pal["faint"] if rng.random() < 0.4 else pal["stroke"],
                       0.5, 2)


def lightning_field(ax, rng, pal, mods):
    """Midpoint-displacement bolts arcing across the card; recurrence
    forks branches, chaos sets the roughness."""
    chaos, rec = mods["chaos"], mods["recurrence"]
    stroke = _make_stroke(ax, rng, mods)

    def bolt(p0, p1, disp, depth, lw, color, z):
        pts = np.array([p0, p1], dtype=float)
        for _ in range(7):
            mid = (pts[:-1] + pts[1:]) / 2
            seg = pts[1:] - pts[:-1]
            nrm = np.stack([-seg[:, 1], seg[:, 0]], axis=1)
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
            mid += nrm * rng.normal(0, disp, (len(mid), 1))
            merged = np.empty((2 * len(pts) - 1, 2))
            merged[0::2], merged[1::2] = pts, mid
            pts = merged
            disp *= 0.55
        stroke(pts[:, 0], pts[:, 1], color, lw, z)
        if depth > 0:
            for _ in range(int(round(1 + 2 * rec))):
                i = int(rng.integers(len(pts) // 4, 3 * len(pts) // 4))
                d = pts[i] - pts[i - 1]
                ang = np.arctan2(d[1], d[0]) + float(rng.uniform(-1.2, 1.2))
                length = float(np.hypot(*(np.array(p1) - p0))) * float(rng.uniform(0.25, 0.5))
                end = pts[i] + length * np.array([np.cos(ang), np.sin(ang)])
                bolt(pts[i], end, disp * 4, depth - 1, lw * 0.6, color, z)

    rough = 3.0 + 9.0 * chaos
    for _ in range(int(rng.integers(4, 8))):
        edge = rng.uniform(-15, 1.15 * CARD_W, 2), rng.uniform(-15, 1.15 * CARD_H, 2)
        bolt((float(edge[0][0]), float(edge[1][0])),
             (float(edge[0][1]), float(edge[1][1])),
             rough * 0.4, 1, 0.5, pal["faint"], 1)
    for _ in range(int(rng.integers(2, 4))):
        x0, x1 = rng.uniform(-10, CARD_W + 10, 2)
        y0, y1 = rng.uniform(-10, CARD_H + 10, 2)
        bolt((float(x0), float(y0)), (float(x1), float(y1)),
             rough, 1 + int(rec > 0.4), float(rng.uniform(0.9, 1.5)),
             pal["stroke"], 3)


def wave_field(ax, rng, pal, mods):
    """Stacked signal rows, amplitude pooled under a drifting envelope -
    neural activity traces. chaos blends carrier sine into fractal noise."""
    chaos = mods["chaos"]
    stroke = _make_stroke(ax, rng, mods)
    rows = int(rng.integers(18, 30))
    n = 500
    x = np.linspace(-10, CARD_W + 10, n)
    env_c = float(rng.uniform(0.2, 0.8)) * CARD_W
    env_w = float(rng.uniform(12, 30))
    drift = float(rng.uniform(-0.8, 0.8))
    for i in range(rows):
        y0 = -8 + (CARD_H + 16) * i / (rows - 1)
        center = env_c + drift * (y0 - CARD_H / 2) + float(rng.normal(0, 4))
        env = np.exp(-((x - center) ** 2) / (2 * env_w ** 2))
        amp = float(rng.uniform(2, 9))
        carrier = np.sin(2 * np.pi * x / float(rng.uniform(5, 14))
                         + float(rng.uniform(0, 6)))
        sig = (1 - chaos) * carrier + chaos * 2.2 * _fbm1(rng, n)
        y = y0 + amp * env * sig + 0.4 * _fbm1(rng, n, octaves=3)
        color = pal["faint"] if i % 4 == 3 else pal["stroke"]
        stroke(x, y, color, 0.6, 2 + (i % 4 != 3))


def fibonacci_field(ax, rng, pal, mods):
    """One golden spiral arm, cloned through golden-angle rotations and
    phi-shrunk generation by generation - the same curve repeated over
    time, fading as it recedes. recurrence sets the generation depth,
    chaos the usual cut-to-fractal blend."""
    chaos, rec = mods["chaos"], mods["recurrence"]
    stroke = _make_stroke(ax, rng, mods)
    cx = float(rng.uniform(10, CARD_W - 10))
    cy = float(rng.uniform(5, CARD_H - 5))
    arms = int(rng.integers(3, 8))
    generations = 2 + int(round(3 * rec))
    facets = int(rng.choice([8, 13, 21]))
    base_rot = float(rng.uniform(0, 2 * np.pi))
    turns = float(rng.uniform(2.4, 3.4))
    base = float(rng.uniform(5, 12))
    for gen in range(generations):
        scale = base * PHI ** (-gen)
        color = pal["stroke"] if gen < 2 else pal["faint"]
        lw = max(0.4, 1.1 * PHI ** (-gen))
        for k in range(arms):
            x, y = _spiral_strand(rng, chaos, cx, cy, scale, turns,
                                  facets, n=700)
            rot = base_rot + GOLDEN * (k + gen)
            dx, dy = x - cx, y - cy
            ca, sa = np.cos(rot), np.sin(rot)
            stroke(cx + dx * ca - dy * sa, cy + dx * sa + dy * ca,
                   color, lw, 4 - min(gen, 2))
    # Phyllotaxis nodes: seed points at golden-angle steps, sqrt spacing.
    if rng.random() < 0.7:
        t = np.linspace(0, 2 * np.pi, 24)
        c = float(rng.uniform(1.5, 3.0))
        for i in range(int(rng.integers(40, 90))):
            r = c * np.sqrt(i)
            a = i * GOLDEN + base_rot
            nr = 0.25 + 0.5 * (i % 3 == 0)
            stroke(cx + r * np.cos(a) + nr * np.cos(t),
                   cy + r * np.sin(a) + nr * np.sin(t),
                   pal["faint"], 0.5, 1)


def glyph_field(ax, rng, pal, mods):
    """Basic closed geometries, drawn and shaded: filled vectors scattered
    nearly full-bleed in some samples, sparse in others. Outlines deform
    with chaos; fills cycle solid tint, solid accent, and hatch shading."""
    chaos, ov = mods["chaos"], mods["overshoot"]
    density = float(rng.uniform(0.1, 1.0))
    count = int(round(4 + 26 * density))
    t = np.linspace(0, 2 * np.pi, 240)
    glyphs = []
    for _ in range(count):
        cx = float(rng.uniform(-5, CARD_W + 5))
        cy = float(rng.uniform(-5, CARD_H + 5))
        rad = float(rng.uniform(2.5, 15)) * (1.6 - 0.6 * density)
        k = int(rng.choice([3, 4, 5, 6, 8, 13, 0]))  # 0 = circle
        rot = float(rng.uniform(0, 2 * np.pi))
        if k:
            # Regular k-gon by apothem, then chaos warps the radius.
            r = rad * np.cos(np.pi / k) / np.cos(
                ((t + rot) % (2 * np.pi / k)) - np.pi / k)
        else:
            r = np.full_like(t, rad)
        r = r * (1 + 0.35 * chaos * _fbm1(rng, len(t), octaves=3))
        glyphs.append((rad, cx + r * np.cos(t), cy + r * np.sin(t)))

    for rad, x, y in sorted(glyphs, key=lambda g: -g[0]):  # big ones behind
        style = rng.random()
        if style < 0.45:
            patch, = ax.fill(x, y, facecolor=pal["fill"],
                             edgecolor=pal["stroke"], lw=0.7, zorder=2)
        elif style < 0.7:
            patch, = ax.fill(x, y, facecolor=pal["stroke"],
                             edgecolor=pal["paper"], lw=0.7, zorder=3)
        else:
            patch, = ax.fill(x, y, facecolor="none", hatch="///",
                             edgecolor=pal["faint"], lw=0.6, zorder=2)
        if ov > 0 and rng.random() < 0.6 * ov:
            patch.set_gid("overshoot")


PROJECTION_REGISTRY = {
    "strands": strand_field,
    "shatter": shatter_field,
    "lightning": lightning_field,
    "waves": wave_field,
    "fibonacci": fibonacci_field,
    "glyphs": glyph_field,
}


def _caps(rng, text, weights):
    style = rng.choice(["title", "upper", "lower"], p=weights)
    return {"title": text.title(), "upper": text.upper(), "lower": text.lower()}[style]


def _sample_mods(rng, mods):
    """Draw the modulation axes in a fixed order; overrides win but the
    draws still happen so a seed renders identically either way."""
    sampled = {
        "chaos": float(rng.uniform(0.05, 0.95)),
        "recurrence": float(rng.uniform(0, 1)),
        "arc": float(rng.uniform(0, 1)),
        "wobble": float(rng.uniform(0, 1)) * float(rng.random() < 0.6),
        "overshoot": float(rng.uniform(0, 1)) * float(rng.random() < 0.6),
    }
    for k, v in (mods or {}).items():
        if v is not None:
            sampled[k] = min(max(float(v), 0.0), 1.0)
    return sampled


def _draw_card(ax, side, seed, theme, hue, authors, donations, run_hash,
               mods=None):
    from matplotlib import patheffects
    from matplotlib.patches import Rectangle

    pal = _palette(theme, hue)
    if side == "back":
        # Inverted two-tone: paper-colored strokes on an accent ground.
        paper, accent = pal["paper"], pal["stroke"]
        pal = {**pal, "paper": accent, "stroke": paper, "ink": paper,
               "faint": paper, "fill": accent}
    rng = np.random.default_rng([seed, 0 if side == "front" else 1])

    ax.set_xlim(0, CARD_W)
    ax.set_ylim(0, CARD_H)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(Rectangle((0, 0), CARD_W, CARD_H, facecolor=pal["border"],
                           edgecolor="none", zorder=0))
    inner = Rectangle((BORDER_MM, BORDER_MM), CARD_W - 2 * BORDER_MM,
                      CARD_H - 2 * BORDER_MM, facecolor=pal["paper"],
                      edgecolor="none", zorder=0.5)
    ax.add_patch(inner)

    names = list(PROJECTION_REGISTRY)
    field = PROJECTION_REGISTRY[names[int(rng.integers(len(names)))]]
    sampled = _sample_mods(rng, mods)
    field(ax, rng, pal, sampled)
    # Clip geometry to the inner panel so the border stays clean;
    # overshoot-tagged strands get the full card and cross the frame.
    bleed = Rectangle((0, 0), CARD_W, CARD_H, facecolor="none",
                      edgecolor="none")
    ax.add_patch(bleed)
    for artist in list(ax.lines) + list(ax.patches) + list(ax.collections):
        if artist is not inner and artist is not bleed:
            artist.set_clip_path(bleed if artist.get_gid() == "overshoot"
                                 else inner)

    halo = [patheffects.withStroke(linewidth=2.2, foreground=pal["paper"])]
    m = BORDER_MM + 4
    # Sometimes the text block mirrors to the top of the card.
    flip_v = rng.random() < 0.35
    vy = (lambda y: CARD_H - y) if flip_v else (lambda y: y)
    va = "top" if flip_v else "baseline"
    if side == "front":
        for i, author in enumerate(authors):
            ax.text(m, vy(m + 6 * (len(authors) - 1 - i)),
                    _caps(rng, author, [0.5, 0.35, 0.15]),
                    fontsize=11, color=pal["ink"], family="DejaVu Sans",
                    weight="bold" if i == 0 else "normal", va=va,
                    path_effects=halo, zorder=5)
        if rng.random() < 0.6:  # a signature stroke above the name, sometimes
            chaos = sampled["chaos"]
            n = 140
            xs = np.linspace(m, m + float(rng.uniform(16, 30)), n)
            env = np.sin(np.linspace(0, np.pi, n)) ** 0.6  # pen taper
            teeth = np.tanh(6 * np.sin(2 * np.pi * np.arange(n)
                                       / float(rng.uniform(15, 40))))
            d = ((1 - chaos) * 0.5 * teeth
                 + chaos * 1.4 * _fbm1(rng, n, octaves=4)) * env
            ax.plot(xs, vy(m + 6 * len(authors) + 1.0) + d,
                    color=pal["stroke"], lw=1.2, zorder=5)
    else:
        if donations:
            ax.text(CARD_W / 2, CARD_H / 2,
                    _caps(rng, donations, [0.1, 0.1, 0.8]),
                    fontsize=7.5, color=pal["ink"], family="DejaVu Sans Mono",
                    ha="center", va="center", path_effects=halo, zorder=5)
        if run_hash:
            ax.text(CARD_W - m, vy(m), run_hash, fontsize=5.5,
                    color=pal["stroke"], family="DejaVu Sans Mono", ha="right",
                    va=va, path_effects=halo, zorder=5)


def _new_fig(w_mm, h_mm, facecolor):
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "praxis-card"
    import matplotlib.pyplot as plt

    return plt.figure(figsize=(w_mm / 25.4, h_mm / 25.4), facecolor=facecolor)


def _mm_axes(fig, x, y, w, h, page_w, page_h):
    ax = fig.add_axes([x / page_w, y / page_h, w / page_w, h / page_h])
    return ax


def _crop_marks(ax, x, y, w, h, color, length=4.0, gap=1.0):
    for px, sx in ((x, -1), (x + w, 1)):
        for py, sy in ((y, -1), (y + h, 1)):
            ax.plot([px + sx * gap, px + sx * (gap + length)], [py, py],
                    color=color, lw=0.5)
            ax.plot([px, px], [py + sy * gap, py + sy * (gap + length)],
                    color=color, lw=0.5)


def _save(fig, fmt):
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    meta = {"Date": None} if fmt == "svg" else {"CreationDate": None}
    fig.savefig(buf, format=fmt, facecolor=fig.get_facecolor(), metadata=meta)
    plt.close(fig)
    return buf.getvalue()


def _merge_mods(mods, chaos):
    mods = dict(mods or {})
    if chaos is not None:
        mods.setdefault("chaos", chaos)
    return mods


def render_card(side, seed, theme, hue, authors, donations, run_hash,
                fmt="svg", chaos=None, mods=None):
    """One bare card at exact size (preview or print-and-cut)."""
    fig = _new_fig(CARD_W, CARD_H, "none")
    ax = _mm_axes(fig, 0, 0, CARD_W, CARD_H, CARD_W, CARD_H)
    _draw_card(ax, side, seed, theme, hue, authors, donations, run_hash,
               _merge_mods(mods, chaos))
    return _save(fig, fmt)


def _page(cells, side, seed_for, theme, hue, authors, donations, run_hash,
          mods=None):
    """A4 page with cards at the given (x, y) mm cells, crop marks, margin note."""
    pal = _palette(theme, hue)
    fig = _new_fig(PAGE_W, PAGE_H, "#ffffff")
    page = _mm_axes(fig, 0, 0, PAGE_W, PAGE_H, PAGE_W, PAGE_H)
    page.set_xlim(0, PAGE_W)
    page.set_ylim(0, PAGE_H)
    page.axis("off")
    for i, (x, y) in enumerate(cells):
        _crop_marks(page, x, y, CARD_W, CARD_H, "#999999")
        ax = _mm_axes(fig, x, y, CARD_W, CARD_H, PAGE_W, PAGE_H)
        _draw_card(ax, side, seed_for(i), theme, hue, authors, donations,
                   run_hash, mods)
    page.text(PAGE_W / 2, 8, PRINT_NOTE, fontsize=6, color="#666666",
              ha="center", va="center", wrap=True)
    page.text(PAGE_W / 2, PAGE_H - 8,
              f"Praxis business card - {side} - seed {seed_for(0)}",
              fontsize=6, color="#666666", ha="center", va="center")
    # Paper guidance runs sideways in the left/right margins.
    page.text(6, PAGE_H / 2, PAPER_NOTE, fontsize=6, color="#666666",
              ha="center", va="center", rotation=90)
    page.text(PAGE_W - 6, PAGE_H / 2, PAPER_LINK_NOTE, fontsize=6,
              color="#666666", ha="center", va="center", rotation=270,
              url=PAPER_URL)
    return _save(fig, "pdf")


def _cell(row, col):
    """Avery 28371 cell origin (mm, bottom-left), row 0 at the page top."""
    return (SHEET_X0 + col * CARD_W,
            PAGE_H - SHEET_Y0 - (row + 1) * CARD_H)


def render_single_pdf(side, seed, theme, hue, authors, donations, run_hash,
                      chaos=None, mods=None):
    # The top-left Avery cell, so a single card prints onto the same stock;
    # the back mirrors to the other column for long-edge duplex.
    col = 0 if side == "front" else SHEET_COLS - 1
    return _page([_cell(0, col)], side, lambda i: seed, theme, hue, authors,
                 donations, run_hash, _merge_mods(mods, chaos))


def render_sheet_pdf(side, seed, theme, hue, authors, donations, run_hash,
                     chaos=None, mods=None):
    """Full Avery 28371 imposition (10-up). Back pages mirror columns for
    long-edge duplex."""
    cells, seeds = [], []
    for row in range(SHEET_ROWS):
        for col in range(SHEET_COLS):
            draw_col = (SHEET_COLS - 1 - col) if side == "back" else col
            cells.append(_cell(row, draw_col))
            seeds.append(seed + row * SHEET_COLS + col)
    return _page(cells, side, lambda i: seeds[i], theme, hue, authors,
                 donations, run_hash, _merge_mods(mods, chaos))
