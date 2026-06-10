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


PHI = (1 + 5**0.5) / 2
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
    return _displace(
        rng,
        x,
        y,
        chaos,
        tooth_h=float(rng.uniform(0.3, 1.0)) * scale,
        period=float(rng.uniform(30, 80)),
        noise_h=float(rng.uniform(0.6, 1.6)) * scale,
    )


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
        sag = bow * (t**2 - half**2) / half
        x, y = x - sag * np.sin(a), y + sag * np.cos(a)
    return _displace(
        rng,
        x,
        y,
        chaos,
        tooth_h=float(rng.uniform(0.8, 2.2)),
        period=float(rng.uniform(25, 70)),
        noise_h=float(rng.uniform(2.0, 6.0)),
    )


# The sampled modulation axes, each in [0, 1]. All endpoints accept them
# as query-param overrides; unset axes are drawn per seed.
MOD_AXES = ("chaos", "recurrence", "arc", "wobble", "overshoot", "symmetry")


def _make_stroke(ax, rng, mods):
    """Shared finishing pipeline: wobble drifts strands off-position, arc
    swirls the whole field, overshoot tags some strands to clip at the
    card edge (through the border frame) instead of the inner panel."""
    arc, wob, ov = mods["arc"], mods["wobble"], mods["overshoot"]
    swirl_cx = float(rng.uniform(0, CARD_W))
    swirl_cy = float(rng.uniform(0, CARD_H))
    swirl_k = float(rng.uniform(0.5, 1.4)) * arc
    falloff = float(rng.uniform(35, 90))
    # symmetry replicates every stroke k-fold around one center: any field
    # becomes a mandala / crystal as the dial rises.
    sym = mods["symmetry"]
    sym_k = 1 + int(round(sym * float(rng.uniform(2, 6))))
    sym_cx = float(rng.uniform(0.25, 0.75)) * CARD_W
    sym_cy = float(rng.uniform(0.25, 0.75)) * CARD_H

    def stroke(x, y, color, lw, z):
        if wob > 0:
            amp = wob * float(rng.uniform(2.0, 8.0))
            # Small marks (bone knobs, boutons, glyph dots) can't survive
            # multi-mm wander; scale the drift to the stroke's own extent.
            ext = float(max(np.ptp(x), np.ptp(y)))
            amp *= min(1.0, max(0.15, ext / 30.0))
            x = x + amp * _fbm1(rng, len(x), octaves=2)
            y = y + amp * _fbm1(rng, len(y), octaves=2)
        if swirl_k > 0:
            dx, dy = x - swirl_cx, y - swirl_cy
            ang = swirl_k * np.exp(-np.hypot(dx, dy) / falloff)
            ca, sa = np.cos(ang), np.sin(ang)
            x = swirl_cx + dx * ca - dy * sa
            y = swirl_cy + dx * sa + dy * ca
        tag = ov > 0 and rng.random() < 0.6 * ov
        for i in range(sym_k):
            if i == 0:
                xi, yi = x, y
            else:
                a = 2 * np.pi * i / sym_k
                ca, sa = np.cos(a), np.sin(a)
                dx, dy = x - sym_cx, y - sym_cy
                xi = sym_cx + dx * ca - dy * sa
                yi = sym_cy + dx * sa + dy * ca
            (line,) = ax.plot(xi, yi, color=color, lw=lw, zorder=z)
            if tag:
                line.set_gid("overshoot")

    return stroke


def strand_field(ax, rng, pal, mods):
    """Vib-Ribbon line art: every mark is one continuous stroke, pushed
    around by the modulation axes. chaos blends discrete golden cuts into
    fractal wander; recurrence spawns child coils."""
    chaos, rec, arc, wob, ov = (
        mods[k] for k in ("chaos", "recurrence", "arc", "wobble", "overshoot")
    )
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
                coil(
                    cx + rr * np.cos(a),
                    cy + rr * np.sin(a),
                    scale / PHI,
                    depth - 1,
                    facets,
                    max(z - 1, 2),
                )

    # Faint background cuts for depth.
    for _ in range(int(rng.integers(5, 9))):
        x, y = _line_strand(rng, chaos, arc, ov)
        stroke(x, y, pal["faint"], 0.5, 1)

    coil(
        float(rng.uniform(-5, CARD_W + 5)),
        float(rng.uniform(-5, CARD_H + 5)),
        1.0,
        1 + int(rec > 0.4),
        int(rng.choice([8, 13, 21])),
        3,
    )

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
                stroke(
                    np.array([x1[j1], x2[j2]]),
                    np.array([y1[j1], y2[j2]]),
                    pal["faint"] if rng.random() < 0.4 else pal["stroke"],
                    0.5,
                    2,
                )


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
                length = float(np.hypot(*(np.array(p1) - p0))) * float(
                    rng.uniform(0.25, 0.5)
                )
                end = pts[i] + length * np.array([np.cos(ang), np.sin(ang)])
                bolt(pts[i], end, disp * 4, depth - 1, lw * 0.6, color, z)

    rough = 3.0 + 9.0 * chaos
    for _ in range(int(rng.integers(4, 8))):
        edge = rng.uniform(-15, 1.15 * CARD_W, 2), rng.uniform(-15, 1.15 * CARD_H, 2)
        bolt(
            (float(edge[0][0]), float(edge[1][0])),
            (float(edge[0][1]), float(edge[1][1])),
            rough * 0.4,
            1,
            0.5,
            pal["faint"],
            1,
        )
    for _ in range(int(rng.integers(2, 4))):
        x0, x1 = rng.uniform(-10, CARD_W + 10, 2)
        y0, y1 = rng.uniform(-10, CARD_H + 10, 2)
        bolt(
            (float(x0), float(y0)),
            (float(x1), float(y1)),
            rough,
            1 + int(rec > 0.4),
            float(rng.uniform(0.9, 1.5)),
            pal["stroke"],
            3,
        )


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
        env = np.exp(-((x - center) ** 2) / (2 * env_w**2))
        amp = float(rng.uniform(2, 9))
        carrier = np.sin(
            2 * np.pi * x / float(rng.uniform(5, 14)) + float(rng.uniform(0, 6))
        )
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
            x, y = _spiral_strand(rng, chaos, cx, cy, scale, turns, facets, n=700)
            rot = base_rot + GOLDEN * (k + gen)
            dx, dy = x - cx, y - cy
            ca, sa = np.cos(rot), np.sin(rot)
            stroke(
                cx + dx * ca - dy * sa,
                cy + dx * sa + dy * ca,
                color,
                lw,
                4 - min(gen, 2),
            )
    # Phyllotaxis nodes: seed points at golden-angle steps, sqrt spacing.
    if rng.random() < 0.7:
        t = np.linspace(0, 2 * np.pi, 24)
        c = float(rng.uniform(1.5, 3.0))
        for i in range(int(rng.integers(40, 90))):
            r = c * np.sqrt(i)
            a = i * GOLDEN + base_rot
            nr = 0.25 + 0.5 * (i % 3 == 0)
            stroke(
                cx + r * np.cos(a) + nr * np.cos(t),
                cy + r * np.sin(a) + nr * np.sin(t),
                pal["faint"],
                0.5,
                1,
            )


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
            r = (
                rad
                * np.cos(np.pi / k)
                / np.cos(((t + rot) % (2 * np.pi / k)) - np.pi / k)
            )
        else:
            r = np.full_like(t, rad)
        r = r * (1 + 0.35 * chaos * _fbm1(rng, len(t), octaves=3))
        glyphs.append((rad, cx + r * np.cos(t), cy + r * np.sin(t)))

    for rad, x, y in sorted(glyphs, key=lambda g: -g[0]):  # big ones behind
        style = rng.random()
        if style < 0.45:
            (patch,) = ax.fill(
                x, y, facecolor=pal["fill"], edgecolor=pal["stroke"], lw=0.7, zorder=2
            )
        elif style < 0.7:
            (patch,) = ax.fill(
                x, y, facecolor=pal["stroke"], edgecolor=pal["paper"], lw=0.7, zorder=3
            )
        else:
            (patch,) = ax.fill(
                x,
                y,
                facecolor="none",
                hatch="///",
                edgecolor=pal["faint"],
                lw=0.6,
                zorder=2,
            )
        if ov > 0 and rng.random() < 0.6 * ov:
            patch.set_gid("overshoot")


def _blob(rng, cx, cy, rad, rough, n=160):
    """Closed organic outline: a circle with fBm-perturbed radius."""
    t = np.linspace(0, 2 * np.pi, n)
    r = rad * (1 + rough * _fbm1(rng, n, octaves=3))
    return cx + r * np.cos(t), cy + r * np.sin(t)


def rubble_field(ax, rng, pal, mods):
    """Particles: fractured stone. A few large cracked slabs, then pebbles
    and rubble scattered in a power-law size cascade, settling toward a
    sampled resting band."""
    chaos, ov = mods["chaos"], mods["overshoot"]
    stroke = _make_stroke(ax, rng, mods)
    band_y = float(rng.uniform(0, CARD_H))
    spread = float(rng.uniform(8, 40))

    def place():
        return (
            float(rng.uniform(-5, CARD_W + 5)),
            float(np.clip(rng.normal(band_y, spread), -5, CARD_H + 5)),
        )

    # Slabs: large angular shards, cracked through by a jagged strand.
    for _ in range(int(rng.integers(1, 4))):
        cx, cy = place()
        rad = float(rng.uniform(8, 20))
        k = int(rng.integers(4, 7))
        t = np.linspace(0, 2 * np.pi, k, endpoint=False)
        r = rad * rng.uniform(0.6, 1.0, k)
        x = np.append(cx + r * np.cos(t), cx + r[0] * np.cos(t[0]))
        y = np.append(cy + r * np.sin(t), cy + r[0] * np.sin(t[0]))
        (patch,) = ax.fill(
            x, y, facecolor=pal["fill"], edgecolor=pal["stroke"], lw=0.9, zorder=2
        )
        if ov > 0 and rng.random() < 0.6 * ov:
            patch.set_gid("overshoot")
        a = float(rng.uniform(0, np.pi))
        n = 80
        tt = np.linspace(-rad, rad, n)
        crack = (1 + 2 * chaos) * _fbm1(rng, n, octaves=4)
        stroke(
            cx + tt * np.cos(a) - crack * np.sin(a),
            cy + tt * np.sin(a) + crack * np.cos(a),
            pal["stroke"],
            0.6,
            3,
        )

    # Pebbles: many small irregular convex stones, smaller = more numerous.
    count = int(rng.integers(25, 70))
    for _ in range(count):
        cx, cy = place()
        rad = 0.8 + 5.0 * float(rng.random()) ** 2.5
        k = int(rng.integers(5, 9))
        t = np.sort(rng.uniform(0, 2 * np.pi, k))
        r = rad * rng.uniform(0.7, 1.0, k)
        x = np.append(cx + r * np.cos(t), cx + r[0] * np.cos(t[0]))
        y = np.append(cy + r * np.sin(t), cy + r[0] * np.sin(t[0]))
        filled = rng.random()
        if filled < 0.25:
            ax.fill(x, y, facecolor=pal["stroke"], edgecolor="none", zorder=2)
        elif filled < 0.5:
            ax.fill(
                x, y, facecolor=pal["fill"], edgecolor=pal["stroke"], lw=0.5, zorder=2
            )
        else:
            ax.plot(x, y, color=pal["stroke"], lw=0.5, zorder=2)


def splatter_field(ax, rng, pal, mods):
    """Fluid: thrown liquid. Main blobs with fBm-rough rims, directional
    droplet spray decaying with distance, and gravity drips trailing off
    the largest masses. chaos sets the violence of the throw."""
    chaos, ov = mods["chaos"], mods["overshoot"]
    rough = 0.15 + 0.5 * chaos
    for _ in range(int(rng.integers(2, 5))):
        cx = float(rng.uniform(0, CARD_W))
        cy = float(rng.uniform(0, CARD_H))
        rad = float(rng.uniform(3, 13))
        solid = rng.random() < 0.7
        x, y = _blob(rng, cx, cy, rad, rough)
        (patch,) = ax.fill(
            x,
            y,
            facecolor=pal["stroke"] if solid else pal["fill"],
            edgecolor="none" if solid else pal["stroke"],
            lw=0.6,
            zorder=3,
        )
        if ov > 0 and rng.random() < 0.6 * ov:
            patch.set_gid("overshoot")

        # Directional spray: droplets shrink and spread with distance.
        throw = float(rng.uniform(0, 2 * np.pi))
        for _ in range(int(10 + 30 * chaos)):
            dist = rad + float(rng.exponential(12 + 25 * chaos))
            a = throw + float(rng.normal(0, 0.35 + 0.4 * chaos))
            dr = max(0.15, rad * 0.25 * float(rng.random()) * rad / (rad + dist))
            dx, dy = _blob(
                rng, cx + dist * np.cos(a), cy + dist * np.sin(a), dr, rough * 0.6, n=40
            )
            ax.fill(dx, dy, facecolor=pal["stroke"], edgecolor="none", zorder=3)

        # Gravity drips off the big masses.
        if rad > 7 and rng.random() < 0.8:
            for _ in range(int(rng.integers(1, 4))):
                dxp = cx + float(rng.uniform(-0.5, 0.5)) * rad
                run = float(rng.uniform(4, 16))
                ax.plot(
                    [dxp, dxp + float(rng.normal(0, 1.5))],
                    [cy, cy - run],
                    color=pal["stroke"],
                    lw=0.7,
                    zorder=2,
                )
                bx, by = _blob(rng, dxp, cy - run, 0.7, 0.3, n=30)
                ax.fill(bx, by, facecolor=pal["stroke"], edgecolor="none", zorder=2)


def snowflake_field(ax, rng, pal, mods):
    """Dendritic crystals: 6-fold arms with paired side branchlets that
    shrink toward the tip. chaos jitters the growth, recurrence adds
    sub-branching, sizes scatter like falling snow."""
    chaos, rec = mods["chaos"], mods["recurrence"]
    stroke = _make_stroke(ax, rng, mods)

    def branchlets(stroke_fn, x0, y0, ang, length, lw, depth):
        n = 30
        t = np.linspace(0, length, n)
        x = x0 + t * np.cos(ang)
        y = y0 + t * np.sin(ang)
        jit = chaos * 0.06 * length * _fbm1(rng, n, octaves=2)
        stroke_fn(x - jit * np.sin(ang), y + jit * np.cos(ang), pal["stroke"], lw, 3)
        if depth > 0:
            for frac in np.arange(0.3, 1.0, float(rng.uniform(0.18, 0.3))):
                bx, by = x0 + frac * length * np.cos(ang), y0 + frac * length * np.sin(
                    ang
                )
                blen = length * (1 - frac) * float(rng.uniform(0.4, 0.7))
                spread = np.pi / 3 * (1 + 0.4 * chaos * float(rng.standard_normal()))
                for s in (-1, 1):
                    branchlets(
                        stroke_fn, bx, by, ang + s * spread, blen, lw * 0.7, depth - 1
                    )

    for _ in range(int(rng.integers(2, 5))):
        cx = float(rng.uniform(0, CARD_W))
        cy = float(rng.uniform(0, CARD_H))
        size = float(rng.uniform(5, 24))
        rot = float(rng.uniform(0, np.pi / 3))
        depth = 1 + int(rec > 0.45)
        for k in range(6):
            branchlets(
                stroke,
                cx,
                cy,
                rot + k * np.pi / 3,
                size,
                max(0.45, 0.09 * size**0.5 * 3),
                depth,
            )
        # Hex core, sometimes.
        if rng.random() < 0.5:
            t = np.linspace(0, 2 * np.pi, 7)
            r = size * 0.18
            stroke(
                cx + r * np.cos(t + rot),
                cy + r * np.sin(t + rot),
                pal["stroke"],
                0.5,
                3,
            )


def flora_field(ax, rng, pal, mods):
    """Branches with leaves and petals: recursive limbs curving as they
    grow, ellipse foliage filled along stems and at tips. recurrence sets
    branching depth, chaos bends the growth."""
    chaos, rec = mods["chaos"], mods["recurrence"]
    stroke = _make_stroke(ax, rng, mods)
    t_leaf = np.linspace(0, 2 * np.pi, 40)

    def leaf(x0, y0, ang, size):
        lx = size * np.cos(t_leaf) * 0.32
        ly = size * np.sin(t_leaf)
        ca, sa = np.cos(ang), np.sin(ang)
        x = x0 + (ly + size) * ca - lx * sa
        y = y0 + (ly + size) * sa + lx * ca
        solid = rng.random() < 0.4
        ax.fill(
            x,
            y,
            facecolor=pal["stroke"] if solid else pal["fill"],
            edgecolor="none" if solid else pal["stroke"],
            lw=0.5,
            zorder=2,
        )

    def limb(x0, y0, ang, length, lw, depth):
        n = 50
        t = np.linspace(0, length, n)
        bend = (0.15 + 0.5 * chaos) * float(rng.standard_normal())
        a = ang + bend * t / length
        x = x0 + np.cumsum(np.cos(a)) * (length / n)
        y = y0 + np.cumsum(np.sin(a)) * (length / n)
        stroke(x, y, pal["stroke"], lw, 3)
        tip_a = float(a[-1])
        if depth > 0:
            for frac in (0.45, 0.7, 0.9):
                i = int(frac * (n - 1))
                limb(
                    x[i],
                    y[i],
                    float(a[i])
                    + float(rng.uniform(0.4, 0.9)) * (1 if rng.random() < 0.5 else -1),
                    length * float(rng.uniform(0.4, 0.6)),
                    lw * 0.7,
                    depth - 1,
                )
            if rng.random() < 0.5:
                leaf(x[-1], y[-1], tip_a, float(rng.uniform(2, 5)))
        else:
            # Terminal foliage: a leaf or a petal whorl.
            if rng.random() < 0.25:
                petals = int(rng.integers(4, 7))
                for p in range(petals):
                    leaf(
                        x[-1],
                        y[-1],
                        tip_a + 2 * np.pi * p / petals,
                        float(rng.uniform(1.5, 3.5)),
                    )
            else:
                leaf(x[-1], y[-1], tip_a, float(rng.uniform(2, 5)))

    for _ in range(int(rng.integers(2, 4))):
        edge_x = float(rng.uniform(0, CARD_W))
        limb(
            edge_x,
            -3.0,
            np.pi / 2 + float(rng.uniform(-0.5, 0.5)),
            float(rng.uniform(18, 42)),
            1.3,
            1 + int(round(2 * rec)),
        )


def bone_field(ax, rng, pal, mods):
    """Ossuary: cartoon long bones (flared shafts, paired condyle knobs)
    strewn at golden angles, with the odd vertebral chain. chaos warps the
    shafts, recurrence scatters smaller bone litter."""
    chaos, rec = mods["chaos"], mods["recurrence"]
    # Rigid geometry: full per-stroke wobble disassembles shapes whose
    # identity lives in stroke alignment (shafts+knobs, strands+rungs).
    mods = {**mods, "wobble": 0.35 * mods["wobble"]}
    stroke = _make_stroke(ax, rng, mods)

    def bone(cx, cy, ang, length, w, lw, z):
        axis = np.array([np.cos(ang), np.sin(ang)])
        perp = np.array([-axis[1], axis[0]])
        center = np.array([cx, cy])
        half = length / 2
        n = int(np.clip(length * 0.8, 50, 160))
        s = np.linspace(-half, half, n)
        # Shaft edges pinch at the middle and flare toward the ends.
        flare = w * (0.55 + 0.45 * (np.abs(s) / half) ** 2.2)
        jit = chaos * 0.18 * w * _fbm1(rng, n, octaves=2)
        for sign in (-1, 1):
            p = center[:, None] + np.outer(axis, s) + np.outer(perp, sign * flare + jit)
            stroke(p[0], p[1], pal["stroke"], lw, z)
        # Each end cap is ONE continuous curve: shaft edge -> around the top
        # lobe -> notch at the tip -> around the bottom lobe -> shaft edge.
        kr = w * float(rng.uniform(0.8, 1.1))
        sep, fwd = 0.62 * kr, 0.45 * kr
        tip = fwd + np.sqrt(kr * kr - sep * sep)  # where the lobe circles meet

        def lobe_arc(p_from, p_to, knob, end_pt):
            """Arc of the lobe circle from p_from to p_to, swept the way
            around that bulges away from the bone center."""
            a0 = np.arctan2(*(p_from - knob)[::-1])
            a1 = np.arctan2(*(p_to - knob)[::-1])
            cw = (a1 - a0) % (2 * np.pi)
            best = None
            for sweep in (cw, cw - 2 * np.pi):
                t = a0 + np.linspace(0, sweep, 40)
                arc = knob[:, None] + kr * np.vstack([np.cos(t), np.sin(t)])
                mid = arc[:, len(t) // 2]
                d = np.hypot(*(mid - center))
                if best is None or d > best[0]:
                    best = (d, arc)
            return best[1]

        for end, j in ((-1, jit[0]), (1, jit[-1])):
            e = center + end * half * axis
            notch = e + end * tip * axis
            a_top = e + (w + j) * perp
            a_bot = e + (-w + j) * perp
            k_top = e + end * fwd * axis + sep * perp
            k_bot = e + end * fwd * axis - sep * perp
            cap = np.hstack(
                [
                    a_top[:, None],
                    lobe_arc(a_top, notch, k_top, e),
                    lobe_arc(notch, a_bot, k_bot, e),
                    a_bot[:, None],
                ]
            )
            stroke(cap[0], cap[1], pal["stroke"], lw, z)

    def spine(cx, cy, ang, n_vert, size, z):
        # Vertebrae as shrinking diamonds along a bowed line.
        bend = (0.2 + 0.6 * chaos) * float(rng.standard_normal())
        a = ang
        for i in range(n_vert):
            r = size * (1 - 0.06 * i)
            t = np.linspace(0, 2 * np.pi, 5)
            stroke(
                cx + r * np.cos(t + a + np.pi / 4),
                cy + r * np.sin(t + a + np.pi / 4),
                pal["stroke"],
                0.7,
                z,
            )
            a += bend / n_vert
            cx += 2.4 * r * np.cos(a)
            cy += 2.4 * r * np.sin(a)

    # Faint sediment lines behind the litter.
    for _ in range(int(rng.integers(2, 5))):
        x, y = _line_strand(rng, chaos, mods["arc"], mods["overshoot"])
        stroke(x, y, pal["faint"], 0.5, 1)

    # A log size spectrum biased toward the colossal: anchor a point the
    # bone must pass through, then slide the length under it - the biggest
    # bones dwarf the card and render only as a cropped fragment.
    for _ in range(int(rng.integers(1, 5))):
        u = float(rng.uniform(0, 1)) ** 0.55
        length = 9.0 * PHI ** (7.2 * u)
        w = min(length * float(rng.uniform(0.055, 0.10)), 22.0)
        ang = float(rng.uniform(0, 2 * np.pi))
        px = float(rng.uniform(0, CARD_W))
        py = float(rng.uniform(0, CARD_H))
        # Anchor biased toward an end, so a giant usually shows a cap
        # fragment rather than an anonymous stretch of mid-shaft.
        t = (
            float(rng.choice([-1, 1]))
            * (0.5 - 0.35 * float(rng.random()) ** 1.5)
            * length
        )
        bone(
            px - t * np.cos(ang),
            py - t * np.sin(ang),
            ang,
            length,
            w,
            0.6 + 0.8 * u,
            3,
        )
    if rng.random() < 0.6:
        spine(
            float(rng.uniform(10, CARD_W - 10)),
            float(rng.uniform(8, CARD_H - 8)),
            float(rng.uniform(0, 2 * np.pi)),
            int(rng.integers(5, 9)),
            float(rng.uniform(1.2, 2.2)),
            2,
        )
    # Recurrence: small bone litter.
    for _ in range(int(round(rec * rng.integers(2, 6)))):
        bone(
            float(rng.uniform(0, CARD_W)),
            float(rng.uniform(0, CARD_H)),
            float(rng.uniform(0, 2 * np.pi)),
            float(rng.uniform(5, 11)),
            float(rng.uniform(0.7, 1.2)),
            0.6,
            2,
        )


def neural_field(ax, rng, pal, mods):
    """Neural growth: somata sprouting tapering dendrites by recursive
    wander, one long axon linking cells, boutons dotting the tips.
    recurrence sets arborization depth, chaos bends the growth cones."""
    chaos, rec = mods["chaos"], mods["recurrence"]
    stroke = _make_stroke(ax, rng, mods)
    t_c = np.linspace(0, 2 * np.pi, 30)
    somata = []

    def dendrite(x0, y0, ang, length, lw, depth):
        n = 40
        t = np.linspace(0, length, n)
        a = ang + (0.4 + 1.4 * chaos) * np.cumsum(rng.standard_normal(n)) * length / (
            n * 14
        )
        x = x0 + np.cumsum(np.cos(a)) * (length / n)
        y = y0 + np.cumsum(np.sin(a)) * (length / n)
        stroke(x, y, pal["stroke"], lw, 3)
        if depth > 0 and length > 2.5:
            for frac in (0.35, 0.65, 0.95):
                i = int(frac * (n - 1))
                if rng.random() < 0.8:
                    dendrite(
                        x[i],
                        y[i],
                        float(a[i])
                        + float(rng.uniform(0.35, 1.0)) * rng.choice([-1, 1]),
                        length * float(rng.uniform(0.45, 0.65)),
                        lw * 0.65,
                        depth - 1,
                    )
        else:
            # Synaptic bouton at the growth tip.
            r = float(rng.uniform(0.25, 0.55))
            stroke(
                x[-1] + r * np.cos(t_c), y[-1] + r * np.sin(t_c), pal["stroke"], lw, 3
            )

    def soma(cx, cy, r):
        somata.append((cx, cy))
        wob = 1 + 0.18 * _fbm1(rng, len(t_c), octaves=2)
        ax.fill(
            cx + r * wob * np.cos(t_c),
            cy + r * wob * np.sin(t_c),
            facecolor=pal["fill"],
            edgecolor=pal["stroke"],
            lw=0.8,
            zorder=3,
        )
        arms = int(rng.integers(4, 7))
        a0 = float(rng.uniform(0, 2 * np.pi))
        for k in range(arms):
            ang = a0 + 2 * np.pi * k / arms + float(rng.uniform(-0.3, 0.3))
            dendrite(
                cx + r * np.cos(ang),
                cy + r * np.sin(ang),
                ang,
                float(rng.uniform(6, 16)),
                0.8,
                1 + int(round(2 * rec)),
            )

    for _ in range(int(rng.integers(2, 4))):
        soma(
            float(rng.uniform(8, CARD_W - 8)),
            float(rng.uniform(6, CARD_H - 6)),
            float(rng.uniform(1.6, 3.2)),
        )
    # Axons: long myelinated runs between somata, faint and beaded.
    for (x0, y0), (x1, y1) in zip(somata, somata[1:]):
        n = 120
        t = np.linspace(0, 1, n)
        sag = float(rng.uniform(4, 14)) * np.sin(np.pi * t)
        dx, dy = x1 - x0, y1 - y0
        norm = max(np.hypot(dx, dy), 1e-6)
        x = x0 + t * dx - sag * dy / norm + chaos * 2.0 * _fbm1(rng, n)
        y = y0 + t * dy + sag * dx / norm + chaos * 2.0 * _fbm1(rng, n)
        stroke(x, y, pal["faint"], 1.1, 2)


def matrix_field(ax, rng, pal, mods):
    """Matrix geometry: a warped lattice under a random linear map, framed
    by oversized brackets - entries dot the intersections, some cells fill
    solid. chaos shears the basis, recurrence nests a sub-matrix."""
    chaos, rec = mods["chaos"], mods["recurrence"]
    # Rigid geometry: full per-stroke wobble disassembles shapes whose
    # identity lives in stroke alignment (shafts+knobs, strands+rungs).
    mods = {**mods, "wobble": 0.35 * mods["wobble"]}
    stroke = _make_stroke(ax, rng, mods)

    def lattice(cx, cy, cols, rows, cell, lw, depth, z):
        ang = GOLDEN * int(rng.integers(0, 13))
        # Random basis: rotation + shear/scale rising with chaos.
        ca, sa = np.cos(ang), np.sin(ang)
        sh = chaos * float(rng.uniform(-0.6, 0.6))
        sc = 1 + chaos * float(rng.uniform(-0.25, 0.25))
        ex = np.array([ca, sa]) * cell
        ey = np.array([-sa * sc + ca * sh, ca * sc + sa * sh]) * cell
        org = np.array([cx, cy]) - (cols / 2) * ex - (rows / 2) * ey

        def P(i, j):
            return org + i * ex + j * ey

        n = 40
        for i in range(cols + 1):
            pts = np.array([P(i, j) for j in np.linspace(0, rows, n)])
            stroke(
                pts[:, 0],
                pts[:, 1],
                pal["stroke" if i % 4 == 0 else "faint"],
                lw * (1.4 if i % 4 == 0 else 1.0),
                z,
            )
        for j in range(rows + 1):
            pts = np.array([P(i, j) for i in np.linspace(0, cols, n)])
            stroke(
                pts[:, 0],
                pts[:, 1],
                pal["stroke" if j % 4 == 0 else "faint"],
                lw * (1.4 if j % 4 == 0 else 1.0),
                z,
            )
        # Entries: dots at intersections, occasional solid cells.
        t_dot = np.linspace(0, 2 * np.pi, 12)
        for i in range(cols):
            for j in range(rows):
                r = rng.random()
                if r < 0.12:
                    quad = np.array(
                        [P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)]
                    )
                    ax.fill(
                        quad[:, 0],
                        quad[:, 1],
                        facecolor=pal["fill"],
                        edgecolor=pal["stroke"],
                        lw=0.4,
                        zorder=z,
                    )
                elif r < 0.4:
                    c = P(i + 0.5, j + 0.5)
                    rr = cell * 0.07
                    stroke(
                        c[0] + rr * np.cos(t_dot),
                        c[1] + rr * np.sin(t_dot),
                        pal["stroke"],
                        lw,
                        z,
                    )
        # Oversized brackets along the left/right columns.
        for side, i in ((-1, 0), (1, cols)):
            lip = 0.35 * cell * side
            col = np.array([P(i, j) for j in np.linspace(0, rows, n)])
            stroke(col[:, 0], col[:, 1], pal["stroke"], lw * 2.2, z)
            for j in (0, rows):
                tip = np.array([P(i, j), P(i, j) - lip * ex / cell])
                stroke(tip[:, 0], tip[:, 1], pal["stroke"], lw * 2.2, z)
        if depth > 0:
            sub = P(float(rng.uniform(0, cols)), float(rng.uniform(0, rows)))
            lattice(
                sub[0],
                sub[1],
                max(2, cols // 2),
                max(2, rows // 2),
                cell / PHI,
                lw * 0.8,
                depth - 1,
                max(z - 1, 2),
            )

    lattice(
        float(rng.uniform(0.3, 0.7)) * CARD_W,
        float(rng.uniform(0.3, 0.7)) * CARD_H,
        int(rng.integers(4, 8)),
        int(rng.integers(3, 6)),
        float(rng.uniform(6, 11)),
        0.55,
        int(rec > 0.4),
        3,
    )


def helix_field(ax, rng, pal, mods):
    """DNA: double helices crossing the card - two phase-offset strands
    woven by depth (the back pass thins at each crossing), rungs for base
    pairs. chaos denatures the geometry, recurrence buds child helices."""
    chaos, rec = mods["chaos"], mods["recurrence"]
    # Rigid geometry: full per-stroke wobble disassembles shapes whose
    # identity lives in stroke alignment (shafts+knobs, strands+rungs).
    mods = {**mods, "wobble": 0.35 * mods["wobble"]}
    stroke = _make_stroke(ax, rng, mods)

    def helix(cx, cy, ang, length, amp, pitch, lw, depth, z):
        n = 500
        ca, sa = np.cos(ang), np.sin(ang)
        s = np.linspace(-length / 2, length / 2, n)
        ph = float(rng.uniform(0, 2 * np.pi))
        wander = chaos * amp * 0.8 * _fbm1(rng, n)
        amp_s = amp * (1 + 0.25 * chaos * _fbm1(rng, n, octaves=3))
        for k, phase in enumerate((0.0, np.pi)):
            w = np.sin(2 * np.pi * s / pitch + ph + phase)
            depth_cue = np.cos(2 * np.pi * s / pitch + ph + phase)
            d = amp_s * w + wander
            x, y = cx + s * ca - d * sa, cy + s * sa + d * ca
            # Split the strand into front (thick) and back (thin) arcs so
            # the pair reads as woven, not just two sine waves.
            front = depth_cue >= 0
            for mask, width in ((front, lw * 1.5), (~front, lw * 0.6)):
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    continue
                for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
                    stroke(x[run], y[run], pal["stroke"], width, z)
        # Base-pair rungs between the strands, skipped near crossings.
        n_rungs = int(length / pitch * float(rng.uniform(4, 6)))
        for i in range(n_rungs):
            si = -length / 2 + (i + 0.5) * length / n_rungs
            w = np.sin(2 * np.pi * si / pitch + ph)
            if abs(w) < 0.25:
                continue
            di = amp * w
            base = np.array([cx + si * ca, cy + si * sa])
            p0 = base + np.array([-di * sa, di * ca])
            p1 = base - np.array([-di * sa, di * ca])
            mid = (p0 + p1) / 2
            for pa, pb in ((p0, mid), (mid, p1)):
                stroke(
                    np.linspace(pa[0], pb[0], 8),
                    np.linspace(pa[1], pb[1], 8),
                    pal["faint"],
                    lw * 0.9,
                    z,
                )
        if depth > 0:
            # A child buds off one end, rotated by the golden angle.
            end = 1 if rng.random() < 0.5 else -1
            helix(
                cx + end * length / 2 * ca,
                cy + end * length / 2 * sa,
                ang + GOLDEN,
                length / PHI,
                amp / PHI**0.5,
                pitch / PHI**0.5,
                lw * 0.8,
                depth - 1,
                max(z - 1, 2),
            )

    for _ in range(int(rng.integers(1, 3))):
        helix(
            float(rng.uniform(0.2, 0.8)) * CARD_W,
            float(rng.uniform(0.2, 0.8)) * CARD_H,
            GOLDEN * int(rng.integers(0, 13)),
            float(rng.uniform(60, 130)),
            float(rng.uniform(3.0, 6.5)),
            float(rng.uniform(12, 26)),
            0.8,
            int(round(rec * 2)),
            3,
        )


# Leaf species as pointed-kernel mixtures: r(theta) = base + sum of
# L*exp(-(|dth|/w)^p) lobes (p=1 pointed, p=2 rounded), a negative kernel
# notching the petiole, optional marginal teeth. Tips double as vein targets.
# Kernel: (angle, length, width, p). Tuned by eye against real silhouettes.
_LEAF_SPECIES = (
    dict(  # maple: 5 pointed lobes, deep sinuses, toothed margin
        lobes=[
            (0.0, 1.0, 0.42, 1.2),
            (1.13, 0.88, 0.38, 1.2),
            (-1.13, 0.88, 0.38, 1.2),
            (2.27, 0.60, 0.30, 1.2),
            (-2.27, 0.60, 0.30, 1.2),
        ],
        base=0.24,
        notch=0.16,
        teeth=40,
        tooth=0.045,
        aspect=1.0,
    ),
    dict(  # oak: seven rounded finger lobes
        lobes=[
            (0.0, 1.0, 0.20, 2.0),
            (0.70, 0.80, 0.20, 2.0),
            (-0.70, 0.80, 0.20, 2.0),
            (1.40, 0.72, 0.20, 2.0),
            (-1.40, 0.72, 0.20, 2.0),
            (2.09, 0.60, 0.20, 2.0),
            (-2.09, 0.60, 0.20, 2.0),
        ],
        base=0.30,
        notch=0.10,
        teeth=0,
        tooth=0.0,
        aspect=0.95,
    ),
    dict(  # ovate: smooth teardrop with a drawn-out apex
        lobes=[(0.0, 0.92, 1.00, 2.0), (0.0, 1.05, 0.25, 1.0)],
        base=0.30,
        notch=0.06,
        teeth=0,
        tooth=0.0,
        aspect=0.68,
    ),
    dict(  # birch: ovate body, serrated margin
        lobes=[(0.0, 0.90, 0.90, 2.0), (0.0, 1.02, 0.28, 1.0)],
        base=0.28,
        notch=0.08,
        teeth=26,
        tooth=0.035,
        aspect=0.72,
    ),
)


def leaf_field(ax, rng, pal, mods):
    """Foliage: leaves grown from one polar function - a base disc plus
    pointed angular kernels per lobe - with species presets (maple, oak,
    ovate, birch), veins traced to the lobe tips, and the bones' size
    spectrum: the occasional giant shows only a cropped fragment."""
    chaos, rec = mods["chaos"], mods["recurrence"]
    # Outline and veins must stay mutually aligned.
    mods = {**mods, "wobble": 0.35 * mods["wobble"]}
    stroke = _make_stroke(ax, rng, mods)

    def leaf(cx, cy, ang, size, sp, lw, z):
        n = 360
        th = np.linspace(-np.pi, np.pi, n)
        r = np.full(n, sp["base"])
        for a, L, w, p in sp["lobes"]:
            d = np.abs(np.angle(np.exp(1j * (th - a))))
            r = r + (L - sp["base"]) * np.exp(-((d / w) ** p))
        d = np.abs(np.angle(np.exp(1j * (th - np.pi))))
        r = r - sp["notch"] * np.exp(-((d / 0.35) ** 2))
        if sp["teeth"]:
            saw = 2 * np.abs((sp["teeth"] * th / (2 * np.pi)) % 1 - 0.5)
            r = r * (1 - sp["tooth"] * saw)
        r = np.maximum(r + chaos * 0.05 * _fbm1(rng, n, octaves=3), 0.02)

        ca, sa = np.cos(ang), np.sin(ang)
        asp = sp["aspect"]

        def to_xy(rr, tt):
            u, v = rr * np.cos(tt), rr * asp * np.sin(tt)
            return cx + size * (u * ca - v * sa), cy + size * (u * sa + v * ca)

        x, y = to_xy(r, th)
        if rng.random() < 0.3:
            ax.fill(x, y, facecolor=pal["fill"], edgecolor="none", zorder=z)
        stroke(x, y, pal["stroke"], lw, z)

        # Petiole + veins: from the basal junction to each major lobe tip,
        # gently bowed. The lobe list already knows where the tips are.
        bx, by = to_xy(np.array([r[0]]), np.array([np.pi]))
        sx, sy = to_xy(np.array([r[0] + 0.30]), np.array([np.pi]))
        stroke(
            np.linspace(bx[0], sx[0], 8),
            np.linspace(by[0], sy[0], 8),
            pal["stroke"],
            lw,
            z,
        )
        if size > 5.5:
            for a, L, w, p in sp["lobes"]:
                if L < 0.5 or w > 0.95:
                    continue  # minor lobe / body kernel, not a vein target
                i = int((a + np.pi) / (2 * np.pi) * (n - 1))
                tx, ty = to_xy(np.array([r[i] * 0.92]), np.array([a]))
                t = np.linspace(0, 1, 24)
                bow = 0.06 * size * np.sin(np.pi * t) * float(rng.uniform(-1, 1))
                vx = bx[0] + t * (tx[0] - bx[0]) - bow * sa
                vy = by[0] + t * (ty[0] - by[0]) + bow * ca
                stroke(vx, vy, pal["faint" if L < 0.9 else "stroke"], lw * 0.7, z)

    # Faint drift lines behind the foliage.
    for _ in range(int(rng.integers(2, 4))):
        x, y = _line_strand(rng, chaos, mods["arc"], mods["overshoot"])
        stroke(x, y, pal["faint"], 0.5, 1)

    # Size spectrum biased large, pass-through anchored like the bones;
    # recurrence thickens the scatter of small companions.
    n_leaves = 2 + int(round(rec * 4)) + int(rng.integers(0, 3))
    for _ in range(n_leaves):
        u = float(rng.uniform(0, 1)) ** 0.65
        size = 4.0 * PHI ** (5.2 * u)
        sp = _LEAF_SPECIES[int(rng.integers(len(_LEAF_SPECIES)))]
        ang = float(rng.uniform(0, 2 * np.pi))
        px = float(rng.uniform(0, CARD_W))
        py = float(rng.uniform(0, CARD_H))
        t = float(rng.uniform(0.1, 0.9)) * size
        leaf(
            px - t * np.cos(ang),
            py - t * np.sin(ang),
            ang,
            size,
            sp,
            0.55 + 0.6 * u,
            3,
        )


def _noise2(rng, gx=5, gy=4):
    """Smooth 2D value noise over the card, as a callable (x, y) -> value."""
    g = rng.standard_normal((gy + 1, gx + 1))

    def f(x, y):
        u = np.clip(np.asarray(x) / CARD_W, 0, 1) * gx
        v = np.clip(np.asarray(y) / CARD_H, 0, 1) * gy
        i0 = np.minimum(np.floor(u).astype(int), gx - 1)
        j0 = np.minimum(np.floor(v).astype(int), gy - 1)
        fu, fv = u - i0, v - j0
        fu = fu * fu * (3 - 2 * fu)
        fv = fv * fv * (3 - 2 * fv)
        return (
            g[j0, i0] * (1 - fu) * (1 - fv)
            + g[j0, i0 + 1] * fu * (1 - fv)
            + g[j0 + 1, i0] * (1 - fu) * fv
            + g[j0 + 1, i0 + 1] * fu * fv
        )

    return f


def chimera_field(ax, rng, pal, mods):
    """TWO styles sharing one card as discrete territories: pool candidate
    samples of the other fields, vote (counts become dominance), render
    both winners, then keep each stroke only where its style's organic
    noise mask wins. The boundary is a hard fbm contour, so each region
    reads as a pure, unmixed sample of its style; only high chaos
    occasionally earns a narrow seam where the two interleave."""
    names = [n for n in PROJECTION_REGISTRY if n != "chimera"]
    if not names:  # nothing to blend (e.g. a stripped-down registry)
        return
    # Candidate pooling + vote: 5 draws, counts weight the winners.
    votes = np.bincount(rng.integers(0, len(names), size=5), minlength=len(names))
    chosen = [int(i) for i in np.argsort(-votes) if votes[i] > 0][:2]
    if len(chosen) < 2 and len(names) > 1:
        # Unanimous vote still pairs with one dissenter.
        extra = int(rng.integers(0, len(names)))
        while extra in chosen:
            extra = int(rng.integers(0, len(names)))
        chosen.append(extra)

    chaos = mods["chaos"]
    sharp = 8.0  # near-argmax weights: territories, not gradients
    # Pure discreteness by default; only high chaos sometimes opens a
    # narrow seam where both styles survive.
    margin = 0.18 * chaos if rng.random() < 0.1 + 0.3 * chaos else 0.0
    # Few, large noise cells: coherent territories instead of confetti.
    masks = [_noise2(rng, gx=3, gy=2) for _ in chosen]
    bias = 0.5 * np.log1p(votes[chosen])  # pooled votes bias dominance

    def weights(x, y):
        m = np.stack([bias[k] + masks[k](x, y) for k in range(len(chosen))])
        e = np.exp(sharp * (m - m.max(axis=0, keepdims=True)))
        return e / e.sum(axis=0, keepdims=True)

    styled = []  # (style index, artist)
    for k, ci in enumerate(chosen):
        n_lines, n_patches = len(ax.lines), len(ax.patches)
        sub = np.random.default_rng(int(rng.integers(2**62)))
        PROJECTION_REGISTRY[names[ci]](ax, sub, pal, dict(mods))
        styled += [(k, a) for a in list(ax.lines)[n_lines:]]
        styled += [(k, a) for a in list(ax.patches)[n_patches:]]

    from matplotlib.lines import Line2D

    for k, artist in styled:
        if not isinstance(artist, Line2D):
            # Filled shapes live or die whole - no fading; a kept shape
            # looks exactly as its pure style would draw it.
            verts = artist.get_xy()
            if float(np.mean(weights(verts[:, 0], verts[:, 1])[k])) < 0.5:
                artist.remove()
            continue
        x, y = np.asarray(artist.get_xdata()), np.asarray(artist.get_ydata())
        if len(x) < 2:
            continue
        w = weights(x, y)
        keep = w[k] >= w.max(axis=0) - margin
        if keep.all():
            continue
        idx = np.where(keep)[0]
        runs = [
            r for r in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if len(r) > 1
        ]
        color, lw, z, gid = (
            artist.get_color(),
            artist.get_linewidth(),
            artist.get_zorder(),
            artist.get_gid(),
        )
        artist.remove()
        for r in runs:
            # Full width to the cut: strokes stop at the territory line
            # instead of dissolving, keeping each side's look intact.
            (line,) = ax.plot(
                x[r[0] : r[-1] + 1],
                y[r[0] : r[-1] + 1],
                color=color,
                lw=lw,
                zorder=z,
            )
            if gid:
                line.set_gid(gid)


PROJECTION_REGISTRY = {
    "strands": strand_field,
    "shatter": shatter_field,
    "lightning": lightning_field,
    "waves": wave_field,
    "fibonacci": fibonacci_field,
    "glyphs": glyph_field,
    "rubble": rubble_field,
    "splatter": splatter_field,
    "snowflake": snowflake_field,
    "flora": flora_field,
    "bones": bone_field,
    "neural": neural_field,
    "matrix": matrix_field,
    "helix": helix_field,
    "leaves": leaf_field,
    "chimera": chimera_field,
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
        "symmetry": float(rng.uniform(0, 1)) * float(rng.random() < 0.35),
    }
    for k, v in (mods or {}).items():
        if v is not None:
            sampled[k] = min(max(float(v), 0.0), 1.0)
    return sampled


def _draw_card(ax, side, seed, theme, hue, authors, donations, run_hash, mods=None):
    from matplotlib import patheffects
    from matplotlib.patches import Rectangle

    pal = _palette(theme, hue)
    if side == "back":
        # Inverted two-tone: paper-colored strokes on an accent ground.
        paper, accent = pal["paper"], pal["stroke"]
        pal = {
            **pal,
            "paper": accent,
            "stroke": paper,
            "ink": paper,
            "faint": paper,
            "fill": accent,
        }
    rng = np.random.default_rng([seed, 0 if side == "front" else 1])

    ax.set_xlim(0, CARD_W)
    ax.set_ylim(0, CARD_H)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(
        Rectangle(
            (0, 0), CARD_W, CARD_H, facecolor=pal["border"], edgecolor="none", zorder=0
        )
    )
    inner = Rectangle(
        (BORDER_MM, BORDER_MM),
        CARD_W - 2 * BORDER_MM,
        CARD_H - 2 * BORDER_MM,
        facecolor=pal["paper"],
        edgecolor="none",
        zorder=0.5,
    )
    ax.add_patch(inner)

    names = list(PROJECTION_REGISTRY)
    field = PROJECTION_REGISTRY[names[int(rng.integers(len(names)))]]
    sampled = _sample_mods(rng, mods)
    field(ax, rng, pal, sampled)
    # Clip geometry to the inner panel so the border stays clean;
    # overshoot-tagged strands get the full card and cross the frame.
    bleed = Rectangle((0, 0), CARD_W, CARD_H, facecolor="none", edgecolor="none")
    ax.add_patch(bleed)
    for artist in list(ax.lines) + list(ax.patches) + list(ax.collections):
        if artist is not inner and artist is not bleed:
            artist.set_clip_path(bleed if artist.get_gid() == "overshoot" else inner)

    halo = [patheffects.withStroke(linewidth=2.2, foreground=pal["paper"])]
    m = BORDER_MM + 4
    # Sometimes the text block mirrors to the top of the card.
    flip_v = rng.random() < 0.35
    vy = (lambda y: CARD_H - y) if flip_v else (lambda y: y)
    va = "top" if flip_v else "baseline"
    if side == "front":
        for i, author in enumerate(authors):
            ax.text(
                m,
                vy(m + 6 * (len(authors) - 1 - i)),
                _caps(rng, author, [0.5, 0.35, 0.15]),
                fontsize=11,
                color=pal["ink"],
                family="DejaVu Sans",
                weight="bold" if i == 0 else "normal",
                va=va,
                path_effects=halo,
                zorder=5,
            )
        if rng.random() < 0.6:  # a signature stroke above the name, sometimes
            chaos = sampled["chaos"]
            n = 140
            xs = np.linspace(m, m + float(rng.uniform(16, 30)), n)
            env = np.sin(np.linspace(0, np.pi, n)) ** 0.6  # pen taper
            teeth = np.tanh(
                6 * np.sin(2 * np.pi * np.arange(n) / float(rng.uniform(15, 40)))
            )
            d = (
                (1 - chaos) * 0.5 * teeth + chaos * 1.4 * _fbm1(rng, n, octaves=4)
            ) * env
            ax.plot(
                xs,
                vy(m + 6 * len(authors) + 1.0) + d,
                color=pal["stroke"],
                lw=1.2,
                zorder=5,
            )
    else:
        if donations:
            ax.text(
                CARD_W / 2,
                CARD_H / 2,
                _caps(rng, donations, [0.1, 0.1, 0.8]),
                fontsize=7.5,
                color=pal["ink"],
                family="DejaVu Sans Mono",
                ha="center",
                va="center",
                path_effects=halo,
                zorder=5,
            )
        if run_hash:
            ax.text(
                CARD_W - m,
                vy(m),
                run_hash,
                fontsize=5.5,
                color=pal["stroke"],
                family="DejaVu Sans Mono",
                ha="right",
                va=va,
                path_effects=halo,
                zorder=5,
            )


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
            ax.plot(
                [px + sx * gap, px + sx * (gap + length)], [py, py], color=color, lw=0.5
            )
            ax.plot(
                [px, px], [py + sy * gap, py + sy * (gap + length)], color=color, lw=0.5
            )


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


def render_card(
    side,
    seed,
    theme,
    hue,
    authors,
    donations,
    run_hash,
    fmt="svg",
    chaos=None,
    mods=None,
):
    """One bare card at exact size (preview or print-and-cut)."""
    fig = _new_fig(CARD_W, CARD_H, "none")
    ax = _mm_axes(fig, 0, 0, CARD_W, CARD_H, CARD_W, CARD_H)
    _draw_card(
        ax,
        side,
        seed,
        theme,
        hue,
        authors,
        donations,
        run_hash,
        _merge_mods(mods, chaos),
    )
    return _save(fig, fmt)


def _page(cells, side, seed_for, theme, hue, authors, donations, run_hash, mods=None):
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
        _draw_card(
            ax, side, seed_for(i), theme, hue, authors, donations, run_hash, mods
        )
    page.text(
        PAGE_W / 2,
        8,
        PRINT_NOTE,
        fontsize=6,
        color="#666666",
        ha="center",
        va="center",
        wrap=True,
    )
    page.text(
        PAGE_W / 2,
        PAGE_H - 8,
        f"Praxis business card - {side} - seed {seed_for(0)}",
        fontsize=6,
        color="#666666",
        ha="center",
        va="center",
    )
    # Paper guidance runs sideways in the left/right margins.
    page.text(
        6,
        PAGE_H / 2,
        PAPER_NOTE,
        fontsize=6,
        color="#666666",
        ha="center",
        va="center",
        rotation=90,
    )
    page.text(
        PAGE_W - 6,
        PAGE_H / 2,
        PAPER_LINK_NOTE,
        fontsize=6,
        color="#666666",
        ha="center",
        va="center",
        rotation=270,
        url=PAPER_URL,
    )
    return _save(fig, "pdf")


def _cell(row, col):
    """Avery 28371 cell origin (mm, bottom-left), row 0 at the page top."""
    return (SHEET_X0 + col * CARD_W, PAGE_H - SHEET_Y0 - (row + 1) * CARD_H)


def render_single_pdf(
    side, seed, theme, hue, authors, donations, run_hash, chaos=None, mods=None
):
    # The top-left Avery cell, so a single card prints onto the same stock;
    # the back mirrors to the other column for long-edge duplex.
    col = 0 if side == "front" else SHEET_COLS - 1
    return _page(
        [_cell(0, col)],
        side,
        lambda i: seed,
        theme,
        hue,
        authors,
        donations,
        run_hash,
        _merge_mods(mods, chaos),
    )


def render_sheet_pdf(
    side, seed, theme, hue, authors, donations, run_hash, chaos=None, mods=None
):
    """Full Avery 28371 imposition (10-up). Back pages mirror columns for
    long-edge duplex."""
    cells, seeds = [], []
    for row in range(SHEET_ROWS):
        for col in range(SHEET_COLS):
            draw_col = (SHEET_COLS - 1 - col) if side == "back" else col
            cells.append(_cell(row, draw_col))
            seeds.append(seed + row * SHEET_COLS + col)
    return _page(
        cells,
        side,
        lambda i: seeds[i],
        theme,
        hue,
        authors,
        donations,
        run_hash,
        _merge_mods(mods, chaos),
    )
