extends RefCounted
class_name Layer

## Layer - the registry of reusable visual components.
##
## The visual sibling of [Primitives] (the force registry). Where Primitives lets a
## scene compose *physics* by key, Layer lets a scene compose *appearance* by key: a
## drift of snow, a bank of rolling fog, a swarm of fireflies, a field of stars - each
## a small self-contained class that seeds itself, advances on the audio, and draws
## itself onto the scene's canvas. The point is **integration**: the same snow that is
## a scene on its own also falls over a city or a hillside, because it is a component,
## not bespoke per-scene code.
##
## A scene composes layers through [GhostScene] helpers (add_layer / update_layers /
## draw_layers); see snowfall.gd, fog_bank.gd, etc. Layers draw in ghost's centred
## unit-fraction space (multiply by the scene's unit() at draw time) and are handed the
## visible half-extents each frame, so they fill any aspect ratio / fullscreen / 4K
## without showing an edge.
##
## Two methods to override:
##   update(f, dt, half) - advance state; `half` is the visible half-size in unit
##                         fractions (Vector2), so a layer knows where the frame ends.
##   draw(ci, u)         - draw onto CanvasItem `ci` at unit `u` (pixels per unit).

const FlowField := preload("res://scripts/flow.gd")


class Base:
	extends RefCounted
	var cfg: Dictionary = {}
	var rng := RandomNumberGenerator.new()
	var half := Vector2(0.9, 0.55)     ## visible half-extent (unit fractions); set each update
	var t := 0.0

	func _init(seed_rng: RandomNumberGenerator, c: Dictionary = {}) -> void:
		cfg = c
		# Derive a private, deterministic stream from the scene's build rng so a layer's
		# own runtime randomness (gusts, retargets) is reproducible per session.
		rng.seed = seed_rng.randi()

	func num(k: String, d: float) -> float:
		return float(cfg.get(k, d))
	func flag(k: String, d: bool) -> bool:
		return bool(cfg.get(k, d))

	## Base hue this layer themes around (a scene can tint a layer to its palette).
	func hue() -> float:
		return num("hue", 0.58)

	## Depth band for compositing onto a geometry scene: "back" (drawn behind the
	## geometry, e.g. stars in the sky) or "front" (drawn over it, e.g. falling snow).
	## Standalone weather scenes ignore this and just draw everything in add order.
	func z() -> String:
		return String(cfg.get("z", "front"))

	func update(_f: AudioFeatures, dt: float, h: Vector2) -> void:
		half = h
		t += dt

	func draw(_ci: CanvasItem, _u: float) -> void:
		pass


# ---------------------------------------------------------------------------------
# Shared drawing helpers (used by several layers and by scenes drawing flakes).
# ---------------------------------------------------------------------------------

## A soft glowing disc: concentric circles, bright tight centre to a wide faint halo.
static func glow(ci: CanvasItem, c: Vector2, radius: float, color: Color, layers := 6) -> void:
	for i in layers:
		var frac := float(i) / float(layers - 1)
		var r := radius * (1.0 - 0.82 * frac)
		var al := color.a * (0.05 + 0.30 * frac)
		ci.draw_circle(c, r, Color(color.r, color.g, color.b, al))


## A wide very soft disc - one puff of fog / haze. The alpha fades smoothly to zero at
## the rim (the outermost ring is invisible), so blobs read as soft gradients rather
## than nested hard-edged circles - which banded visibly over dark backgrounds.
static func soft_blob(ci: CanvasItem, c: Vector2, radius: float, color: Color, layers := 8) -> void:
	for i in layers:
		var frac := float(i) / float(layers - 1)     # 0 outer .. 1 inner
		var r := radius * (1.0 - 0.78 * frac)
		var al := color.a * smoothstep(0.0, 1.0, frac)
		ci.draw_circle(c, r, Color(color.r, color.g, color.b, al))


# A reusable soft radial-gaussian sprite (white, alpha falling smoothly to nothing at the edge),
# built once. Drawing it tinted + scaled is a TRUE soft particle: smooth (no concentric-ring
# banding the stacked-circle blob shows) and cheap (one textured quad). The base primitive for
# fire tongues, cloud puffs, fog, and soft glows.
static var _puff_tex: Texture2D = null
static func puff_texture() -> Texture2D:
	if _puff_tex == null:
		var s := 64
		var img := Image.create(s, s, false, Image.FORMAT_RGBA8)
		var hf := float(s) * 0.5
		for y in s:
			for x in s:
				var dx := (float(x) + 0.5 - hf) / hf
				var dy := (float(y) + 0.5 - hf) / hf
				var a := clampf(exp(-3.6 * (dx * dx + dy * dy)) - 0.03, 0.0, 1.0)   # gaussian -> 0 at rim
				img.set_pixel(x, y, Color(1.0, 1.0, 1.0, a))
		_puff_tex = ImageTexture.create_from_image(img)
	return _puff_tex


## Draw a soft gaussian puff of `radius` at `c`, tinted (and alpha-scaled) by `color`.
static func puff(ci: CanvasItem, c: Vector2, radius: float, color: Color) -> void:
	ci.draw_texture_rect(puff_texture(),
		Rect2(c - Vector2(radius, radius), Vector2(radius * 2.0, radius * 2.0)), false, color)


## Drifting gaussian "squall" centres in normalized [-1, 1] frame space - the wandering hot-spots
## where a harmonic-coverage veil (snow, rain, ...) piles heaviest. Deterministic from time so it
## never flickers; shared so every obscuring effect moves the same way.
static func squall_centers(time: float) -> Array:
	return [
		Vector2(sin(time * 0.13) * 0.6, cos(time * 0.11) * 0.4),
		Vector2(sin(time * 0.09 + 2.0) * 0.7, sin(time * 0.15 + 1.0) * 0.5),
		Vector2(cos(time * 0.07 + 4.0) * 0.5, cos(time * 0.12 + 3.0) * 0.45),
	]


## Harmonic "coverage" drive in [0, 1]: how thick an obscuring veil should be right now, from
## musical intensity (energy / bass / spectral flux / beat). Pair with an EMA in the caller
## (fast swell, slow clear) so squalls build and fade rather than strobe.
static func coverage_drive(f: AudioFeatures) -> float:
	return clampf(0.25 * f.energy + 0.55 * f.bass + 1.6 * f.flux + 0.2 * f.beat, 0.0, 1.0)


## An ellipse outline as a point ring (for a wobbling bubble rim, etc.).
static func ellipse(c: Vector2, rx: float, ry: float, segs := 22) -> PackedVector2Array:
	var pts := PackedVector2Array()
	for i in segs:
		var th := TAU * float(i) / float(segs)
		pts.append(c + Vector2(cos(th) * rx, sin(th) * ry))
	return pts


## Draw a procedural six-fold snow-crystal dendrite, centred at `c`, of `radius` pixels,
## rotated by `ang`. Generated, not hard-coded: a main arm per fold with a few side
## branches at the natural 60° dendrite angle, their length and placement varied by
## `shape` (0..1) so every flake differs. This is the elegant-procedural answer to the
## old hard-coded Koch stars (feedback): symmetry + sampled detail, never a fixed glyph.
static func draw_flake(ci: CanvasItem, c: Vector2, radius: float, ang: float,
		color: Color, width := 1.5, folds := 6, shape := 0.5) -> void:
	var branch_at := 0.42 + 0.30 * shape          # how far out the side branches sit
	var branch_len := 0.26 + 0.34 * shape          # side-branch length (fraction of arm)
	var tip := 0.30 + 0.30 * (1.0 - shape)         # little terminal V at the arm's end
	for s in folds:
		var a := ang + TAU * float(s) / float(folds)
		var dir := Vector2(cos(a), sin(a))
		var tipv := c + dir * radius
		ci.draw_line(c, tipv, color, width, true)
		# Two symmetric side branches partway along the arm.
		var bp := c + dir * (radius * branch_at)
		var ll := radius * branch_len
		for sgn in [-1.0, 1.0]:
			var ba: float = a + sgn * deg_to_rad(60.0)
			ci.draw_line(bp, bp + Vector2(cos(ba), sin(ba)) * ll, color, width * 0.8, true)
		# A small terminal fork at the very tip.
		for sgn2 in [-1.0, 1.0]:
			var ta: float = a + sgn2 * deg_to_rad(45.0)
			ci.draw_line(tipv, tipv + Vector2(cos(ta), sin(ta)) * (radius * 0.16 * tip),
				color, width * 0.7, true)


# ---------------------------------------------------------------------------------
# Bed - a full-frame colour wash: a soft vertical gradient plus a few slow colour
# pools that breathe with the spectrum. The "colours underneath / inside" that fog,
# snow, or stars sit on; on its own it is a calm aurora-of-colour ground.
# ---------------------------------------------------------------------------------
class Bed:
	extends Base
	var _pools: Array = []
	var _hue := 0.0
	var _energy := 0.0

	func _init(seed_rng: RandomNumberGenerator, c: Dictionary = {}) -> void:
		super(seed_rng, c)
		_hue = num("hue", seed_rng.randf())
		var n := int(num("pools", 3))
		for i in n:
			_pools.append({
				# Spread wider (partly off-frame) and larger, so the pools read as a soft
				# wash rather than discrete giant lights sitting at the edges.
				"home": Vector2(seed_rng.randf_range(-0.95, 0.95), seed_rng.randf_range(-0.6, 0.6)),
				"size": seed_rng.randf_range(0.7, 1.25),
				"hue_off": seed_rng.randf_range(-0.12, 0.16),
				"band": seed_rng.randf(),
				"px": seed_rng.randf() * TAU, "py": seed_rng.randf() * TAU,
				"rx": seed_rng.randf_range(0.05, 0.12), "ry": seed_rng.randf_range(0.04, 0.10),
			})

	func update(f: AudioFeatures, dt: float, h: Vector2) -> void:
		super(f, dt, h)
		_hue += dt * 0.012
		_energy = lerpf(_energy, f.energy, 1.0 - exp(-3.0 * dt))

	func draw(ci: CanvasItem, u: float) -> void:
		# Vertical gradient quad (per-vertex colours), darker at the edges.
		var sat: float = num("sat", 0.5)
		var val: float = num("val", 0.32) * (0.7 + 0.5 * _energy)
		var top := Color.from_hsv(fposmod(_hue, 1.0), sat, val * 0.5)
		var midc := Color.from_hsv(fposmod(_hue + 0.04, 1.0), sat, val)
		var bot := Color.from_hsv(fposmod(_hue - 0.06, 1.0), sat * 0.8, val * 0.35)
		# Oversized so a pull-back shot never reveals an edge of the wash.
		var x := half.x * u * 1.5
		var y := half.y * u * 1.5
		# Two stacked quads so the brightest band sits in the middle.
		ci.draw_polygon(PackedVector2Array([
			Vector2(-x, -y), Vector2(x, -y), Vector2(x, 0), Vector2(-x, 0)]),
			PackedColorArray([top, top, midc, midc]))
		ci.draw_polygon(PackedVector2Array([
			Vector2(-x, 0), Vector2(x, 0), Vector2(x, y), Vector2(-x, y)]),
			PackedColorArray([midc, midc, bot, bot]))
		# Slow colour pools breathing with their band.
		for p in _pools:
			var c: Vector2 = (p.home + Vector2(
				p.rx * sin(t * 0.18 + p.px), p.ry * cos(t * 0.15 + p.py))) * u
			var bright: float = _energy * 0.4 + 0.6
			var col := Color.from_hsv(fposmod(_hue + p.hue_off, 1.0), sat + 0.1, val * 1.4 * bright,
				0.10 + 0.12 * bright)
			Layer.soft_blob(ci, c, float(p.size) * u, col, 6)


# ---------------------------------------------------------------------------------
# Fog - big faint blobs rolling across the frame, diffusing whatever is behind them.
# A tempo-kicked swirl gives the bank velocity that decays (feedback: not uniform).
# ---------------------------------------------------------------------------------
class Fog:
	extends Base
	var _blobs: Array = []
	var _swirl := 0.0
	var _swirl_vel := 0.06
	var _beat_prev := 0.0
	var _drift := 0.0

	func _init(seed_rng: RandomNumberGenerator, c: Dictionary = {}) -> void:
		super(seed_rng, c)
		var n := int(num("count", 7))
		for i in n:
			_blobs.append({
				"home": Vector2(seed_rng.randf_range(-0.8, 0.8), seed_rng.randf_range(-0.5, 0.5)),
				"size": seed_rng.randf_range(0.35, 0.8),
				"px": seed_rng.randf() * TAU, "py": seed_rng.randf() * TAU,
				"speed": seed_rng.randf_range(0.04, 0.12) * (1.0 if seed_rng.randf() < 0.5 else -1.0),
			})

	func update(f: AudioFeatures, dt: float, h: Vector2) -> void:
		super(f, dt, h)
		var beat_edge: bool = f.beat > 0.55 and _beat_prev <= 0.55
		_beat_prev = f.beat
		if beat_edge and flag("swirl", true):
			_swirl_vel += 0.5 * (0.4 + f.energy)
		_swirl_vel = 0.05 + (_swirl_vel - 0.05) * exp(-1.2 * dt)
		_swirl += _swirl_vel * dt
		_drift += dt

	func draw(ci: CanvasItem, u: float) -> void:
		var tint_h: float = num("hue", 0.6)
		var sat: float = num("sat", 0.12)
		var alpha: float = num("alpha", 0.045)
		var tint := Color.from_hsv(tint_h, sat, 0.92, alpha)
		for b in _blobs:
			var p: Vector2 = b.home
			# Wrap far off-screen (±2.4, well past the frame + the blob's own radius) so a
			# big blob drifts fully out of view before reappearing on the other side - no
			# pop-in of giant lights at the edges.
			p.x = wrapf(p.x + _drift * float(b.speed), -2.4, 2.4)
			var sway := Vector2(0.06 * sin(t * 0.2 + b.px), 0.05 * cos(t * 0.17 + b.py))
			var pos := Vector2((p.x + sway.x) * half.x, (b.home.y + sway.y) * half.y) \
				.rotated(_swirl * 0.4) * u
			Layer.soft_blob(ci, pos, float(b.size) * u, tint, 5)


# ---------------------------------------------------------------------------------
# Snow - a drift of falling flakes. Soft out-of-focus dots for the many small ones,
# a procedural six-fold dendrite for the few large near ones. Async by construction:
# each flake has its own depth (parallax fall speed), sway phase, and spin. Gusts ride
# the energy/treble. Wraps seamlessly, so it is an endless field at any aspect.
# ---------------------------------------------------------------------------------
class Snow:
	extends Base
	var _flakes: Array = []
	var _fall := 0.0
	var _gust := 0.0
	var _coverage := 0.0     # harmonic-driven density: swells into obscuring squalls, then clears

	func _init(seed_rng: RandomNumberGenerator, c: Dictionary = {}) -> void:
		super(seed_rng, c)
		var n := int(num("count", 140))
		# Most flakes are tiny flecks; squaring the size roll skews the distribution small
		# so a few near ones read bigger without the field looking like a scatter of large
		# shapes (feedback: "many more small flecks, far fewer large").
		var smax: float = num("size", 0.006)
		for i in n:
			var depth := seed_rng.randf_range(0.3, 1.0)        # near flakes are bigger + faster
			var sz := pow(seed_rng.randf(), 2.0)               # skew toward small
			_flakes.append({
				"x": seed_rng.randf(), "y": seed_rng.randf(),  # normalized 0..1 within the frame
				"depth": depth,
				"size": (0.0014 + smax * sz) * (0.5 + depth),
				"sway_amp": seed_rng.randf_range(0.01, 0.05),
				"sway_rate": seed_rng.randf_range(0.3, 0.9),
				"phase": seed_rng.randf() * TAU,
				"spin": seed_rng.randf_range(-1.0, 1.0),
				"shape": seed_rng.randf(),
				# Only the nearest, largest flakes ever become drawn crystals, and rarely -
				# so detailed dendrites are an occasional accent, not the norm.
				"crystal": seed_rng.randf() < num("crystal_frac", 0.06) and depth > 0.8 and sz > 0.6,
			})

	func update(f: AudioFeatures, dt: float, h: Vector2) -> void:
		super(f, dt, h)
		_gust = lerpf(_gust, f.treble * 0.5 + f.beat * 0.3, 1.0 - exp(-4.0 * dt))
		# Harmonic-driven coverage: swells FAST with musical intensity (energy / bass / spectral
		# flux), clears SLOWLY - so the snow thickens into an obscuring squall on a big moment, then
		# thins back out. This is what fades the view toward a heavy whiteout and back.
		_coverage = lerpf(_coverage, Layer.coverage_drive(f), 1.0 - exp(-(3.0 if Layer.coverage_drive(f) > _coverage else 0.8) * dt))
		var speed: float = num("fall", 0.10) * (1.0 + 0.5 * f.energy)
		_fall += speed * dt
		for fl in _flakes:
			fl.y = fposmod(fl.y + speed * float(fl.depth) * dt, 1.0)

	func draw(ci: CanvasItem, u: float) -> void:
		var base_h: float = num("hue", 0.58)
		var sat: float = num("sat", 0.06)
		var centers := Layer.squall_centers(t)
		# Whiteout veil: soft gaussian masses of heavy snow centred on the squalls, swelling with
		# the harmonic coverage - at a peak they merge into a near-white that obscures the view.
		var va := clampf((_coverage - 0.12) * 0.85, 0.0, 0.62)
		if va > 0.004:
			for sc in centers:
				Layer.puff(ci, Vector2(sc.x * half.x, sc.y * half.y) * u,
					(0.55 + 0.45 * _coverage) * maxf(half.x, half.y) * u,
					Color.from_hsv(base_h, sat * 0.5, 1.0, va))
		for fl in _flakes:
			var sway: float = (fl.sway_amp + 0.04 * _gust) * sin(t * fl.sway_rate + fl.phase)
			var px: float = (fl.x * 2.0 - 1.0) * half.x + sway
			var py: float = (fl.y * 2.0 - 1.0) * half.y
			var pos := Vector2(px, py) * u
			var r: float = fl.size * u
			var bright: float = 0.7 + 0.3 * sin(t * 1.3 + fl.phase)
			# Gaussian coverage: a flake fades up where a squall passes over it and the harmonics
			# are loud, so transparency rides BOTH the spatial pattern and the music.
			var nrm := Vector2(fl.x * 2.0 - 1.0, fl.y * 2.0 - 1.0)
			var sw := 0.0
			for sc in centers:
				sw = maxf(sw, exp(-nrm.distance_squared_to(sc) / 0.35))
			var cov := clampf(0.55 + _coverage * (0.35 + 0.85 * sw), 0.0, 1.4)
			var a0: float = clampf(0.5 + 0.4 * float(fl.depth), 0.3, 0.95) * cov
			var col := Color.from_hsv(base_h, sat, 1.0, clampf(a0, 0.0, 1.0))
			if fl.crystal and r > 3.5:
				var ang: float = t * 0.4 * float(fl.spin) + fl.phase
				Layer.draw_flake(ci, pos, r * 2.1, ang,
					Color(col.r, col.g, col.b, col.a * 0.9), maxf(1.0, r * 0.28), 6, fl.shape)
			else:
				ci.draw_circle(pos, r * 1.8, Color(col.r, col.g, col.b, col.a * 0.18))
				ci.draw_circle(pos, r, Color(col.r, col.g, col.b, col.a * bright))


# ---------------------------------------------------------------------------------
# Rain - fast streaks falling at a wind-blown slant, with faint splash flecks near the
# floor. Density and slant ride the audio. Cheap line draws, many of them.
# ---------------------------------------------------------------------------------
class Rain:
	extends Base
	var _drops: Array = []
	var _slant := 0.18

	func _init(seed_rng: RandomNumberGenerator, c: Dictionary = {}) -> void:
		super(seed_rng, c)
		var n := int(num("count", 130))
		for i in n:
			var depth := seed_rng.randf_range(0.4, 1.0)
			_drops.append({
				"x": seed_rng.randf(), "y": seed_rng.randf(), "depth": depth,
				"len": seed_rng.randf_range(0.03, 0.07) * depth,
				"speed": seed_rng.randf_range(0.9, 1.5) * depth,
			})

	func update(f: AudioFeatures, dt: float, h: Vector2) -> void:
		super(f, dt, h)
		_slant = num("slant", 0.16) + 0.12 * sin(t * 0.3) + 0.1 * f.bass
		var spd: float = num("fall", 1.1) * (1.0 + 0.3 * f.energy)
		for d in _drops:
			d.y = fposmod(d.y + spd * float(d.speed) * dt, 1.0)

	func draw(ci: CanvasItem, u: float) -> void:
		var col := Color.from_hsv(num("hue", 0.58), num("sat", 0.14), 0.9)
		var w: float = maxf(1.0, num("width", 1.5))
		for d in _drops:
			var py: float = (d.y * 2.0 - 1.0) * half.y
			# Shear x by slant*y so each drop travels ALONG its streak (dx/dy == slant),
			# i.e. the rain falls in the same direction the lines point - wind-blown - not
			# straight down past a tilted streak.
			var px: float = (d.x * 2.0 - 1.0) * half.x + _slant * py
			var top := Vector2(px, py) * u
			var bottom := top + Vector2(_slant, 1.0).normalized() * float(d.len) * u
			var a: float = 0.15 + 0.35 * float(d.depth)
			ci.draw_line(top, bottom, Color(col.r, col.g, col.b, a), w * float(d.depth), true)


# ---------------------------------------------------------------------------------
# Fireflies - warm motes wandering a curl-noise flow field, blinking async. Each has
# its own glow phase; a beat lights the subset whose threshold it crosses (the embers
# trick), so the meadow sparkles in ripples, not in unison.
# ---------------------------------------------------------------------------------
class Fireflies:
	extends Base
	var _bugs: Array = []
	var _flow: FlowField

	func _init(seed_rng: RandomNumberGenerator, c: Dictionary = {}) -> void:
		super(seed_rng, c)
		_flow = FlowField.new(seed_rng.randi(), 2.2, 0.08)
		var n := int(num("count", 40))
		for i in n:
			_bugs.append({
				"pos": Vector2(seed_rng.randf_range(-0.8, 0.8), seed_rng.randf_range(-0.5, 0.5)),
				"phase": seed_rng.randf() * TAU,
				"rate": seed_rng.randf_range(1.0, 3.0),
				"thresh": 0.2 + 0.5 * seed_rng.randf(),
				"hue_off": seed_rng.randf_range(-0.04, 0.06),
				"size": seed_rng.randf_range(0.006, 0.013),
				"glow": 0.0,
			})

	func update(f: AudioFeatures, dt: float, h: Vector2) -> void:
		super(f, dt, h)
		_flow.advance(dt)
		var spd: float = num("speed", 0.06)
		var beat_drive := f.beat + 0.3 * f.energy
		for b in _bugs:
			var v := _flow.at(b.pos) * spd
			# Curl advection alone slowly channels them into knots; a small diffusion term
			# (a random walk) counteracts the collapse so the field stays spread out.
			var jit := Vector2(rng.randf_range(-1.0, 1.0), rng.randf_range(-1.0, 1.0)) * num("jitter", 0.16) * dt
			b.pos += v * dt + jit + Vector2(0.0, -0.004 * dt)   # slight rise
			b.pos.x = wrapf(b.pos.x, -half.x - 0.05, half.x + 0.05)
			b.pos.y = clampf(b.pos.y, -half.y - 0.05, half.y + 0.05)
			var twinkle := 0.5 + 0.5 * sin(t * float(b.rate) + b.phase)
			var flare := Nonlinear.apply("spike", clampf(beat_drive - float(b.thresh), 0.0, 1.0), 3.0)
			b.glow = clampf(0.25 * twinkle + 0.85 * flare, 0.0, 1.0)

	func draw(ci: CanvasItem, u: float) -> void:
		var base_h: float = num("hue", 0.16)
		var real: bool = flag("real_light", false)
		for b in _bugs:
			var c: Vector2 = b.pos * u
			var v: float = b.glow
			var col := Color.from_hsv(fposmod(base_h + b.hue_off, 1.0), 0.55, clampf(0.35 + 0.65 * v, 0.0, 1.0))
			var r: float = float(b.size) * u
			if real:
				# A real point light, not a pasted halo: a faint wide wash (light reaching
				# into the dark) + a steep additive falloff + a white-hot core. Reads as a
				# source that actually illuminates.
				ci.draw_circle(c, r * (7.0 + 9.0 * v), Color(col.r, col.g, col.b, 0.02 + 0.04 * v))
				for k in 4:
					var fk := float(k) / 3.0
					ci.draw_circle(c, r * (0.8 + (1.0 - fk) * 3.0 * (0.6 + 0.7 * v)),
						Color(col.r, col.g, col.b, (0.05 + 0.20 * fk) * (0.4 + 0.6 * v)))
				ci.draw_circle(c, r * (0.6 + 0.4 * v), Color(1, 1, 1, 0.55 * v))   # hot core
			else:
				# A tighter, less fake glow than before, with a hot core on the flare.
				Layer.glow(ci, c, r * (2.2 + 3.2 * v), Color(col.r, col.g, col.b, 0.10 + 0.26 * v), 5)
				ci.draw_circle(c, r * 0.7, Color(1, 1, 1, 0.4 * v))
			ci.draw_circle(c, r, col)


# ---------------------------------------------------------------------------------
# Stars - a parallax starfield that twinkles, with the occasional shooting star
# streaking across. A nebula-quiet backdrop; pairs with bed/fog/aurora.
# ---------------------------------------------------------------------------------
class Stars:
	extends Base
	var _stars: Array = []
	var _shoot: Dictionary = {}
	var _next_shoot := 3.0

	func _init(seed_rng: RandomNumberGenerator, c: Dictionary = {}) -> void:
		super(seed_rng, c)
		var n := int(num("count", 150))
		for i in n:
			_stars.append({
				"pos": Vector2(seed_rng.randf_range(-1.0, 1.0), seed_rng.randf_range(-0.7, 0.7)),
				"size": seed_rng.randf_range(0.0012, 0.004),
				"phase": seed_rng.randf() * TAU,
				"rate": seed_rng.randf_range(0.4, 1.6),
				"hue": fposmod(num("hue", 0.6) + seed_rng.randf_range(-0.12, 0.12), 1.0),
				"depth": seed_rng.randf_range(0.2, 1.0),
			})
		_next_shoot = seed_rng.randf_range(4.0, 10.0)

	func update(f: AudioFeatures, dt: float, h: Vector2) -> void:
		super(f, dt, h)
		if _shoot.is_empty():
			_next_shoot -= dt * (1.0 + 0.8 * f.beat)
			if _next_shoot <= 0.0:
				# Enter from a RANDOM edge and cross in a varied direction (not always the same
				# diagonal), at a sampled DEPTH so near ones streak big/fast/bright and far ones
				# drift small/slow/faint - the sky has depth, and they are rarer now.
				var from: Vector2
				match rng.randi() % 4:
					0: from = Vector2(rng.randf_range(-1.1, 1.1), -0.8)   # top
					1: from = Vector2(rng.randf_range(-1.1, 1.1), 0.8)    # bottom
					2: from = Vector2(-1.2, rng.randf_range(-0.7, 0.7))   # left
					_: from = Vector2(1.2, rng.randf_range(-0.7, 0.7))    # right
				var aim := Vector2(rng.randf_range(-0.5, 0.5), rng.randf_range(-0.5, 0.5))
				var ang := (aim - from).angle() + rng.randf_range(-0.45, 0.45)
				var depth := rng.randf()                              # 0 far .. 1 near
				_shoot = {"p": from, "v": Vector2.from_angle(ang) * lerpf(0.8, 2.6, depth),
					"life": 1.0, "depth": depth}
				_next_shoot = rng.randf_range(5.0, 13.0)
		else:
			_shoot.p += _shoot.v * dt
			_shoot.life -= dt * lerpf(0.55, 1.0, float(_shoot.depth))   # far ones linger to cross
			if _shoot.life <= 0.0:
				_shoot = {}

	func draw(ci: CanvasItem, u: float) -> void:
		for s in _stars:
			var c: Vector2 = s.pos * Vector2(half.x / 0.9, half.y / 0.55) * u
			var tw: float = 0.5 + 0.5 * sin(t * float(s.rate) + s.phase)
			var v: float = clampf(0.35 + 0.65 * tw * float(s.depth), 0.0, 1.0)
			var col := Color.from_hsv(float(s.hue), 0.25, v)
			var r: float = float(s.size) * u
			if r > 1.5:
				Layer.glow(ci, c, r * 3.0, Color(col.r, col.g, col.b, 0.25 * v), 4)
			ci.draw_circle(c, maxf(0.8, r), col)
		if not _shoot.is_empty():
			var dep: float = _shoot.depth
			var p: Vector2 = _shoot.p * Vector2(half.x / 0.9, half.y / 0.55) * u
			var taillen: float = lerpf(0.06, 0.17, dep)        # near streaks are longer
			var tail: Vector2 = p - _shoot.v.normalized() * taillen * u * clampf(_shoot.life, 0.0, 1.0)
			var a: float = clampf(_shoot.life, 0.0, 1.0) * lerpf(0.4, 1.0, dep)   # far ones fainter
			ci.draw_line(tail, p, Color(1, 1, 1, a), lerpf(1.0, 2.6, dep), true)
			ci.draw_circle(p, lerpf(1.4, 3.6, dep), Color(1, 1, 1, a))


# ---------------------------------------------------------------------------------
# Aurora - slow flowing curtains of light. A few horizontal ribbons, each a wavy band
# whose vertical wander and brightness ride a band of the spectrum. Green / violet by
# default; deeply atmospheric over stars.
# ---------------------------------------------------------------------------------
class Aurora:
	extends Base
	var _ribbons: Array = []
	var _f_cache: AudioFeatures = AudioFeatures.new()

	func _init(seed_rng: RandomNumberGenerator, c: Dictionary = {}) -> void:
		super(seed_rng, c)
		var n := int(num("count", 4))
		for i in n:
			# A few sparse, slowly drifting nodes along the wavelength where a gentle vibrato
			# in the wave amplitude is allowed to act (elsewhere the ribbon stays calm).
			var nodes := []
			for j in seed_rng.randi_range(2, 4):
				nodes.append({
					"pos": seed_rng.randf(),
					"w": seed_rng.randf_range(0.04, 0.10),
					"drift": seed_rng.randf_range(-0.03, 0.03),
				})
			_ribbons.append({
				"y": seed_rng.randf_range(-0.35, 0.1),
				"amp": seed_rng.randf_range(0.06, 0.16),
				"thick": seed_rng.randf_range(0.06, 0.14),
				"freq": seed_rng.randf_range(1.2, 2.6),
				"speed": seed_rng.randf_range(0.1, 0.3) * (1.0 if seed_rng.randf() < 0.5 else -1.0),
				"phase": seed_rng.randf() * TAU,
				"hue_off": seed_rng.randf_range(0.0, 0.18),
				"band": seed_rng.randf(),
				# Harmonic vibrato on the amplitude: a gentle quiver (rate in rad/s) that rides
				# the high band like the shimmer in a held vocal note, only near the nodes above.
				"vib_rate": seed_rng.randf_range(3.0, 7.0),
				"vib_depth": seed_rng.randf_range(0.20, 0.45),
				"vib_phase": seed_rng.randf() * TAU,
				"vib_sx": seed_rng.randf_range(1.5, 4.0),
				"nodes": nodes,
			})

	func update(f: AudioFeatures, dt: float, h: Vector2) -> void:
		super(f, dt, h)
		_f_cache = f

	# The wave amplitude at position fx along a ribbon: the base amp, plus a gentle harmonic
	# VIBRATO that is sparsely active along the wavelength - only near a few slowly drifting
	# nodes - and rides the high band, like the shimmer in a held vocal note. Away from the
	# nodes the envelope is ~0 and the ribbon keeps its calm wave.
	func _amp_at(rb: Dictionary, fx: float) -> float:
		var env := 0.0
		for nd in rb.nodes:
			var pos: float = fposmod(float(nd.pos) + t * float(nd.drift), 1.0)
			var d: float = fx - pos
			var w: float = float(nd.w)
			env = maxf(env, exp(-(d * d) / (2.0 * w * w)))
		var voice: float = 0.4 + 0.6 * _f_cache.high
		var vib: float = float(rb.vib_depth) * env * voice \
			* sin(float(rb.vib_rate) * t + fx * float(rb.vib_sx) * TAU + float(rb.vib_phase))
		return float(rb.amp) * (1.0 + vib)

	func draw(ci: CanvasItem, u: float) -> void:
		var base_h: float = num("hue", 0.38)
		var sat: float = num("sat", 0.55)
		var f := _f_cache
		var steps := 26
		for rb in _ribbons:
			var bright: float = 0.45 + 0.55 * f.sample(float(rb.band))
			var hue := fposmod(base_h + rb.hue_off, 1.0)
			# Build the ribbon as a vertical strip of translucent quads following a wave.
			for k in steps:
				var fx0 := float(k) / float(steps)
				var fx1 := float(k + 1) / float(steps)
				var x0 := (fx0 * 2.0 - 1.0) * half.x
				var x1 := (fx1 * 2.0 - 1.0) * half.x
				var y0: float = rb.y + _amp_at(rb, fx0) * sin(rb.freq * TAU * fx0 + t * rb.speed * TAU + rb.phase)
				var y1: float = rb.y + _amp_at(rb, fx1) * sin(rb.freq * TAU * fx1 + t * rb.speed * TAU + rb.phase)
				var th: float = rb.thick * (0.6 + 0.6 * bright)
				var top := Color.from_hsv(hue, sat, bright, 0.0)
				var midc := Color.from_hsv(hue, sat, bright, 0.16 + 0.30 * bright)
				ci.draw_polygon(
					PackedVector2Array([
						Vector2(x0, (y0 - th) * 1.0) * u, Vector2(x1, (y1 - th)) * u,
						Vector2(x1, y1) * u, Vector2(x0, y0) * u]),
					PackedColorArray([top, top, midc, midc]))
				ci.draw_polygon(
					PackedVector2Array([
						Vector2(x0, y0) * u, Vector2(x1, y1) * u,
						Vector2(x1, (y1 + th)) * u, Vector2(x0, (y0 + th)) * u]),
					PackedColorArray([midc, midc, top, top]))


# ---------------------------------------------------------------------------------
# Petals - drifting petals / leaves that tumble as they fall, riding a curl-noise
# breeze. Flat quads rotating slowly (the flat-subject discipline: drift + spin, never
# fake depth). Warm or pink by default; a soft botany weather.
# ---------------------------------------------------------------------------------
class Petals:
	extends Base
	var _petals: Array = []
	var _flow: FlowField

	func _init(seed_rng: RandomNumberGenerator, c: Dictionary = {}) -> void:
		super(seed_rng, c)
		_flow = FlowField.new(seed_rng.randi(), 1.6, 0.06)
		var n := int(num("count", 46))
		for i in n:
			var depth := seed_rng.randf_range(0.4, 1.0)
			_petals.append({
				"x": seed_rng.randf(), "y": seed_rng.randf(), "depth": depth,
				"size": seed_rng.randf_range(0.012, 0.026) * depth,
				"ang": seed_rng.randf() * TAU,
				"spin": seed_rng.randf_range(-1.2, 1.2),
				"hue_off": seed_rng.randf_range(-0.05, 0.05),
				"flutter": seed_rng.randf() * TAU,
			})

	func update(f: AudioFeatures, dt: float, h: Vector2) -> void:
		super(f, dt, h)
		_flow.advance(dt)
		var spd: float = num("fall", 0.08) * (1.0 + 0.3 * f.energy)
		for p in _petals:
			p.y = fposmod(p.y + spd * float(p.depth) * dt, 1.0)
			p.ang += p.spin * dt * (0.6 + 0.4 * f.energy)

	func draw(ci: CanvasItem, u: float) -> void:
		var base_h: float = num("hue", 0.95)
		var sat: float = num("sat", 0.5)
		for p in _petals:
			var breeze := _flow.at(Vector2(p.x * 2.0 - 1.0, p.y * 2.0 - 1.0)) * 0.06
			var px: float = (p.x * 2.0 - 1.0) * half.x + breeze.x + 0.05 * sin(t * 0.7 + p.flutter)
			var py: float = (p.y * 2.0 - 1.0) * half.y
			var c: Vector2 = Vector2(px, py) * u
			var s: float = float(p.size) * u
			# A petal = a thin diamond, foreshortened by its flutter so it tumbles.
			var fold := 0.35 + 0.65 * absf(sin(t * 1.1 + p.flutter))
			var col := Color.from_hsv(fposmod(base_h + p.hue_off, 1.0), sat,
				0.85, 0.5 + 0.4 * float(p.depth))
			var pts := PackedVector2Array([
				Vector2(0, -s), Vector2(s * 0.6 * fold, 0), Vector2(0, s), Vector2(-s * 0.6 * fold, 0)])
			var tp := PackedVector2Array()
			for v in pts:
				tp.append(c + v.rotated(p.ang))
			ci.draw_colored_polygon(tp, col)


# ---------------------------------------------------------------------------------
# Dust - tiny slow motes adrift in a soft light shaft. Subtle; reads as floating
# particles in a sunbeam. Great as a faint overlay on any scene.
# ---------------------------------------------------------------------------------
class Dust:
	extends Base
	var _motes: Array = []
	var _flow: FlowField

	func _init(seed_rng: RandomNumberGenerator, c: Dictionary = {}) -> void:
		super(seed_rng, c)
		_flow = FlowField.new(seed_rng.randi(), 1.4, 0.04)
		var n := int(num("count", 110))
		for i in n:
			_motes.append({
				"pos": Vector2(seed_rng.randf_range(-0.9, 0.9), seed_rng.randf_range(-0.6, 0.6)),
				"size": seed_rng.randf_range(0.0008, 0.0026),
				"phase": seed_rng.randf() * TAU,
				"rate": seed_rng.randf_range(0.3, 0.9),
			})

	func update(f: AudioFeatures, dt: float, h: Vector2) -> void:
		super(f, dt, h)
		_flow.advance(dt)
		var spd: float = num("speed", 0.02)
		for m in _motes:
			var jit := Vector2(rng.randf_range(-1.0, 1.0), rng.randf_range(-1.0, 1.0)) * num("jitter", 0.10) * dt
			m.pos += _flow.at(m.pos) * spd * dt + jit + Vector2(num("drift", 0.006) * dt, 0.0)
			m.pos.x = wrapf(m.pos.x, -half.x - 0.05, half.x + 0.05)
			m.pos.y = wrapf(m.pos.y, -half.y - 0.05, half.y + 0.05)

	func draw(ci: CanvasItem, u: float) -> void:
		# An optional soft shaft of light behind the motes.
		if flag("shaft", true):
			# A soft shaft of light from above, built from a column of overlapping soft gaussian
			# puffs that widen and fade as they descend. Its bright end sits well ABOVE the frame
			# (only the soft falloff reaches in), so there is NO hard flat top to reveal even when
			# the scene is panned or zoomed during a transition.
			var sh := num("shaft_x", -0.2)
			var hue := num("hue", 0.12)
			var x := sh * half.x * u
			for i in 7:
				var fy := float(i) / 6.0                        # 0 high (off-screen) .. 1 mid-low
				var py := lerpf(-half.y * 1.4, half.y * 0.55, fy) * u
				var pw := lerpf(0.22, 0.7, fy) * half.x * u     # the beam spreads downward
				Layer.puff(ci, Vector2(x + sh * 0.08 * fy * half.x * u, py), pw,
					Color.from_hsv(hue, 0.2, 1.0, 0.055 * (1.0 - 0.85 * fy)))
		var mc := Color.from_hsv(num("hue", 0.12), 0.15, 1.0)
		for m in _motes:
			var c: Vector2 = m.pos * u
			var v: float = 0.4 + 0.6 * (0.5 + 0.5 * sin(t * float(m.rate) + m.phase))
			ci.draw_circle(c, maxf(0.7, float(m.size) * u), Color(mc.r, mc.g, mc.b, 0.15 + 0.4 * v))


# ---------------------------------------------------------------------------------
# Bubbles - underwater bubbles released in BURSTS (gurgles) from bed emitters, rising,
# meandering, and POPPING near the surface. Not a tidy field of identical discs: a few
# vents trickle bubbles and periodically (and on beats - a burp) belch a cluster of
# mostly-tiny ones, which ascend at size-dependent speeds and burst into a quick ring.
# ---------------------------------------------------------------------------------
class Bubbles:
	extends Base
	var _bubbles: Array = []
	var _pops: Array = []
	var _emitters: Array = []        # vent x positions, normalized -1..1
	var _flow: FlowField
	var _trickle := 0.0
	var _gurgle := 0.0
	var _beat_prev := 0.0
	const CAP := 240

	func _init(seed_rng: RandomNumberGenerator, c: Dictionary = {}) -> void:
		super(seed_rng, c)
		_flow = FlowField.new(seed_rng.randi(), 1.4, 0.05)
		for i in seed_rng.randi_range(2, 4):
			_emitters.append(seed_rng.randf_range(-0.8, 0.8))
		_gurgle = seed_rng.randf_range(0.4, 1.8)

	# Release one bubble from vent `ex` (normalized x). `scale` shrinks a burst's bubbles.
	func _spawn(ex: float, scale: float) -> void:
		if _bubbles.size() >= CAP:
			return
		var size: float = pow(rng.randf(), 1.8) * 0.026 * scale + 0.0022   # skew small, but enough mid bubbles to read
		_bubbles.append({
			"x": ex * half.x + rng.randf_range(-0.025, 0.025),
			"y": half.y * rng.randf_range(0.82, 0.98),         # near the bed, clearly inside the frame
			"size": size,
			# Terminal-velocity rise: buoyancy balances drag, so bubbles climb at a calm, bounded
			# speed (bigger ones a little faster) - not an accelerating cartoon shot upward.
			"vy": num("rise", 0.075) * (0.45 + 9.0 * size) * rng.randf_range(0.85, 1.15),
			"wob_amp": rng.randf_range(0.012, 0.045),          # they spiral as they rise
			"wob_rate": rng.randf_range(0.6, 1.6),
			"squash_rate": rng.randf_range(1.2, 2.8),
			"phase": rng.randf() * TAU,
			"age": 0.0,
			# Variable-length lifetimes, skewed so most are short and a few linger.
			"life": 1.4 + pow(rng.randf(), 0.7) * 7.0,
		})

	func update(f: AudioFeatures, dt: float, h: Vector2) -> void:
		super(f, dt, h)
		_flow.advance(dt)
		# A steady trickle from the vents.
		_trickle -= dt
		if _trickle <= 0.0:
			_trickle = rng.randf_range(0.05, 0.16)
			_spawn(_emitters[rng.randi() % _emitters.size()], 1.0)
			_spawn(_emitters[rng.randi() % _emitters.size()], 1.0)
		# Gurgles: a modest timed burst, and a slightly bigger burp on a STRONG beat -
		# gated through a spike curve so it only belches on real hits, not every tick.
		var beat_edge: bool = f.beat > 0.6 and _beat_prev <= 0.6
		_beat_prev = f.beat
		var burp: float = Nonlinear.apply("spike", clampf(f.beat - 0.5, 0.0, 1.0), 3.0)
		_gurgle -= dt
		if _gurgle <= 0.0 or beat_edge:
			_gurgle = rng.randf_range(2.2, 4.8)
			var ex: float = _emitters[rng.randi() % _emitters.size()]
			var k := rng.randi_range(3, 6) + int(round(3.0 * burp))
			for j in k:
				_spawn(ex, rng.randf_range(0.35, 1.0))
		# Advance: rise (size-dependent), meander, age; pop near the surface / at end of life.
		var boost: float = 1.0 + 0.08 * f.energy        # audio barely nudges the rise - bubbles stay calm
		var live: Array = []
		for b in _bubbles:
			b.age += dt
			b.y -= float(b.vy) * boost * dt
			b.x += _flow.at(Vector2(b.x / maxf(0.01, half.x), b.y / maxf(0.01, half.y))).x * 0.04 * dt
			var pop: bool = b.age > float(b.life) or b.y < -half.y * 0.85 \
				or (b.age > 0.7 and rng.randf() < 1.2 * dt)
			if pop:
				_pops.append({"x": b.x, "y": b.y, "size": float(b.size), "age": 0.0})
			else:
				live.append(b)
		_bubbles = live
		var lp: Array = []
		for p in _pops:
			p.age += dt
			if p.age < 0.22:
				lp.append(p)
		_pops = lp

	func draw(ci: CanvasItem, u: float) -> void:
		var tint := Color.from_hsv(num("hue", 0.55), 0.25, 1.0)   # the water's colour bleeds into the glass
		for b in _bubbles:
			# Ease in at birth AND ease out in the last moments, so a bubble doesn't blink
			# out abruptly - it thins away just before it pops.
			var fin: float = clampf(b.age * 3.0, 0.0, 1.0) * clampf((float(b.life) - b.age) * 4.0, 0.0, 1.0)
			# Spiral path: a smooth two-frequency lateral sway, so the bubble corkscrews up.
			var wob: float = float(b.wob_amp) * (sin(t * float(b.wob_rate) + b.phase)
				+ 0.4 * sin(t * float(b.wob_rate) * 1.7 + float(b.phase) * 2.0))
			var c: Vector2 = Vector2(b.x + wob, b.y) * u
			var r: float = float(b.size) * u
			if r < 1.6:
				ci.draw_circle(c, maxf(0.8, r), Color(0.85, 0.93, 1.0, 0.28 * fin))   # distant fleck
				continue
			# A glassy, transparent sphere: a soft refraction halo, a barely-there fill you see
			# through, a dim full rim brightening to a lit crescent on the upper-left, a sharp
			# specular highlight, and a faint spot of light TRANSMITTED through to the lower-right.
			Layer.puff(ci, c, r * 1.7, Color(tint.r, tint.g, tint.b, 0.09 * fin))
			ci.draw_colored_polygon(Layer.ellipse(c, r * 0.92, r * 0.92, 18),
				Color(tint.r * 0.7, tint.g * 0.85, 1.0, 0.07 * fin))
			var rim := Layer.ellipse(c, r, r, 22)
			rim.append(rim[0])
			ci.draw_polyline(rim, Color(0.80, 0.90, 1.0, 0.34 * fin), maxf(1.0, r * 0.06), true)
			ci.draw_arc(c, r, PI * 0.9, PI * 1.55, 12, Color(0.97, 0.99, 1.0, 0.7 * fin), maxf(1.3, r * 0.12), true)
			ci.draw_circle(c + Vector2(-r * 0.34, -r * 0.34), maxf(1.0, r * 0.18), Color(1, 1, 1, 0.85 * fin))
			ci.draw_circle(c + Vector2(r * 0.30, r * 0.32), maxf(0.8, r * 0.11), Color(0.9, 0.96, 1.0, 0.38 * fin))
		# Pops: a soft, quick wisp fading out - a bubble's skin breaking, not an explosion.
		for p in _pops:
			var k: float = float(p.age) / 0.22
			Layer.puff(ci, Vector2(p.x, p.y) * u, float(p.size) * u * (1.4 + 1.0 * k),
				Color(0.85, 0.93, 1.0, (1.0 - k) * 0.18))


# ---------------------------------------------------------------------------------
# Embers - warm sparks rising on the wind, twinkling and flaring async (the embers
# scene, as a reusable layer). A beat lights only the subset above its own threshold.
# ---------------------------------------------------------------------------------
class Embers:
	extends Base
	var _sparks: Array = []
	var _flow: FlowField

	func _init(seed_rng: RandomNumberGenerator, c: Dictionary = {}) -> void:
		super(seed_rng, c)
		_flow = FlowField.new(seed_rng.randi(), 2.0, 0.07)
		var n := int(num("count", 110))
		var base_h := num("hue", 0.06)
		for i in n:
			_sparks.append({
				"pos": Vector2(seed_rng.randf_range(-0.7, 0.7), seed_rng.randf_range(-0.4, 0.6)),
				"size": seed_rng.randf_range(0.004, 0.011),
				"hue": fposmod(base_h + seed_rng.randf_range(-0.03, 0.06), 1.0),
				"phase": seed_rng.randf() * TAU,
				"rate": seed_rng.randf_range(0.8, 2.4),
				"thresh": 0.18 + 0.5 * seed_rng.randf(),
				"glow": 0.0,
			})

	func update(f: AudioFeatures, dt: float, h: Vector2) -> void:
		super(f, dt, h)
		_flow.advance(dt)
		var beat_drive := f.beat + 0.4 * f.energy
		var spd: float = num("speed", 0.05)
		for s in _sparks:
			var jit := Vector2(rng.randf_range(-1.0, 1.0), rng.randf_range(-1.0, 1.0)) * num("jitter", 0.12) * dt
			s.pos += _flow.at(s.pos) * spd * dt + jit + Vector2(0.0, num("lift", -0.05) * dt)
			s.pos.x = wrapf(s.pos.x, -half.x - 0.05, half.x + 0.05)
			if s.pos.y < -half.y - 0.05:
				s.pos.y = half.y + 0.05
			var twinkle := 0.5 + 0.5 * sin(t * float(s.rate) + s.phase)
			var flare := Nonlinear.apply("spike", clampf(beat_drive - float(s.thresh), 0.0, 1.0), 3.0)
			s.glow = clampf(0.28 + 0.34 * twinkle + 0.75 * flare, 0.05, 1.0)

	func draw(ci: CanvasItem, u: float) -> void:
		for s in _sparks:
			var c: Vector2 = s.pos * u
			var v: float = s.glow
			var col := Color.from_hsv(float(s.hue), 0.6, v)
			var r: float = float(s.size) * u
			ci.draw_circle(c, r * (2.0 + 2.0 * v), Color(col.r, col.g, col.b, 0.10 + 0.18 * v))
			ci.draw_circle(c, r, col)


# ---------------------------------------------------------------------------------
# Cosmos - large, distant background bodies that give a star field DEPTH instead of just
# dots: a shaded PLANET (a lit crescent over a dark globe, sometimes ringed), a soft
# coloured NEBULA cloud, or a slowly turning spiral GALAXY. Few, dim, drawn behind the
# stars (z = back). Seeded; a scene adds it sometimes so the void isn't barren.
# ---------------------------------------------------------------------------------
class Cosmos:
	extends Base
	var _bodies: Array = []

	func _init(seed_rng: RandomNumberGenerator, c: Dictionary = {}) -> void:
		super(seed_rng, c)
		var n := int(num("count", 1))
		for i in n:
			var kind: String = "nebula" if seed_rng.randf() < 0.5 else "galaxy"   # planet is its own 3D layer now
			var blobs := []
			for k in 6:
				blobs.append(Vector2(seed_rng.randf_range(-1.0, 1.0), seed_rng.randf_range(-1.0, 1.0))
					* seed_rng.randf_range(0.2, 0.7))
			_bodies.append({
				"kind": kind,
				"pos": Vector2(seed_rng.randf_range(-0.85, 0.85), seed_rng.randf_range(-0.6, 0.6)),
				"size": seed_rng.randf_range(0.10, 0.24) * (1.0 if kind == "planet" else 1.4),
				"depth": seed_rng.randf_range(0.18, 0.7),      # far/dim .. nearer/brighter
				"hue": fposmod(num("hue", 0.6) + seed_rng.randf_range(-0.25, 0.25), 1.0),
				"light": seed_rng.randf() * TAU,               # planet sun direction
				"spin": seed_rng.randf_range(-0.04, 0.04),     # galaxy turn
				"arms": seed_rng.randi_range(2, 4),
				"wind": seed_rng.randf_range(3.5, 6.5),        # galaxy arm winding
				"phase": seed_rng.randf() * TAU,
				"ring": seed_rng.randf() < 0.4,
				"blobs": blobs,
			})

	func draw(ci: CanvasItem, u: float) -> void:
		for b in _bodies:
			var c: Vector2 = (b.pos as Vector2) * half * u
			match String(b.kind):
				"planet": _planet(ci, u, c, b)
				"nebula": _nebula(ci, u, c, b)
				_: _galaxy(ci, u, c, b)

	# A dark globe with a soft lit crescent toward the sun direction, sometimes ringed.
	func _planet(ci: CanvasItem, u: float, c: Vector2, b: Dictionary) -> void:
		var r: float = float(b.size) * u
		var dep: float = float(b.depth)
		var hue: float = float(b.hue)
		var ld := Vector2(cos(float(b.light)), sin(float(b.light)))
		ci.draw_circle(c, r, Color.from_hsv(hue, 0.5, 0.10 * dep))           # the shadowed globe
		for k in 6:                                                          # lit side: a soft terminator
			var f := float(k) / 5.0
			ci.draw_circle(c + ld * r * 0.55 * f, r * (1.0 - 0.55 * f),
				Color.from_hsv(hue, lerpf(0.5, 0.25, f), lerpf(0.12, 0.7, f) * dep))
		if bool(b.ring):
			var pts := Layer.ellipse(c, r * 1.7, r * 0.5, 30)
			pts.append(pts[0])
			ci.draw_polyline(pts, Color.from_hsv(hue, 0.3, 0.55 * dep, 0.5), maxf(1.0, r * 0.03), true)

	# A soft coloured cloud from a few overlapping faint blobs, drifting slowly.
	func _nebula(ci: CanvasItem, u: float, c: Vector2, b: Dictionary) -> void:
		var r: float = float(b.size) * u
		var hue: float = float(b.hue)
		var dep: float = float(b.depth)
		var drift := Vector2(sin(t * 0.05 + float(b.phase)), cos(t * 0.04)) * r * 0.1
		for off: Vector2 in b.blobs:
			var col := Color.from_hsv(fposmod(hue + off.x * 0.05, 1.0), 0.55, 0.6 * dep, 0.06 * dep)
			Layer.soft_blob(ci, c + off * r + drift, r * (0.9 + 0.5 * off.length()), col, 8)

	# A glowing core with a few logarithmic arms of fading dots, turning slowly.
	func _galaxy(ci: CanvasItem, u: float, c: Vector2, b: Dictionary) -> void:
		var r: float = float(b.size) * u
		var hue: float = float(b.hue)
		var dep: float = float(b.depth)
		var rot := t * float(b.spin) + float(b.phase)
		var arms := int(b.arms)
		var wind: float = float(b.wind)
		var arm_h := fposmod(hue * 0.2 + 0.58, 1.0)            # arms cool blue-white (mostly), lightly tinted
		# Spiral arms first (under the core): bands of soft stars and dust jittered off the spiral
		# line, brightening toward the centre. Soft sprites, not hard dots - no flat "vortex".
		for arm in arms:
			var a0 := TAU * float(arm) / float(arms) + rot
			for k in 80:
				var f := float(k) / 79.0
				var ang := a0 + f * wind
				var on := Vector2(cos(ang), sin(ang))
				var perp := Vector2(-on.y, on.x)
				var jit := perp * sin(float(k) * 12.9898 + float(arm) * 7.5) * r * 0.07 * (1.0 - 0.5 * f)
				var p := c + on * (r * f) + jit
				var bv := 0.45 + 0.55 * absf(sin(float(k) * 4.73 + float(arm) * 2.1))   # star/gap variation
				var v := bv * (1.0 - 0.55 * f) * dep
				var ph := fposmod(arm_h + 0.10 * sin(float(k) * 0.7), 1.0)              # cool, the odd warm knot
				Layer.puff(ci, p, maxf(1.0, r * 0.05 * bv * (1.0 - 0.4 * f)),
					Color.from_hsv(ph, 0.45, v, 0.5 * bv))
		# Core last, on top: a warm, bright bulge - a galaxy's heart is yellow-white, not purple.
		Layer.puff(ci, c, r * 0.55, Color.from_hsv(fposmod(hue * 0.2 + 0.09, 1.0), 0.45, 0.9 * dep, 0.30))
		Layer.puff(ci, c, r * 0.24, Color.from_hsv(0.11, 0.30, dep, 0.6))


# ---------------------------------------------------------------------------------
# Clouds - soft cloud masses drifting horizontally across the sky, wrapping at the edges.
# Each cloud is a clump of overlapping soft puffs; the wind picks up with energy.
# ---------------------------------------------------------------------------------
class Clouds:
	extends Base
	var _clouds: Array = []

	func _init(seed_rng: RandomNumberGenerator, c: Dictionary = {}) -> void:
		super(seed_rng, c)
		for i in int(num("count", 5)):
			var puffs := []
			for k in seed_rng.randi_range(4, 7):
				puffs.append({"off": Vector2(seed_rng.randf_range(-1.0, 1.0),
					seed_rng.randf_range(-0.45, 0.45)), "r": seed_rng.randf_range(0.5, 1.0)})
			_clouds.append({
				"pos": Vector2(seed_rng.randf_range(-1.15, 1.15), seed_rng.randf_range(-0.6, 0.25)),
				"size": seed_rng.randf_range(0.18, 0.40),
				"depth": seed_rng.randf_range(0.3, 1.0),
				"speed": seed_rng.randf_range(0.012, 0.05) * (1.0 if seed_rng.randf() < 0.5 else -1.0),
				"puffs": puffs,
			})

	func update(f: AudioFeatures, dt: float, h: Vector2) -> void:
		super(f, dt, h)
		var wind := 0.6 + 0.9 * f.energy
		for cl in _clouds:
			cl.pos.x += float(cl.speed) * dt * wind
			if cl.pos.x > 1.35:
				cl.pos.x = -1.35
			elif cl.pos.x < -1.35:
				cl.pos.x = 1.35

	func draw(ci: CanvasItem, u: float) -> void:
		var hue := num("hue", 0.6)
		var sat := num("sat", 0.12)
		var val := num("val", 0.62)
		var alpha := num("alpha", 0.07)
		for cl in _clouds:
			var c: Vector2 = Vector2(cl.pos.x * half.x, cl.pos.y * half.y) * u
			var base: float = float(cl.size) * u
			var dep: float = cl.depth
			var col := Color.from_hsv(hue, sat, val * (0.4 + 0.6 * dep), alpha * dep)
			for pf in cl.puffs:
				Layer.soft_blob(ci, c + Vector2(pf.off) * base, base * float(pf.r), col, 8)


# ---------------------------------------------------------------------------------
# Fire - a burning flame: sparks rise from the base, flicker sideways, and cool from a
# white-yellow hot core through orange to a dim red as they climb and die, then respawn.
# ---------------------------------------------------------------------------------
class Fire:
	extends Base
	var _sparks: Array = []
	var _hot_x := 0.0      # a wandering hot spot: flames near it leap taller (uneven across)
	var _burst := 0.0      # beat-driven surge envelope: kicks the tall licks up, then decays

	func _init(seed_rng: RandomNumberGenerator, c: Dictionary = {}) -> void:
		super(seed_rng, c)
		for i in int(num("count", 160)):
			var s := {}
			_reseed(s, seed_rng, true)
			_sparks.append(s)

	func _reseed(s: Dictionary, r: RandomNumberGenerator, anywhere: bool) -> void:
		s["x"] = r.randf_range(-1.05, 1.05) * num("spread", 1.0)        # spans the FULL width
		# A sparse minority are tall LICKS: fast, long-lived flames that leap high up the screen
		# (then dissipate), so the fire is not a flat band in the bottom third.
		var lick := r.randf() < 0.10
		s["lick"] = lick
		s["vy"] = r.randf_range(1.2, 2.4) if lick else r.randf_range(0.35, 1.05)
		s["size"] = r.randf_range(0.010, 0.024) if lick else r.randf_range(0.006, 0.018)
		s["flen"] = r.randf_range(0.05, 0.13)      # flame-tongue length (fraction of unit)
		s["amp"] = r.randf_range(0.03, 0.10)       # how much this tongue weaves
		s["flicker"] = r.randf_range(6.0, 16.0)    # per-spark flicker rate
		s["phase"] = r.randf() * TAU
		s["decay"] = (0.40 + 0.45 * float(s["vy"])) * (0.42 if lick else 1.0)   # licks last longer
		if anywhere:
			# Seed each spark partway through its life so the very FIRST frame already looks like
			# steady-state fire - no flicker as a uniform scatter corrects itself down to the base.
			var prog := r.randf()
			s["life"] = 1.0 - prog
			var rise := float(s["vy"]) * 0.7 / float(s["decay"])      # full-life climb (drive ~0.7)
			s["y"] = clampf(r.randf_range(0.8, 1.05) - rise * prog, -1.1, 1.05)
		else:
			s["life"] = 1.0
			s["y"] = r.randf_range(0.8, 1.05)

	func update(f: AudioFeatures, dt: float, h: Vector2) -> void:
		super(f, dt, h)
		var drive := 0.7 + 0.9 * f.energy + 0.5 * f.beat
		_hot_x = sin(t * 0.13) * 0.6 + sin(t * 0.31 + 1.7) * 0.3      # the hot side wanders
		# Beats kick the burst envelope up; it falls away between them, so tall flames surge and die.
		_burst = maxf(_burst * exp(-2.2 * dt), clampf(1.2 * f.beat + 0.5 * f.energy, 0.0, 1.0))
		for s in _sparks:
			s.y -= float(s.vy) * dt * drive
			s.life -= dt * float(s.decay)
			if float(s.life) <= 0.0 or float(s.y) < -1.1:
				_reseed(s, rng, false)

	func draw(ci: CanvasItem, u: float) -> void:
		var base_h := num("hue", 0.04)
		# A warm ember BED glowing along the base, brighter where the hot spot sits, so the rising
		# licks read as one anchored fire and not a field of loose sparks.
		var by := half.y * u * 0.96
		for i in 7:
			var gx := lerpf(-0.92, 0.92, float(i) / 6.0)
			var glow := 0.06 + 0.06 * exp(-pow((gx - _hot_x) / 0.5, 2.0))
			Layer.puff(ci, Vector2(gx * half.x * u, by), half.x * u * 0.42,
				Color.from_hsv(0.06, 0.7, 0.7, glow))
		for s in _sparks:
			var heat := clampf(float(s.life), 0.0, 1.0)
			var flick := 0.7 + 0.3 * sin(t * float(s.flicker) + float(s.phase) * 2.0)   # shimmer
			var c: Vector2 = Vector2(float(s.x) * half.x, float(s.y) * half.y) * u
			var w: float = float(s.size) * u * (1.5 + 0.9 * heat)
			# Height: a base tongue, taller near the wandering hot spot, and massively taller for
			# the sparse licks - more so on a burst - so tall flames leap up one side and dissipate.
			var hot := exp(-pow((float(s.x) - _hot_x) / 0.45, 2.0))   # 0..1 proximity to the hot side
			var tall := (1.6 + 4.5 * _burst) * (0.55 + hot) if bool(s.lick) else 1.0
			var hgt: float = float(s.flen) * u * (0.5 + 0.7 * heat) * flick * (1.0 + 0.7 * hot) * tall
			# Weave is a COHERENT sideways sway, the same for the whole tongue, so the flame leans/
			# curls as one - it does not point in a random direction.
			var weave := (sin(t * 2.0 + float(s.phase) + float(s.y) * 4.0)
				+ 0.5 * sin(t * 3.3 + float(s.phase) * 1.7)) \
				* float(s.amp) * u * (1.0 + 2.0 * float(s.lick))
			# Draw the flame as a COLUMN of soft gaussian puffs: a bright, wide white-hot HEAD that
			# leads at the top (where the flame is climbing to) tapering down to a thin, dim, red
			# TAIL that trails DOWNWARD behind it - so the wisp flows down, not up. fk=0 is the tail
			# (bottom), fk=1 the head (top); the head curls with the coherent sway.
			var steps: int = clampi(int(hgt / maxf(w * 0.7, 1.0)) + 3, 4, 16)
			for k in steps:
				var fk := float(k) / float(steps - 1)            # 0 tail (bottom) .. 1 head (top)
				var pp := c + Vector2(weave * fk * fk, -hgt * fk)  # up toward the head + a coherent curl
				var pr := w * (0.45 + 1.05 * fk)                 # thin tail -> wide rounded head
				var hue := fposmod(0.02 + 0.09 * fk, 1.0)        # red tail -> orange-yellow head
				var sat := clampf(lerpf(1.0, 0.45, fk), 0.0, 1.0)           # red tail -> white-hot head
				var val := clampf(lerpf(0.40, 1.0, fk) * (0.55 + 0.5 * heat) * flick, 0.0, 1.0)
				var a := clampf(lerpf(0.12, 0.55, fk) * (0.6 + 0.4 * heat), 0.0, 1.0)
				Layer.puff(ci, pp, pr, Color.from_hsv(hue, sat, val, a))
			# A soft warm glow around the HEAD - the flame's brightest light (a halo, not a dot).
			Layer.puff(ci, c + Vector2(weave * 0.9, -hgt * 0.85), w * 3.0,
				Color.from_hsv(0.10, 0.5, clampf((0.6 + 0.5 * heat) * flick, 0.0, 1.0), 0.13 * (0.6 + 0.4 * heat)))


# ---------------------------------------------------------------------------------
# Rays - god-rays / light shafts angling down from the top edge, swaying slowly, brighter
# at the top and fading to nothing below. The underwater (and fog-lit) light cue.
# ---------------------------------------------------------------------------------
class Rays:
	extends Base
	var _rays: Array = []
	var _e := 0.0

	func _init(seed_rng: RandomNumberGenerator, c: Dictionary = {}) -> void:
		super(seed_rng, c)
		for i in int(num("count", 5)):
			_rays.append({
				"x": seed_rng.randf_range(-1.0, 1.0), "ang": seed_rng.randf_range(-0.3, 0.3),
				"w": seed_rng.randf_range(0.05, 0.13), "sway": seed_rng.randf_range(0.08, 0.25),
				"phase": seed_rng.randf() * TAU, "bright": seed_rng.randf_range(0.4, 1.0),
			})

	func update(f: AudioFeatures, dt: float, h: Vector2) -> void:
		super(f, dt, h)
		_e = f.energy

	func draw(ci: CanvasItem, u: float) -> void:
		var hue := num("hue", 0.52)
		for ry in _rays:
			var sway := sin(t * float(ry.sway) + float(ry.phase)) * 0.07
			var top := Vector2((float(ry.x) + sway) * half.x, -half.y) * u   # from the top edge
			var ang := float(ry.ang) + sway * 0.6
			var dir := Vector2(sin(ang), 1.0).normalized()
			var perp := Vector2(-dir.y, dir.x)
			var length := 2.4 * half.y * u
			var w0: float = float(ry.w) * half.x * u * 0.35
			var w1: float = float(ry.w) * half.x * u * 1.7
			var br: float = float(ry.bright) * (0.6 + 0.5 * _e)
			var col := Color.from_hsv(hue, 0.28, 1.0)
			var c_top := Color(col.r, col.g, col.b, 0.22 * br)
			var c_bot := Color(col.r, col.g, col.b, 0.0)
			ci.draw_polygon(
				PackedVector2Array([top - perp * w0, top + perp * w0,
					top + dir * length + perp * w1, top + dir * length - perp * w1]),
				PackedColorArray([c_top, c_top, c_bot, c_bot]))


# ---------------------------------------------------------------------------------
# Planet - a REAL 3D sphere: a smooth-shaded [Mesh3D] icosphere drawn through a private
# [Lens3D], exactly the way the eye's sclera is built, so it is genuinely round and lit with a
# true terminator (not a flat stack of discs). Opaque, so the stars behind it are occluded.
# Drawn AFTER the stars by the scene for that occlusion.
# ---------------------------------------------------------------------------------
class Planet:
	extends Base
	var _lens := Lens3D.new()
	var _sphere: Mesh3D
	var _pos := Vector2.ZERO
	var _size := 0.2
	var _hue := 0.6
	var _sat := 0.5
	var _basis := Basis.IDENTITY
	var _spin := Vector3.ZERO

	func _init(seed_rng: RandomNumberGenerator, c: Dictionary = {}) -> void:
		super(seed_rng, c)
		_sphere = Mesh3D.icosphere(3)
		_sphere.compute_normals()
		_pos = Vector2(seed_rng.randf_range(-0.7, 0.7), seed_rng.randf_range(-0.4, 0.35))
		_size = seed_rng.randf_range(0.12, 0.28)
		_hue = fposmod(num("hue", 0.6) + seed_rng.randf_range(-0.3, 0.3), 1.0)
		_sat = seed_rng.randf_range(0.30, 0.7)
		_basis = Basis.from_euler(Vector3(seed_rng.randf() * TAU, seed_rng.randf() * TAU, 0.0))
		_spin = Vector3(seed_rng.randf_range(-0.04, 0.04), seed_rng.randf_range(0.03, 0.12), 0.0)

	func update(f: AudioFeatures, dt: float, h: Vector2) -> void:
		super(f, dt, h)
		_basis = _basis * Basis.from_euler(_spin * dt)        # the planet turns

	func draw(ci: CanvasItem, u: float) -> void:
		var fov := 50.0
		var dist := 6.0
		_lens.eye = Vector3(0.0, 0.0, dist)
		_lens.look = Vector3.ZERO
		_lens.fov = fov
		_lens.prepare()
		var sc := dist * tan(deg_to_rad(fov) * 0.5)           # world units per screen unit-fraction
		# Place the sphere in 3D so it projects to its screen spot at its screen size, then draw
		# it as a smooth-shaded, opaque, real sphere (the eye's sclera trick): true 3D lighting.
		var wpos := Vector3(_pos.x * half.x * sc, -_pos.y * half.y * sc, 0.0)
		_sphere.draw_through(ci, _lens, u, _basis, wpos, _size * sc,
			_hue, _sat, 0, 1.0, 0.0, 0.0, 0.18, 0.7, Color(0, 0, 0, 0), true)


# ---------------------------------------------------------------------------------
# Volumetric - REAL 3D clouds / fog. A field of soft gaussian puffs placed in 3D space, lit
# VOLUMETRICALLY (sorted from the sun inward, accumulating optical depth, so puffs facing the sun
# are bright and those buried behind others fall into shadow), depth-sorted and projected through
# a private [Lens3D], drifting and billowing over time. Simulated dynamics, not a flat 2D sprite -
# and being atmospheric it washes well over other scenes (the LAYER transition can stack it).
# ---------------------------------------------------------------------------------
class Volumetric:
	extends Base
	var _lens := Lens3D.new()
	var _puffs: Array = []
	var _sun := Vector3(0.4, 0.85, 0.3)
	var _driftx := 0.05
	var _fov := 62.0
	var _mode := "cloud"
	var _energy := 0.0

	func _init(seed_rng: RandomNumberGenerator, c: Dictionary = {}) -> void:
		super(seed_rng, c)
		_mode = String(cfg.get("mode", "cloud"))
		_sun = Vector3(rng.randf_range(-0.5, 0.5), rng.randf_range(0.55, 1.0), rng.randf_range(-0.2, 0.5)).normalized()
		_driftx = rng.randf_range(0.03, 0.09) * (1.0 if rng.randf() < 0.5 else -1.0)
		_fov = rng.randf_range(56.0, 70.0)
		if _mode == "fog":
			_build_fog()
		else:
			_build_clouds()
		_relight()

	func _build_clouds() -> void:
		for ci in rng.randi_range(3, 4):           # several billowing masses, spread across the sky
			var ctr := Vector3(rng.randf_range(-2.2, 2.2), rng.randf_range(0.1, 1.3), rng.randf_range(-1.8, -3.5))
			var ext := Vector3(rng.randf_range(0.8, 1.4), rng.randf_range(0.30, 0.55), rng.randf_range(0.6, 1.0))
			for pi in rng.randi_range(44, 72):     # many overlapping puffs so they merge into a mass
				var off := Vector3(rng.randfn(0.0, 0.5), rng.randfn(0.0, 0.42), rng.randfn(0.0, 0.5))
				var pos := ctr + Vector3(off.x * ext.x, off.y * ext.y * 0.7 + ext.y * 0.15, off.z * ext.z)
				var rad := rng.randf_range(0.35, 0.70) * clampf(1.0 - 0.30 * off.length(), 0.45, 1.0)
				_puffs.append({"home": pos, "pos": pos, "r": rad,
					"dens": rng.randf_range(0.6, 1.0), "ph": rng.randf() * TAU, "lit": 1.0})

	func _build_fog() -> void:
		for pi in rng.randi_range(70, 110):        # a wide bank of haze receding into depth
			var pos := Vector3(rng.randf_range(-3.5, 3.5), rng.randf_range(-0.6, 0.2), rng.randf_range(-1.0, -5.0))
			_puffs.append({"home": pos, "pos": pos, "r": rng.randf_range(0.6, 1.3),
				"dens": rng.randf_range(0.3, 0.6), "ph": rng.randf() * TAU, "lit": 1.0})

	# Self-shadowing: sort puffs from the sun inward and accumulate optical depth, so the sunlit
	# edge is bright and the buried core darkens - the cue that reads as real cloud volume.
	func _relight() -> void:
		var n := _puffs.size()
		if n == 0:
			return
		var order := range(n)
		order.sort_custom(func(a, b): return _puffs[a].pos.dot(_sun) > _puffs[b].pos.dot(_sun))
		var total := 0.0
		for p in _puffs:
			total += float(p.dens)
		# Walk from the sun inward; lit = exp(-K x fraction-of-the-cloud-already-passed). Using the
		# fraction (not raw count) keeps the bright-edge-to-dark-core gradient the same whatever the
		# puff count - so adding puffs makes the cloud denser, not uniformly black.
		var cum := 0.0
		for idx in order:
			_puffs[idx].lit = exp(-2.2 * (cum / maxf(total, 0.001)))
			cum += float(_puffs[idx].dens)

	func update(f: AudioFeatures, dt: float, h: Vector2) -> void:
		super(f, dt, h)
		_energy = lerpf(_energy, clampf(f.energy, 0.0, 1.0), 1.0 - exp(-3.0 * dt))
		for p in _puffs:
			var hm: Vector3 = p.home
			hm.x += _driftx * dt                   # the field drifts across
			p.home = hm
			# a slow billow: each puff bobs and breathes around its home
			p.pos = hm + Vector3(sin(t * 0.30 + float(p.ph)) * 0.06,
				sin(t * 0.40 + float(p.ph) * 1.3) * 0.05, cos(t * 0.25 + float(p.ph)) * 0.06)
		_relight()

	func draw(ci: CanvasItem, u: float) -> void:
		_lens.eye = Vector3.ZERO
		_lens.look = Vector3(0.0, 0.25, -1.0)      # camera looks slightly up into the sky
		_lens.fov = _fov
		_lens.prepare()
		var focal := 1.0 / tan(deg_to_rad(_fov) * 0.5)
		var tint := num("hue", -1.0)
		var sun_c := Color(1.0, 0.96, 0.88)        # warm sunlit white ...
		var shad_c := Color(0.42, 0.46, 0.6)       # ... cool shadow
		if tint >= 0.0:                            # optional palette tint (e.g. sunset clouds)
			sun_c = Color.from_hsv(tint, 0.12, 1.0)
			shad_c = Color.from_hsv(fposmod(tint + 0.55, 1.0), 0.4, 0.55)
		var vis: Array = []
		for p in _puffs:
			var d := _lens.depth(p.pos)
			if d > _lens.near:
				vis.append({"p": p, "d": d})
		vis.sort_custom(func(a, b): return a.d > b.d)        # far first (painter's)
		var base_a := 0.30 if _mode == "fog" else 0.42
		for it in vis:
			var p: Dictionary = it.p
			var pr := _lens.project(p.pos)
			var rad := float(p.r) * focal / float(it.d) * u
			if rad < 1.0:
				continue
			var lit := float(p.lit)
			var b := (0.5 + 0.55 * lit) * (0.9 + 0.25 * _energy)
			var col := shad_c.lerp(sun_c, lit)
			var a := clampf(float(p.dens) * base_a * (0.85 + 0.3 * _energy), 0.0, 0.95)
			Layer.puff(ci, Vector2(pr.x, pr.y) * u, rad, Color(col.r * b, col.g * b, col.b * b, a))


# ---------------------------------------------------------------------------------
# Flare - a lens flare drawn IN FRONT: a bright off-frame source with a starburst and anamorphic
# streak, plus a chain of translucent "ghost" discs and rings marching along the line from the
# source through the optical centre to the far side (where real lens flares sit). Its brightness
# pulses with the harmonics, and a barrel "fisheye" bows the whole chain outward. Composable, so
# any scene can wear a flare over the top. Add it LAST (front).
# ---------------------------------------------------------------------------------
class Flare:
	extends Base
	var _src := Vector2(0.7, -0.5)
	var _ghosts: Array = []
	var _energy := 0.0
	var _drift := 0.0

	func _init(seed_rng: RandomNumberGenerator, c: Dictionary = {}) -> void:
		super(seed_rng, c)
		var ang := seed_rng.randf() * TAU
		_src = Vector2(cos(ang), sin(ang)) * seed_rng.randf_range(0.5, 0.85)   # source toward an edge
		for i in seed_rng.randi_range(6, 10):
			_ghosts.append({
				"t": seed_rng.randf_range(-0.3, 1.7),         # position along source->centre->far line
				"size": seed_rng.randf_range(0.02, 0.09),
				"hue_off": seed_rng.randf_range(-0.18, 0.22),
				"sat": seed_rng.randf_range(0.1, 0.5),
				"ring": seed_rng.randf() < 0.45,
			})
		_drift = seed_rng.randf_range(-1.0, 1.0)

	func update(f: AudioFeatures, dt: float, h: Vector2) -> void:
		super(f, dt, h)
		_energy = lerpf(_energy, clampf(0.35 + 0.55 * f.energy + 0.5 * f.beat, 0.0, 1.0), 1.0 - exp(-4.5 * dt))

	# Map a normalized [-1,1] point to pixels, bowing it outward by the barrel "fisheye" first.
	func _px(p: Vector2, k: float, u: float) -> Vector2:
		var d := p * (1.0 + k * p.length_squared())
		return Vector2(d.x * half.x, d.y * half.y) * u

	func draw(ci: CanvasItem, u: float) -> void:
		var base_h: float = num("hue", 0.58)
		var k_fish: float = num("fisheye", 0.22)
		var e := _energy
		var src := _src.rotated(0.08 * sin(t * 0.2 + _drift))        # the source drifts slowly
		var src_px := _px(src, k_fish, u)
		var diag := maxf(half.x, half.y) * u
		# Source: a broad halo, a bright core, a radial starburst, and an anamorphic streak.
		Layer.puff(ci, src_px, diag * 0.55 * (0.6 + 0.5 * e), Color.from_hsv(base_h, 0.30, 1.0, 0.10 * e))
		Layer.puff(ci, src_px, diag * 0.12, Color.from_hsv(base_h, 0.12, 1.0, 0.55 * e))
		for s in 10:
			var a := TAU * float(s) / 10.0 + 0.1 * sin(t * 0.3)
			var ln := diag * (0.16 + 0.13 * float(s % 2)) * (0.5 + 0.7 * e)
			ci.draw_line(src_px, src_px + Vector2(cos(a), sin(a)) * ln,
				Color.from_hsv(base_h, 0.2, 1.0, 0.16 * e), maxf(1.0, diag * 0.0035), true)
		ci.draw_line(src_px - Vector2(diag * 0.95, 0), src_px + Vector2(diag * 0.95, 0),
			Color.from_hsv(base_h, 0.25, 1.0, 0.09 * e), maxf(1.0, diag * 0.006), true)
		# Ghosts along the source -> centre -> far line.
		for g in _ghosts:
			var p := src * (1.0 - 2.0 * float(g.t))
			var pp := _px(p, k_fish, u)
			var col := Color.from_hsv(fposmod(base_h + float(g.hue_off), 1.0), float(g.sat), 1.0)
			var gr: float = float(g.size) * diag
			if bool(g.ring):
				var ring := Layer.ellipse(pp, gr, gr, 22)
				ring.append(ring[0])
				ci.draw_polyline(ring, Color(col.r, col.g, col.b, 0.22 * e), maxf(1.0, gr * 0.06), true)
			else:
				Layer.puff(ci, pp, gr, Color(col.r, col.g, col.b, 0.18 * e))


# ---------------------------------------------------------------------------------
# Veil - a composable harmonic OBSCURING layer: drifting gaussian masses (snow squalls, rain
# sheets, haze) that swell with musical intensity and fade out between, softening and hiding the
# scene behind. The general form of the snow whiteout, so ANY scene can add moving patterns of
# visibility and obscuration. Tint with hue/sat/val; floor/gain/max shape how the swell maps to
# opacity. Add it last (front) so it veils what is beneath.
# ---------------------------------------------------------------------------------
class Veil:
	extends Base
	var _coverage := 0.0

	func update(f: AudioFeatures, dt: float, h: Vector2) -> void:
		super(f, dt, h)
		var drive := Layer.coverage_drive(f)
		_coverage = lerpf(_coverage, drive, 1.0 - exp(-(3.0 if drive > _coverage else 0.8) * dt))

	func draw(ci: CanvasItem, u: float) -> void:
		var va := clampf((_coverage - num("floor", 0.12)) * num("gain", 0.9), 0.0, num("max", 0.6))
		if va <= 0.004:
			return
		var hue := num("hue", 0.58)
		var sat := num("sat", 0.05)
		var val := num("val", 1.0)
		for sc in Layer.squall_centers(t):
			Layer.puff(ci, Vector2(sc.x * half.x, sc.y * half.y) * u,
				(0.55 + 0.45 * _coverage) * maxf(half.x, half.y) * u,
				Color.from_hsv(hue, sat, val, va))


const REGISTRY := {
	"bed": Bed,
	"fog": Fog,
	"snow": Snow,
	"rain": Rain,
	"fireflies": Fireflies,
	"stars": Stars,
	"aurora": Aurora,
	"petals": Petals,
	"dust": Dust,
	"bubbles": Bubbles,
	"embers": Embers,
	"cosmos": Cosmos,
	"clouds": Clouds,
	"fire": Fire,
	"rays": Rays,
	"planet": Planet,
	"volumetric": Volumetric,
	"veil": Veil,
	"flare": Flare,
}


## Build a layer by registry key, seeding it from `rng` with its constants baked in.
static func make(key: String, rng: RandomNumberGenerator, cfg := {}) -> Base:
	return REGISTRY[key].new(rng, cfg)
