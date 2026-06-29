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
		var speed: float = num("fall", 0.10) * (1.0 + 0.5 * f.energy)
		_fall += speed * dt
		for fl in _flakes:
			fl.y = fposmod(fl.y + speed * float(fl.depth) * dt, 1.0)

	func draw(ci: CanvasItem, u: float) -> void:
		var base_h: float = num("hue", 0.58)
		var sat: float = num("sat", 0.06)
		for fl in _flakes:
			var sway: float = (fl.sway_amp + 0.04 * _gust) * sin(t * fl.sway_rate + fl.phase)
			var px: float = (fl.x * 2.0 - 1.0) * half.x + sway
			var py: float = (fl.y * 2.0 - 1.0) * half.y
			var pos := Vector2(px, py) * u
			var r: float = fl.size * u
			var bright: float = 0.7 + 0.3 * sin(t * 1.3 + fl.phase)
			var col := Color.from_hsv(base_h, sat, 1.0, clampf(0.5 + 0.4 * float(fl.depth), 0.3, 0.95))
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
		_next_shoot = seed_rng.randf_range(2.0, 6.0)

	func update(f: AudioFeatures, dt: float, h: Vector2) -> void:
		super(f, dt, h)
		if _shoot.is_empty():
			_next_shoot -= dt * (1.0 + 2.0 * f.beat)
			if _next_shoot <= 0.0:
				var from := Vector2(rng.randf_range(-1.0, 0.2), rng.randf_range(-0.6, -0.1))
				_shoot = {"p": from, "v": Vector2(rng.randf_range(0.5, 1.0),
					rng.randf_range(0.25, 0.5)).normalized() * rng.randf_range(1.2, 2.0), "life": 1.0}
				_next_shoot = rng.randf_range(3.0, 8.0)
		else:
			_shoot.p += _shoot.v * dt
			_shoot.life -= dt * 1.1
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
			var p: Vector2 = _shoot.p * Vector2(half.x / 0.9, half.y / 0.55) * u
			var tail: Vector2 = p - _shoot.v.normalized() * 0.12 * u * clampf(_shoot.life, 0.0, 1.0)
			var a: float = clampf(_shoot.life, 0.0, 1.0)
			ci.draw_line(tail, p, Color(1, 1, 1, a), 2.0, true)
			ci.draw_circle(p, 2.5, Color(1, 1, 1, a))


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
			_ribbons.append({
				"y": seed_rng.randf_range(-0.35, 0.1),
				"amp": seed_rng.randf_range(0.06, 0.16),
				"thick": seed_rng.randf_range(0.06, 0.14),
				"freq": seed_rng.randf_range(1.2, 2.6),
				"speed": seed_rng.randf_range(0.1, 0.3) * (1.0 if seed_rng.randf() < 0.5 else -1.0),
				"phase": seed_rng.randf() * TAU,
				"hue_off": seed_rng.randf_range(0.0, 0.18),
				"band": seed_rng.randf(),
			})

	func update(f: AudioFeatures, dt: float, h: Vector2) -> void:
		super(f, dt, h)
		_f_cache = f

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
				var y0: float = rb.y + rb.amp * sin(rb.freq * TAU * fx0 + t * rb.speed * TAU + rb.phase)
				var y1: float = rb.y + rb.amp * sin(rb.freq * TAU * fx1 + t * rb.speed * TAU + rb.phase)
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
			var sh := num("shaft_x", -0.2)
			var col := Color.from_hsv(num("hue", 0.12), 0.2, 1.0, 0.04)
			var x := sh * half.x * u
			var w := 0.5 * half.x * u
			ci.draw_polygon(PackedVector2Array([
				Vector2(x - w * 0.3, -half.y * u), Vector2(x + w * 0.3, -half.y * u),
				Vector2(x + w, half.y * u), Vector2(x - w, half.y * u)]),
				PackedColorArray([col, col, Color(col.r, col.g, col.b, 0.0), Color(col.r, col.g, col.b, 0.0)]))
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
		var size: float = pow(rng.randf(), 2.4) * 0.024 * scale + 0.0018   # skew strongly small
		_bubbles.append({
			"x": ex * half.x + rng.randf_range(-0.025, 0.025),
			"y": half.y * rng.randf_range(0.9, 1.02),          # near the bed
			"size": size,
			"vy": num("rise", 0.09) * (0.5 + 16.0 * size) * rng.randf_range(0.8, 1.3),
			"wob_amp": rng.randf_range(0.008, 0.035),
			"wob_rate": rng.randf_range(0.8, 2.2),
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
			_trickle = rng.randf_range(0.12, 0.45)
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
		var boost: float = 1.0 + 0.4 * f.energy
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
		var base_h: float = num("hue", 0.55)
		for b in _bubbles:
			# Ease in at birth AND ease out in the last moments, so a bubble doesn't blink
			# out abruptly - it thins away just before it pops.
			var fin: float = clampf(b.age * 3.0, 0.0, 1.0) * clampf((float(b.life) - b.age) * 4.0, 0.0, 1.0)
			var wob: float = b.wob_amp * sin(t * float(b.wob_rate) + b.phase)
			var c: Vector2 = Vector2(b.x + wob, b.y) * u
			var r: float = float(b.size) * u
			var sq: float = 1.0 + 0.12 * sin(t * float(b.squash_rate) + b.phase)
			var rx := r
			var ry := r / sq
			ci.draw_colored_polygon(Layer.ellipse(c, rx * 0.95, ry * 0.95, 14),
				Color.from_hsv(base_h, 0.28, 0.9, 0.04 * fin))
			var rim := Layer.ellipse(c, rx, ry, 16)
			rim.append(rim[0])
			ci.draw_polyline(rim, Color(0.85, 0.93, 1.0, 0.22 * fin), maxf(1.0, r * 0.06), true)
			if r > 3.0:
				ci.draw_circle(c + Vector2(-r * 0.32, -r * 0.34), r * 0.09, Color(1, 1, 1, 0.5 * fin))
		# Pops: a small, brief ring - just a wisp, not a wild explosion.
		for p in _pops:
			var k: float = float(p.age) / 0.22
			var pr: float = float(p.size) * u * (1.0 + 1.4 * k)
			var a: float = (1.0 - k) * 0.25
			var ring := Layer.ellipse(Vector2(p.x, p.y) * u, pr, pr, 12)
			ring.append(ring[0])
			ci.draw_polyline(ring, Color(0.9, 0.95, 1.0, a), maxf(1.0, float(p.size) * u * 0.2), true)


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
}


## Build a layer by registry key, seeding it from `rng` with its constants baked in.
static func make(key: String, rng: RandomNumberGenerator, cfg := {}) -> Base:
	return REGISTRY[key].new(rng, cfg)
