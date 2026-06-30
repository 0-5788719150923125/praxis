extends GhostScene

## Rooted growth - crawling roots and tendrils that spread from a seed.
##
## Built on the [Filament] primitive: each root follows a curl-noise [Flow2D] field so
## it meanders and curls, branches into trunk-and-limb structure, tapers, and is
## revealed along a growth front so it *crawls* outward rather than appearing whole.
##
## Roots emerge the way real ones do: mostly as **laterals** sprouting from a point
## partway along an *existing* root (at an angle to that parent), only occasionally as a
## fresh taproot from the central seed. So the system spreads from many points across the
## branches instead of every strand radiating from one origin.
##
## Each root runs an independent **lifecycle**, staggered and rate-varied so the field
## is asynchronous (no uniform bands): it grows slowly to full, holds, then **retires**
## gracefully - either FADING out (alpha to zero) or REWINDING (the front retracts back
## toward its attachment point, collapsing inward) - before regrowing from a fresh site.
## It never clears and pops back in. Audio surges the growth front (a nonlinear `spike`,
## so beats lunge it and quiet barely moves) and carries colour and glow.

var _f: AudioFeatures = AudioFeatures.new()
var _rng := RandomNumberGenerator.new()
var _flow: Flow2D
var _roots: Array = []
var _variant := "root"
var _hue := 0.0
var _hue_depth := 0.0
var _glow := 0.0
var _base_len := 0.5
var _width := 7.0
var _draw_life := 1.0        # current root's alpha, set before its draw, read by _color_for
var _max_roots := 20         # the bloom grows the pool up to this, then sustains
var _emit_t := 0.0           # countdown to the next sprout from the seed
var _emit_interval := 0.6    # sampled seconds between new sprouts


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	_rng.seed = rng.randi()
	_hue = rng.randf()
	_hue_depth = rng.randf_range(-0.05, 0.12)
	_variant = "tendril" if rng.randf() < 0.4 else "root"
	_flow = Flow2D.new(rng.randi(), rng.randf_range(2.0, 3.4), 0.06)
	_width = rng.randf_range(5.0, 9.0)
	_base_len = rng.randf_range(0.42, 0.60)
	_max_roots = rng.randi_range(14, 22)
	_emit_interval = rng.randf_range(0.4, 0.9)
	_emit_t = _emit_interval
	# Start SPARSE - a few barely-sprouted shoots from the seed - and bloom outward over time
	# (new shoots keep emitting from the centre in update), rather than filling the frame at
	# once and then holding a static shape.
	var start := rng.randi_range(2, 4)
	for i in start:
		var r := _new_root()
		r.heading = -PI * 0.5 + TAU * float(i) / float(start) + rng.randf_range(-0.4, 0.4)
		_regrow(r, true)                            # from the central seed
		r.grown = rng.randf_range(0.0, 0.25)        # barely out of the ground; they grow from here
		_roots.append(r)
	return {}


# A fresh root record (lifecycle fields); origin/heading/fil are filled in by _regrow.
func _new_root() -> Dictionary:
	return {"fil": null, "grown": 0.0, "life": 1.0, "state": "grow",
		"timer": 0.0, "rate": 1.0, "hold": 2.0, "mode": "fade", "retract_to": 0.0,
		"origin": Vector2.ZERO, "heading": 0.0}


# The seed keeps sprouting: a new shoot emerges from the centre heading outward in any
# direction, so the root system continuously blooms over time instead of pausing once full.
func _emit_root() -> void:
	var r := _new_root()
	r.heading = _rng.randf_range(-PI, PI)
	_regrow(r, true)
	r.grown = 0.0
	_roots.append(r)


# Grow a fresh path for one root from a spawn site (a point along an existing root, or
# the central seed), on a heading set by that site, and re-roll its lifecycle constants
# (rate / hold / retire mode) so each life differs from the last. Laterals start shorter
# and finer than taproots, so the network reads as trunks feeding ever-thinner roots.
func _regrow(r: Dictionary, force_centre := false) -> void:
	var steps := _rng.randi_range(12, 20)
	var site := _spawn_site(force_centre)
	r.origin = site.pos
	r.heading = float(site.ang) + _rng.randf_range(-0.3, 0.3)
	var length := _base_len * _rng.randf_range(0.8, 1.25)
	var w := _width
	if site.lateral:
		length *= _rng.randf_range(0.5, 0.8)
		w *= _rng.randf_range(0.45, 0.7)
	r.fil = Filament.grow(_variant, r.origin, r.heading, length, w, steps, _flow, _rng)
	r.grown = 0.0
	r.life = 1.0
	r.state = "grow"
	r.rate = _gauss_rate()
	r.hold = _rng.randf_range(0.6, 2.2)   # shorter holds: keep crawling, don't freeze full
	r.mode = _roll_mode()   # mostly partial rewind, rarely a full fade-out


# Where a new root sprouts: mostly a point partway along an existing, revealed root
# (a LATERAL, emerging at an angle to its parent), occasionally the central seed (a fresh
# TAPROOT). Returns {pos, ang (heading suggestion), lateral}. Falls back to the centre
# when nothing has grown enough to branch from yet (e.g. the very first roots).
func _spawn_site(force_centre := false) -> Dictionary:
	var centre := {"pos": Vector2.ZERO, "ang": _rng.randf_range(-PI, PI), "lateral": false}
	# More shoots come straight from the seed now (per the bloom note), and emitted shoots
	# force it; the rest emerge as laterals off existing roots.
	if force_centre or _rng.randf() < 0.42:
		return centre
	# Roots with something revealed to branch from.
	var live := []
	for o in _roots:
		if o.fil != null and o.fil.segs.size() > 0 and float(o.grown) > 0.12:
			live.append(o)
	if live.is_empty():
		return centre
	var o = live[_rng.randi() % live.size()]
	var segs: Array = o.fil.segs
	# Find a revealed, fairly basal segment (basal points survive the parent's diebacks and
	# read as laterals off older tissue), then emerge from a point along it at an angle.
	var reveal: float = minf(float(o.grown), 0.65)
	for _attempt in 6:
		var s: Dictionary = segs[_rng.randi() % segs.size()]
		if float(s.born0) <= reveal:
			var t := _rng.randf()
			var pos: Vector2 = (s.a as Vector2).lerp(s.b, t)
			var tang := ((s.b as Vector2) - (s.a as Vector2)).angle()
			var side := 1.0 if _rng.randf() < 0.5 else -1.0
			return {"pos": pos, "ang": tang + side * _rng.randf_range(0.5, 1.1), "lateral": true}
	return centre


# Per-root growth-speed multiplier from a ~normal distribution (sum of three uniforms),
# wide spread - so roots grow at a real variety of speeds, not all at roughly one rate.
func _gauss_rate() -> float:
	var g := (_rng.randf() + _rng.randf() + _rng.randf()) / 3.0
	return 0.3 + 1.7 * g


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.03, 0.05)
	_flow.advance(delta)
	_glow = Nonlinear.flare(_glow, clampf(0.30 * f.energy + 0.70 * f.beat, 0.0, 1.0), delta, 8.0, 1.5)

	# The seed keeps blooming: emit new shoots from the centre over time until the pool fills,
	# so the system is always sprouting rather than settling into one finished shape.
	_emit_t -= delta
	if _emit_t <= 0.0:
		_emit_t = _emit_interval * _rng.randf_range(0.7, 1.4)
		if _roots.size() < _max_roots:
			_emit_root()

	# A slow base creep, surged by beats/energy through a spike curve.
	var drive := 0.55 + 1.1 * Nonlinear.apply("spike", clampf(0.7 * f.energy + f.beat, 0.0, 1.0), 2.0)
	for r in _roots:
		_advance(r, delta, drive)
	queue_redraw()


# One root's lifecycle step: grow -> hold -> (fade | rewind) -> regrow.
func _advance(r: Dictionary, delta: float, drive: float) -> void:
	match r.state:
		"grow":
			r.grown = minf(1.0, r.grown + delta * 0.26 * float(r.rate) * drive)
			if r.grown >= 1.0:
				r.state = "hold"
				r.timer = r.hold
		"hold":
			r.timer -= delta
			if r.timer <= 0.0:
				r.state = r.mode
				if r.state == "rewind":
					r.retract_to = _roll_retract()
		"fade":
			r.life = maxf(0.0, r.life - delta * 0.55)
			if r.life <= 0.0:
				_regrow(r)
		"rewind":
			# Retract the front back toward the seed - the root collapses partly inward.
			var floor_v: float = r.retract_to
			r.grown = maxf(floor_v, r.grown - delta * 0.4 * (0.6 + 0.5 * drive))
			if r.grown <= floor_v + 0.005:
				if floor_v < 0.08:
					_regrow(r)                  # fully retracted -> fresh path
				else:
					r.state = "grow"            # partial -> regrow the same root back up
					r.hold = _rng.randf_range(0.6, 2.2)
					r.mode = _roll_mode()


# How far a rewinding root pulls back: almost always a shallow dieback (then it grows out
# again on the same root); a deep collapse that nearly erases it is rare.
func _roll_retract() -> float:
	if _rng.randf() < 0.10:
		return _rng.randf_range(0.0, 0.30)    # rare: deep collapse (< 0.08 = fresh path)
	return _rng.randf_range(0.55, 0.88)        # usual: a small partial dieback


# Mostly a partial REWIND (dieback + regrow), only occasionally a full FADE + new path.
func _roll_mode() -> String:
	return "fade" if _rng.randf() < 0.12 else "rewind"


func _draw() -> void:
	begin_draw()
	var u := unit()
	# Timelapse twitch: trunks hold, young tips tremble (more with energy).
	var jitter := 0.008 + 0.018 * _f.energy
	for r in _roots:
		if r.fil == null:
			continue
		_draw_life = float(r.life)
		var tip := Color.from_hsv(fposmod(_hue + 0.5, 1.0), 0.25, 1.0, 0.85 * _draw_life)
		r.fil.draw_growing(self, u, float(r.grown), _color_for, tip, jitter, _life)


# Colour with TEXTURE along the strand, not one flat tone: the hue drifts and the strand
# desaturates from base to tip (a woody-to-fresh gradient), brightness ramps toward the tip,
# and a fine sinusoidal band rides along the length so the line reads as grain / pattern
# rather than a uniform stroke. `along` is 0 at the base, 1 at the growing tip. Brightness is
# still carried by bass + beat glow; alpha by the root's current life.
func _color_for(depth: int, along := 0.0) -> Color:
	var h := fposmod(_hue + _hue_depth * float(depth) + 0.10 * along, 1.0)
	var band := 0.11 * sin(along * 38.0 + float(depth) * 1.9)        # fine grain along the length
	var v := clampf(0.28 + 0.40 * along + 0.42 * _f.bass + 0.38 * _glow + band, 0.10, 1.0)
	var sat := clampf(0.72 - 0.30 * along, 0.0, 1.0)                 # fresher / paler toward the tip
	return Color.from_hsv(h, sat, v, 0.92 * _draw_life)
