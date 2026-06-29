extends GhostScene

## Rooted growth - crawling roots and tendrils that spread from a seed.
##
## Built on the [Filament] primitive: each root follows a curl-noise [Flow2D] field so
## it meanders and curls, branches into trunk-and-limb structure, tapers, and is
## revealed along a growth front so it *crawls* outward rather than appearing whole.
##
## Each root runs an independent **lifecycle**, staggered and rate-varied so the field
## is asynchronous (no uniform bands): it grows slowly to full, holds, then **retires**
## gracefully - either FADING out (alpha to zero) or REWINDING (the front retracts back
## toward the seed, collapsing inward) - before regrowing on a fresh path. It never
## clears and pops back in. Audio surges the growth front (a nonlinear `spike`, so beats
## lunge it and quiet barely moves) and carries colour and glow.

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


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	_rng.seed = rng.randi()
	_hue = rng.randf()
	_hue_depth = rng.randf_range(-0.05, 0.12)
	_variant = "tendril" if rng.randf() < 0.4 else "root"
	_flow = Flow2D.new(rng.randi(), rng.randf_range(2.0, 3.4), 0.06)
	_width = rng.randf_range(5.0, 9.0)
	_base_len = rng.randf_range(0.42, 0.60)
	var count := rng.randi_range(4, 7)
	for i in count:
		var r := {"fil": null, "grown": 0.0, "life": 1.0, "state": "grow",
			"timer": 0.0, "rate": 1.0, "hold": 2.0, "mode": "fade", "retract_to": 0.0,
			"heading": -PI * 0.5 + TAU * float(i) / float(count) + rng.randf_range(-0.3, 0.3)}
		_regrow(r)
		# Stagger: start each root at a random point in its own lifecycle, so they are
		# desynchronised from frame zero (and the varied rates keep them apart).
		r.grown = rng.randf_range(0.0, 1.0)
		if r.grown >= 1.0:
			r.state = "hold"
			r.timer = r.hold
		_roots.append(r)
	return {}


# Grow a fresh path for one root from the centre, on a new heading, and re-roll its
# lifecycle constants (rate / hold / retire mode) so each life differs from the last.
func _regrow(r: Dictionary) -> void:
	var steps := _rng.randi_range(12, 20)
	r.heading += _rng.randf_range(-0.6, 0.6)
	var length := _base_len * _rng.randf_range(0.8, 1.25)
	r.fil = Filament.grow(_variant, Vector2.ZERO, r.heading, length, _width, steps, _flow, _rng)
	r.grown = 0.0
	r.life = 1.0
	r.state = "grow"
	r.rate = _gauss_rate()
	r.hold = _rng.randf_range(1.6, 4.5)
	r.mode = "rewind" if _rng.randf() < 0.30 else "fade"   # mostly fade, sometimes collapse inward


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
					# Varied retraction depth: usually a partial pull-back (then it grows
					# out again on the same root), occasionally a full collapse + new path.
					r.retract_to = _rng.randf_range(0.0, 0.65)
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
					r.hold = _rng.randf_range(1.6, 4.5)
					r.mode = "rewind" if _rng.randf() < 0.30 else "fade"


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


# Palette per branch depth, brightness carried by bass + beat glow (colour over scale),
# alpha by the root's current life (so a fade dissolves gracefully, never pops).
func _color_for(depth: int) -> Color:
	var h := fposmod(_hue + _hue_depth * float(depth), 1.0)
	var v := clampf(0.40 + 0.45 * _f.bass + 0.40 * _glow, 0.10, 1.0)
	return Color.from_hsv(h, 0.6, v, 0.92 * _draw_life)
