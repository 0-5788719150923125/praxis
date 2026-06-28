extends VortexScene

## Rooted growth - crawling roots and tendrils that spread from a seed.
##
## Rebuilt on the [Filament] primitive (the old version was a rigid recursive Y of
## straight 90-degree forks - geometric, not alive). Each root follows a curl-noise
## [Flow2D] field, so it meanders and curls; it branches stochastically and tapers;
## and it is revealed along a growth front, so it *crawls* outward rather than
## appearing whole. Roots grow, mature, then dissolve and regrow on a fresh path -
## staggered, so something is always creeping. Audio surges the growth front (the
## drive runs through a nonlinear `spike`, so beats push it hard while quiet barely
## moves) and drives colour and glow. Nonlinearity is what makes it read as living.

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


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	_rng.seed = rng.randi()
	_hue = rng.randf()
	_hue_depth = rng.randf_range(-0.05, 0.12)
	_variant = "tendril" if rng.randf() < 0.4 else "root"
	_flow = Flow2D.new(rng.randi(), rng.randf_range(2.0, 3.4), 0.06)
	_width = rng.randf_range(5.0, 9.0)
	_base_len = rng.randf_range(0.42, 0.60)
	var count := rng.randi_range(3, 6)
	for i in count:
		var heading := -PI * 0.5 + TAU * float(i) / float(count) + rng.randf_range(-0.3, 0.3)
		var r := {"fil": null, "grown": 0.0, "mature": 0.0, "heading": heading}
		_regrow(r)
		r.grown = rng.randf_range(0.5, 1.0)   # stagger: enter mid-growth so it's never bare
		_roots.append(r)
	return {}


# Grow a fresh path for one root from the centre, on a slightly new heading.
func _regrow(r: Dictionary) -> void:
	var steps := _rng.randi_range(11, 18)
	r.heading += _rng.randf_range(-0.5, 0.5)
	r.fil = Filament.grow(_variant, Vector2.ZERO, r.heading, _base_len, _width, steps, _flow, _rng)
	r.grown = 0.0
	r.mature = 0.0


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.03, 0.05)
	_flow.advance(delta)
	_glow = Nonlinear.flare(_glow, clampf(0.30 * f.energy + 0.70 * f.beat, 0.0, 1.0), delta, 8.0, 1.5)

	# Growth drive: a steady creep, surged hard by beats/energy through a spike curve
	# (so a quiet passage barely advances and a hit lunges the front outward).
	var drive := 0.35 + 1.5 * Nonlinear.apply("spike", clampf(0.7 * f.energy + f.beat, 0.0, 1.0), 2.0)
	for r in _roots:
		if r.grown < 1.0:
			r.grown = minf(1.0, r.grown + delta * drive * 0.42)
		else:
			r.mature += delta
			if r.mature > 3.0 and (f.beat > 0.6 or r.mature > 9.0):
				_regrow(r)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()
	var tip := Color.from_hsv(fposmod(_hue + 0.5, 1.0), 0.25, 1.0, 0.85)
	# Timelapse twitch: trunks hold, young tips tremble (more with energy).
	var jitter := 0.008 + 0.018 * _f.energy
	for r in _roots:
		if r.fil != null:
			r.fil.draw_growing(self, u, r.grown, _color_for, tip, jitter, _life)


# Palette per branch depth, brightness carried by bass + beat glow (colour over
# scale - the structure holds, the light moves).
func _color_for(depth: int) -> Color:
	var h := fposmod(_hue + _hue_depth * float(depth), 1.0)
	var v := clampf(0.40 + 0.45 * _f.bass + 0.40 * _glow, 0.10, 1.0)
	return Color.from_hsv(h, 0.6, v, 0.92)
