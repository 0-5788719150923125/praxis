extends VortexScene

## Spectrum ring - the spectrum bent into a circle.
##
## The whole band array is laid around a ring; each band pushes a bar outward by
## its energy. The ring breathes with energy, spins at a seeded rate, flashes on
## the beat, and the whole frame drifts/tilts on its own through the view.

var _spin := 0.0
var _drift := 0.0
var _free := 0.0
var _f: AudioFeatures = AudioFeatures.new()
var _act: Activation
var _light: Lighting


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	var sparsity := 0.0 if rng.randf() < 0.4 else rng.randf_range(0.3, 0.7)
	_act = Activation.new(Spectrum.BAND_COUNT, rng, sparsity)
	_light = Lighting.new(rng)
	_free = 0.0 if rng.randf() < 0.65 else rng.randf_range(0.3, 1.0)
	return {
		"hue": rng.randf(),
		"hue_spread": rng.randf_range(0.1, 0.6),
		"radius": rng.randf_range(0.16, 0.26),
		"bar_len": rng.randf_range(0.12, 0.30),
		"spin_rate": rng.randf_range(-0.10, 0.10),
		"mirror": rng.randi_range(1, 2),
		"thickness": rng.randf_range(2.0, 6.0),
	}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.05, 0.07, 0.06, 0.12)
	_act.update(f.energy + 0.5 * f.beat, delta)
	_light.update(f, delta)
	# Bounded sway about rest; continuous slow drift only when free.
	_drift += float(params.spin_rate) * delta * _free
	_spin = _drift + 0.22 * mod.value("turn")
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()
	var base_r: float = u * params.radius          # fixed - the ring doesn't breathe
	var radius_frac: float = params.radius
	var max_len: float = u * params.bar_len
	var glow := _light.glow()
	var n := _f.bands.size()
	if n == 0:
		n = Spectrum.BAND_COUNT

	var mirror := int(params.mirror)
	var thickness: float = params.thickness
	var hue: float = params.hue
	var hue_spread: float = params.hue_spread
	var count := n * mirror
	for i in count:
		var band_i := i % n if mirror == 1 else (i if i < n else count - 1 - i)
		band_i = clampi(band_i, 0, n - 1)
		var e := _f.bands[band_i] if _f.bands.size() > 0 else 0.0
		# Activation gates how far each bar reaches: rooted bands stay short, active
		# ones swell and decay. Each bar also drifts on its own radius/angle (fluid
		# behavior only); rigid behaviors leave wobble() at 0.
		e *= 0.25 + 0.75 * _act.level(band_i)
		var r_i := base_r * (1.0 + 0.10 * wobble("bar", i))
		var ang := _spin + TAU * float(i) / float(count) + 0.05 * wobble("ang", i)
		var dir := Vector2(cos(ang), sin(ang))
		var inner := dir * r_i
		var outer := dir * (r_i + max_len * e)
		# Colour, not size, carries the audio: a moving hotspot lights the bars it
		# sweeps over, and the global glow lifts everything on a beat.
		var lit := _light.at(dir * radius_frac)
		var h := fposmod(hue + hue_spread * float(band_i) / float(n) + 0.10 * lit + _light.hue_shift(), 1.0)
		var val := clampf(0.35 + 0.4 * e + 0.5 * lit + 0.4 * glow, 0.0, 1.0)
		draw_line(inner, outer, Color.from_hsv(h, 0.7, val), thickness, true)

	# Core glows on the beat (fixed size).
	var core := Color.from_hsv(hue, 0.4, 1.0, 0.2 + 0.6 * glow)
	draw_circle(Vector2.ZERO, base_r * 0.5, core)
