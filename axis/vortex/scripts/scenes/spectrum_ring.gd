extends VortexScene

## Spectrum ring - the spectrum bent into a circle.
##
## The whole band array is laid around a ring; each band pushes a bar outward by
## its energy. The ring breathes with overall energy, spins at a seeded rate,
## and flashes on the beat. The simplest scene that already reads as "music."

var _spin := 0.0
var _f: AudioFeatures = AudioFeatures.new()


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	return {
		"hue": rng.randf(),                       # base color
		"hue_spread": rng.randf_range(0.1, 0.6),  # color drift around the ring
		"radius": rng.randf_range(0.18, 0.28),    # base ring radius (of min axis)
		"bar_len": rng.randf_range(0.12, 0.30),   # how far bars reach
		"spin_rate": rng.randf_range(-0.5, 0.5),  # radians/sec, seeded sign
		"mirror": rng.randi_range(1, 2),          # 1 = full ring, 2 = symmetric
		"thickness": rng.randf_range(2.0, 6.0),
	}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	# Beat nudges the spin so hits feel like a kick of rotation.
	_spin += (params.spin_rate + f.beat * 1.5) * delta
	queue_redraw()


func _draw() -> void:
	var c := center()
	var unit := minf(size.x, size.y)
	var base_r: float = unit * params.radius * (1.0 + 0.15 * _f.energy + 0.1 * _f.beat)
	var max_len: float = unit * params.bar_len
	var n := _f.bands.size()
	if n == 0:
		n = Spectrum.BAND_COUNT  # idle: draw an empty ring so the scene is visible

	var count := n * int(params.mirror)
	for i in count:
		var band_i := i % n if params.mirror == 1 else (i if i < n else count - 1 - i)
		band_i = clampi(band_i, 0, n - 1)
		var e := _f.bands[band_i] if _f.bands.size() > 0 else 0.0
		var ang := _spin + TAU * float(i) / float(count)
		var dir := Vector2(cos(ang), sin(ang))
		var inner := c + dir * base_r
		var outer := c + dir * (base_r + max_len * e)
		var h := fposmod(params.hue + params.hue_spread * float(band_i) / float(n), 1.0)
		var col := Color.from_hsv(h, 0.7, 0.5 + 0.5 * e)
		draw_line(inner, outer, col, params.thickness, true)

	# Soft core that pulses with the bass.
	var core := Color.from_hsv(params.hue, 0.4, 1.0, 0.25 + 0.5 * _f.bass)
	draw_circle(c, base_r * 0.5 * (0.8 + 0.6 * _f.bass), core)
