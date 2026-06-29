extends RefCounted
class_name Palette

## Palette - a colour ramp sampled by a scalar (0..1). The colour half of the texture
## story: a [Field] gives a scalar (height, mottle, development); a Palette turns it into
## colour. Themed presets map terrain elevation to believable bands (water -> shore ->
## green -> rock -> snow), but a Palette colours anything driven by a 0..1 field.

var _stops: Array = []     # [{t: float, c: Color}] sorted by t


static func from_stops(stops: Array) -> Palette:
	var p := Palette.new()
	p._stops = stops.duplicate()
	p._stops.sort_custom(func(a, b): return float(a.t) < float(b.t))
	return p


# A few earthy / dramatic elevation ramps. `rng` jitters the hues so no two are identical.
static func named(name: String, rng: RandomNumberGenerator) -> Palette:
	var j := func(h): return fposmod(h + rng.randf_range(-0.03, 0.03), 1.0)
	match name:
		"desert":
			return _ramp([[0.0, j.call(0.09), 0.5, 0.32], [0.35, j.call(0.10), 0.55, 0.55],
				[0.7, j.call(0.08), 0.45, 0.78], [1.0, j.call(0.11), 0.20, 0.95]])
		"alpine":
			return _ramp([[0.0, j.call(0.58), 0.55, 0.30], [0.28, j.call(0.30), 0.45, 0.45],
				[0.55, j.call(0.10), 0.25, 0.55], [0.78, 0.0, 0.05, 0.78], [1.0, 0.0, 0.0, 1.0]])
		"volcanic":
			return _ramp([[0.0, 0.0, 0.0, 0.06], [0.45, j.call(0.02), 0.7, 0.30],
				[0.7, j.call(0.04), 0.9, 0.65], [0.88, j.call(0.10), 0.95, 0.95], [1.0, 0.15, 0.4, 1.0]])
		"alien":
			return _ramp([[0.0, j.call(0.78), 0.6, 0.25], [0.4, j.call(0.52), 0.6, 0.45],
				[0.7, j.call(0.40), 0.55, 0.6], [1.0, j.call(0.30), 0.25, 0.95]])
		"ocean":
			return _ramp([[0.0, j.call(0.62), 0.7, 0.18], [0.5, j.call(0.55), 0.6, 0.4],
				[0.78, j.call(0.12), 0.45, 0.7], [1.0, j.call(0.10), 0.18, 0.95]])
		_:  # "earth": deep water, shore, grass, rock, snow
			return _ramp([[0.0, j.call(0.60), 0.65, 0.28], [0.30, j.call(0.12), 0.55, 0.62],
				[0.45, j.call(0.32), 0.55, 0.50], [0.7, j.call(0.09), 0.40, 0.45],
				[0.88, j.call(0.07), 0.18, 0.62], [1.0, 0.0, 0.04, 0.98]])


static func _ramp(rows: Array) -> Palette:
	var stops: Array = []
	for r in rows:
		stops.append({"t": float(r[0]), "c": Color.from_hsv(float(r[1]), float(r[2]), float(r[3]))})
	return Palette.from_stops(stops)


## Colour at t (0..1), linearly interpolated between the stops.
func at(t: float) -> Color:
	t = clampf(t, 0.0, 1.0)
	if _stops.is_empty():
		return Color.MAGENTA
	if t <= float(_stops[0].t):
		return _stops[0].c
	for i in range(1, _stops.size()):
		var s0: Dictionary = _stops[i - 1]
		var s1: Dictionary = _stops[i]
		if t <= float(s1.t):
			var k := (t - float(s0.t)) / maxf(1e-5, float(s1.t) - float(s0.t))
			return (s0.c as Color).lerp(s1.c, k)
	return _stops[-1].c
