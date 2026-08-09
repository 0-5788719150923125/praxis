extends GhostScene

## Gaussian landscape - rolling terrain with fog in the valleys.
##
## A heightfield built from a handful of Gaussian bumps, drawn as rows of filled
## ridgelines receding toward a horizon (near rows occlude far ones). Each bump's
## height is driven by a slice of the spectrum, so peaks rise and fall with the
## music. Over the low ground sits a translucent fog whose top edge undulates and
## scrolls sideways - the valleys breathing in a gentle wind while the peaks stay
## clear above it.
##
## The land is not one shape or one colour: a [Scheme] mood colours the ridgelines (with the
## fog and the sky layers on its accent, so the weather belongs to the same evening), and a
## sampled RELIEF picks the landform - broad dunes, rolling hills, a crowded range of narrow
## ridges, or a couple of massifs - along with how finely it is drawn.

## Landforms. Bump count and sigma together decide whether the horizon is a few wide swells
## or a crowded serration, so they are sampled as a set; row/column counts follow, since a
## serrated range needs the resolution a dune field does not.
const RELIEFS := {
	"dunes":   {"bumps": [3, 5],  "sigma": [0.22, 0.40], "height": [0.14, 0.24],
		"rows": [12, 17], "cols": [40, 56]},
	"rolling": {"bumps": [4, 7],  "sigma": [0.12, 0.28], "height": [0.18, 0.30],
		"rows": [16, 22], "cols": [48, 64]},
	"ridges":  {"bumps": [8, 12], "sigma": [0.06, 0.13], "height": [0.26, 0.44],
		"rows": [18, 24], "cols": [56, 72]},
	"massif":  {"bumps": [2, 3],  "sigma": [0.16, 0.28], "height": [0.34, 0.52],
		"rows": [14, 20], "cols": [44, 60]},
}

var _f: AudioFeatures = AudioFeatures.new()
var _t := 0.0                       # wind / fog scroll clock
var _bumps: Array = []
var _amps := PackedFloat32Array()   # per-bump amplitude this frame
var _sch: Scheme
var _rows := 18                     # receding ridgelines - relief picks how many
var _cols := 56                     # samples across one ridgeline (its crispness)
var _fog_bands := 3


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "field"
	_sch = Scheme.pick(rng)
	var relief := String(RELIEFS.keys()[rng.randi() % RELIEFS.size()])
	var rl: Dictionary = RELIEFS[relief]
	_rows = rng.randi_range(int(rl.rows[0]), int(rl.rows[1]))
	_cols = rng.randi_range(int(rl.cols[0]), int(rl.cols[1]))
	var n := rng.randi_range(int(rl.bumps[0]), int(rl.bumps[1]))
	for i in n:
		_bumps.append({
			"x": rng.randf_range(-0.5, 0.5),   # grid x (-0.5..0.5)
			"y": rng.randf_range(0.0, 1.0),    # depth (0 far .. 1 near)
			"amp": rng.randf_range(0.5, 1.0),
			"sigma": rng.randf_range(float(rl.sigma[0]), float(rl.sigma[1])),
			"band": rng.randf(),
		})
	_amps.resize(n)
	# Fog is the scene's other half, so how much of it there is varies too: a thin single
	# band clinging to the low ground, or a deep stack drowning the valleys.
	_fog_bands = rng.randi_range(1, 4)
	# Sky + weather over the hills, composed from the Layer registry: stars behind the
	# ridgelines, snow drifting over them - "snow on a hillside" from shared components.
	# Both take the scheme's ACCENT, so the weather is lit by the same sky as the land.
	var sky := rng.randf()
	if sky < 0.35:
		add_layer("stars", rng, {"z": "back", "count": rng.randi_range(80, 150),
			"hue": _sch.accent, "sat": _sch.sat * 0.3})
		add_layer("snow", rng, {"count": rng.randi_range(60, 110),
			"hue": _sch.accent, "sat": _sch.sat * 0.25, "fall": 0.08})
	elif sky < 0.55:
		add_layer("stars", rng, {"z": "back", "count": rng.randi_range(100, 170),
			"hue": _sch.accent, "sat": _sch.sat * 0.3})
	return {
		"mood": _sch.name,
		"relief": relief,
		"rows": _rows,
		"bumps": n,
		# How far the receding rows walk from the base hue toward the accent - some evenings
		# are one colour into the distance, others grade the whole way across the scheme.
		"reach": rng.randf_range(0.25, 1.0),
		"height": rng.randf_range(float(rl.height[0]), float(rl.height[1])),  # fraction of unit
		"fog_bands": _fog_bands,
		"fog_level": rng.randf_range(0.30, 0.52), # screen fraction the fog sits at
		"wind": rng.randf_range(0.05, 0.18),
	}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.02, 0.04, 0.0, 0.06)
	update_layers(f, delta)
	_t += delta * float(params.wind)
	for i in _bumps.size():
		var b: Dictionary = _bumps[i]
		_amps[i] = float(b.amp) * (0.35 + 1.0 * f.sample(float(b.band)))
	queue_redraw()


# Heightfield value at grid x (-0.5..0.5) and depth (0..1): sum of the bumps.
func _height(gx: float, depth: float) -> float:
	var h := 0.0
	for i in _bumps.size():
		var b: Dictionary = _bumps[i]
		var dx := gx - float(b.x)
		var dy := depth - float(b.y)
		var s: float = float(b.sigma)
		h += _amps[i] * exp(-(dx * dx + dy * dy) / (2.0 * s * s))
	return h


func _draw() -> void:
	begin_draw()
	draw_layers("back")          # stars in the sky behind the ridgelines
	var w := size.x * 1.3
	var left := -w * 0.5
	var foot := size.y * 0.5
	var hpx: float = float(params.height) * unit()
	var horizon := -size.y * 0.15
	var span := size.y * 0.55
	# The near rows walk toward the scheme's accent by the sampled reach - the shortest way
	# round the wheel, so the grade is always the short arc between two related hues.
	var to_accent := fposmod(_sch.accent - _sch.hue + 0.5, 1.0) - 0.5
	var reach: float = float(params.reach) * to_accent

	# Far to near: each row fills down to the foot so it hides the rows behind.
	for r in _rows:
		var depth := float(r) / float(_rows - 1)
		var row_y := horizon + depth * span
		var pts := PackedVector2Array()
		pts.resize(_cols + 2)
		var crest := 0.0
		for c in _cols:
			var fx := float(c) / float(_cols - 1)
			var h := _height(fx - 0.5, depth)
			crest = maxf(crest, h)
			var y := minf(row_y - h * hpx, foot - 1.0)
			pts[c] = Vector2(left + fx * w, y)
		pts[_cols] = Vector2(left + w, foot)
		pts[_cols + 1] = Vector2(left, foot)
		var hh := fposmod(_sch.hue + reach * depth, 1.0)
		var val := 0.12 + 0.35 * depth + 0.25 * clampf(crest, 0.0, 1.0)
		draw_colored_polygon(pts, _sch.color(hh, 0.72, val * 1.2))

	_draw_fog(left, w, foot)
	draw_layers("front")         # snow drifting over the hills


# Translucent fog layers pooling over the low ground, flowing sideways.
func _draw_fog(left: float, w: float, foot: float) -> void:
	var base_y := -size.y * 0.5 + float(params.fog_level) * size.y
	for layer in _fog_bands:
		var pts := PackedVector2Array()
		pts.resize(_cols + 2)
		for c in _cols:
			var fx := float(c) / float(_cols - 1)
			var ripple := sin((fx * 4.0 + _t + layer * 0.7) * TAU) + 0.5 * sin((fx * 7.0 - _t * 0.6) * TAU)
			var y := base_y + layer * size.y * 0.045 + ripple * size.y * 0.02
			pts[c] = Vector2(left + fx * w, minf(y, foot - 1.0))
		pts[_cols] = Vector2(left + w, foot)
		pts[_cols + 1] = Vector2(left, foot)
		var a := 0.06 + 0.025 * _f.low_mid
		# Fog on the scheme's accent, barely saturated: it belongs to the same weather as
		# the hills instead of being an unrelated second hue.
		draw_colored_polygon(pts, _sch.color(_sch.accent, 0.15, 1.15, a))
