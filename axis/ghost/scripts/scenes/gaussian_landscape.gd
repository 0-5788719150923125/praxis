extends GhostScene

## Gaussian landscape - rolling terrain with fog in the valleys.
##
## A heightfield built from a handful of Gaussian bumps, drawn as rows of filled
## ridgelines receding toward a horizon (near rows occlude far ones). Each bump's
## height is driven by a slice of the spectrum, so peaks rise and fall with the
## music. Over the low ground sits a translucent fog whose top edge undulates and
## scrolls sideways - the valleys breathing in a gentle wind while the peaks stay
## clear above it.

const ROWS := 18
const COLS := 56

var _f: AudioFeatures = AudioFeatures.new()
var _t := 0.0                       # wind / fog scroll clock
var _bumps: Array = []
var _amps := PackedFloat32Array()   # per-bump amplitude this frame


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "field"
	var n := rng.randi_range(4, 7)
	for i in n:
		_bumps.append({
			"x": rng.randf_range(-0.5, 0.5),   # grid x (-0.5..0.5)
			"y": rng.randf_range(0.0, 1.0),    # depth (0 far .. 1 near)
			"amp": rng.randf_range(0.5, 1.0),
			"sigma": rng.randf_range(0.12, 0.28),
			"band": rng.randf(),
		})
	_amps.resize(n)
	# Sky + weather over the hills, composed from the Layer registry: stars behind the
	# ridgelines, snow drifting over them - "snow on a hillside" from shared components.
	var sky := rng.randf()
	if sky < 0.35:
		add_layer("stars", rng, {"z": "back", "count": rng.randi_range(80, 150), "hue": 0.6})
		add_layer("snow", rng, {"count": rng.randi_range(60, 110), "hue": 0.58, "fall": 0.08})
	elif sky < 0.55:
		add_layer("stars", rng, {"z": "back", "count": rng.randi_range(100, 170), "hue": 0.62})
	return {
		"hue": rng.randf(),
		"hue_span": rng.randf_range(0.10, 0.35),
		"height": rng.randf_range(0.18, 0.30),   # peak height, fraction of unit
		"fog_hue": rng.randf(),
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
	var hue: float = params.hue
	var hue_span: float = params.hue_span

	# Far to near: each row fills down to the foot so it hides the rows behind.
	for r in ROWS:
		var depth := float(r) / float(ROWS - 1)
		var row_y := horizon + depth * span
		var pts := PackedVector2Array()
		pts.resize(COLS + 2)
		var crest := 0.0
		for c in COLS:
			var fx := float(c) / float(COLS - 1)
			var h := _height(fx - 0.5, depth)
			crest = maxf(crest, h)
			var y := minf(row_y - h * hpx, foot - 1.0)
			pts[c] = Vector2(left + fx * w, y)
		pts[COLS] = Vector2(left + w, foot)
		pts[COLS + 1] = Vector2(left, foot)
		var hh := fposmod(hue + hue_span * depth, 1.0)
		var val := 0.12 + 0.35 * depth + 0.25 * clampf(crest, 0.0, 1.0)
		draw_colored_polygon(pts, Color.from_hsv(hh, 0.5, val))

	_draw_fog(left, w, foot)
	draw_layers("front")         # snow drifting over the hills


# Translucent fog layers pooling over the low ground, flowing sideways.
func _draw_fog(left: float, w: float, foot: float) -> void:
	var fog_hue: float = params.fog_hue
	var base_y := -size.y * 0.5 + float(params.fog_level) * size.y
	for layer in 3:
		var pts := PackedVector2Array()
		pts.resize(COLS + 2)
		for c in COLS:
			var fx := float(c) / float(COLS - 1)
			var ripple := sin((fx * 4.0 + _t + layer * 0.7) * TAU) + 0.5 * sin((fx * 7.0 - _t * 0.6) * TAU)
			var y := base_y + layer * size.y * 0.045 + ripple * size.y * 0.02
			pts[c] = Vector2(left + fx * w, minf(y, foot - 1.0))
		pts[COLS] = Vector2(left + w, foot)
		pts[COLS + 1] = Vector2(left, foot)
		var a := 0.06 + 0.025 * _f.low_mid
		draw_colored_polygon(pts, Color.from_hsv(fog_hue, 0.10, 0.9, a))
