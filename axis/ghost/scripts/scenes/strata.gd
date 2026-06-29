extends GhostScene

## Strata - stacked waveform planes receding into depth.
##
## Horizontal planes are stacked back-to-front; each is a waveform whose height
## comes from a slice of the spectrum plus a traveling sine, and each is filled
## down to the foot of the frame as a translucent sheet. Nearer planes sit over
## farther ones, and the view's tilt skews the whole stack, so it reads as planes
## of light lying in space. Far planes scroll slower than near ones - parallax.

const OVER := 1.4
const COLS := 72       # samples across each plane

var _f: AudioFeatures = AudioFeatures.new()
var _t := 0.0


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "field"
	return {
		"planes": rng.randi_range(5, 9),
		"hue": rng.randf(),
		"hue_span": rng.randf_range(0.15, 0.5),
		"wave_k": rng.randf_range(1.5, 4.0),    # spatial frequency across x
		"amp": rng.randf_range(0.06, 0.13),     # crest height, fraction of unit
		"scroll": rng.randf_range(0.15, 0.45),
		"alpha": rng.randf_range(0.35, 0.55),
	}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	# Tilt-forward bias so the stack always reads as receding planes. This is an
	# absolute skew *target* (the view eases toward it) - it must be set, not
	# accumulated: `+=` ran the skew away every frame and sheared the whole stack.
	drift_view(f, 0.03, 0.04, 0.04, 0.08)
	view.skew = 0.18 + 0.05 * mod.value("tilt2")
	_t += delta * float(params.scroll)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var field := size * OVER
	var planes := int(params.planes)
	var step := field.y / float(planes)
	var top := -field.y * 0.5
	var left := -field.x * 0.5
	var foot := field.y * 0.5
	var amp_u := unit() * float(params.amp)
	var hue: float = params.hue
	var hue_span: float = params.hue_span
	var wave_k: float = params.wave_k
	var alpha: float = params.alpha

	# Far (top) to near (bottom): later draws cover earlier -> depth ordering.
	for i in planes:
		var depth := float(i) / float(planes - 1)        # 0 far .. 1 near
		var base_y := top + (i + 0.5) * step
		var band := 1.0 - depth                          # bass near, treble far
		var loud := _f.sample(band)
		var phase := _t * (0.4 + 0.6 * depth)            # parallax: near scrolls faster
		var crest := amp_u * (0.4 + 1.4 * loud + 0.3 * _f.beat)

		var pts := PackedVector2Array()
		pts.resize(COLS + 2)
		for c in COLS:
			var fx := float(c) / float(COLS - 1)
			var x := left + fx * field.x
			var wave := 0.6 * sin(wave_k * fx * TAU + phase) + 0.4 * (_f.sample(fx) - 0.5) * 2.0
			# Clamp every crest above the closing edge, so the band is always a
			# simple (non-self-intersecting) polygon that can be triangulated.
			var y := minf(base_y - wave * crest, foot - 1.0)
			pts[c] = Vector2(x, y)
		# Close the band down to the foot of the frame.
		pts[COLS] = Vector2(left + field.x, foot)
		pts[COLS + 1] = Vector2(left, foot)

		var h := fposmod(hue + hue_span * depth, 1.0)
		var fill := Color.from_hsv(h, 0.6, 0.25 + 0.6 * (0.3 + loud) * (0.4 + depth), alpha)
		draw_colored_polygon(pts, fill)

		# A brighter crest line for definition.
		var line := PackedVector2Array()
		line.resize(COLS)
		for c in COLS:
			line[c] = pts[c]
		var lcol := Color.from_hsv(h, 0.4, 0.7 + 0.3 * loud, 0.7)
		draw_polyline(line, lcol, 1.5 + 2.0 * depth, true)
