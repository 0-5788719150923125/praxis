extends GhostScene

## Harmonic lattice - a grid of cells that breathe with the spectrum.
##
## A rows x cols lattice. Each cell samples the spectrum by its column (low bands
## left, high right) and a traveling wave whose phase is pushed by the bass, so
## energy visibly sweeps across the grid. Cells scale, spin, and shift hue. The
## field is overscanned so the view's tilt and drift never expose the edges.

const OVER := 1.35   # draw this much beyond the screen, for view motion headroom

var _phase := 0.0
var _f: AudioFeatures = AudioFeatures.new()
var _act: Activation
var _light: Lighting


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "field"
	var cols := rng.randi_range(8, 18)
	var rows := rng.randi_range(5, 12)
	var sparsity := 0.0 if rng.randf() < 0.4 else rng.randf_range(0.35, 0.7)
	_act = Activation.new(cols * rows, rng, sparsity)
	_light = Lighting.new(rng)
	return {
		"cols": cols,
		"rows": rows,
		"hue": rng.randf(),
		"hue_flow": rng.randf_range(0.2, 0.8),
		"wave_freq": rng.randf_range(1.5, 4.0),
		"wave_speed": rng.randf_range(0.5, 2.0),
		"cell_fill": rng.randf_range(0.55, 0.85),
		"spin": rng.randf_range(0.0, 1.0),
		"diamond": rng.randf() < 0.5,
	}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.03, 0.05, 0.04, 0.10)
	_act.update(f.energy + 0.4 * f.beat, delta)
	_light.update(f, delta)
	_phase += (float(params.wave_speed) * (0.4 + 0.6 * f.bass) + f.treble) * 0.5 * delta
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()
	var cols := int(params.cols)
	var rows := int(params.rows)
	var field := size * OVER
	var cell_w := field.x / float(cols)
	var cell_h := field.y / float(rows)
	var spacing := minf(cell_w, cell_h)
	var max_size: float = spacing * float(params.cell_fill)
	var hue: float = params.hue
	var hue_flow: float = params.hue_flow
	var wave_freq: float = params.wave_freq
	var spin: float = params.spin
	var diamond: bool = params.diamond
	var origin := -field * 0.5

	for r in rows:
		for col in cols:
			var idx := r * cols + col
			var pos := origin + Vector2((col + 0.5) * cell_w, (r + 0.5) * cell_h)
			# Per-cell independent drift (fluid behavior only; 0 otherwise).
			pos += Vector2(wobble("cx", idx), wobble("cy", idx)) * spacing * 0.3
			var t := float(col) / float(maxi(1, cols - 1))
			var wave := 0.5 + 0.5 * sin(wave_freq * (t + float(r) / float(rows)) * TAU - _phase)
			var e: float = _f.sample(t) * 0.7 + wave * 0.3 * _f.energy
			e = clampf(e + _f.beat * 0.2, 0.0, 1.0)
			# Rooted cells hold a small base size; activated cells swell and decay.
			e *= 0.2 + 0.8 * _act.level(idx)
			# Size stays mostly stable; colour and brightness carry the audio - a
			# moving hotspot lights the cells it sweeps, and the beat glow lifts all.
			var s := max_size * (0.55 + 0.30 * e)
			var lit := _light.at(pos / u)
			var h := fposmod(hue + hue_flow * t + 0.12 * lit + _light.hue_shift(), 1.0)
			var val := clampf(0.28 + 0.3 * e + 0.55 * lit + 0.4 * _light.glow(), 0.05, 1.0)
			_draw_cell(pos, s, e * spin, Color.from_hsv(h, 0.65, val), diamond)


func _draw_cell(pos: Vector2, s: float, rot: float, col: Color, diamond: bool) -> void:
	var half := s * 0.5
	var base_rot := rot * PI + (PI * 0.25 if diamond else 0.0)
	var pts := PackedVector2Array([
		Vector2(-half, -half), Vector2(half, -half),
		Vector2(half, half), Vector2(-half, half),
	])
	var out := PackedVector2Array()
	for p in pts:
		out.append(pos + p.rotated(base_rot))
	draw_colored_polygon(out, col)
