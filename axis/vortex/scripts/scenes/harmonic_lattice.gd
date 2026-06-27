extends VortexScene

## Harmonic lattice - a grid of cells that breathe with the spectrum.
##
## A rows x cols lattice. Each cell samples the spectrum by its position (low
## bands left, high bands right) and a traveling wave whose phase is pushed by
## the named bands, so energy visibly sweeps across the grid. Cells scale, spin,
## and shift hue with the sound. A different shape of motion from the ring -
## same AudioFeatures, same contract.

var _phase := 0.0
var _f: AudioFeatures = AudioFeatures.new()


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	return {
		"cols": rng.randi_range(8, 18),
		"rows": rng.randi_range(5, 12),
		"hue": rng.randf(),
		"hue_flow": rng.randf_range(0.2, 0.8),     # hue shift across the grid
		"wave_freq": rng.randf_range(1.5, 4.0),    # spatial frequency of the wave
		"wave_speed": rng.randf_range(0.5, 2.0),
		"cell_fill": rng.randf_range(0.55, 0.85),  # max cell size vs. spacing
		"spin": rng.randf_range(0.0, 1.0),         # how much cells rotate
		"diamond": rng.randf() < 0.5,              # square vs. diamond cells
	}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	# Bass drives the wave forward; treble adds shimmer to its speed.
	_phase += (params.wave_speed * (0.5 + f.bass) + f.treble * 2.0) * delta
	queue_redraw()


func _draw() -> void:
	var cols: int = params.cols
	var rows: int = params.rows
	var cell_w := size.x / float(cols)
	var cell_h := size.y / float(rows)
	var spacing := minf(cell_w, cell_h)
	var max_size: float = spacing * params.cell_fill

	for r in rows:
		for col in cols:
			var pos := Vector2((col + 0.5) * cell_w, (r + 0.5) * cell_h)
			# Spectrum by column (low -> high), modulated by a traveling wave.
			var t := float(col) / float(maxi(1, cols - 1))
			var wave := 0.5 + 0.5 * sin(
				params.wave_freq * (t + float(r) / float(rows)) * TAU - _phase)
			var e: float = _f.sample(t) * 0.7 + wave * 0.3 * _f.energy
			e = clampf(e + _f.beat * 0.2, 0.0, 1.0)

			var s := max_size * (0.15 + 0.85 * e)
			var h := fposmod(params.hue + params.hue_flow * t, 1.0)
			var col_color := Color.from_hsv(h, 0.65, 0.4 + 0.6 * e)
			_draw_cell(pos, s, e * params.spin, col_color)


# A square or diamond centered at `pos`, rotated by `rot`.
func _draw_cell(pos: Vector2, s: float, rot: float, col: Color) -> void:
	var half := s * 0.5
	var pts := PackedVector2Array([
		Vector2(-half, -half), Vector2(half, -half),
		Vector2(half, half), Vector2(-half, half),
	])
	var base_rot := rot * PI + (PI * 0.25 if params.diamond else 0.0)
	var out := PackedVector2Array()
	for p in pts:
		out.append(pos + p.rotated(base_rot))
	draw_colored_polygon(out, col)
