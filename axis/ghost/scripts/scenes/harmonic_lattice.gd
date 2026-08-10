extends GhostScene

## Harmonic lattice - a grid of cells that breathe with the spectrum.
##
## A rows x cols lattice. Each cell samples the spectrum by its column (low bands
## left, high right) and a traveling wave whose phase is pushed by the bass, so
## energy visibly sweeps across the grid. Cells scale, spin, and shift hue. The
## field is overscanned so the view's tilt and drift never expose the edges.

const OVER := 1.35   # draw this much beyond the screen, for view motion headroom

# How finely the field is divided. This is the one choice that decides whether the
# lattice reads as a few big panels or a fine mesh, and every lattice used to be
# built inside the same middling band, so they all read alike.
const DENSITIES := {
	"coarse": {"cols": [4, 8], "rows": [3, 6], "fill": [0.60, 0.95]},
	"medium": {"cols": [8, 18], "rows": [5, 12], "fill": [0.55, 0.85]},
	"fine": {"cols": [18, 28], "rows": [10, 17], "fill": [0.45, 0.80]},
}

# The cell itself. `turn` is the polygon's resting rotation in whole turns - an
# eighth stands a 4-gon on its edge (the old "square"), none leaves it on its point
# (the old "diamond"). Those two were the entire vocabulary before.
const FORMS := {
	"square": {"sides": 4, "turn": 0.125},
	"diamond": {"sides": 4, "turn": 0.0},
	"triangle": {"sides": 3, "turn": 0.25},
	"hex": {"sides": 6, "turn": 0.0},
	"disc": {"sides": 14, "turn": 0.0},
}

var _phase := 0.0
var _hue_t := 0.0     # flowing-hue clock: advances with the audio so the palette drifts
var _f: AudioFeatures = AudioFeatures.new()
var _act: Activation
var _light: Lighting
var _ch := Vector2.ZERO            # live tonal colour (hue, strength) from the harmonic signature
var _sig := PackedFloat32Array()   # the 12 chroma channels + coarse shape (continuous)


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "field"
	# A lattice is abstract geometry - no mood is wrong for it, so take the whole set.
	var sch := Scheme.pick(rng)
	var dname := String(DENSITIES.keys()[rng.randi() % DENSITIES.size()])
	var den: Dictionary = DENSITIES[dname]
	var fname := String(FORMS.keys()[rng.randi() % FORMS.size()])
	var form: Dictionary = FORMS[fname]
	var cr: Array = den["cols"]
	var rr: Array = den["rows"]
	var fr: Array = den["fill"]
	var cols := rng.randi_range(int(cr[0]), int(cr[1]))
	var rows := rng.randi_range(int(rr[0]), int(rr[1]))
	var sparsity := 0.0 if rng.randf() < 0.4 else rng.randf_range(0.35, 0.7)
	_act = Activation.new(cols * rows, rng, sparsity)
	_light = Lighting.new(rng)
	# The gradients across the field travel from the scheme's base hue toward its
	# accent rather than an arbitrary distance, so the whole grid stays one family.
	var to_accent := fposmod(sch.accent - sch.hue + 0.5, 1.0) - 0.5
	return {
		"cols": cols,
		"rows": rows,
		"mood": sch.name,
		"density": dname,
		"form": fname,
		"hue": sch.hue,
		"hue_flow": to_accent * rng.randf_range(0.6, 1.6),    # horizontal hue gradient
		"hue_flow_y": to_accent * rng.randf_range(0.10, 0.60),  # vertical hue gradient
		"hue_wave": sch.spread * rng.randf_range(0.8, 2.2),   # amplitude of the travelling hue wave
		"hue_drift": rng.randf_range(0.10, 0.40),    # how fast the whole palette flows
		"sat": sch.sat,
		"sat_var": sch.sat * rng.randf_range(0.08, 0.28),   # saturation variance across the field
		"val": 0.75 + 0.30 * sch.val,                # the mood's brightness character
		"wave_freq": rng.randf_range(1.5, 4.0),
		"wave_speed": rng.randf_range(0.5, 2.0),
		"cell_fill": rng.randf_range(float(fr[0]), float(fr[1])),
		"spin": rng.randf_range(0.0, 1.0),
		"sides": int(form["sides"]),
		"turn": float(form["turn"]),
		"aspect": rng.randf_range(0.70, 1.45),       # stretched cells: bars, not only squares
	}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.03, 0.05, 0.04, 0.10)
	_act.update(f.energy + 0.4 * f.beat, delta)
	_light.update(f, delta)
	_phase += (float(params.wave_speed) * (0.4 + 0.6 * f.bass) + f.treble) * 0.5 * delta
	_hue_t += float(params.hue_drift) * (0.5 + 0.8 * f.energy + 0.6 * f.beat) * delta
	# Continuous harmonic seeding: the chroma channels and the tonality, read live each frame.
	_sig = Spectrum.harmonic_signature()
	_ch = chroma_hue()
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
	var hue_flow_y: float = params.hue_flow_y
	var hue_wave: float = params.hue_wave
	var sat0: float = params.sat
	var sat_var: float = params.sat_var
	var wave_freq: float = params.wave_freq
	var spin: float = params.spin
	var val_mul: float = params.val
	var sides: int = params.sides
	var turn: float = params.turn
	var aspect: float = params.aspect
	var origin := -field * 0.5

	for r in rows:
		var rt := float(r) / float(maxi(1, rows - 1))
		for col in cols:
			var idx := r * cols + col
			var pos := origin + Vector2((col + 0.5) * cell_w, (r + 0.5) * cell_h)
			# Per-cell independent drift (fluid behavior only; 0 otherwise).
			pos += Vector2(wobble("cx", idx), wobble("cy", idx)) * spacing * 0.3
			var t := float(col) / float(maxi(1, cols - 1))
			var wave := 0.5 + 0.5 * sin(wave_freq * (t + float(r) / float(rows)) * TAU - _phase)
			var e: float = _f.sample(t) * 0.7 + wave * 0.3 * _f.energy
			e = clampf(e + _f.beat * 0.2, 0.0, 1.0)
			# Each COLUMN maps to a pitch class: that chroma channel lights the column, so the
			# grid reads the music's harmonics tonally (not just raw band energy left-to-right).
			var pc := int(t * 12.0) % 12
			if _sig.size() > pc:
				e = clampf(e + 0.85 * _sig[pc], 0.0, 1.0)
			# Rooted cells hold a small base size; activated cells swell and decay.
			e *= 0.2 + 0.8 * _act.level(idx)
			# Size stays mostly stable; colour and brightness carry the audio - a
			# moving hotspot lights the cells it sweeps, and the beat glow lifts all.
			var s := max_size * (0.55 + 0.30 * e)
			var lit := _light.at(pos / u)
			# A flowing hue *map*: a diagonal gradient across the field (hue_flow x +
			# hue_flow_y y) plus a travelling wave that the audio drives forward (_hue_t),
			# so colour is never static - it gradients across the grid and drifts over time.
			var h := fposmod(hue + hue_flow * t + hue_flow_y * rt
				+ hue_wave * sin((t + rt) * TAU - _hue_t * TAU) + 0.06 * _hue_t
				+ 0.12 * lit + _light.hue_shift(), 1.0)
			# Pull the palette toward the live tonal hue, scaled by how tonal the moment is.
			var dh: float = _ch.x - h
			h = fposmod(h + (dh - round(dh)) * 0.4 * _ch.y, 1.0)
			var sat := clampf(sat0 + sat_var * sin((t - rt) * PI + _hue_t * 1.3) - 0.25 * lit, 0.0, 1.0)
			var val := clampf((0.28 + 0.3 * e + 0.55 * lit + 0.4 * _light.glow()) * val_mul, 0.05, 1.0)
			_draw_cell(pos, s, e * spin, Color.from_hsv(h, sat, val), sides, turn, aspect)


func _draw_cell(pos: Vector2, s: float, rot: float, col: Color, sides: int, turn: float, aspect: float) -> void:
	# Size every form by its INRADIUS, so a cell spans the same width across its flats
	# whatever it has for corners (a 4-gon comes out exactly the old square). Capped,
	# because a triangle's true circumradius would spill into its neighbours.
	var r := s * 0.5 * minf(1.3, 1.0 / cos(PI / float(sides)))
	var base_rot := rot * PI + turn * TAU
	var out := PackedVector2Array()
	for i in sides:
		# The cell as a regular n-gon, stretched on one axis - a 4-gon at an eighth
		# turn is the original square, so the old look is still in the set.
		var a := TAU * float(i) / float(sides)
		out.append(pos + Vector2(cos(a) * r * aspect, sin(a) * r / aspect).rotated(base_rot))
	fill_aa(out, col)
