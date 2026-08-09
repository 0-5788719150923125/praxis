extends GhostScene

## Orbits - harmonograph curves that morph with the music.
##
## Each curve is a damped sum of sines in x and y - the figure a harmonograph pen traces.
## A slow global phase advances every frame, so the figure breathes and folds through
## itself continuously, and the per-axis amplitudes are driven by the spectrum, so loud
## passages swell the curve outward.
##
## The seed picks a FIGURE first, and the four figures are different curves, not different
## tunings of one: incommensurate frequencies fold a rosette that never repeats, small
## integer ratios close into a braid, a carrier plus a fast epicycle loops petals around
## an open ring, and a heavily damped pair spirals inward to a point. Colour comes from a
## shared [Scheme] mood, so a session is not one hue every time.

const SAMPLES := 480

# The four ways a pen on two pendulums can move. `ratio` names how the frequencies relate
# (that is what decides whether the trace ever closes), `w2` how much of the second term
# each axis carries (small = a ripple on a ring, large = a fold), and `damp`/`span` how far
# the trace runs before it dies. Change one of these and the SILHOUETTE changes, not the tint.
const FIGURES := {
	"harmonograph": {
		# Incommensurate frequencies, lightly damped: the classic folding rosette that
		# never quite returns to where it started.
		"curves": [2, 3], "ratio": "free", "w2": [0.6, 1.0], "damp": [0.02, 0.06],
		"span": [20.0, 26.0], "amp": [0.20, 0.34], "width": [1.5, 3.0], "bed": 0.5,
	},
	"lissajous": {
		# Small integer ratios and no damping at all: a closed braid that holds its shape
		# and only shears as the phase advances.
		"curves": [1, 3], "ratio": "integer", "w2": [0.0, 0.18], "damp": [0.0, 0.004],
		"span": [1.0, 1.0], "amp": [0.26, 0.40], "width": [2.0, 4.0], "bed": 0.6,
	},
	"spirograph": {
		# A dominant carrier plus a fast epicycle: petal loops marching around an open
		# ring, so this figure has a HOLE in the middle the others do not.
		"curves": [1, 2], "ratio": "epicycle", "w2": [0.18, 0.42], "damp": [0.0, 0.003],
		"span": [1.0, 1.0], "amp": [0.28, 0.42], "width": [1.2, 2.4], "bed": 0.65,
	},
	"pendulum": {
		# Heavily damped and nearly in tune with itself: an ellipse that spirals in and
		# dies at the centre, the way a real harmonograph ends its run.
		"curves": [2, 4], "ratio": "detune", "w2": [0.08, 0.22], "damp": [0.14, 0.26],
		"span": [26.0, 34.0], "amp": [0.30, 0.44], "width": [1.2, 2.2], "bed": 0.45,
	},
}

var _f: AudioFeatures = AudioFeatures.new()
var _gphase := 0.0
var _curves: Array = []
var _ch := Vector2.ZERO       # live tonal colour (hue, strength) from the harmonic signature
var _sch: Scheme = null


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	var keys := FIGURES.keys()
	var figure := String(keys[rng.randi() % keys.size()])
	var fig: Dictionary = FIGURES[figure]
	_sch = Scheme.pick(rng)                  # any mood suits a line drawing
	var n := rng.randi_range(int(fig["curves"][0]), int(fig["curves"][1]))
	var alpha := rng.randf_range(0.75, 0.95)
	for c in n:
		var cv := _shape(figure, fig, rng)
		cv["amp"] = rng.randf_range(float(fig["amp"][0]), float(fig["amp"][1]))
		cv["damp"] = rng.randf_range(float(fig["damp"][0]), float(fig["damp"][1]))
		cv["width"] = rng.randf_range(float(fig["width"][0]), float(fig["width"][1]))
		# Related hues rather than an arbitrary spacing: the i-th of n from base to accent.
		cv["hue"] = fposmod(_sch.hue_at(c, maxi(n, 2)) + rng.randf_range(-0.02, 0.02), 1.0)
		cv["alpha"] = alpha
		_curves.append(cv)
	# A faint wash behind the line work, on the mood - the void was black every session.
	if rng.randf() < float(fig["bed"]):
		add_layer("bed", rng, {
			"hue": _sch.accent,
			"sat": clampf(_sch.sat * rng.randf_range(0.5, 0.9), 0.05, 1.0),
			"val": rng.randf_range(0.05, 0.13),
			"pools": rng.randi_range(1, 3),
		})
	return {"hue": _sch.hue, "mood": _sch.name, "figure": figure, "curves": n}


# Frequencies, phases and per-term weights for one curve. The RELATION between the four
# frequencies is the whole difference between the figures: free ratios never close, integer
# ratios close, a 1:k pair with quarter-phase offsets is a carrier and its epicycle.
func _shape(figure: String, fig: Dictionary, rng: RandomNumberGenerator) -> Dictionary:
	var f := PackedFloat32Array([1.0, 1.0, 1.0, 1.0])
	var ph := PackedFloat32Array([0.0, 0.0, 0.0, 0.0])
	var w2 := rng.randf_range(float(fig["w2"][0]), float(fig["w2"][1]))
	var w := PackedFloat32Array([1.0, w2, 1.0, w2])
	var span := rng.randf_range(float(fig["span"][0]), float(fig["span"][1]))
	match String(fig["ratio"]):
		"integer":
			# x and y on different small whole multiples of one fundamental, with the
			# quarter-turn offset that opens the braid out instead of collapsing it to a line.
			var p := rng.randi_range(2, 5)
			var q := p + rng.randi_range(1, 4)
			f = PackedFloat32Array([float(p), float(p * 2), float(q), float(q * 2)])
			ph = PackedFloat32Array([rng.randf() * TAU, rng.randf() * TAU,
				0.0, rng.randf() * TAU])
			ph[2] = ph[0] + PI * 0.5
			span *= TAU                       # exactly one closed pass
		"epicycle":
			# Carrier at 1, epicycle at k, y a quarter turn behind x on BOTH terms - the
			# parametric form of an epitrochoid.
			var k := float(rng.randi_range(3, 9))
			var p0 := rng.randf() * TAU
			var p1 := rng.randf() * TAU
			f = PackedFloat32Array([1.0, k, 1.0, k])
			ph = PackedFloat32Array([p0, p1, p0 + PI * 0.5, p1 + PI * 0.5])
			span *= TAU
		"detune":
			# The two axes almost in tune: the figure precesses slowly as it decays.
			var b := rng.randf_range(1.6, 3.4)
			f = PackedFloat32Array([b, b * 2.0, b * rng.randf_range(0.97, 1.03), b * 2.0])
			for k2 in 4:
				ph[k2] = rng.randf() * TAU
		_:
			for k2 in 4:
				f[k2] = rng.randf_range(1.0, 5.0)
				ph[k2] = rng.randf() * TAU
	return {"f": f, "ph": ph, "w": w, "span": span, "figure": figure}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.04, 0.06, 0.06, 0.12)
	_gphase += delta * (0.07 + 0.35 * f.treble + 0.1 * mod.unit("rate"))
	_ch = chroma_hue()        # the music's tonality, as a hue + strength (continuous modulation)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
	var u := unit()
	for ci in _curves.size():
		var cv: Dictionary = _curves[ci]
		_draw_curve(cv, u)


func _draw_curve(cv: Dictionary, u: float) -> void:
	var f: PackedFloat32Array = cv.f
	var ph: PackedFloat32Array = cv.ph
	var w: PackedFloat32Array = cv.w
	var span: float = cv.span
	# Tonal swell: the curve reaches wider when the moment is strongly tonal.
	var scale: float = u * float(cv.amp) * (1.0 + 0.3 * _ch.y)
	# Amplitudes leaning on different bands so the curve pulses asymmetrically.
	var ax := scale * (0.6 + 0.7 * _f.bass)
	var ay := scale * (0.6 + 0.7 * _f.mid)
	var pts := PackedVector2Array()
	pts.resize(SAMPLES)
	for k in SAMPLES:
		var s := float(k) / float(SAMPLES - 1) * span
		var d := exp(-s * float(cv.damp))
		var x := (w[0] * sin(f[0] * s + ph[0] + _gphase) + w[1] * sin(f[1] * s + ph[1])) * d
		var y := (w[2] * sin(f[2] * s + ph[2] + _gphase) + w[3] * sin(f[3] * s + ph[3])) * d
		pts[k] = Vector2(x * ax, y * ay)
	# Pull the curve's hue toward the live tonal hue (circular nudge, scaled by how tonal it is),
	# so the palette tracks the music's key - the continuous half of harmonic seeding.
	var dh: float = _ch.x - float(cv.hue)
	dh = dh - round(dh)                              # shortest way around the wheel
	var hue := fposmod(float(cv.hue) + dh * 0.5 * _ch.y, 1.0)
	# Value still rides the energy, but through the mood's own character rather than a
	# hardcoded 0.7/0.6 - an ash figure stays wan where a toxic one burns.
	var col := _sch.color(hue, 1.0, 0.75 + 0.5 * _f.energy, float(cv.alpha))
	draw_polyline(pts, col, float(cv.width), true)
