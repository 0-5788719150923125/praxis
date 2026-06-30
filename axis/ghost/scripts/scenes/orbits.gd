extends GhostScene

## Orbits - harmonograph curves that morph with the music.
##
## Each curve is a damped sum of sines in x and y - the figure a harmonograph
## pen traces. A slow global phase advances every frame, so the figure breathes
## and folds through itself continuously, and the per-axis amplitudes are driven
## by the spectrum, so loud passages swell the curve outward. Incommensurate
## frequencies mean the trace rarely returns to the same shape.

const SAMPLES := 480
const SPAN := 22.0      # parametric length of the trace

var _f: AudioFeatures = AudioFeatures.new()
var _gphase := 0.0
var _curves: Array = []
var _ch := Vector2.ZERO       # live tonal colour (hue, strength) from the harmonic signature


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	var base_hue := rng.randf()
	var n := rng.randi_range(2, 3)
	for c in n:
		var f := PackedFloat32Array()
		var ph := PackedFloat32Array()
		for k in 4:
			f.append(rng.randf_range(1.0, 5.0))
			ph.append(rng.randf_range(0.0, TAU))
		_curves.append({
			"f": f,
			"ph": ph,
			"amp": rng.randf_range(0.20, 0.34),     # fraction of unit
			"damp": rng.randf_range(0.02, 0.06),
			"hue": fposmod(base_hue + float(c) * rng.randf_range(0.12, 0.3), 1.0),
			"width": rng.randf_range(1.5, 3.0),
		})
	return {"base_hue": base_hue}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.04, 0.06, 0.06, 0.12)
	_gphase += delta * (0.07 + 0.35 * f.treble + 0.1 * mod.unit("rate"))
	_ch = chroma_hue()        # the music's tonality, as a hue + strength (continuous modulation)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()
	for ci in _curves.size():
		var cv: Dictionary = _curves[ci]
		_draw_curve(cv, u)


func _draw_curve(cv: Dictionary, u: float) -> void:
	var f: PackedFloat32Array = cv.f
	var ph: PackedFloat32Array = cv.ph
	# Tonal swell: the curve reaches wider when the moment is strongly tonal.
	var scale: float = u * float(cv.amp) * (1.0 + 0.3 * _ch.y)
	# Amplitudes leaning on different bands so the curve pulses asymmetrically.
	var ax := scale * (0.6 + 0.7 * _f.bass)
	var ay := scale * (0.6 + 0.7 * _f.mid)
	var pts := PackedVector2Array()
	pts.resize(SAMPLES)
	for k in SAMPLES:
		var s := float(k) / float(SAMPLES - 1) * SPAN
		var d := exp(-s * float(cv.damp))
		var x := (sin(f[0] * s + ph[0] + _gphase) + sin(f[1] * s + ph[1])) * d
		var y := (sin(f[2] * s + ph[2] + _gphase) + sin(f[3] * s + ph[3])) * d
		pts[k] = Vector2(x * ax, y * ay)
	# Pull the curve's hue toward the live tonal hue (circular nudge, scaled by how tonal it is),
	# so the palette tracks the music's key - the continuous half of harmonic seeding.
	var dh: float = _ch.x - float(cv.hue)
	dh = dh - round(dh)                              # shortest way around the wheel
	var hue := fposmod(float(cv.hue) + dh * 0.5 * _ch.y, 1.0)
	var col := Color.from_hsv(hue, 0.7, 0.6 + 0.4 * _f.energy, 0.85)
	draw_polyline(pts, col, float(cv.width), true)
