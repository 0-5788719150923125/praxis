extends RefCounted
class_name EyeBody

## EyeBody - a floating human eyeball (no lids, no blink), used by the eye scenes.
##
## A shaded sclera sphere with an iris/pupil that LOOKS AROUND in saccades: it holds
## a fixation, then jumps - fast and jerky - to a new target and holds again. The
## targets are drawn *center-biased* (a squared radius plus a frequent snap back to
## dead-centre), so the gaze prefers the neutral forward look even as it wanders.
## The nonlinearities are the whole point: the squared radius makes small glances far
## more likely than big ones, and the fast exponential snap + dwell makes the motion
## read as a living, deciding eye rather than a smooth drift. Pupil dilates with the
## audio. Reusable: the eye and two_eyes scenes both render these.

const LIGHT := Vector2(-0.34, -0.40)   # catchlight + shading direction (upper-left)

var gaze := Vector2.ZERO               # current look offset (small, ~ -0.42..0.42)
var hue := 0.58                        # public: a morph handoff can copy/override it
var _target := Vector2.ZERO
var _dwell := 0.0
var _rng := RandomNumberGenerator.new()
var _sat := 0.6
var _dilate := 0.3


func _init(seed_value := 0, hue_override := -1.0) -> void:
	_rng.seed = seed_value
	hue = hue_override if hue_override >= 0.0 else _rng.randf_range(0.05, 0.62)
	_sat = _rng.randf_range(0.5, 0.8)
	_dwell = _rng.randf_range(0.6, 1.5)


func update(dt: float, energy: float) -> void:
	_dilate = lerpf(_dilate, clampf(0.25 + 0.6 * energy, 0.0, 1.0), 1.0 - exp(-4.0 * dt))
	_dwell -= dt
	if _dwell <= 0.0:
		_saccade()
	gaze = gaze.lerp(_target, 1.0 - exp(-32.0 * dt))   # fast snap = a jerky saccade


# Pick the next fixation: usually a small glance, biased toward centre.
func _saccade() -> void:
	if _rng.randf() < 0.38:
		_target = Vector2.ZERO                          # prefer dead-centre / neutral
	else:
		var a := _rng.randf() * TAU
		var r := _rng.randf()
		r = r * r * 0.42                                # squared -> strong centre bias
		_target = Vector2(cos(a), sin(a)) * r
	_dwell = _rng.randf_range(0.35, 1.7)                # fixation hold (stop-start)


## Draw the eyeball at [param center], radius [param scale] px. [param fade] scales
## opacity (for the split morph's emergence, etc.).
func draw(ci: CanvasItem, center: Vector2, scale: float, fade := 1.0) -> void:
	var a := fade
	# Sclera: matte white sphere with a rim vignette, a bottom occlusion, an upper-
	# left wet sheen, and a couple of faint veins.
	ci.draw_circle(center, scale, Color(0.93, 0.92, 0.90, a))
	ci.draw_arc(center, scale * 0.95, 0.0, TAU, 48, Color(0, 0, 0, 0.10 * a), scale * 0.12, true)
	ci.draw_circle(center + Vector2(0, scale * 0.55), scale * 0.7, Color(0.7, 0.69, 0.68, 0.18 * a))
	ci.draw_circle(center + LIGHT * scale * 0.45, scale * 0.55, Color(1, 1, 1, 0.12 * a))
	for s in [-1.0, 1.0]:
		ci.draw_line(center + Vector2(s * scale * 0.85, -scale * 0.1),
			center + Vector2(s * scale * 0.4, scale * 0.05), Color(0.8, 0.3, 0.3, 0.16 * a), 1.5, true)

	# Iris: shifts toward the gaze, foreshortens a touch as it turns.
	var ipos := center + gaze * scale * 0.6
	var ir := scale * 0.46 * (1.0 - 0.2 * gaze.length())
	for i in 8:
		var t := float(i) / 7.0
		ci.draw_circle(ipos, lerpf(ir, ir * 0.35, t),
			Color.from_hsv(hue, _sat * (1.0 - 0.3 * t), lerpf(0.32, 0.78, t), a))
	for k in 44:
		var ang := TAU * float(k) / 44.0 + 0.2 * sin(float(k))
		var d := Vector2(cos(ang), sin(ang))
		var fv := 0.5 + 0.4 * sin(float(k) * 2.3)
		ci.draw_line(ipos + d * ir * 0.4, ipos + d * ir, Color.from_hsv(hue, _sat, fv, 0.5 * a), 1.0, true)
	ci.draw_arc(ipos, ir, 0.0, TAU, 44, Color.from_hsv(hue, _sat, 0.15, 0.9 * a), 2.0, true)

	# Pupil (dilating) + wet catchlight.
	var pr := ir * (0.28 + 0.46 * _dilate)
	ci.draw_circle(ipos, pr, Color(0.02, 0.02, 0.03, a))
	var cl := ipos + LIGHT * ir * 0.7
	ci.draw_circle(cl, ir * 0.22, Color(1, 1, 1, 0.95 * a))
	ci.draw_circle(cl + Vector2(0.22, 0.2) * ir, ir * 0.09, Color(1, 1, 1, 0.55 * a))
