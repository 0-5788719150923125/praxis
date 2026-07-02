extends Scene3D

## Two eyes - the single eye split into two (the-point, scene 2).
##
## Two real-3D eyeballs ([EyeBody]) that *verge*: both aim at one shared 3D focus point,
## so they toe in on a near point and run parallel on a far one - real binocular gaze,
## not two eyes locked to the same direction. The focus drifts in depth (near -> far ->
## extreme distance) and lingers at the extremes, with the pupils accommodating. Declares
## `morph_in = "eye"`: arriving from the single eye it plays the split (the same eye
## dividing). Occasionally one eye diverges - the nonlinear deviation, not the default.

var _f: AudioFeatures = AudioFeatures.new()
var _rng := RandomNumberGenerator.new()
var _left: EyeBody
var _right: EyeBody
var _split := 1.0       # 0 = one centred eye, 1 = two apart (1 unless morphed in)
var _start_radius := 0.34  # world radius the split begins at (the source eye's)
var _focus := Vector3(0, 0, 6.0)        # the shared point both eyes look at
var _focus_target := Vector3(0, 0, 6.0)
var _focus_dwell := 0.0
var _ldiv := Vector2.ZERO   # rare per-eye gaze divergence
var _rdiv := Vector2.ZERO
var _eye_off := 0.0         # current world x separation (kept in sync with _draw)
var _eye_rad := 0.27


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "plane"
	morph_in = "eye"
	morph_out = "eyes"
	_rng.seed = rng.randi()
	var h := rng.randf_range(0.05, 0.6)        # two IDENTICAL eyes: same colour
	_left = EyeBody.new(rng.randi(), h)
	_right = EyeBody.new(rng.randi(), h)
	_left.autonomous = false                   # driven by the shared focus below
	_right.autonomous = false
	lens.eye = Vector3(0, 0, 4.0)
	lens.look = Vector3.ZERO
	lens.fov = 48.0
	_eye_rad = rng.randf_range(0.24, 0.30)
	return {"radius": _eye_rad, "offset": rng.randf_range(0.55, 0.72)}


# Arrived from the single eye: become that exact eye (colour, gaze, size) at centre,
# then split apart - the SAME eye dividing, not two new ones.
func begin_morph(from: GhostScene) -> void:
	_split = 0.0
	var p := from.morph_payload()
	if p.is_empty():
		return
	_start_radius = float(p.get("size", _start_radius))
	var h := float(p.get("hue", _left.hue))
	_left.hue = h
	_right.hue = h
	var g: Vector2 = p.get("gaze", Vector2.ZERO)
	_left.gaze = g
	_right.gaze = g
	# Continue from where the single eye was looking: a forward focus along that gaze.
	var front: Vector3 = Basis.from_euler(Vector3(g.y, g.x, 0.0)) * Vector3(0, 0, 1)
	_focus = front * 6.0
	_focus_target = _focus


# Hand the two eyes' identity to a morph target (eye_prism): the LIVE EyeBody instances
# (so the surviving eye literally continues, no re-seeding), their world separation, radius,
# hue, and the shared focus. eye_prism keeps the LEFT eye and crystallizes the RIGHT into a
# blue prism at the same slot - so nothing jumps at the swap.
func morph_payload() -> Dictionary:
	return {"left": _left, "right": _right, "offset": _eye_off, "radius": _eye_rad,
		"hue": _left.hue, "focus": _focus}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.006, 0.012)
	var drive := clampf(f.energy * 0.7 + f.beat * 0.4, 0.0, 1.0)

	# Drift the shared focus in depth and across the frame, lingering at the extremes.
	_focus_dwell -= delta
	if _focus_dwell <= 0.0:
		_new_focus()
	_focus = _focus_target   # snap to the target; each eye's spring gives the saccade momentum + overshoot

	# The split eases open over ~45% of the scene's hold, so it always finishes on screen no matter how
	# short the scene is (a plain cut opens already-split: _split starts at 1.0, so this is a no-op then).
	_split = minf(1.0, _split + delta / maxf(0.3, phase_span(2.0) * 0.45))
	var s01 := smoothstep(0.0, 1.0, _split)
	_eye_off = float(params.offset) * s01
	_eye_rad = lerpf(_start_radius, float(params.radius), s01)

	# Both eyes aim at the shared focus from their own positions - this is the vergence.
	_left.look_at_point(Vector3(-_eye_off, 0, 0), _focus)
	_right.look_at_point(Vector3(_eye_off, 0, 0), _focus)
	_left.target += _ldiv                        # rare one-eye wander on top
	_right.target += _rdiv
	_left.update(delta, drive)
	_right.update(delta, drive)
	queue_redraw()


# Choose a new focus: a depth tier (near / mid / far-extreme, the extremes held longer)
# plus a lateral offset. A near point is eccentric (the eyes converge hard); a far point
# is nearly straight ahead - the geometry does the vergence for free.
func _new_focus() -> void:
	var tier := _rng.randf()
	var d: float
	var dwell: float
	# Hold each fixation much longer so the gaze reads as calm and deliberate, not flickering:
	# real eyes rest on a point for whole seconds between saccades.
	if tier < 0.32:
		d = _rng.randf_range(2.2, 4.0)
		dwell = _rng.randf_range(3.0, 5.5)       # near - linger
	elif tier < 0.68:
		d = _rng.randf_range(5.0, 11.0)
		dwell = _rng.randf_range(2.0, 3.6)       # mid (was a twitchy 0.5-1.4)
	else:
		d = _rng.randf_range(22.0, 80.0)
		dwell = _rng.randf_range(3.5, 7.0)       # far / extreme distance - linger
	var sacc := EyeBody.saccade_target(_rng)
	# Smaller lateral excursions keep the eyes near forward, so refixations are short hops,
	# not wild swings across the orbit.
	var lat := _rng.randf_range(0.12, 0.45)
	_focus_target = Vector3(sacc.x * lat, sacc.y * lat * 0.7, d)
	_focus_dwell = dwell
	_ldiv = Vector2.ZERO
	_rdiv = Vector2.ZERO
	if _rng.randf() < 0.07:                       # the nonlinear deviation: one eye wanders (rare)
		var dv := Vector2(_rng.randf_range(-1, 1), _rng.randf_range(-1, 1)).normalized() \
			* _rng.randf_range(0.06, 0.16)
		if _rng.randf() < 0.5:
			_ldiv = dv
		else:
			_rdiv = dv


func _draw() -> void:
	begin_draw()
	lens.prepare()
	var u := unit()
	_left.draw(self, lens, u, Vector3(-_eye_off, 0, 0), _eye_rad)
	_right.draw(self, lens, u, Vector3(_eye_off, 0, 0), _eye_rad)
