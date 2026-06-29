extends Scene3D

## Eye - a single human eye in the black void (the-point, scene 1).
##
## A real-3D floating eyeball ([EyeBody]) - sclera sphere, recessed iris, glossy
## cornea - drawn through the [Scene3D] camera with a real light, no eyelids, no
## blink. It looks around in centre-preferring saccades by rotating in 3D. The pupil
## dilates with the audio. Declares `morph_out = "eye"` so the Director can morph it
## into two_eyes (the split) rather than cutting.

var _f: AudioFeatures = AudioFeatures.new()
var _eye: EyeBody
var _rng := RandomNumberGenerator.new()
var _focus := Vector3(0, 0, 6.0)        # the point the eye is looking at
var _focus_target := Vector3(0, 0, 6.0)
var _focus_dwell := 0.0


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "plane"
	morph_out = "eye"
	_rng.seed = rng.randi()
	_eye = EyeBody.new(rng.randi())
	_eye.autonomous = false              # driven by the focus point below
	lens.eye = Vector3(0, 0, 4.0)        # static, forward-facing camera (per the brief)
	lens.look = Vector3.ZERO
	lens.fov = 48.0
	return {"radius": rng.randf_range(0.30, 0.40)}   # world radius -> a golf-ball on screen


## Hand the eye's identity (colour, gaze, size) to a morph target, so the split is
## continuous - two_eyes becomes the SAME eye, not new ones.
func morph_payload() -> Dictionary:
	return {"hue": _eye.hue, "gaze": _eye.gaze, "size": float(params.radius)}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.006, 0.012)
	# Look at a point that drifts near -> far -> extreme distance and lingers there; the
	# eye tracks it and the pupil accommodates (a near focus constricts it).
	_focus_dwell -= delta
	if _focus_dwell <= 0.0:
		_new_focus()
	_focus = _focus.lerp(_focus_target, 1.0 - exp(-3.0 * delta))
	_eye.look_at_point(Vector3.ZERO, _focus)
	_eye.update(delta, clampf(f.energy * 0.7 + f.beat * 0.4, 0.0, 1.0))
	queue_redraw()


func _new_focus() -> void:
	var tier := _rng.randf()
	var d: float
	var dwell: float
	if tier < 0.32:
		d = _rng.randf_range(1.6, 3.2)
		dwell = _rng.randf_range(1.6, 3.0)       # near - linger
	elif tier < 0.68:
		d = _rng.randf_range(4.0, 10.0)
		dwell = _rng.randf_range(0.5, 1.4)       # mid
	else:
		d = _rng.randf_range(20.0, 80.0)
		dwell = _rng.randf_range(2.0, 4.2)       # far / extreme distance - linger
	var sacc := EyeBody.saccade_target(_rng)
	var lat := _rng.randf_range(0.3, 1.0)
	_focus_target = Vector3(sacc.x * lat, sacc.y * lat * 0.7, d)
	_focus_dwell = dwell


func _draw() -> void:
	begin_draw()
	lens.prepare()
	_eye.draw(self, lens, unit(), Vector3.ZERO, float(params.radius))
