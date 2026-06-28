extends Scene3D

## Two eyes - the single eye split into two (the-point, scene 2).
##
## Two real-3D eyeballs ([EyeBody]) side by side, each looking around independently,
## drawn through the [Scene3D] camera. Declares `morph_in = "eye"`: arriving from the
## single `eye` scene it plays the *split* - starting as that exact eye at centre (its
## colour, gaze, and size) and easing apart into two identical copies as it shrinks
## to pair size. Entered by a plain cut it simply opens already split.

var _f: AudioFeatures = AudioFeatures.new()
var _rng := RandomNumberGenerator.new()
var _left: EyeBody
var _right: EyeBody
var _split := 1.0       # 0 = one centred eye, 1 = two apart (1 unless morphed in)
var _start_radius := 0.34  # world radius the split begins at (the source eye's)
# Conjugate gaze: both eyes lock to one shared target (as real eyes do); occasionally
# one diverges - the async wander is the nonlinear deviation, not the default.
var _gaze_target := Vector2.ZERO
var _gaze_dwell := 0.0
var _ldiv := Vector2.ZERO
var _rdiv := Vector2.ZERO


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "plane"
	morph_in = "eye"
	morph_out = "eyes"
	_rng.seed = rng.randi()
	var h := rng.randf_range(0.05, 0.6)        # two IDENTICAL eyes: same colour
	_left = EyeBody.new(rng.randi(), h)
	_right = EyeBody.new(rng.randi(), h)
	_left.autonomous = false                   # driven by the shared gaze below
	_right.autonomous = false
	lens.eye = Vector3(0, 0, 4.0)
	lens.look = Vector3.ZERO
	lens.fov = 48.0
	return {"radius": rng.randf_range(0.24, 0.30), "offset": rng.randf_range(0.55, 0.72)}


# Arrived from the single eye: become that exact eye (colour, gaze, size) at centre,
# then split apart - the SAME eye dividing, not two new ones.
func begin_morph(from: VortexScene) -> void:
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
	_gaze_target = g            # continue the shared gaze from where the single eye was


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.006, 0.012)
	var drive := clampf(f.energy * 0.7 + f.beat * 0.4, 0.0, 1.0)
	# Shared (locked) gaze, with a rare divergence on one eye.
	_gaze_dwell -= delta
	if _gaze_dwell <= 0.0:
		_gaze_target = EyeBody.saccade_target(_rng)
		_ldiv = Vector2.ZERO
		_rdiv = Vector2.ZERO
		if _rng.randf() < 0.16:                # the nonlinear deviation: one eye wanders
			var d := Vector2(_rng.randf_range(-1, 1), _rng.randf_range(-1, 1)).normalized() * _rng.randf_range(0.15, 0.4)
			if _rng.randf() < 0.5:
				_ldiv = d
			else:
				_rdiv = d
		_gaze_dwell = _rng.randf_range(0.4, 1.6)
	_left.target = _gaze_target + _ldiv
	_right.target = _gaze_target + _rdiv
	_left.update(delta, drive)
	_right.update(delta, drive)
	_split = minf(1.0, _split + delta * 1.1)   # the split eases open over ~1s
	queue_redraw()


func _draw() -> void:
	begin_draw()
	lens.prepare()
	var u := unit()
	var s01 := smoothstep(0.0, 1.0, _split)
	var off := float(params.offset) * s01                       # world x separation
	var rad := lerpf(_start_radius, float(params.radius), s01)  # start at source size, shrink
	_left.draw(self, lens, u, Vector3(-off, 0, 0), rad)
	_right.draw(self, lens, u, Vector3(off, 0, 0), rad)
