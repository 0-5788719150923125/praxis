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


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "plane"
	morph_out = "eye"
	_eye = EyeBody.new(rng.randi())
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
	_eye.update(delta, clampf(f.energy * 0.7 + f.beat * 0.4, 0.0, 1.0))
	queue_redraw()


func _draw() -> void:
	begin_draw()
	lens.prepare()
	_eye.draw(self, lens, unit(), Vector3.ZERO, float(params.radius))
