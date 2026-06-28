extends VortexScene

## Eye - a single human eye in the black void (the-point, scene 1).
##
## A floating eyeball ([EyeBody]) - no eyelids, no blink - that looks around in
## centre-preferring saccades, starting from the neutral forward gaze. The pupil
## dilates with the audio. It declares `morph_out = "eye"`, so the Director can
## morph it into the two-eyes scene (the split) instead of cutting.

var _f: AudioFeatures = AudioFeatures.new()
var _eye: EyeBody


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "plane"
	morph_out = "eye"
	_eye = EyeBody.new(rng.randi())
	return {"size": rng.randf_range(0.32, 0.42)}


## Hand the eye's identity to a morph target (two_eyes), so the split is continuous.
func morph_payload() -> Dictionary:
	return {"hue": _eye.hue, "gaze": _eye.gaze, "size": float(params.size)}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.008, 0.015)        # nearly static, per the brief
	_eye.update(delta, clampf(f.energy * 0.7 + f.beat * 0.4, 0.0, 1.0))
	queue_redraw()


func _draw() -> void:
	begin_draw()
	_eye.draw(self, Vector2.ZERO, float(params.size) * unit())
