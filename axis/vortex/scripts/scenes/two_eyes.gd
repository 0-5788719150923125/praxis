extends VortexScene

## Two eyes - the single eye split into two (the-point, scene 2).
##
## Two independent floating eyeballs ([EyeBody]) side by side, each looking around on
## its own. It declares `morph_in = "eye"`: when the Director changes here from the
## single `eye` scene, it plays the *split* - starting as one eye at centre (matching
## the previous shot's size) and easing apart into two as it shrinks to pair size.
## Entered any other way (a plain cut), it simply opens already split. `morph_out =
## "eyes"` leaves the door open for a later morph out of the pair.

var _f: AudioFeatures = AudioFeatures.new()
var _left: EyeBody
var _right: EyeBody
var _split := 1.0       # 0 = one centred eye, 1 = two apart. 1 unless morphed in.
var _start_size := 0.37  # size the split begins at (the source eye's, set on morph)


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "plane"
	morph_in = "eye"
	morph_out = "eyes"
	# Two IDENTICAL eyes: same colour (per the brief), different saccade seeds.
	var h := rng.randf_range(0.05, 0.62)
	_left = EyeBody.new(rng.randi(), h)
	_right = EyeBody.new(rng.randi(), h)
	return {"size": rng.randf_range(0.24, 0.30), "offset": rng.randf_range(0.27, 0.34)}


# Arrived from the single eye: start as that exact eye (its colour, gaze, size) at
# centre and split apart - the SAME eye dividing, not two new ones.
func begin_morph(from: VortexScene) -> void:
	_split = 0.0
	var p := from.morph_payload()
	if p.is_empty():
		return
	_start_size = float(p.get("size", _start_size))
	var h := float(p.get("hue", _left.hue))
	_left.hue = h
	_right.hue = h
	var g: Vector2 = p.get("gaze", Vector2.ZERO)
	_left.gaze = g
	_right.gaze = g


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.008, 0.015)
	var drive := clampf(f.energy * 0.7 + f.beat * 0.4, 0.0, 1.0)
	_left.update(delta, drive)
	_right.update(delta, drive)
	_split = minf(1.0, _split + delta * 1.1)   # the split eases open over ~1s
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()
	var s01 := smoothstep(0.0, 1.0, _split)
	var off := float(params.offset) * u * s01
	# Start at the source eye's size and shrink to pair size as they part - so the
	# morph from `eye` is continuous (no size pop at the swap).
	var sc := lerpf(_start_size, float(params.size), s01) * u
	_left.draw(self, Vector2(-off, 0), sc)
	_right.draw(self, Vector2(off, 0), sc)
