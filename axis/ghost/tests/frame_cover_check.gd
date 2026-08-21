extends Node

## Gate for the rule that stops scenes showing their own edges:
##
##   tests/run_boot_probe.sh tests/frame_cover_check.gd 90
##
## THE FAILURE, reported over and over and about a different scene every time: "the shifting
## camera reveals the fakery behind each scene... I can see the hard edges at the top and left
## sides of the water... I can watch as birds bounce off of invisible boundaries. This happens
## over and over, and the problem is clear: most scenes have no awareness of where the camera is."
##
## THE CAUSE WAS ONE CONSTANT. Scenes draw around the origin and a [SceneView] then zooms, pans,
## rolls and skews the whole canvas, so anything sized to the raw viewport has an edge just
## outside the frame. The convention for keeping it out there was a fixed 1.15 overdraw, written
## into [method GhostScene.update_layers], [method GhostScene.paint_ground] and a scatter of
## individual scenes. A CONSTANT CANNOT BE RIGHT: it has no idea whether the shot has pulled back
## to 0.7 zoom, panned a third of a frame, or rolled - and every one of those makes 1.15 too
## small. A slightly better version divided by the zoom, which covers a pull-back and nothing
## else.
##
## [method SceneView.visible_half] answers it exactly instead, by pulling the screen rectangle
## back through the very transform the renderer is handed. This gate holds that: over a sweep of
## camera states - zoomed out, panned, rolled, skewed, mid-stinger, and all of those together - a
## quad of that half-extent, pushed through that same matrix, must cover every pixel of the frame.
##
## THE TWO CONTROLS ARE THE RULES IT REPLACED, measured on the identical states. If the fixed
## 1.15 ever passed this sweep, the sweep would not be reaching the cameras that broke it.

const W := 1920.0
const H := 1080.0

var _fails: Array = []


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	var size := Vector2(W, H)
	var states := _states()
	var worst_new := 1.0
	var worst_fixed := 1.0
	var worst_zoom := 1.0
	var bad: Array = []
	for st in states:
		var v := _view(st as Dictionary)
		var m := v.matrix(size)
		# The rule under test, and the two it replaced.
		var cov_new := _covers(m, size, v.visible_half(size))
		var u := minf(size.x, size.y)
		var cov_fixed := _covers(m, size, size * 0.5 * 1.15)
		var cov_zoom := _covers(m, size, size * 0.5 * 1.15 / maxf(0.001, v.zoom_actual()))
		worst_new = minf(worst_new, cov_new)
		worst_fixed = minf(worst_fixed, cov_fixed)
		worst_zoom = minf(worst_zoom, cov_zoom)
		if cov_new < 0.9999:
			bad.append("%s (%.3f)" % [String((st as Dictionary)["name"]), cov_new])
		# The unit-fraction form the layers are handed has to agree with the pixel form, or a
		# layer and the ground it sits on are sized off two different cameras.
		var uf := v.visible_half(size) / u
		if absf(uf.x * u - v.visible_half(size).x) > 0.001:
			_fails.append("the unit-fraction and pixel forms of visible_half disagree")
	print("")
	print("over %d camera states, the worst frame coverage was:" % states.size())
	print("  visible_half (the rule under test): %.4f" % worst_new)
	print("  a fixed 1.15 overdraw (the control): %.4f" % worst_fixed)
	print("  1.15 divided by the zoom (the control): %.4f" % worst_zoom)
	_ok(bad.is_empty(), "visible_half left frame uncovered on %d states (%s) - a scene sized "
		% [bad.size(), ", ".join(bad.slice(0, 6))] + "from it still has an edge the camera can "
		+ "reach, which is the whole failure this exists to end")
	_ok(worst_fixed < 0.999, "the fixed 1.15 overdraw covered every state in this sweep - so "
		+ "the sweep is not reaching the cameras that actually broke it and this gate proves "
		+ "nothing")
	_ok(worst_zoom < 0.999, "1.15-over-zoom covered every state in this sweep - it handles a "
		+ "pull-back and nothing else, so a sweep it survives is missing the pans and rolls")
	print("")
	if _fails.is_empty():
		print("frame_cover_check: ALL OK - content sized from visible_half cannot show its edge.")
	else:
		print("frame_cover_check: %d FAILURE(S)" % _fails.size())
		for f in _fails:
			print("   ", f)
	for _i in 4:
		await get_tree().process_frame
	get_tree().quit(1 if not _fails.is_empty() else 0)


## The camera states to sweep. Deliberately includes the extremes the shot bag and the stinger
## can actually reach together, because that combination is what the reports were of - a shot
## already panned when a punch pulls the zoom back is well past anything either does alone.
func _states() -> Array:
	var out: Array = []
	out.append({"name": "neutral"})
	out.append({"name": "zoom-out", "zoom": 0.62})
	out.append({"name": "zoom-in", "zoom": 1.8})
	out.append({"name": "pan", "off": Vector2(0.22, -0.17)})
	out.append({"name": "roll", "rot": 0.42})
	out.append({"name": "skew", "skew": 0.30})
	out.append({"name": "bias", "bias_off": Vector2(-0.30, 0.24), "bias_zoom": 0.80})
	out.append({"name": "punch", "pulse_zoom": 0.72, "pulse_rot": 0.22, "pulse_skew": 0.18})
	out.append({"name": "pan+roll", "off": Vector2(0.19, 0.15), "rot": -0.35})
	out.append({"name": "out+pan", "zoom": 0.70, "off": Vector2(-0.20, 0.18)})
	out.append({"name": "everything", "zoom": 0.68, "off": Vector2(0.18, -0.16),
		"rot": 0.33, "skew": 0.22, "bias_off": Vector2(-0.12, 0.10), "bias_zoom": 0.88,
		"pulse_zoom": 0.85, "pulse_rot": 0.15})
	return out


## A SceneView settled on one state. `snap()` puts the eased actuals onto the targets, which is
## what the pre-warm does and what a held shot converges to.
func _view(st: Dictionary) -> SceneView:
	var v := SceneView.new()
	v.zoom = float(st.get("zoom", 1.0))
	v.rotation = float(st.get("rot", 0.0))
	v.skew = float(st.get("skew", 0.0))
	v.offset = st.get("off", Vector2.ZERO)
	v.bias_offset = st.get("bias_off", Vector2.ZERO)
	v.bias_zoom = float(st.get("bias_zoom", 1.0))
	v.snap()
	v.pulse_zoom = float(st.get("pulse_zoom", 1.0))
	v.pulse_rot = float(st.get("pulse_rot", 0.0))
	v.pulse_skew = float(st.get("pulse_skew", 0.0))
	return v


## What share of the frame a quad of half-extent [param half], drawn in scene coordinates and
## pushed through [param m], actually covers. Sampled on a grid rather than solved analytically,
## because the transform can be skewed and the honest question is about pixels.
func _covers(m: Transform2D, size: Vector2, half: Vector2) -> float:
	var inv := m.affine_inverse()
	var n := 0
	var inside := 0
	for j in 61:
		for i in 61:
			var s := Vector2(size.x * float(i) / 60.0, size.y * float(j) / 60.0)
			var p: Vector2 = inv * s
			n += 1
			if absf(p.x) <= half.x and absf(p.y) <= half.y:
				inside += 1
	return float(inside) / float(maxi(1, n))


func _ok(cond: bool, msg: String) -> void:
	if not cond:
		_fails.append(msg)
