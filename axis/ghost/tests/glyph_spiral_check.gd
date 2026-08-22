extends Node

## glyph_spiral_check - that writing, once written, STAYS WHERE IT WAS WRITTEN.
##
## The complaint this gates: "glyphs were rendered into a spiral shape, but they were not
## moving at all. They remained rooted for ~30 seconds, and then the entire scene full of them
## started rotating rapidly."
##
## Both halves of that were one line. The coil had a shift along its own arc (`_sshift`), zero
## while there was still room and advancing once per character the moment the coil filled - and
## a glyph's angle on an Archimedean spiral is `sqrt(2 s / coil)`, so subtracting from every
## glyph's `s` turns every glyph at once. Worst at the middle, where that square root is
## steepest: about a radian per character for the innermost turn, which at eight characters a
## second is more than a revolution a second. Hence rooted, and then suddenly spinning.
##
## The coil is drawn to FIT now (`_refit`): what is written keeps the arc length it was written
## at forever, and the whole picture is scaled down as the writing reaches the frame. A scale is
## uniform, so nothing moves relative to anything else, and it eases over a second rather than
## arriving with a character.
##
## TWO-SIDED, because "nothing rotated" is also what a scene that never draws would report:
##   - the shipped rule, measured on the real scene over a minute of writing;
##   - the OLD rule's rotation for the same glyphs, computed from the same numbers, which has
##     to be large or this file is asserting nothing;
##   - and the coil must still be growing and still be in frame, or it is holding still for
##     the wrong reason.
##
## Run: tests/run_boot_probe.sh tests/glyph_spiral_check.gd 180

const DT := 1.0 / 60.0
const SECONDS := 60.0
## Radians per frame. A written glyph should not move at all; this is a floating-point floor,
## not a tolerance - the measured worst case is 0.0000.
const STILL := 0.002
## What the old rule managed, in radians per frame, and the bar the control has to clear to
## prove this check can see rotation at all.
const CONTROL_MIN := 0.05

var _fails: Array = []


func _ready() -> void:
	_run.call_deferred()


func _features() -> AudioFeatures:
	var f := AudioFeatures.new()
	f.energy = 0.5
	f.bands = PackedFloat32Array()
	for _i in 64:
		f.bands.append(0.4)
	return f


func _run() -> void:
	var vp := SubViewport.new()
	vp.size = Vector2i(1280, 720)
	vp.disable_3d = true
	add_child(vp)
	var sc = load("res://scripts/scenes/glyphs.gd").new()
	vp.add_child(sc)
	var seed_used := -1
	for sv in range(1, 400):
		sc.init_with_seed(sv, "drift")
		if sc._layout == "spiral":
			seed_used = sv
			break
	if seed_used < 0:
		print("glyph_spiral_check: no seed in 400 produced the spiral layout")
		get_tree().quit(1)
		return

	# Follow ONE early glyph - the fifth ever placed, which sits in the innermost turn where
	# the old rule moved hardest - and watch the angle it is actually drawn at.
	var tracked := 5
	var prev := 0.0
	var have_prev := false
	var worst := 0.0
	var s_first := 0.0
	var s_last := 0.0
	var t := 0.0
	var frames := int(SECONDS / DT)
	for i in frames:
		sc.update(_features(), DT)
		t += DT
		if i % 6 == 0:
			await get_tree().process_frame
		if sc._placed.size() <= tracked:
			continue
		var e: Dictionary = sc._placed[tracked]
		if s_first <= 0.0:
			s_first = sc._s
		s_last = sc._s
		if not sc._pose(e, 1.0):
			continue
		var ang: float = sc._pxf.get_rotation()
		if have_prev:
			worst = maxf(worst, absf(angle_difference(ang, prev)))
		prev = ang
		have_prev = true

	var placed: int = sc._placed.size()
	var fit: float = sc._coil_fit
	# The outermost writing, as drawn, against the frame it has to stay inside.
	var theta_out: float = sqrt(2.0 * maxf(sc._s, sc._s_min) / maxf(0.000001, sc._coil))
	var r_out: float = sc._coil * theta_out * fit
	var half: float = minf(sc._half.x, sc._half.y) / maxf(0.001, sc._zoom_in)

	# THE CONTROL. What the retired rule would have done to that same glyph, per character, at
	# the arc length it sits at: the shift advanced by one advance-width per character written.
	var s_g: float = float((sc._placed[mini(tracked, placed - 1)] as Dictionary)["s"])
	var old_a: float = sqrt(2.0 * maxf(s_g, 0.0001) / sc._coil)
	var old_b: float = sqrt(2.0 * maxf(s_g - sc._adv, 0.0001) / sc._coil)
	var old_step := absf(old_a - old_b)

	print("glyph_spiral_check: seed %d, %.0fs of writing - %d glyphs, arc %.2f -> %.2f, fit %.3f"
		% [seed_used, SECONDS, placed, s_first, s_last, fit])
	print("glyph_spiral_check: tracked glyph moved %.4f rad/frame; the old rule would move it %.4f rad per CHARACTER"
		% [worst, old_step])
	print("glyph_spiral_check: outermost writing at %.3f of a %.3f half-frame" % [r_out, half])

	if worst > STILL:
		_fails.append("a written glyph turns %.4f rad in one frame - writing does not move once it is written"
			% worst)
	if old_step < CONTROL_MIN:
		_fails.append("the OLD rule only moved this glyph %.4f rad per character, so the control proves nothing"
			% old_step)
	if s_last <= s_first + 1.0:
		_fails.append("the coil barely grew (%.2f -> %.2f) - it is still because nothing is being written"
			% [s_first, s_last])
	if r_out > half * 1.05:
		_fails.append("the writing reaches %.3f, past the %.3f half-frame - the fit is not keeping it in shot"
			% [r_out, half])
	if placed < 50:
		_fails.append("only %d glyphs survived a minute - the retirement valve is eating the coil" % placed)

	vp.queue_free()
	await get_tree().process_frame
	if _fails.is_empty():
		print("glyph_spiral_check: ALL OK")
		get_tree().quit()
		return
	print("glyph_spiral_check: %d FAILURE(S)" % _fails.size())
	for f in _fails:
		print("   ", f)
	get_tree().quit(1)
