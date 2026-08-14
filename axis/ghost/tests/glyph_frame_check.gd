extends Node

## glyph_frame_check - that the glyphs camera actually frames the writing, in EVERY layout.
##
## The bug this gates: `_track_pen` aimed at the page cursor `_cx` / `_cy`, and only three of
## the six layouts write there. A SPIRAL advances `_s` along its coil and never touches
## `_cx` / `_cy` after _write_opening, so the camera sat on the page's top-left corner watching
## blank paper for the whole hold while the writing went round the middle of the frame. A BANNER
## recentres its live line by `_shift_cur`, which the camera did not apply, so at the start of
## every line the pen was about half a measure outside the frame.
##
## A third defect, older than either and shared by all four layouts, only showed up once this
## probe existed: the tracker eased at ONE loose rate, so a line break - a carriage return, not
## a drift - had the frame gliding back across the page for 1.47 s while the hand wrote at the
## start of the new line. That is the `_chasing` band in _track_pen.
##
## MEASURED, both rules at once. The probe runs the shipped camera and, beside it, a shadow
## camera on the old rule, and reports for each what fraction of 900 frames had the nib in shot:
##
##   page    68.3% -> 97.9%
##   column  87.6% -> 99.7%
##   banner  31.3% -> 100.0%
##   spiral    n/a -> 100.0%
##
## The spiral has no honest before/after in that column: the fix changed what the camera AIMS at
## (the coil's centre, not the orbiting nib) and how far it is zoomed, so the shadow camera is no
## longer running the old system. Its recorded numbers are 3.7% under the old rule, and 1.3%
## under a first attempt that tracked the nib along the coil - worse than doing nothing, which is
## what sent the fix to framing the composition instead.
##
## It asserts on the SUSTAINED case, not the instantaneous one. The tracking is deliberately
## eased over about 1.4 s so the nib drifts across the frame and the camera catches up between
## words - a frame or two of overshoot is the feature - but the nib being gone for a second at a
## time is the bug that was reported.
##
## The scene reaches Spectrum and Director, so it needs a REAL boot - autoloads do not exist
## under a bare `--script` run (see CLAUDE.md's validation notes, and run_boot_probe's header).
##
## Run: tests/run_boot_probe.sh tests/glyph_frame_check.gd 120

const W := 1920.0
const H := 1080.0
const FPS := 60.0
const FRAMES := 900
## The nib may leave the frame for this long while the eased camera catches up. 0.5 s is a third
## of the tracker's own 1.4 s time constant, which is as long as a catch-up can honestly take.
const MAX_GONE := 30
## And it has to be in frame for most of the hold, not merely return to it.
const MIN_INSIDE := 0.90

var _fails: Array = []
var _seen: Dictionary = {}


func _ready() -> void:
	var script := load("res://scripts/scenes/glyphs.gd")
	# The layout is a seeded roll, so sweep seeds until every one of the four has been measured.
	for s in range(0, 400):
		if _seen.size() >= 4:
			break
		_run(script, s)
	for want in ["page", "column", "banner", "spiral"]:
		if not _seen.has(want):
			_fails.append("no seed in 0..399 produced the %s layout - the probe never tested it" % want)
	if _fails.is_empty():
		print("glyph_frame_check: ALL OK")
		get_tree().quit()
		return
	print("glyph_frame_check: %d FAILURE(S)" % _fails.size())
	for f in _fails:
		print("   ", f)
	get_tree().quit(1)


func _run(script, seed_value: int) -> void:
	var sc = script.new()
	sc.init_with_seed(seed_value, "drift")
	if _seen.has(sc._layout):
		sc.free()
		return
	_seen[sc._layout] = true
	# _ready never fires outside a tree, so hand it the frame it would have measured.
	sc.size = Vector2(W, H)
	sc._metrics()
	sc._open = true
	sc._write_opening()
	sc._cam = sc._clamp_cam(sc._pen_at())
	# DRIVE THE PATH THAT SHIPS. An earlier version of this probe called _commit() on a timer
	# instead of running _step, and _step is where `_scroll` is eased toward `_scroll_to` - so
	# `_cy` walked off the bottom of a page that never scrolled and every layout measured about
	# 50%, an artifact of the harness rather than of the scene. Audio the pen will actually
	# write to: above Director.silence_floor, with a plausible tempo.
	sc._f = AudioFeatures.new()
	sc._f.energy = 0.25
	sc._f.beat_period = 0.5
	sc._f.flux = 0.02

	var dt := 1.0 / FPS
	var old_cam: Vector2 = sc._cam          # the shadow camera, on the rule that was there
	var inside := 0
	var old_inside := 0
	var gone := 0
	var worst_gone := 0
	for i in FRAMES:
		if i % 180 == 90:
			sc._pending_word = true         # a section change, as update() would raise it
		sc._step(dt)
		sc._track_pen(dt)
		# The OLD target: the raw page cursor, no centring shift, and nothing for the spiral.
		var old_target := Vector2(sc._cx + sc._adv * 0.5,
			sc._cy - sc._scroll - sc._line_step * 0.25)
		old_cam = old_cam.lerp(old_target, 1.0 - exp(-0.7 * dt))

		var h: Vector2 = sc._half / maxf(0.001, sc._zoom_in)
		var pen: Vector2 = sc._pen_at()
		var off: Vector2 = pen - sc._cam
		var ok := absf(off.x) <= h.x and absf(off.y) <= h.y
		if ok:
			inside += 1
			gone = 0
		else:
			gone += 1
			worst_gone = maxi(worst_gone, gone)
		var oo: Vector2 = pen - old_cam
		if absf(oo.x) <= h.x and absf(oo.y) <= h.y:
			old_inside += 1

	var frac := float(inside) / float(FRAMES)
	var old_frac := float(old_inside) / float(FRAMES)
	print("glyph_frame_check: %-7s seed=%-4d nib in frame %5.1f%% (old rule %5.1f%%), longest gone %d frames"
		% [sc._layout, seed_value, frac * 100.0, old_frac * 100.0, worst_gone])
	if frac < MIN_INSIDE:
		_fails.append("%s: the nib is inside the frame only %.1f%% of the time (want %.0f%%) - the camera is not following the writing"
			% [sc._layout, frac * 100.0, MIN_INSIDE * 100.0])
	if worst_gone > MAX_GONE:
		_fails.append("%s: the nib left the frame for %d consecutive frames (%.2f s, want at most %.2f s)"
			% [sc._layout, worst_gone, float(worst_gone) / FPS, float(MAX_GONE) / FPS])
	sc.free()
