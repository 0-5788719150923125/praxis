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
## AND THAT THE ROW ITSELF HOLDS STILL, which is a second report on the same layout: "the entire
## row of glyphs kept shifting in hard, discrete jumps. The text wasn't stable... you would expect
## the row of glyphs to remain where they were written, and you would expect the camera to track
## the writing. The camera was working here: but the jumping row was the part that needs fixing."
## The banner's centring shift was recomputed at draw time off the page CURSOR, which advances a
## whole character at a time, so the row stepped by half an advance at every commit. It is
## integrated on the fixed clock now, off a target that moves with the reveal (see
## `Glyphs._ease_shift`). The probe records the shift every frame and reports the worst
## single-frame step as a fraction of one advance, beside the same measurement on the OLD rule -
## the numbers are 0.5 of an advance in one frame before, and under 0.05 after.
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
## THE ROW'S STABILITY, as a shape rather than a size. An absolute per-frame limit cannot tell a
## jump from a fast hand: the centring legitimately travels at half the pen's own rate, which at
## the shortest character spans is about 0.08 of an advance per frame all by itself. What separates
## a glide from a jump is the DISTRIBUTION - both rules move the row by the same total amount over
## a line (they must; it is the same centring), but one spreads it over every frame and the other
## delivers it in one step per character. Measured on the same seed: peak/mean 5.6 against 36.8.
## So: a peak no more than this many times the mean step, with an absolute backstop for the case
## where everything is jumping equally.
const MAX_PEAK_RATIO := 12.0
const MAX_STEP := 0.20
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
	# The row's own stability: the worst single-frame step of the centring shift, and the same
	# for the rule that was there, both as a fraction of one character advance.
	var shift_prev: float = sc._shift_cur
	var raw_prev: float = -(sc._cx + sc._adv + sc._left) * 0.5
	var worst_step := 0.0
	var worst_raw := 0.0
	var travel := 0.0
	var steps := 0
	var raw_travel := 0.0
	var raw_steps := 0
	var line_prev: int = sc._line
	for i in FRAMES:
		if i % 180 == 90:
			sc._pending_word = true         # a section change, as update() would raise it
		sc._step(dt)
		sc._track_pen(dt)
		# The OLD target: the raw page cursor, no centring shift, and nothing for the spiral.
		var old_target := Vector2(sc._cx + sc._adv * 0.5,
			sc._cy - sc._scroll - sc._line_step * 0.25)
		old_cam = old_cam.lerp(old_target, 1.0 - exp(-0.7 * dt))

		if sc._centred:
			var adv: float = maxf(0.0001, sc._adv)
			var now_shift: float = sc._shift_cur
			# MEASURED ONLY WHILE THE LIVE LINE HAS INK ON IT, and never across a carriage
			# return. The claim is that a row of glyphs does not move once written - and at a
			# line break the new line is empty, so its centring may (and should) snap: nothing
			# is drawn with it yet. Measuring that frame would report the one jump that moves no
			# ink and hide the ones that do.
			var mid_line: bool = sc._line == line_prev and sc._cx > sc._left + 0.0001
			if mid_line:
				worst_step = maxf(worst_step, absf(now_shift - shift_prev) / adv)
				travel += absf(now_shift - shift_prev) / adv
				steps += 1
			line_prev = sc._line
			shift_prev = now_shift
			# THE CONTROL: the expression that used to be evaluated at draw time.
			var raw_now: float = -(sc._cx + sc._adv + sc._left) * 0.5
			if mid_line:
				worst_raw = maxf(worst_raw, absf(raw_now - raw_prev) / adv)
				raw_travel += absf(raw_now - raw_prev) / adv
				raw_steps += 1
			raw_prev = raw_now
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
	if sc._centred:
		var mean_step := travel / float(maxi(1, steps))
		var raw_mean := raw_travel / float(maxi(1, raw_steps))
		print("                   row shift: worst step %.3f advances/frame, mean %.4f, "
			% [worst_step, mean_step] + "peak/mean %.1f  |  old rule: worst %.3f, mean %.4f, "
			% [worst_step / maxf(0.00001, mean_step), worst_raw, raw_mean]
			+ "peak/mean %.1f" % (worst_raw / maxf(0.00001, raw_mean)))
		var ratio := worst_step / maxf(0.00001, mean_step)
		var raw_ratio := worst_raw / maxf(0.00001, raw_mean)
		if ratio > MAX_PEAK_RATIO:
			_fails.append("%s: the row's worst frame is %.1fx its mean (want under %.0f) - the "
				% [sc._layout, ratio, MAX_PEAK_RATIO]
				+ "centring is arriving in steps, not as a glide")
		if worst_step > MAX_STEP:
			_fails.append("%s: the row moved %.2f of a character advance in ONE frame (want under "
				% [sc._layout, worst_step] + "%.2f)" % MAX_STEP)
		if travel < 1.0:
			_fails.append("%s: the centring shift moved %.2f advances over %d frames - it is not "
				% [sc._layout, travel, FRAMES] + "recentring at all, so this measures nothing")
		# BOTH CONTROLS. The old rule has to fail both of the checks above, or they are not
		# measuring the reported bug - and its MEAN has to match, or the comparison is between
		# two different amounts of centring rather than between two ways of delivering it.
		if raw_ratio < MAX_PEAK_RATIO * 1.5:
			_fails.append("%s: the OLD rule's peak/mean is only %.1f - this measure cannot see "
				% [sc._layout, raw_ratio] + "the jump it exists for")
		if absf(raw_mean - mean_step) > 0.35 * maxf(raw_mean, mean_step):
			_fails.append("%s: the two rules travel different distances (%.4f vs %.4f per frame), "
				% [sc._layout, raw_mean, mean_step] + "so the comparison is not about jumpiness")
	sc.free()
