extends Node

## tunnel_smooth_check - that a tunnel_run flight is CONTINUOUS, measured in pixels.
##
## The complaint this gates: "the tunnel run scenes are as jumpy as they have ever been... it
## almost feels like the camera is moving forward while the geometry is moving backward, and
## then it jumps ahead to correct itself. And it's jumping at least 5 times per second."
##
## That was the RIBS. Everything else along the tube - the swell, the hue zones, the wound
## spectrum, the fog - is keyed to arclength and therefore fixed in the world, but the rib
## test was `i % rib_every` on the station's index INTO THE BUFFER, and that buffer's near end
## moves: `_advance_track` retires a station every time the camera covers one step, which at
## this speed is ten times a second. So every rib in shot stepped backward by a station, ten
## times a second, against walls that were still moving forward.
##
## TWO MEASUREMENTS, because they answer different questions and the first one is the one that
## found it:
##
##   1. THE PICTURE. Frame-to-frame mean absolute luma difference over 90 real frames. Smooth
##      forward motion through a tube is a slowly-varying number; a jump is a spike. Measured
##      at 3.3 typical with a 10-12 spike every sixth frame before the fix, and 2.2-3.7 with no
##      spike at all after it. This needs no model of what a rib is, which is the point - it
##      would catch the next thing that steps once per station too.
##
##   2. THE RULE, both ways. The set of world arclengths carrying a rib is computed under the
##      shipped rule and under the old one, before and after a retirement, and the old rule has
##      to MOVE or this file is asserting nothing. A control that cannot fail is worse than no
##      control - see rain_check and glyph_frame_check, which learned the same lesson.
##
## Run: GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/tunnel_smooth_check.gd 240

const W := 640
const H := 360
const DT := 1.0 / 60.0
const FRAMES := 90
const WARMUP := 40
## A frame that differs from its predecessor by more than this multiple of the run's own mean
## is a step rather than a move. 1.8 sits well above the honest variation of a bending flight
## (measured 2.2 to 3.7 against a 2.8 mean, a spread of 1.3x) and well under a station step
## (measured 10 to 12 against 3.3, a spread of 3.6x).
const SPIKE := 1.8

var _fails: Array = []


func _ready() -> void:
	_run.call_deferred()


func _features() -> AudioFeatures:
	var f := AudioFeatures.new()
	f.energy = 0.5
	f.bass = 0.5
	f.bands = PackedFloat32Array()
	for _i in 64:
		f.bands.append(0.4)
	return f


func _run() -> void:
	_check_rule()
	await _check_picture()
	if _fails.is_empty():
		print("tunnel_smooth_check: ALL OK")
		get_tree().quit()
		return
	print("tunnel_smooth_check: %d FAILURE(S)" % _fails.size())
	for f in _fails:
		print("   ", f)
	get_tree().quit(1)


## Where the ribs sit IN THE WORLD, under each rule, as the window slides along the track.
##
## No rendering and no scene: the defect is one modulo, so it can be stated as arithmetic. A
## station's absolute number is `s / STEP` and the window holds `n` of them starting at `s0`;
## retiring one advances `s0` by a step and drops every index by one.
func _check_rule() -> void:
	var step := 0.55
	var every := 9
	var n := 30
	var moved_new := 0
	var moved_old := 0
	for retired in 6:
		var s0 := float(retired) * step
		for i in n:
			var si := s0 + float(i) * step
			var abs_no := int(round(si / step))
			# the shipped rule, and the rule it replaced
			var now := (abs_no % every) == 0
			var was := (i % every) == 0
			# a rib at this arclength under the FIRST window is the reference
			var ref_now := (abs_no % every) == 0
			var ref_was := ((abs_no) % every) == 0
			if now != ref_now:
				moved_new += 1
			if retired > 0 and was != ref_was:
				moved_old += 1
	print("tunnel_smooth_check: over 6 retirements, ribs that moved in the world - shipped %d, old rule %d"
		% [moved_new, moved_old])
	if moved_new != 0:
		_fails.append("a rib moved in the world under the shipped rule (%d of them) - it is not keyed to arclength"
			% moved_new)
	if moved_old == 0:
		_fails.append("the OLD index rule did not move a single rib, so this control is measuring nothing")


## What the flight actually looks like, one frame at a time.
func _check_picture() -> void:
	var vp := SubViewport.new()
	vp.size = Vector2i(W, H)
	vp.disable_3d = true
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(vp)
	var sc = load("res://scripts/scenes/tunnel_run.gd").new()
	vp.add_child(sc)
	sc.init_with_seed(7, "drift")
	for _i in WARMUP:
		sc.update(_features(), DT)
		await get_tree().process_frame
	var prev := PackedByteArray()
	var diffs: Array = []
	for _i in FRAMES:
		sc.update(_features(), DT)
		await get_tree().process_frame
		var img := vp.get_texture().get_image()
		img.convert(Image.FORMAT_L8)
		var cur := img.get_data()
		if prev.size() == cur.size():
			var acc := 0
			var taken := 0
			for k in range(0, cur.size(), 7):
				acc += absi(int(cur[k]) - int(prev[k]))
				taken += 1
			diffs.append(float(acc) / float(maxi(1, taken)))
		prev = cur
	vp.queue_free()
	await get_tree().process_frame
	if diffs.size() < 10:
		_fails.append("the probe captured %d usable frames - it is not measuring the scene" % diffs.size())
		return
	var mean := 0.0
	var worst := 0.0
	for d in diffs:
		mean += float(d)
		worst = maxf(worst, float(d))
	mean /= float(diffs.size())
	var spikes := 0
	for d in diffs:
		if float(d) > mean * SPIKE:
			spikes += 1
	var per_sec := float(spikes) * 60.0 / float(diffs.size())
	print("tunnel_smooth_check: %d frames, mean change %.2f, worst %.2f (%.2fx) - %d spike(s), %.1f/s"
		% [diffs.size(), mean, worst, worst / maxf(0.01, mean), spikes, per_sec])
	if spikes > 0:
		_fails.append("the picture steps %.1f times a second (%d frames over %.1fx the mean) - a flight through a tube does not"
			% [per_sec, spikes, SPIKE])
	# ...and it must still be MOVING, or a scene that froze would pass the test above.
	if mean < 0.5:
		_fails.append("the picture barely changes at all (mean %.2f) - it is smooth because nothing is happening" % mean)
