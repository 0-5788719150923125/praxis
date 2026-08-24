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
## How many engine frames to wait for the worker to deliver a packet before giving up on it.
## Generous: the point is to notice a forge that has stopped, not to time one that is merely
## busy. At the rates measured here a packet lands within a dozen frames.
const SPIN_LIMIT := 240

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


## Wait until the worker hands over a NEW packet, so one call to this is exactly one step of
## the flight however fast the harness happens to be spinning. False if it never arrived.
func _next_packet(sc) -> bool:
	var src: Variant = sc._forge.packet_source()
	for _spin in SPIN_LIMIT:
		await get_tree().process_frame
		if sc._forge.packet_source() != src:
			return true
	return false


## What the flight actually looks like, one step at a time.
func _check_picture() -> void:
	var vp := SubViewport.new()
	vp.size = Vector2i(W, H)
	vp.disable_3d = true
	vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(vp)
	var sc = load("res://scripts/scenes/tunnel_run.gd").new()
	vp.add_child(sc)
	sc.init_with_seed(7, "drift")
	# WARMED UP THE SAME WAY IT IS MEASURED. Spinning `update` without waiting for packets
	# leaves the forge a long way behind, and the first measured packet then arrives carrying
	# all of that backlog at once - a 10.5 against a 2.9 mean, which is a startup transient
	# being reported as a step in the flight. Entering the measurement in the regime the
	# measurement assumes is the fix; discarding the first sample would only hide it.
	for _i in WARMUP:
		sc.update(_features(), DT)
		await _next_packet(sc)
	var prev := PackedByteArray()
	var diffs: Array = []
	var starved := 0
	for _i in FRAMES:
		sc.update(_features(), DT)
		# ONE SAMPLE PER PACKET, not one per engine frame, and getting that wrong made this
		# file measure the harness instead of the scene. tunnel_run builds its geometry on a
		# worker (see FrameForge) and the picture only changes when a packet lands. This probe
		# runs unthrottled - measured at 224 to 293 fps under xvfb - while the forge delivers
		# 30 to 50 packets a second, so nine samples in ten were of the SAME picture and the
		# tenth carried nine frames of travel at once. That reads as a step every fourth frame
		# and it is nothing of the kind; the scene had not moved between the samples that
		# showed nothing and it had moved nine times between the ones that showed a lot.
		#
		# It also explains why this passed the day it was written and failed later on the
		# identical commit: nothing about the scene changed, the machine's spare capacity did.
		# A gate whose verdict depends on how fast the probe happens to spin is not measuring
		# the thing it names.
		#
		# Waiting for the packet source to CHANGE makes every sample exactly one `update` of
		# travel apart, whatever the frame rate, so the diffs are comparable to each other -
		# which is the whole basis of calling one of them a spike.
		if not await _next_packet(sc):
			starved += 1
		await RenderingServer.frame_post_draw
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
	if starved > 0:
		_fails.append("the forge failed to deliver a packet within %d frames on %d of %d samples - the scene is not building"
			% [SPIN_LIMIT, starved, FRAMES])
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
	# THE INSTRUMENT'S OWN CONTROL, and it costs nothing to run: what WOULD a step score? A
	# skipped station delivers two steps of travel in one frame, so the smallest step this
	# measure has to catch is the sum of two adjacent diffs. If that does not clear the spike
	# threshold then the threshold is too loose to see the fault the file exists for, and a
	# green run above means nothing.
	#
	# Written as arithmetic on the samples already taken rather than as a second render: the
	# quantity is exactly defined, and a probe that deliberately drops packets would be a
	# second, differently-wrong harness to keep in step with the first.
	var pair := 0.0
	for k in range(1, diffs.size()):
		pair = maxf(pair, float(diffs[k]) + float(diffs[k - 1]))
	print("tunnel_smooth_check: a doubled step would score %.2f, %.2fx the mean (threshold %.1fx)"
		% [pair, pair / maxf(0.01, mean), SPIKE])
	if pair <= mean * SPIKE:
		_fails.append("two steps at once would only score %.2fx the mean - this measure cannot see a skipped station, so it is not gating anything"
			% (pair / maxf(0.01, mean)))
	# ...and it must still be MOVING, or a scene that froze would pass the test above.
	if mean < 0.5:
		_fails.append("the picture barely changes at all (mean %.2f) - it is smooth because nothing is happening" % mean)
