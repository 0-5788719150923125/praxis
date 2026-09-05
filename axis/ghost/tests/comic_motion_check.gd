extends Node

## WHAT THE EYE ACTUALLY SEES, measured on the OUTPUT rather than on the plan.
##
##   tests/run_boot_probe.sh tests/comic_motion_check.gd 120
##
## THIS EXISTS BECAUSE EVERY OTHER GATE HERE WAS GREEN THROUGH A NIGHT OF REPORTS.
## comic_camera_check measures the PLAN - does a move finish before the cut, does a chain go
## forward, does a shot re-roll its zoom - and comic_look_probe measures single FRAMES. Both
## stayed green while the reports were "the camera jumps to correct its position, landing on
## the exact same frame", "it never stops moving", "wild angles". Neither instrument could
## have caught any of it, because the defects live BETWEEN frames and BELOW the move layer:
## _place_eye derives the rake from the panel being read, the scale from whichever panel the
## aim is inside, and the distance from a solve that changes branch at the film panel's edge.
## All three are step functions the move vocabulary cannot see, so no amount of measuring the
## vocabulary reaches them.
##
## So this measures the SHEET'S SILHOUETTE IN SCREEN SPACE, frame to frame. That is the
## picture. Whatever moved it - a move, a rake step, a solve changing branch, a follower - it
## shows up here as screen-space displacement, and nothing that moves the picture can hide.
##
## THREE NUMBERS, each the direct form of one report:
##   TELEPORTS  frames whose displacement is a large multiple of the running median. "The
##              camera jumps." A cut is allowed to be one of these; four in six seconds is the
##              defect. Reported with WHETHER THE FRAMING PANEL CHANGED, because a jump that
##              lands on the panel it was already on is the specific complaint.
##   HELD       the fraction of frames whose displacement is under a stillness threshold.
##              "It never holds a position." Measured on the picture, NOT on the name of the
##              active move - a gate that counts a move called `breathe` as held will report
##              72% while the camera thrashes, which is exactly what happened.
##   REVERSALS  sign changes in the rendered SCALE (the silhouette's area), with the number of
##              held frames separating them. "Pulling gently outward, only to IMMEDIATELY
##              start pushing in again at its apex... no reversing direction allowed, without
##              first holding at that position first."
##
## THE BASELINE, measured on 654e9dab - the build described as "a kind of decent camera with a
## few small quirks", and the one the reported export was cut from. 90 s per setting:
##
##   camera  p99/frame   teleports (same panel)   held   zoom reversals (no hold between)
##   0.00      0.0066            0  (0)            71%              5  (3)
##   0.50      0.0505            5  (4)            87%             12  (9)
##   1.00      0.0378            5  (4)            86%              9  (7)
##
## It agrees with an independent frame-by-frame reading of the exported video, which counted 11
## discontinuities in 136 s with 7 landing on the panel already framed; this finds 5 per 90 s
## with 4 on the same panel. Three defects fall out of it, each one a report made concrete:
##   FOUR IN FIVE JUMPS LAND ON THE PANEL ALREADY FRAMED. "The camera jumps to correct its
##     position... landing on the exact same frame."
##   MOST ZOOM REVERSALS HAVE NO HOLD BETWEEN THEM. "Pulling gently outward, only to
##     IMMEDIATELY start pushing in again at its apex."
##   AND THE SLIDER IS INVERTED FOR HOLDING: camera 0 holds 71% where camera 1 holds 86%. The
##     calm end holds LESS than the busy end, which is why turning it down never helped.
##
## It asserts nothing on the first run. Print the numbers, look at them, and set the bars from
## a build that has been WATCHED - a threshold invented before anyone has seen the baseline is
## how a gate ends up green against a broken picture.

var _fails: Array = []
## Seconds of camera driven per setting, at FPS. Long enough to contain several Director holds
## at the default pacing, so the numbers cover cuts and the tails between them.
## Kept modest on purpose: each level drives a real vehicle with a full spread of live scenes
## in it, and this shares a machine with whatever else is running. Long enough to contain
## several Director holds, which is what the numbers are about.
const SECONDS := 90.0
const FPS := 30.0
## The Camera slider settings compared. The calm end is where the reports came from.
const LEVELS := [0.0, 1.0, 2.0]
## A TELEPORT is a SPIKE: one frame displacing at least this much of the frame, with quiet
## frames either side. Absolute, not a multiple of the median - the first version of this used
## `median * 12`, and at camera 0 the median is 0.00000 because the camera really is
## motionless most frames, so the bar was zero and every ordinary move counted. It reported 594
## teleports at camera 0 against 33 at camera 1 and that ordering was the instrument, not the
## camera.
##
## 0.04 of the frame in one frame is 1.2 frame widths per second, which no move in the
## vocabulary travels at. The quiet-neighbour test is what distinguishes a cut from a fast
## move: a fast move is many consecutive large frames, a cut is exactly one.
const TELEPORT := 0.04
const TELEPORT_QUIET := 0.01
## ...and as HELD when it moves less than this fraction of the frame's short side. A camera
## creeping at under a thousandth of the frame per frame is a held shot with life in it.
const STILL := 0.0012
## ...and FROZEN is the tighter bar under which the picture is not merely held but stopped.
## A camera creeping at even a ten-thousandth of the frame per frame is alive on screen; below
## this it is a photograph.
const FROZEN := 0.00004
## Frames the camera must be HELD for before a change of zoom direction counts as separated.
const HOLD_FRAMES := 15
## ...and how far the rendered scale must actually TRAVEL in one direction before a change of
## direction is a reversal at all. Without this the count is float jitter around a static
## value: the first version reported 1034 reversals in 90 seconds at camera 0, on a camera that
## was motionless for most of them. Expressed as a fraction of the page's on-screen size, so it
## means the same thing at every framing.
const ZOOM_DEADBAND := 0.02


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	var stage := SubViewport.new()
	stage.size = Vector2i(320, 180)
	add_child(stage)
	Director.detach()
	var v: Vehicle = Vehicle.make("comic")
	v.mount(stage)
	Director.attach(stage, v)
	var comic: ComicVehicle = v as ComicVehicle
	_check_pinned(comic)
	var was := Director.camera
	for sev: float in LEVELS:
		Director.set_camera(sev)
		_drive(comic, sev)
	Director.set_camera(was)
	Director.detach()
	stage.queue_free()
	for _i in 3:
		await get_tree().process_frame
	if _fails.is_empty():
		print("comic_motion_check: ALL OK")
	else:
		for f in _fails:
			print("comic_motion_check: FAIL - %s" % f)
	get_tree().quit(0 if _fails.is_empty() else 1)


## HOW MUCH DOES THE PICTURE MOVE WITH THE CAMERA PINNED?
##
## The rig is page-local: `_place_eye` builds the eye offset in SPREAD space and rotates it by
## the sheet's attitude, so the sheet's own drift is supposed to rotate the camera and the page
## by the same matrix and leave the projection invariant. comic.gd states that as measured fact
## - "1.1e-12 px over 60 s". If it is not true, the page slides under a camera that has been
## asked to hold, which is indistinguishable from a camera that will not hold, and no amount of
## work in the shot layer can fix it.
##
## So: pin the camera and the target, run the sheet's drift, and measure. Anything above the
## float noise floor means the invariance has been lost.
func _check_pinned(comic: ComicVehicle) -> void:
	comic._turn_spread(0)
	var dt := 1.0 / FPS
	# WARM UP PROPERLY BEFORE PINNING. `_flat_s` and `_dist_s` are followers, so the first
	# second after a spread turn is them converging, not the page drifting - measured, the same
	# code reported 0.003522, then 0.000604, then 0.000075 per frame as this warm-up lengthened.
	# A transient read as a steady state is how this check first accused the rig of losing its
	# projection invariance, which it has not: freezing the sheet's attitude makes the residual
	# slightly WORSE, so what is left is float noise in the solve, not the drift.
	for _w in int(8.0 / dt):
		comic.advance(Spectrum.current, dt, 0.0)
	var pinned: Dictionary = (comic._cam as Dictionary).duplicate()
	# A/B: the same run with the sheet's ATTITUDE also frozen. If the motion vanishes it is the
	# page drift and the rig is not invariant to it; if it survives, it is something in
	# _place_eye that is still converging or still stepping.
	var worst := 0.0
	var worst_still := 0.0
	var eye_move := 0.0
	for pass_i in 2:
		var att_frozen: Basis = comic._att_basis
		var prev := PackedVector2Array()
		var prev_eye: Vector3 = comic._eye
		for i in int(20.0 / dt):
			comic._cam = pinned.duplicate()
			comic._tgt = pinned.duplicate()
			# ...AND THE SHOT MUST NOT BE HOLDING, or _ease overwrites the target we just
			# pinned with the hold creep and the camera chases a point that flips between the
			# two every frame. That is the probe fighting the vehicle, and it showed up as a
			# residual that was identical with the sheet's attitude frozen - which should have
			# been the clue, since a drift artefact would have to change when the drift stops.
			comic._shot["arrived"] = false
			comic.advance(Spectrum.current, dt, 0.0)
			if pass_i == 1:
				comic._att_basis = att_frozen
				comic._place_eye(dt)
			comic._prepare_lens()
			var q := _grid(comic)
			if q.size() == prev.size() and q.size() > 0:
				for k in q.size():
					if pass_i == 0:
						worst = maxf(worst, q[k].distance_to(prev[k]))
					else:
						worst_still = maxf(worst_still, q[k].distance_to(prev[k]))
			if pass_i == 0:
				eye_move = maxf(eye_move, comic._eye.distance_to(prev_eye))
			prev = q
			prev_eye = comic._eye
	print("  camera+target PINNED 20s -> picture moved %.6f of frame per frame (eye moved %.6f world)"
		% [worst, eye_move])
	print("  ...and with the sheet's ATTITUDE frozen too -> %.6f" % worst_still)
	# PRINTED, NOT ASSERTED. This measurement has returned 0.000075, 0.0006, 0.001, 0.0035 and
	# 0.019 per frame across runs of the same code, so whatever it is measuring is not a stable
	# property and a gate built on it would fail at random. What it DID establish is worth
	# keeping: freezing the sheet's attitude does not remove the residual, so the page-local
	# rig's projection invariance is not the cause of anything - which is what it was written
	# to test. Left in as a diagnostic to read, not as a bar to clear.


## Drive the vehicle for SECONDS and report what the picture did.
##
## The Director runs FREE here - it must, because the whole question is what happens across
## cuts and in the tail of a hold, and a probe that pins the hold (as comic_look_probe does)
## makes every deadline infinite and every one of those behaviours unreachable.
func _drive(comic: ComicVehicle, sev: float) -> void:
	comic._turn_spread(0)
	var dt := 1.0 / FPS
	var steps := int(SECONDS / dt)
	var prev := PackedVector2Array()
	var prev_panel := -1
	var moves: Array = []          # per-frame screen displacement
	var areas: Array = []          # per-frame silhouette area, for the zoom direction
	var panels: Array = []         # per-frame framing panel, to tell a jump's subject
	var live: GhostScene = null
	var cuts := 0
	# HOW LONG A SHOT TAKES TO ARRIVE is what `held` is made of, and measuring it directly is
	# the difference between knowing and guessing. A shot that never reaches its target holds
	# for none of its scene however gentle the slider says it is.
	# HOW WIDE THE SHOTS ARE, across the run. "It never pulls-back, to reveal the greater page"
	# is a complaint about the DISTRIBUTION of framings, and nothing here was measuring it - a
	# camera stuck on one close-up value and a camera using its whole vocabulary produce
	# identical teleport and hold numbers.
	var fills: Array = []
	var arrivals: Array = []
	var shots := 0
	var never := 0
	var was_shot := -1
	var this_arrived := false
	for i in steps:
		Director._elapsed += dt
		# THE DIRECTOR MUST ACTUALLY CUT. The first version of this loop advanced the clock and
		# then did nothing with it, so the vehicle sat on ONE shot for the whole run - and
		# under a follower a shot that is never replaced simply arrives and holds, which
		# reported zero teleports and 90% held for a camera that had never been asked to move.
		# The cut-related half of the measurement was reading a camera doing nothing.
		#
		# A cut on a CAST-OWNING vehicle is a HANDOVER and nothing else: ask it to take over,
		# adopt what it hands back, and build or free nothing. The panels were cast when the
		# page turned and they all stay on the paper - freeing "the outgoing scene" here would
		# delete a panel out of the comic.
		if Director._elapsed >= Director.hold_remaining():
			# THE CLOCK IS RESET BEFORE THE HANDOVER, NOT AFTER. The vehicle sizes its shot
			# from Director.hold_remaining() at the moment it is asked to take over, so
			# resetting afterwards hands it the exhausted hold it is being cut out of and it
			# plans for a scene that is already over.
			Director._elapsed = 0.0
			var handed := comic.take_over(live)
			if handed != null:
				live = handed
			cuts += 1
		comic.advance(Spectrum.current, dt, 0.0)
		# SAMPLED EVERY FRAME, because _begin_shot replaces `_shot` with a fresh dict AND bumps
		# `_shot_n` in the same call - so a probe that waits for the counter to change is
		# reading the NEW shot's `arrived`, which is false by construction. This reported "no
		# shot ever arrived" for a camera that was arriving perfectly well.
		if comic._shot_n != was_shot:
			if not comic._tgt.is_empty():
				fills.append(float(comic._tgt.fill))
			if was_shot >= 0:
				shots += 1
				if not this_arrived:
					never += 1
			was_shot = comic._shot_n
			this_arrived = false
		if not this_arrived and bool(comic._shot.get("arrived", false)):
			this_arrived = true
			arrivals.append(comic._shot_t)
		comic._prepare_lens()
		var q := _grid(comic)
		var fp: int = comic._framing_panel(comic._clamp_aim(comic._cam.aim))
		if q.size() == prev.size() and q.size() > 0:
			# THE MEDIAN SAMPLE, NOT THE WORST. One point grazing the near plane projects to a
			# huge coordinate, and a max over the grid is then a measurement of that one point
			# rather than of the picture - the first version of this reported a "worst" of 44
			# frame widths, which is a projection blowing up, not a camera moving.
			var ds: Array = []
			for k in q.size():
				ds.append(q[k].distance_to(prev[k]))
			ds.sort()
			moves.append(float(ds[ds.size() / 2]))
			areas.append(_spread_px(q))
			panels.append(fp)
		prev = q
		prev_panel = fp
	if moves.size() < 10:
		_fails.append("camera %.2f produced no measurable picture" % sev)
		return
	# A RUN WITH NO CUTS IN IT MEASURES NOTHING ABOUT CUTS, and silently: see the note in the
	# loop above. If this ever trips, the numbers below are about a camera that was never asked
	# to look anywhere else.
	if cuts < 2:
		_fails.append("camera %.2f: only %d Director cuts in %.0fs - the run is not exercising them"
			% [sev, cuts, SECONDS])
	var mean_arr := 0.0
	for a in arrivals:
		mean_arr += float(a)
	mean_arr /= maxf(1.0, float(arrivals.size()))
	# THE SPREAD, NOT JUST THE MEAN. "The transitions all seem rather fast... I wouldn't expect
	# them all to be that way" is a complaint about VARIANCE, and a mean cannot show it: every
	# shot arriving in exactly 8 s and a healthy mix averaging 8 s print the same number.
	var lo := INF
	var hi := 0.0
	for a in arrivals:
		lo = minf(lo, float(a))
		hi = maxf(hi, float(a))
	var flo := INF
	var fhi := 0.0
	for f in fills:
		flo = minf(flo, float(f))
		fhi = maxf(fhi, float(f))
	print("  camera %.2f over %.0fs, %d cuts, %d shots: arrived in %.1fs mean (%.1f-%.1f), %d never arrived, framing %.2f-%.2f"
		% [sev, SECONDS, cuts, shots, mean_arr, 0.0 if lo == INF else lo, hi, never,
			0.0 if flo == INF else flo, fhi])
	# A SHOT THAT NEVER ARRIVES NEVER HOLDS, and no amount of tuning the planner can show up as
	# holding while this is non-zero. It was the whole of a 60 s run once.
	# MOST of them, not all. A shot cut before its approach could possibly finish is legitimate -
	# the Director's hold varies, and a three-second scene cannot contain a six-second arrival -
	# so the bar is that arriving is the NORM. It was 100% never-arriving once, which is the
	# failure this is really guarding against.
	if shots > 0 and float(never) / float(shots) > 0.5:
		_fails.append("camera %.2f: %d of %d shots never reached their target"
			% [sev, never, shots])
	_report(sev, moves, areas, panels)


func _report(sev: float, moves: Array, areas: Array, panels: Array) -> void:
	var sorted := moves.duplicate()
	sorted.sort()
	var med: float = float(sorted[sorted.size() / 2])
	var p99: float = float(sorted[int(float(sorted.size()) * 0.99)])
	var worst: float = float(sorted[sorted.size() - 1])
	# TELEPORTS: a single loud frame between quiet ones, and whether it changed the subject.
	var jumps := 0
	var same_panel := 0
	for i in range(1, moves.size() - 1):
		if float(moves[i]) < TELEPORT:
			continue
		if float(moves[i - 1]) > TELEPORT_QUIET or float(moves[i + 1]) > TELEPORT_QUIET:
			continue                       # a fast move, not a cut
		jumps += 1
		if int(panels[i]) == int(panels[i - 1]):
			same_panel += 1
	# HELD, measured on the picture...
	var held := 0
	for m in moves:
		if float(m) < STILL:
			held += 1
	# ...AND THE LONGEST UNBROKEN STILL RUN, which is the number `held` cannot give you. A good
	# hold and a dead freeze are both "held": the difference is whether it ever ends. A
	# frame-by-frame reading of the exported video found 33.8 seconds of literally identical
	# picture inside a scene that scored as well-held, so this is the complement that catches
	# it - and it caught a hold creep that saturated on a clock and then stopped dead.
	var run := 0
	var longest := 0
	for m in moves:
		if float(m) < FROZEN:
			run += 1
			longest = maxi(longest, run)
		else:
			run = 0
	# REVERSALS of the rendered scale, and whether a hold separated them.
	# REVERSALS, with a DEADBAND. A direction is only in force once the scale has actually
	# travelled ZOOM_DEADBAND that way, and a reversal only counts once it has travelled that
	# far back - so a shot creeping in, holding, and creeping out is one reversal, and a static
	# shot whose projected size wobbles in the last decimal place is none.
	var dir := 0
	var since_hold := 0
	var flips := 0
	var unheld := 0
	var anchor: float = float(areas[0])
	for i in range(1, areas.size()):
		if float(moves[i]) < STILL:
			since_hold += 1
		var scale: float = maxf(float(areas[i]), 1e-4)
		var travel: float = (float(areas[i]) - anchor) / scale
		if absf(travel) < ZOOM_DEADBAND:
			continue
		var nd := 1 if travel > 0.0 else -1
		anchor = float(areas[i])
		if dir != 0 and nd != dir:
			flips += 1
			if since_hold < HOLD_FRAMES:
				unheld += 1
			since_hold = 0
		dir = nd
	print("      median %.5f  p99 %.5f  worst %.5f of frame" % [med, p99, worst])
	print("      teleports %d (%d landed on the panel already framed)   held %.0f%%   frozen run %.1fs   zoom reversals %d (%d without a hold between)" % [
		jumps, same_panel, 100.0 * float(held) / float(moves.size()),
		float(longest) / FPS, flips, unheld])


## A FIXED GRID OF POINTS ON THE PAPER, projected. This is the honest way to ask "how much
## did the picture move": the samples are nailed to the sheet, so their screen displacement IS
## the content sliding across the frame, in frame widths.
##
## Sampling the SHEET rather than its silhouette is the whole point. The silhouette's corners
## are the least stable points in the scene - they are the farthest from the aim, the first to
## cross the near plane, and the first to be clipped - so a metric built on them measures the
## geometry of the page's edge instead of the picture. These sit across the printed area, where
## the camera is actually looking. Points behind the eye are dropped, and a frame whose set of
## survivors differs from the last one is skipped rather than compared against a different set.
func _grid(comic: ComicVehicle) -> PackedVector2Array:
	var out := PackedVector2Array()
	for iy in 3:
		for ix in 5:
			var sp := Vector2(0.2 + 1.6 * float(ix) / 4.0,
				comic._spread.aspect * (0.2 + 0.6 * float(iy) / 2.0))
			var w: Vector3 = comic._world(sp, comic._att_basis)
			if comic._lens.depth(w) <= comic._lens.near:
				return PackedVector2Array()     # any sample behind the eye voids the frame
			var pr: Vector3 = comic._lens.project(w)
			out.append(Vector2(pr.x, pr.y))
	return out


## HOW BIG THE PAGE IS ON SCREEN: the spread of the projected grid. The rendered SCALE - what
## "zoomed in" means as a number, without asking the camera what it intended.
func _spread_px(q: PackedVector2Array) -> float:
	var lo := Vector2(INF, INF)
	var hi := Vector2(-INF, -INF)
	for p in q:
		lo = lo.min(p)
		hi = hi.max(p)
	return (hi - lo).length()
