extends Node

## Does the camera severity knob actually move the camera along one axis, and does the angular
## walk actually spread a swing over several shots?
##
##   tests/run_boot_probe.sh tests/comic_camera_check.gd 90
##
## Needs a real boot (it reaches Director and Settings) but NOT a real renderer: everything
## here is the planning arithmetic, read out of `_mv` before anything draws.
##
## IT PLANS MOVES WITHOUT CUTTING. `take_over` would cast a real scene per panel, and four
## hundred of those is minutes of scene construction to measure a number that never touches a
## pixel. Setting `_spread` and calling `_plan_move` is the same code path the Director's cut
## reaches, minus the casting - the same shortcut tests/film_clock_check.gd takes to walk
## hundreds of spreads.
##
## WHAT IT IS GUARDING. Two reports, both about the same thing from different ends: "the
## camera can rotate/turn A LOT every time it shifts from one frame to another", and a wish
## for one slider from "gentle, slow movements" to "faster, more chaotic, more cinematic".
## The first is a bound on ONE shot's swing; the second is that the bound is tunable. Neither
## is checkable by looking at a frame, because both are about the DISTRIBUTION over many
## shots - which is exactly the kind of thing that gets tuned by eye, shipped, and reported
## again.

var _fails: Array = []
## How many moves to plan per severity. Enough that the means are steady; this is arithmetic,
## so it costs nothing.
const MOVES := 600
## The severities compared. The ends of Director's range and the tuned middle.
const LEVELS := [0.0, 1.0, 2.0]
## Moves planned and DISCARDED at the start of each severity, before anything is counted.
##
## The camera carries state across a change of setting - that is the point of it; the slider
## is not supposed to teleport the picture. So the first shots after a change are the old
## severity's pose being walked off, and counting them measures the handover rather than the
## setting. Measured: without this, severity 0 reported a residual 0.002 rad per shot, all of
## it one 0.1 rad correction on the third move as the camera returned to its base azimuth.
## One spread's worth is comfortably enough - at severity 0 the walk reaches its target on the
## first station.
const WARMUP := 8


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
	Director.hold(true)          # nothing but this probe advances the reading
	var comic: ComicVehicle = v

	var was := Director.camera
	var swing: Array = []
	var dur: Array = []
	var jumps: Array = []
	for sev in LEVELS:
		Director.set_camera(float(sev))
		var r := _measure(comic)
		swing.append(r["swing"])
		dur.append(r["dur"])
		jumps.append(r["jump"])
		print("  severity %.1f -> mean shot-to-shot swing %.1f deg, mean move %.1fs, "
			% [sev, rad_to_deg(float(r["swing"])), float(r["dur"])]
			+ "%.0f%% jump cuts, walk stayed within %.1f deg"
			% [100.0 * float(r["jump"]), rad_to_deg(float(r["worst"]))])
		# THE WALK IS BOUNDED BY ITS ARC, at every setting. If this ever fails the clamp in
		# _walk has been lost and the whole "spread over several shots" guarantee with it.
		if float(r["worst"]) > ComicVehicle.AZ_ARC * float(sev) + 1e-4:
			_fails.append("severity %.1f: the azimuth walk reached %.3f rad, outside its %.3f arc"
				% [sev, float(r["worst"]), ComicVehicle.AZ_ARC * float(sev)])

	# THE KNOB IS MONOTONE. Not "different at the ends" - every step of it has to move the
	# camera the same way, or it is a switch with a slider drawn on it.
	for i in LEVELS.size() - 1:
		if float(swing[i + 1]) <= float(swing[i]):
			_fails.append("swing did not grow from severity %.1f to %.1f (%.4f -> %.4f rad)"
				% [LEVELS[i], LEVELS[i + 1], swing[i], swing[i + 1]])
		if float(dur[i + 1]) >= float(dur[i]):
			_fails.append("moves did not get shorter from severity %.1f to %.1f (%.2f -> %.2f s)"
				% [LEVELS[i], LEVELS[i + 1], dur[i], dur[i + 1]])

	# AT ZERO THE CAMERA NEVER JUMPS AND NEVER TURNS. This is the end of the range the report
	# asked for by name ("gentle, slow movements"), and it is the one that has to be exact
	# rather than merely smaller - a camera that jump-cuts 2% of the time still reads as a
	# camera that jump-cuts.
	if float(jumps[0]) > 0.0:
		_fails.append("severity 0 still jump-cut on %.1f%% of shots" % (100.0 * float(jumps[0])))
	if float(swing[0]) > 1e-6:
		_fails.append("severity 0 still swung the camera %.4f rad per shot" % float(swing[0]))

	Director.set_camera(was)
	Director.hold(false)
	Director.detach()
	stage.queue_free()
	for _i in 3:
		await get_tree().process_frame
	if _fails.is_empty():
		print("comic_camera_check: ALL OK")
	else:
		for f in _fails:
			print("comic_camera_check: FAIL - %s" % f)
	get_tree().quit(0 if _fails.is_empty() else 1)


## Plan MOVES moves across a run of spreads and report the distribution.
##
## `swing` is the ANGULAR DISTANCE ONE SHOT TRAVELS - the azimuth from where the camera
## starts to where the move puts it - which is the quantity the report is about. Measuring
## the walk's own position instead would say nothing: the walk is bounded by construction,
## and a bounded quantity re-rolled every shot is exactly the behaviour being removed.
func _measure(comic: ComicVehicle) -> Dictionary:
	var swing := 0.0
	var dur := 0.0
	var hard := 0
	var worst := 0.0
	var n := 0
	for m in MOVES:
		if m % 8 == 0:                       # a fresh spread every eight shots, as a page turn
			comic._spread_i = m / 8
			comic._spread = ComicSpread.new(hash(["cam", m]))
			comic._read = 0
			comic._choose_spread_look()
		else:
			comic._read = m % 8
		comic._plan_move(hash(["cam-move", m]), comic._read % comic._spread.panels.size(), "")
		var a: Dictionary = comic._mv["a"]
		var b: Dictionary = comic._mv["b"]
		worst = maxf(worst, absf(comic._az_walk))
		# See WARMUP: the opening draw and the walk-off of the previous setting are the
		# handover, not the behaviour.
		if m >= WARMUP:
			swing += absf(wrapf(float(b["az"]) - float(a["az"]), -PI, PI))
			dur += float(comic._mv["dur"])
			if bool(ComicVehicle.MOVES[String(comic._mv["kind"])].get("hard", false)):
				hard += 1
			n += 1
		# CARRY THE CAMERA FORWARD, or every move starts from the same place and a
		# continuity-preserving move measures as a jump. This is what advance() does with the
		# result of a move, compressed to its end state.
		comic._cam = b.duplicate()
	return {"swing": swing / float(n), "dur": dur / float(n),
		"jump": float(hard) / float(n), "worst": worst}
