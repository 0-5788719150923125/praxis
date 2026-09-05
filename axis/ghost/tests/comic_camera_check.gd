extends Node

## Does the Camera knob move the shot planner along one axis, and is the angular walk bounded?
##
##   tests/run_boot_probe.sh tests/comic_camera_check.gd 120
##
## Needs a real boot (it reaches Director and Settings) but NOT a real renderer: everything
## here is the planning arithmetic, read out of `_tgt` before anything draws.
##
## THIS MEASURES THE PLAN, AND THAT IS ALL IT CLAIMS. It is the right instrument for "is the
## slider monotone" - a statement about a distribution over hundreds of shots, which no amount
## of watching can judge - and the WRONG one for "does the camera behave", which is
## tests/comic_motion_check.gd's job: that one drives the real vehicle and measures the picture.
##
## FOUR CHECKS THAT USED TO LIVE HERE WENT WITH THE MOVE LAYER THEY TESTED. They asked whether
## a move fitted its deadline, whether a chain went forward, whether a settle repeated, whether
## a chained move re-rolled its zoom - all bookkeeping internal to a mechanism that no longer
## exists. Every one of them was green through a night of reports that the camera jumped, never
## held, and swung wildly, because none of them looked at the picture. That is the lesson worth
## keeping from them: a gate that measures a mechanism only tells you the mechanism ran.

var _fails: Array = []
## How many shots to plan per setting. Enough that the means are steady; this is arithmetic,
## so it costs nothing.
const SHOTS := 600
## The Camera settings compared. The ends of Director's range and the tuned middle.
const LEVELS := [0.0, 1.0, 2.0]
## Shots planned and DISCARDED at the start of each setting, before anything is counted.
##
## The camera carries state across a change of setting - that is the point of it; the slider is
## not supposed to teleport the picture. So the first shots after a change are the old
## setting's pose being walked off, and counting them measures the handover rather than the
## setting. Forty, because most shots preserve the azimuth rather than taking a fresh one, so a
## camera that starts off-anchor is re-anchored only by the minority that swing it, and the
## handover can run for a couple of dozen shots.
const WARMUP := 40


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
	var reach: Array = []
	var jumps: Array = []
	for sev in LEVELS:
		Director.set_camera(float(sev))
		var r := _measure(comic)
		swing.append(r["swing"])
		reach.append(comic._reach())
		jumps.append(r["jump"])
		print("  camera %.1f -> mean shot-to-shot swing %.1f deg, %.0f%% jump cuts, "
			% [sev, rad_to_deg(float(r["swing"])), 100.0 * float(r["jump"])]
			+ "reach %.2f, walk stayed within %.1f deg"
			% [comic._reach(), rad_to_deg(float(r["worst"]))])
		# THE WALK IS BOUNDED BY ITS ARC, at every setting. If this ever fails the clamp in
		# _walk has been lost and the whole "spread over several shots" guarantee with it.
		if float(r["worst"]) > ComicVehicle.AZ_ARC * float(sev) + 1e-4:
			_fails.append("camera %.1f: the azimuth walk reached %.3f rad, outside its %.3f arc"
				% [sev, float(r["worst"]), ComicVehicle.AZ_ARC * float(sev)])

	# THE KNOB IS MONOTONE. Not "different at the ends" - every step of it has to move the
	# camera the same way, or it is a switch with a slider drawn on it.
	for i in LEVELS.size() - 1:
		if float(swing[i + 1]) <= float(swing[i]):
			_fails.append("swing did not grow from camera %.1f to %.1f (%.4f -> %.4f rad)"
				% [LEVELS[i], LEVELS[i + 1], swing[i], swing[i + 1]])
		# ...AND SO IS REACH, which is the axis that decides holding. Under a follower there is
		# no move duration to compare: a shot arrives when the picture gets there, so what
		# separates a calm camera from a busy one is how FAR it goes, not how slowly. Scaling
		# only the speed is what made the previous build hold 71% at camera 0 against 86% at
		# camera 1 - the calm end holding LESS, which is why turning the slider down never
		# helped.
		if float(reach[i + 1]) <= float(reach[i]):
			_fails.append("reach did not grow from camera %.1f to %.1f (%.2f -> %.2f)"
				% [LEVELS[i], LEVELS[i + 1], reach[i], reach[i + 1]])

	# AT ZERO THE CAMERA NEVER JUMPS AND NEVER TURNS. This is the end of the range the report
	# asked for by name ("gentle, slow movements"), and it has to be exact rather than merely
	# smaller - a camera that jump-cuts 2% of the time still reads as a camera that jump-cuts.
	if float(jumps[0]) > 0.0:
		_fails.append("camera 0 still jump-cut on %.1f%% of shots" % (100.0 * float(jumps[0])))
	if float(swing[0]) > 1e-6:
		_fails.append("camera 0 still swung the camera %.4f rad per shot" % float(swing[0]))

	_check_film_fit(comic)
	_check_focal_live(comic)

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


## Plan SHOTS shots at the current setting and report what the planner chose.
func _measure(comic: ComicVehicle) -> Dictionary:
	comic._spread = ComicSpread.new(hash(["cam", Director.camera]))
	comic._spread_i = 3
	comic._film_at = -1
	comic._roll_plan()
	comic._read = int(comic._plan[0])
	comic._choose_spread_look()
	var swing := 0.0
	var jumps := 0
	var worst := 0.0
	var n := 0
	for m in SHOTS + WARMUP:
		var before: float = float(comic._cam.az)
		comic._begin_shot(comic._read % comic._spread.panels.size())
		# The planner's answer is the TARGET. Where the camera is on the way there is the
		# follower's business, and is continuous by construction.
		var az: float = float(comic._tgt.az)
		if m >= WARMUP:
			swing += absf(angle_difference(before, az))
			worst = maxf(worst, absf(comic._az_walk))
			if bool(ComicVehicle.SHOTS[String(comic._shot["kind"])].get("hard", false)):
				jumps += 1
			n += 1
		# Stand the camera on its target so the next shot is planned from where this one led -
		# what the running vehicle does once the follower has closed.
		comic._cam = (comic._tgt as Dictionary).duplicate()
	return {
		"swing": swing / maxf(1.0, float(n)),
		"jump": float(jumps) / maxf(1.0, float(n)),
		"worst": worst,
	}


## IS THE WHOLE VIDEO ON SCREEN WHEN THE VIDEO IS WHAT WE ARE LOOKING AT?
##
## "It can look very strange to zoom in to a specific corner of the embedded video, where
## nothing is happening in it... a video with a girl speaking in it, you really don't want to
## focus the camera on her door to the side." Every other panel is framed on HEIGHT alone and
## allowed to overflow the sides, which is right for an abstract field and wrong for footage.
##
## Asserted here rather than in the look probe because the look probe cannot see it: the film
## panel is the FIRST entry of the reading plan, so it has been read and left behind before the
## probe's first capture. This drives _place_eye directly on the widest panel of a spread, with
## and without the film flag, so the contain path is compared against the shot it replaces.
func _check_film_fit(comic: ComicVehicle) -> void:
	comic._spread = ComicSpread.new(hash(["filmfit"]))
	comic._spread_i = 3
	comic._att = Vector3(0.3, 0.35, 0.15)
	comic._att_basis = Basis.from_euler(comic._att)
	# The widest panel on the sheet - the one a height-only fit crops hardest.
	var wide := 0
	for i in comic._spread.panels.size():
		if comic._spread.panel_aspect(i) > comic._spread.panel_aspect(wide):
			wide = i
	comic._read = wide
	comic._plan = [wide]
	comic._step = 0
	var out := {}
	for as_film in [false, true]:
		comic._film_at = wide if as_film else -1
		# PIN THE WHOLE CAMERA STATE, not just the aim. az/el/roll/fov carry over from whatever
		# the checks above left in _cam, so without this the two framings below are measured
		# from different stations and the comparison - and the run-to-run numbers - drift.
		comic._cam["aim"] = comic._spread.panel_center(wide)
		comic._cam["fill"] = 1.2
		comic._cam["az"] = 0.6
		comic._cam["el"] = deg_to_rad(52.0)
		comic._cam["roll"] = 0.08
		comic._cam["fov"] = 46.0
		comic._place_eye()
		comic._prepare_lens()
		out[as_film] = comic.read_panel_fit()
		out["cov%s" % as_film] = comic.page_coverage()
	print("  widest panel (aspect %.2f): as a scene %.2f of frame (page covers %.2f), "
		% [comic._spread.panel_aspect(wide), out[false], out["covfalse"]]
		+ "as film %.2f (page covers %.2f)" % [out[true], out["covtrue"]])
	if float(out[true]) > 1.0:
		_fails.append("a film panel being read is cropped by the frame (%.2f of it)"
			% float(out[true]))
	# ...and the contain path must actually be DOING something, or the assertion above is
	# passing for the wrong reason on a panel that happened to fit anyway.
	if float(out[false]) <= 1.0:
		print("  (note: this panel fits even without containing, so the contrast is weak)")
	comic._film_at = -1


## IS THE PANEL BEING READ EVER FROZEN?
##
## "The scene in that comic book frame becomes stuck, frozen, and stops moving at all." The
## liveness sort puts every panel below its warm-up ahead of every warm one, and LIVE_MAX is
## three - so the three panels cast by a page turn could take the whole budget and freeze the
## Director's own current scene. This casts a full spread (so every panel is cold, the worst
## case) and asserts the read panel survives it.
func _check_focal_live(comic: ComicVehicle) -> void:
	comic._spread = ComicSpread.new(hash(["focal"]))
	comic._spread_i = 11
	comic._film_at = -1
	var n := comic._spread.panels.size()
	comic._cast = []
	comic._cast.resize(n)
	comic._warm = []
	comic._warm.resize(n)
	comic._warm.fill(0)
	# Stand in for a cast spread: every panel holds a scene and none has drawn yet.
	for i in n:
		var sc := GhostScene.new()
		comic._slots[comic._pool * ComicVehicle.POOL + i].add_child(sc)
		comic._cast[i] = sc
	var missed := 0
	for i in n:
		comic._read = i
		comic._update_liveness()
		if not comic._live.has(i):
			missed += 1
	print("  every panel of a freshly cast %d-panel spread read in turn -> focal frozen %d time(s)"
		% [n, missed])
	if missed > 0:
		_fails.append("the panel being READ was frozen on %d of %d panels - its scene stops"
			% [missed, n])
	for i in n:
		if comic._cast[i] != null and is_instance_valid(comic._cast[i]):
			comic._cast[i].queue_free()
	comic._cast = []
	comic._warm = []
