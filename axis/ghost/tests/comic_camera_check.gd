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
## FORTY, not the eight this started at. Most moves in the bag now PRESERVE the azimuth rather
## than taking it from a fresh station - that is what keeps them continuous - so a camera that
## starts off-anchor is re-anchored only by the minority of moves that do take a station, and
## the handover can run for a couple of dozen shots. Eight left a residue of up to 0.0045 rad
## in the severity-0 figure whose true value is exactly zero, and it came and went between
## runs, which is worse than being wrong: a gate that fails one run in three is a gate people
## learn to re-run.
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

	_check_deadline(comic)
	_check_film_fit(comic)
	_check_chain_commit(comic)
	_check_zoom_stability(comic)
	_check_focal_live(comic)
	_check_settle_runs(comic)

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


## DOES THE CAMERA SETTLE ONCE, OR OVER AND OVER?
##
## A settle holds the shot still until the cut arrives, and it eases the aim a fifth of the way
## toward its panel as it does. Planning a FRESH one each time the last expires therefore
## re-corrects onto a frame the camera is already on, repeatedly - "the camera corrects and
## jumps to focus on the exact same frame it's already on... it never holds long enough to look
## at anything". Measured in a real export log before the fix: 19 of 42 moves were settles, in
## runs of up to seven.
##
## The cause was upstream, in Director.hold_remaining(), which reported ZERO for the whole tail
## of any hold outliving its median - most of the second half of every scene - so the vehicle
## was told the cut was imminent for tens of seconds and settled again and again. This drives a
## scene right through its window and counts how many distinct settles come out.
func _check_settle_runs(comic: ComicVehicle) -> void:
	Director.hold(false)
	var spec_was: Dictionary = Director._current.exit_spec
	comic._spread = ComicSpread.new(hash(["settle"]))
	comic._spread_i = 13
	comic._film_at = -1
	comic._roll_plan()
	comic._step = 0
	comic._read = int(comic._plan[0])
	Director._current.exit_spec = {"hold": 30.0}
	Director._elapsed = 0.0
	comic._choose_move()
	var settles := 0
	var dt := 1.0 / 30.0
	# Run a 30 s scene right to its backstop, ticking the camera as the app does.
	for f in int(30.0 / dt):
		Director._elapsed += dt
		var was := String(comic._mv.get("kind", ""))
		comic._ease(dt)
		if String(comic._mv.get("kind", "")) == "settle" and was != "settle":
			settles += 1
	print("  one 30s scene driven to its backstop -> %d distinct settle(s)" % settles)
	if settles > 1:
		_fails.append("the camera settled %d separate times inside one scene - it re-corrects"
			% settles)
	Director._current.exit_spec = spec_was
	Director._elapsed = 0.0
	Director.hold(true)


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


## DOES THE ZOOM SIT STILL BETWEEN CUTS?
##
## "It zoomed-in then immediately zoomed-out again, but barely. It barely moved in either
## direction; that felt strange, and it would have been better to just remain stable, or to
## zoom and keep zooming."
##
## Every move takes its target framing from a fresh _station, which SAMPLES `fill`. At a cut
## that is the point - a new shot. Between cuts it is not: the chain is the same shot
## continuing, so re-rolling the distance nudged the picture in and then out by a few percent
## with no gesture behind it. Only `push` and `pull`, whose whole subject is the zoom, may
## change it now.
func _check_zoom_stability(comic: ComicVehicle) -> void:
	comic._spread = ComicSpread.new(hash(["zoom"]))
	comic._spread_i = 9
	comic._film_at = -1
	comic._roll_plan()
	comic._step = 0
	comic._read = int(comic._plan[0])
	comic._choose_move()
	var moved := 0
	var worst := 0.0
	var zooms := 0
	for c in 60:
		var before: float = comic._cam["fill"]
		comic._chain_move()
		var after: float = comic._mv["b"]["fill"]
		var kind := String(comic._mv["kind"])
		if kind == "push" or kind == "pull":
			zooms += 1
			continue
		if not is_equal_approx(before, after):
			moved += 1
			worst = maxf(worst, absf(after - before))
		# ...and carry the camera to the move's end, as _ease would.
		comic._cam = (comic._mv["b"] as Dictionary).duplicate()
	print("  60 chained moves -> %d re-rolled the zoom (worst %.3f frames), %d were push/pull"
		% [moved, worst, zooms])
	if moved > 0:
		_fails.append("%d chained move(s) changed the zoom without being a push or a pull"
			% moved)


## DOES A RUN OF CHAINED MOVES KEEP CHANGING ITS MIND?
##
## "The camera was bouncing between two diagonal corners - clearly because of some kind of
## repulsion mechanism. But repulsion is not the right approach here: the camera should have
## never selected the top-right frame again, in the first place."
##
## There is no repulsion in this vehicle. What there was is a destination recomputed on every
## chain from where the camera happened to BE - ahead when it was on a panel, back to the one
## being read when it was over a gutter - so a traverse that crossed a gutter re-targeted
## itself mid-flight, arrived, and re-targeted back. This drives many chains between one cut
## and the next and asserts the target never moves once chosen, which is the property that
## makes an oscillation impossible rather than merely unlikely.
func _check_chain_commit(comic: ComicVehicle) -> void:
	comic._spread = ComicSpread.new(hash(["chain"]))
	comic._spread_i = 5
	comic._film_at = -1
	comic._roll_plan()
	comic._step = 0
	comic._read = int(comic._plan[0])
	comic._choose_move()
	var targets: Array = []
	for c in 40:
		# Walk the camera to somewhere ARBITRARY between chains, including out over the paper
		# where the old code would have called it lost. The target must not care.
		comic._cam["aim"] = comic._spread.panel_center(int(comic._plan[c % comic._plan.size()])) \
			if c % 3 else Vector2(ComicSpread.SPINE, comic._spread.aspect * 0.5)
		comic._chain_move()
		targets.append(comic._chain_to)
	# MONOTONE, not constant. The target may advance once - from the panel being read to the
	# panel the next cut will land on - and may never go back. Monotone is what makes an
	# oscillation inexpressible; constant would also stop the camera dead.
	var back := 0
	var steps := 0
	for i in range(1, targets.size()):
		if int(targets[i]) != int(targets[i - 1]):
			steps += 1
			if comic._plan.find(int(targets[i])) < comic._plan.find(int(targets[i - 1])):
				back += 1
	print("  40 chained moves between cuts -> target advanced %d time(s), went back %d"
		% [steps, back])
	if back > 0:
		_fails.append("the chain went BACKWARD %d time(s) - it can oscillate" % back)
	if steps > 1:
		_fails.append("the chain advanced %d times between two cuts - it should advance once"
			% steps)


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


## DOES A MOVE FINISH BEFORE THE CUT THAT ENDS IT?
##
## The defect this guards: "you can see it converging, then BOOM. The camera cuts to something
## else... we never gave the camera enough time to slow down, and settle somewhere, before the
## transition." A move sampled seven to twenty seconds against a hold whose median is five and
## a half on a driving passage is interrupted more often than not, and no still frame can show
## that - it is a relationship between two durations that live in different files.
##
## What it was before the deadline existed, computed from the bag's weights and its uniform
## duration ranges at severity 1: with 4s left 3.6% of moves could finish, with 8s 17.4%, with
## 16s 95.2%. The first two are the driving and middling passages, which is to say most of a
## song. Now every move finishes with the rest of the hold left over to be settled in.
##
## Driven through the REAL [method Director.hold_remaining], not a stub: a scene whose exit
## spec carries a fixed `hold` takes that function's deterministic branch, so setting one and
## zeroing the clock gives an exact, controlled "seconds left" without inventing a second
## implementation of the thing being tested.
func _check_deadline(comic: ComicVehicle) -> void:
	Director.hold(false)                       # hold_remaining reports 1e9 while frozen
	var spec_was: Dictionary = Director._current.exit_spec
	comic._spread = ComicSpread.new(hash(["deadline"]))
	comic._spread_i = 1
	comic._read = 0
	# The holds a real session actually produces: a driving passage, a middling one, and calm.
	for room in [4.0, 8.0, 16.0]:
		Director._current.exit_spec = {"hold": float(room)}
		Director._elapsed = 0.0
		var fits := 0
		var longest := 0.0
		for m in 240:
			comic._plan_move(hash(["deadline", room, m]), m % comic._spread.panels.size(), "")
			var d := float(comic._mv["dur"])
			longest = maxf(longest, d)
			if d <= float(room):
				fits += 1
		var pct := 100.0 * float(fits) / 240.0
		print("  %4.1fs until the cut -> %.0f%% of moves finish in time, longest %.1fs"
			% [room, pct, longest])
		# EVERY move must fit, not most of them. A move that cannot finish is the report.
		if fits < 240:
			_fails.append("with %.1fs left, %d of 240 moves ran past the cut (longest %.1fs)"
				% [room, 240 - fits, longest])
	Director._current.exit_spec = spec_was
	Director.hold(true)


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
