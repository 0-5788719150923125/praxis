extends Node

## NOT a gate - it asserts almost nothing. It drives a real comic-vehicle session and
## writes PNGs, because "does this read as a comic page" is a question a picture answers
## and a source file does not. (city_look_probe.gd and clown_look_probe.gd exist for the
## same reason, and the same reason again.)
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/comic_look_probe.gd 240 \
##       --out /tmp/comic --cuts 9 --seed 404
##
## GHOST_PROBE_GPU is not optional: this reaches Director and Spectrum (so it needs a real
## boot) and reads pixels back (so it needs a real renderer - --headless is the dummy
## driver, whose readback returns nothing).
##
## It drives the vehicle DIRECTLY rather than waiting for the Director's musical cues: a
## cut here is `host_for` plus a scene, which is exactly what the Director does at a cut
## and nothing else. Waiting for real cues would have meant minutes of silent audio per
## page, and the schedule is not what is being looked at.
##
## The one thing it does assert is the trivially checkable one: that the frame is not
## uniform. A page that failed to project, or a texture that never arrived, comes out as
## a flat field, and a flat field is the failure this is most likely to ship.

const W := 1280
const H := 720
const DT := 1.0 / 30.0

var _out := "user://comic"
## Which vehicle to drive. `comic` is the point of the probe; `full` is here so the same
## harness can answer "is this the comic, or is it ghost?" about anything odd it turns up.
var _vehicle_key := "comic"
var _cuts := 9
var _shots: Array = []          # cut indices to photograph; default = every third
var _wrote := 0
var _flat := 0
var _thin := 0                    # frames where the page did not cover the frame
var _static := 0                  # shots where the camera had stopped moving
var _cropped := 0                 # shots where a FILM panel was cropped by the frame
## How many frames to let the camera travel before photographing. The default covers the
## slowest pan in the vocabulary; a LONG settle is the interesting one, because the defect
## this probe exists to catch is what the camera does AFTER its move has run out - see
## _static below.
var _settle := 90
## Spread-coordinate units per second of AIM travel, under which the camera counts as
## STOPPED. One unit is one page width, so this is a hundredth of a page a second - a
## framing that is not going anywhere.
##
## THE AIM, NOT THE EYE, and that distinction is the whole measurement. The sheet's attitude
## drifts forever (ComicVehicle.DRIFT), and the eye is placed relative to the sheet, so the
## eye keeps moving at 0.05 world units a second even when the framing has been locked on
## one corner for twenty seconds. Measuring the eye says the camera is alive; measuring what
## it is LOOKING AT says it is not. The second one is what the report was about.
const STILL := 0.010
var _vehicle: Vehicle = null
var _live: GhostScene = null      # the scene in the open panel; this probe drives it


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	_parse_args()
	await _shoot()
	print("comic_look_probe: wrote %d frame(s) under %s" % [_wrote, _out])
	if _flat > 0:
		print("comic_look_probe: %d frame(s) were UNIFORM - the page did not draw." % _flat)
	if _thin > 0:
		print("comic_look_probe: %d frame(s) showed DESK - the page did not cover." % _thin)
	if _static > 0:
		print("comic_look_probe: %d shot(s) had a STOPPED camera - the move ran out and nothing replaced it." % _static)
	if _cropped > 0:
		print("comic_look_probe: %d shot(s) CROPPED THE FILM - the whole video was not on screen." % _cropped)
	for _i in 6:
		await get_tree().process_frame
	get_tree().quit(1 if _flat > 0 else 0)


func _parse_args() -> void:
	var args := OS.get_cmdline_user_args()
	for i in args.size():
		if i + 1 >= args.size():
			break
		match args[i]:
			"--out": _out = args[i + 1]
			"--vehicle": _vehicle_key = args[i + 1]
			"--cuts": _cuts = int(args[i + 1])
			"--settle": _settle = int(args[i + 1])
			"--shots":
				_shots = []
				for s in String(args[i + 1]).split(","):
					_shots.append(int(s))
			# FOOTAGE ON THE PAGE. Pass a prepared .ogv and every page gives a panel to it,
			# so "does a film panel read as part of the comic" is a question these pictures
			# answer. It goes through Films' probe seam rather than the library, because
			# the library is in Settings and a probe may not write that.
			"--film":
				var path := String(args[i + 1])
				var dur := Films._probe_duration(path)
				Films.use_for_test([{"source": path, "slug": Films._slug(path),
					"name": path.get_file().get_basename(), "duration": maxf(dur, 1.0)}], 1.0)
				print("comic_look_probe: film %s (%.1fs) on every page" % [path, dur])


## Shoots one session. The seed is the DIRECTOR'S - pass `--seed N` on the command line
## and it resolves it the way every songless boot does. It used to be pinned here by
## writing Director._session_seed and re-rolling the vehicle afterwards, which is a state
## the app can never be in, and testing a sequence the app does not perform is how the
## last two defects in this file got in.
func _shoot() -> void:
	var stage := SubViewport.new()
	stage.size = Vector2i(W, H)
	stage.transparent_bg = false
	stage.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(stage)

	# A real session, so the vehicle samples its paper and its pages off a real seed. The
	# Director's own scheduling is not driven here (there is no audio); the cuts below are.
	Director.detach()
	var vehicle: Vehicle = Vehicle.make(_vehicle_key)
	_vehicle = vehicle
	vehicle.mount(stage)
	Director.attach(stage, vehicle)
	# PIN THE SEED, which attach() has just resolved at random (there is no audio here, so
	# there is no fingerprint to derive it from). Without this the probe names its files
	# after a seed it never set, and two runs of `--seeds 3` are two different shows - which
	# is the one thing a look probe must not be, because comparing before and after is the
	# whole reason it exists. Re-rolling begin_session is what makes it take: the vehicle
	# samples its paper, its ink and its first page there, off exactly this number.
	# FREEZE THE DIRECTOR'S OWN CUTTING. It is attached and running, so without this it
	# advances the show on its max-hold backstop at the same time as this probe advances it
	# explicitly - two drivers on one reading, and the panel it lands on is whichever won
	# the frame. hold() is the same freeze the feedback console uses; the probe already
	# ticks the focal scene itself, which is the other half of what hold() stops.
	Director.hold(true)
	# ADOPT what attach() set up rather than replacing it. Under a cast-owning vehicle
	# Director._current IS one of the page's panels, so an earlier version of this probe
	# freeing it to "start clean" was reaching into the comic and deleting a panel out of
	# its own page - the reading then had nowhere to go and stuck on the panel behind.
	_live = Director._current
	print("--- %s seed %d: %s ---" % [
		_vehicle_key, Director.session_seed(), _page_line(vehicle)])

	var t := 0.0
	for cut in _cuts:
		var comic := vehicle is ComicVehicle
		var was_spread: int = vehicle._spread_i if comic else 0
		# STAND IN FOR main._process. A probe scene REPLACES main, so the one place that
		# promotes a finished window cut is not in the tree - without this the probe waits
		# forever on `.part` files it started itself, a deadlock the app cannot have but
		# every probe can.
		Films.pump()
		_cut(vehicle)
		if comic:
			var turned: bool = vehicle._spread_i != was_spread
			# WHICH PANEL HOLDS FOOTAGE, on every line. "the video is only ever shown ONE
			# time, on ONE page" was reported from watching, and could not be checked from
			# this probe's output at all - the one number that would have shown it was the
			# only one not printed.
			var film := ""
			if vehicle._film_at >= 0:
				film = "[film p%d] " % (vehicle._film_at + 1)
			print("    cut %d -> spread %d, reading panel %d of %d, plan %s, live %s %s%s" % [
				cut, vehicle._spread_i, vehicle._read + 1, vehicle._spread.panels.size(),
				vehicle._plan, vehicle._live, film,
				("(leaf turned) " + _page_line(vehicle)) if turned else ""])

		else:
			print("    cut %d" % cut)
		# Let the panel's scene build and the camera ease toward the new framing.
		#
		# THE SCENE MUST BE TICKED HERE. A GhostScene has no _process of its own - the
		# Director calls update() on it every frame (see Director._tick_animation) - so a
		# probe that only waits draws each scene exactly once, on the frame it enters the
		# tree, and any scene that builds its picture over time freezes into its panel as
		# pure black. That is what the first run of this produced, and it looked exactly
		# like a broken freeze.
		# LONG ENOUGH FOR THE MOVE TO LAND. A slow pan eases over several seconds, so a
		# 26-frame settle photographed the camera halfway there and every frame came out
		# looser than the shot actually is - which reads exactly like a framing bug and is
		# not one. Three seconds covers the slowest pan in the vocabulary.
		# CAMERA TRAVEL, measured over the LAST SECOND of the settle. A move has a duration
		# and the Director's hold is longer than most of them, so the interesting question is
		# not where the camera goes but whether it is still going anywhere when it arrives.
		var was_aim := Vector2.ZERO
		var travel := 0.0
		for i in _settle:
			# The FOCAL scene only - a cast-owning vehicle drives its other live panels
			# itself in advance(), exactly as the Director drives only its current one.
			if _live != null and is_instance_valid(_live):
				_live.update(Spectrum.current, DT)
				_live.view.commit(DT)
			vehicle.advance(Spectrum.current, DT, 1.0)
			if comic:
				var aim: Vector2 = vehicle._cam["aim"]
				if i >= _settle - 30:
					if i > _settle - 30:
						travel += was_aim.distance_to(aim)
					was_aim = aim
			t += DT
			await get_tree().process_frame
		if _shots.is_empty() or _shots.has(cut):
			await _capture(stage, cut, travel)
	_live = null
	Director.hold(false)
	Director.detach()
	# The stage OWNS the vehicle (Vehicle.mount parents itself), so freeing the stage frees
	# it. Freeing both is a double queue_free of the same subtree.
	stage.queue_free()
	for _i in 3:
		await get_tree().process_frame


## One cut, exactly as the Director performs one.
##
## For a CAST-OWNING vehicle that is a HANDOVER and nothing else (see Director._handover):
## ask it to take over, adopt what it hands back, and build/free nothing - the panels were
## cast when the page turned, and they all stay on the paper.
##
## Getting this wrong is not cosmetic. An earlier version of this function kept the
## full-frame behaviour - mint a scene, parent it, free the outgoing one - and under a
## cast-owning vehicle "the outgoing one" is a panel of the live page, so every cut deleted
## a panel out of the comic and the reading never moved. The probe reported a stuck,
## half-dead page for a mechanism that was working correctly.
func _cut(vehicle: Vehicle) -> void:
	if vehicle.owns_cast():
		var handed := vehicle.take_over(_live)
		if handed != null:
			_live = handed
		return
	var entry: Dictionary = Director.SCENES[randi() % Director.SCENES.size()]
	var sc: GhostScene = (entry["script"] as Resource).new()
	# THE DIRECTOR'S ORDER, exactly: init_with_seed FIRST, add_child second (see
	# Director._make_scene). Reversed, the node enters the tree uninitialised and Node2D
	# draws it once immediately, so a scene that indexes its own built arrays in _draw
	# reads an empty one - a probe artefact that never happens on the real path.
	sc.init_with_seed(randi(), String(entry["behavior"]))
	var prev: GhostScene = _live
	vehicle.host_for(sc).add_child(sc)
	_live = sc
	if prev != null and is_instance_valid(prev):
		prev.queue_free()


## [param travel] is how far the eye moved over the last second of the settle, in world
## units - see the loop in _shoot.
func _capture(stage: SubViewport, cut: int, travel := -1.0) -> void:
	var img := stage.get_texture().get_image()
	var path := "%s_cut%02d.png" % [_out, cut]
	img.save_png(path)
	_wrote += 1
	var spread := _spread(img)
	if spread < 0.02:
		_flat += 1
	# PAGE COVERAGE, reported per frame. "Is there dead space in the picture" was the
	# defect that took three passes to settle, and every one of those passes was judged by
	# looking. A number makes it checkable: below 1.0 the sheet does not reach both frame
	# edges and the surface it lies on is in shot.
	var cov := 1.0
	var still := false
	var fit := -1.0
	if _vehicle_key == "comic":
		cov = float(_vehicle.page_coverage())
		# A hair under 1 is a pixel of rounding at the frame edge, not desk in shot.
		if cov < 0.995:
			_thin += 1
		still = travel >= 0.0 and travel < STILL
		if still:
			_static += 1
		# THE READ PANEL'S SHARE OF THE FRAME, on its worst axis. Over 1 the panel overflows
		# the picture and is being cropped by the screen - which is correct and wanted for an
		# abstract field, and wrong for a piece of footage: "you really don't want to focus the
		# camera on her door to the side". A film panel that is the one being read must come
		# out at or under 1.
		fit = float(_vehicle.read_panel_fit())
		if _vehicle._read == _vehicle._film_at and fit > 1.02:
			_cropped += 1
	if _vehicle_key == "comic":
		print("      panels: %s" % _panel_line(_vehicle))
	print("    %s  (luma %.3f, spread %.3f, page covers %.2f, aim moved %.4f/s, read panel %.2f of frame)%s%s%s" % [
		path, _luma(img), spread, cov, maxf(travel, 0.0), maxf(fit, 0.0),
		"  <-- UNIFORM" if spread < 0.02 else "",
		"  <-- DESK IN SHOT" if cov < 0.995 else "",
		"  <-- CAMERA STOPPED" if still else ""])


func _page_line(vehicle: Vehicle) -> String:
	if not (vehicle is ComicVehicle):
		return "(no spread - %s draws the scene straight onto the stage)" % vehicle.key
	var sp: ComicSpread = vehicle._spread
	if sp == null:
		return "(no spread yet)"
	var aspects := ""
	for i in sp.panels.size():
		aspects += "%.2f%s " % [sp.panel_aspect(i), "|" if sp.side_of(i) == 0 else ""]
	var left: int = sp.pages[0].panels.size()
	var right: int = sp.pages[1].panels.size()
	return "%d+%d panels, aspect %.2f, gutter %.3f, radius %.4f, panel aspects [ %s]" % [
		left, right, sp.aspect, sp.gutter, sp.radius, aspects]


## EVERY PANEL'S STATE, per cut. "4 of the 6 frames are pure black, and empty" is a report
## about panels this probe was drawing but never describing - the picture showed the symptom
## and nothing in the log said whether the panel had a scene, whether its viewport was
## running, or how big its render target was.
##
## `c` = a live GhostScene is parented in it. `L` = its viewport is running this frame.
## Then the render target size. A panel that is black with no `c` was never cast; one that is
## black at 64x64 is a slot that was shrunk and never grown back; one that is black with a `c`
## at full size is a scene drawing black, which is the scene's business and not the page's.
func _panel_line(vehicle: ComicVehicle) -> String:
	var out := ""
	for i in vehicle._spread.panels.size():
		var vp: SubViewport = vehicle._slots[vehicle._pool * ComicVehicle.POOL + i]
		var cast_ok: bool = i < vehicle._cast.size() and vehicle._cast[i] != null \
			and is_instance_valid(vehicle._cast[i])
		var warm: int = int(vehicle._warm[i]) if i < vehicle._warm.size() else 0
		# `w` = the panel has rendered enough frames to have a picture worth freezing. A
		# COLD panel that is frozen is the "pure black, and empty" defect.
		out += "%d[%s%s%s %d] " % [i + 1, "c" if cast_ok else "-",
			"L" if vehicle._live.has(i) else " ",
			"w" if warm >= ComicVehicle.WARM_FRAMES else "-", warm]
	return out


func _luma(img: Image) -> float:
	var acc := 0.0
	var n := 0
	for y in range(0, img.get_height(), 7):
		for x in range(0, img.get_width(), 7):
			var c := img.get_pixel(x, y)
			acc += c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722
			n += 1
	return acc / float(maxi(1, n))


## Standard deviation of luma. A page that drew has paper, ink and pictures on it; a page
## that did not is one flat colour, and that is what this separates.
func _spread(img: Image) -> float:
	var mean := _luma(img)
	var acc := 0.0
	var n := 0
	for y in range(0, img.get_height(), 7):
		for x in range(0, img.get_width(), 7):
			var c := img.get_pixel(x, y)
			var l := c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722
			acc += (l - mean) * (l - mean)
			n += 1
	return sqrt(acc / float(maxi(1, n)))
