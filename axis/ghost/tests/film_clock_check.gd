extends Node

## Gate for FOOTAGE IN A COMIC PANEL - specifically, for the one thing that was asked for
## and that could quietly not happen: a clip must NOT start from the beginning every time
## it is sampled.
##
## The requirement is that a film reads as something that has been running all along,
## which the comic occasionally cuts into. [Films.position_at] answers "where would this
## clip be if it had been looping since the show started" and [FilmScene] seeks there -
## but a seek that silently does nothing looks exactly like a clip that plays from zero,
## and at a glance so does a clip that is simply short. So this measures the picture.
##
## HOW IT MEASURES. The fixture is a clip whose COLOUR ENCODES ITS OWN TIMESTAMP - red
## ramps 0..1 across its length while blue ramps the other way - built here with ffmpeg
## and cached. Reading the mean red off the rendered panel therefore reads the clip's
## position back out of the pixels, which is the only way to be sure the seek reached the
## decoder AND that what was drawn is the frame it landed on. A gate that asked
## `stream_position` alone would pass on a player that seeks correctly and draws nothing.
##
## THE FIXTURE IS LONGER THAN A WINDOW, deliberately. A clip is never transcoded whole -
## it is cut [constant Films.WINDOW] seconds at a time, on demand - so a position past the
## first window is playing a DIFFERENT FILE, seeked to a local offset, and the arithmetic
## that maps film time onto window time is the part with somewhere to be wrong. A fixture
## that fitted in one window would exercise none of it.
##
## THE CLAIMS:
##   the seek lands where the show clock says (including past the loop, and below zero -
##   a bookend hold runs the clock negative at the head of a render and fmod would hand
##   back a seek VideoStreamPlayer refuses);
##   what is DRAWN is the frame at that position, not a stale first frame, and it is right
##   in the SECOND and THIRD windows too, where a window-local offset error would show;
##   a panel that runs over a window boundary carries on across it;
##   and after a panel has been frozen - its clock stopped while the show's ran on - it
##   catches back up rather than playing on at a permanent lag.
##
## Needs a real renderer (it reads pixels back) and a real boot (Films reads Settings):
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/film_clock_check.gd 180

const W := 256
const H := 192
## The fixture's length: long enough to span three windows, so the boundary arithmetic is
## exercised rather than assumed. Flat colour, so it encodes in a moment despite the length.
const DUR := 100.0
const FIXTURE := "user://films_test/ramp.mp4"
## Frames to let the decoder settle before reading pixels. The seek is deferred a frame
## (VideoStreamPlayer refuses one on the frame it starts playing), and libtheora needs a
## few more to hand back the frame it landed on.
const SETTLE_FRAMES := 24

var _clip: Dictionary = {}
var _fails: Array = []


func _ready() -> void:
	if not Deps.has("ffmpeg"):
		print("film_clock_check: SKIPPED - ffmpeg is not installed, so the fixture "
			+ "cannot be built")
		get_tree().quit()
		return
	if not _ensure_fixture():
		printerr("film_clock_check: could not build the fixture")
		get_tree().quit(1)
		return
	_clip = {"source": ProjectSettings.globalize_path(FIXTURE), "slug": "ramp_test",
		"name": "ramp", "duration": DUR}
	Films.use_for_test([_clip], 1.0)

	_check_pure()
	_check_windows()
	_check_fit()
	if not await _cut_windows():
		printerr("film_clock_check: the fixture's windows could not be cut")
		get_tree().quit(1)
		return
	_check_rate()
	await _check_keeps_up()
	# 0 is the control: the one show time whose correct answer IS the start, so a seek that
	# never fires cannot pass the whole gate by accident. 50 and 95 are in the second and
	# third windows - a different file each, seeked to a local offset.
	for t in [0.0, 13.0, 50.0, 95.0, 124.0, -3.0]:
		await _check_at(float(t))
	await _check_boundary()
	await _check_catch_up()

	if _fails.is_empty():
		print("film_clock_check: ALL OK")
	else:
		for f in _fails:
			printerr("film_clock_check: " + String(f))
		printerr("film_clock_check: %d FAILURE(S)" % _fails.size())
	get_tree().quit(0 if _fails.is_empty() else 1)


## The arithmetic on its own, where a wrong answer is unambiguous. Everything below is
## about whether the player obeys it.
func _check_pure() -> void:
	var cases := [[0.0, 0.0], [5.0, 5.0], [99.9, 99.9], [100.0, 0.0], [124.0, 24.0],
		[-3.0, 97.0], [-100.0, 0.0], [243.5, 43.5]]
	for c in cases:
		var got := Films.position_at(_clip, float(c[0]))
		if absf(got - float(c[1])) > 0.001:
			_fails.append("position_at(%.1f) = %.3f, expected %.3f" % [c[0], got, c[1]])
	# A clip too short to be a clip must not be seeked into at all - a fractional seek on
	# a one-frame file is how a decoder gets asked for a frame that does not exist.
	if Films.position_at({"duration": 0.2}, 9.0) != 0.0:
		_fails.append("a sub-minimum clip was given a nonzero position")


## THE WINDOW ARITHMETIC on its own. `window_local` is what a player is actually seeked
## to, and getting it wrong by a window is a clip that plays the wrong minute of itself
## while every other check still passes.
func _check_windows() -> void:
	var w := Films.WINDOW
	var cases := [[0.0, 0, 0.0], [w - 0.5, 0, w - 0.5], [w, 1, 0.0], [w + 7.0, 1, 7.0],
		[2.0 * w + 1.0, 2, 1.0]]
	for c in cases:
		var pos := float(c[0])
		var i := Films.window_index(pos)
		var local := Films.window_local(pos)
		if i != int(c[1]) or absf(local - float(c[2])) > 0.001:
			_fails.append("%.2fs is window %d+%.2f, expected %d+%.2f"
				% [pos, i, local, int(c[1]), float(c[2])])
	# A window past the end of the film must not exist, or a request for it starts an
	# ffmpeg run that produces an empty file and a panel that waits on it forever.
	var last := Films.window_index(DUR - 0.1)
	if not Films.window_exists(_clip, last):
		_fails.append("the last window (%d) of a %.0fs clip does not exist" % [last, DUR])
	if Films.window_exists(_clip, last + 1):
		_fails.append("window %d exists past the end of a %.0fs clip" % [last + 1, DUR])
	if Films.next_window(_clip, last) != 0:
		_fails.append("the window after the last is %d, not 0 - the clip does not loop"
			% Films.next_window(_clip, last))


## THE FIT: footage covers its panel, and crops on ONE axis only.
##
## Reported as "videos that are cropped on ALL edges except at the bottom". Two causes, both
## arithmetic. [FilmScene] sized itself with `GhostScene.view_half_px`, which is a deliberate
## OVERDRAW bound - a 1.06 margin, measured about the origin so any pan inflates it - and it
## then drew through the view transform, which the Director's shot bias had panned. Measured
## with a fixture painted a different colour on each edge: at a 2.4-aspect panel ALL FOUR
## edges were gone.
##
## This is the arithmetic half, kept honest here because it is cheap and exact; the pixels
## themselves were checked with that fixture (scratchpad/fit_probe.gd, kept out of the gate
## because it needs a clip built for it). Cover means: scale by the LARGER ratio so both
## axes are covered, and then exactly one axis has anything to spare.
func _check_fit() -> void:
	var src := Vector2(640.0, 360.0)
	for panel: Vector2 in [Vector2(162, 360), Vector2(360, 360), Vector2(864, 360),
			Vector2(640, 360)]:
		var k := maxf(panel.x / src.x, panel.y / src.y)
		var dst := src * k
		var over_x := dst.x - panel.x
		var over_y := dst.y - panel.y
		# Both must cover - a negative overflow is a bar down that side of the panel.
		if over_x < -0.01 or over_y < -0.01:
			_fails.append("panel %v: the footage does not cover it (%.1f x %.1f)"
				% [panel, dst.x, dst.y])
		# ...and only ONE axis may have anything spare, or the scale is bigger than cover
		# needs and both pairs of edges are being trimmed for nothing.
		if over_x > 0.5 and over_y > 0.5:
			_fails.append("panel %v: cropped on BOTH axes (%.1f, %.1f spare) - the scale "
				% [panel, over_x, over_y] + "is larger than covering requires")
	print("  fit: cover crops one axis, never both")


## Cut every window of the fixture up front, so the checks below measure PLAYBACK rather
## than how fast this machine encodes. In the app they are cut as the show asks for them.
func _cut_windows() -> bool:
	var n := Films.window_index(DUR - 0.1) + 1
	var waited := 0.0
	while waited < 120.0:
		Films.pump()
		# ASK EVERY FRAME, not once. Films.PIPELINE caps how many cuts may run at a time and
		# a request over that cap is refused - so asking once for more windows than the cap
		# allows leaves the rest never started, which is a wait that never ends. The app has
		# the same shape and is fine because `warm` runs again on every page.
		for i in n:
			Films.request_window(_clip, i)
		var ready := 0
		for i in n:
			if Films.window_ready(_clip, i):
				ready += 1
		if ready == n:
			print("  cut %d window(s) of %.0fs in %.1fs" % [n, Films.WINDOW, waited])
			return true
		await get_tree().process_frame
		waited += 1.0 / 60.0
	return false


## FILM KEEPS APPEARING, PAGE AFTER PAGE. Reported at the top of the dial: "the video is
## only ever shown ONE time, on ONE page."
##
## Two separate causes, and this covers the one that lives in [Films]. The window the show
## wants advances at exactly 1x realtime, because the position IS the clock, while cutting
## one takes tens of seconds - so asking only for the window needed RIGHT NOW is a race
## that is lost whenever a page turn and a window boundary land near each other, and lost
## again on the next page, forever. [method Films.warm] therefore keeps one window ahead,
## and this asserts that directly: once a clip is playable at some moment, the window after
## that moment is already cut or already cutting.
##
## (The other cause was that nothing PUMPED - a finished cut was never promoted out of its
## `.part` name because the only two pollers were a live film panel and an open Generative
## panel, which are exactly the two things usually absent. That one is wired in `main.gd`
## and checked by docs.py, because a probe replaces the main scene and cannot see it.)
func _check_keeps_up() -> void:
	var clock := 0.0
	var served := 0
	var pages := 30
	for p in pages:
		# pages turn every 40 seconds or so, and the clock never goes backwards
		clock += 40.0
		var ok := Films.warm(_clip, clock)
		if ok:
			served += 1
			var i := Films.window_index(Films.position_at(_clip, clock))
			var nxt := Films.next_window(_clip, i)
			if nxt != i and not (Films.window_ready(_clip, nxt) or Films.busy(_clip)):
				_fails.append("page %d: window %d is playable but %d is neither cut nor "
					% [p, i, nxt] + "cutting - the cache is not staying ahead")
		for _f in 30:
			Films.pump()
			await get_tree().process_frame
	print("  %d/%d pages could be served film across %.0fs of clock"
		% [served, pages, clock])
	# The fixture's windows are all cut by now, so anything less than every page means the
	# asking itself is broken rather than the machine being slow.
	if served < pages - 1:
		_fails.append("only %d of %d pages could be served film" % [served, pages])


## A PANEL THAT OUTLIVES ITS WINDOW carries on into the next one. This is the case the
## second player exists for: a window is a file, and reaching its end without a handover
## would freeze the panel on the last frame it decoded.
func _check_boundary() -> void:
	# Open just before the first boundary and run the clock past it.
	var rig := _open(Films.WINDOW - 4.0)
	var sc: GhostScene = rig["scene"]
	for i in SETTLE_FRAMES:
		sc.update(null, 0.0)
		await get_tree().process_frame
	var before := await _read_position(rig)
	# Cross it, in the small steps a live panel actually gets - and well past it, since a
	# handover is only proved by the window that comes after.
	for i in 360:
		sc.update(null, 1.0 / 30.0)
		await get_tree().process_frame
	await RenderingServer.frame_post_draw
	var after := await _read_position(rig)
	var want := Films.position_at(_clip, (sc as FilmScene)._show_t)
	print("  boundary: %5.2fs -> %5.2fs (window %d, show says %5.2fs)"
		% [before, after, (sc as FilmScene)._win, want])
	if (sc as FilmScene)._win != 1:
		_fails.append("after crossing the boundary the panel is still on window %d"
			% (sc as FilmScene)._win)
	if absf(after - want) > 2.0:
		_fails.append("across the boundary the panel draws %.2fs, the show says %.2fs"
			% [after, want])
	_close(rig)


## HOW OFTEN FOOTAGE ACTUALLY APPEARS, driven through the real chooser over many pages.
##
## The worry this answers is a reasonable one to have: with seventy-odd scene types in the
## catalogue, a film sounds like it would be picked one time in seventy and never seen. It
## is not in that draw at all - [method ComicVehicle._choose_film] runs when the page turns
## and decides before the Director is asked for anything - but "not in the lottery" is a
## claim about control flow, and the thing worth pinning is the RATE that comes out of it.
##
## The endpoints are what is asserted, because they are the ones with a right answer: 0
## must mean never (a dial that cannot turn the feature off is a bug) and 1 must mean every
## page (a ceiling below that would make the dial's top end a lie). The middle is printed
## rather than bounded - it is a sampled probability, and a gate that fails on an unlucky
## seed is a gate people learn to ignore.
##
## THE ONE-AT-A-TIME INVARIANT rides along here, since this is the only place that walks
## hundreds of spreads: `_film_at` is a single index, so a spread cannot carry two - and this
## says so out of the running code rather than out of the type declaration.
func _check_rate() -> void:
	var stage := SubViewport.new()
	stage.size = Vector2i(320, 180)
	add_child(stage)
	Director.detach()
	var v: Vehicle = Vehicle.make("comic")
	v.mount(stage)
	Director.attach(stage, v)
	Director.hold(true)          # the probe turns the pages; nothing else may
	var comic: ComicVehicle = v
	var spreads := 400
	for freq in [0.0, 0.5, 1.0]:
		Films.use_for_test([_clip], float(freq))
		var with := 0
		var panels := 0
		for p in spreads:
			comic._spread_i = p
			comic._spread = ComicSpread.new(
				hash([Director.session_seed(), "comic-spread", p]))
			comic._choose_film()
			panels += comic._spread.panels.size()
			if comic._film_at >= 0:
				with += 1
				if comic._film_at >= comic._spread.panels.size():
					_fails.append("freq %.2f spread %d: film panel %d is off the spread"
						% [freq, p, comic._film_at])
		print("  freq %.2f -> %d/%d spreads carry film (%.0f%%), %d/%d panels (%.1f%%)"
			% [freq, with, spreads, 100.0 * with / spreads, with, panels,
				100.0 * float(with) / float(panels)])
		if freq <= 0.0 and with != 0:
			_fails.append("frequency 0 still put film on %d spreads - the dial cannot "
				% with + "turn the feature off")
		if freq >= 1.0 and with != spreads:
			_fails.append("frequency 1 put film on only %d of %d spreads" % [with, spreads])
	# ...and with nothing imported, nothing changes at all.
	Films.use_for_test([], 1.0)
	comic._spread_i = 7
	comic._spread = ComicSpread.new(hash([Director.session_seed(), "comic-spread", 7]))
	comic._choose_film()
	if comic._film_at >= 0:
		_fails.append("an empty library still produced a film panel")
	Films.use_for_test([_clip], 1.0)
	# HELD AND DETACHED. Releasing the hold here let the Director run a real show behind the
	# rest of the gate - every later frame then carried a live scene, and the checks that
	# only wait for frames took minutes instead of seconds.
	Director.detach()
	stage.queue_free()


## Open a panel at `show_t` and read the clip's position back out of the drawn pixels.
func _check_at(show_t: float) -> void:
	var rig := _open(show_t)
	for i in SETTLE_FRAMES:
		(rig["scene"] as GhostScene).update(null, 0.0)
		await get_tree().process_frame
	await RenderingServer.frame_post_draw
	var drawn := await _read_position(rig)
	# THE DECODER'S POSITION IS WINDOW-LOCAL - it is playing a file that starts at zero
	# however far into the film that file was cut from - so it has to be lifted back into
	# film time before it means anything. Getting this wrong is not a hypothetical: the
	# first version of this check compared the two spaces directly and called a working
	# player broken at every position past the first window.
	var decoder := _film_time_of(rig)
	var want := Films.position_at(_clip, show_t)
	print("  show_t %6.1f  want %5.2fs   decoder %5.2fs (window %d)   drawn %5.2fs"
		% [show_t, want, decoder, int((rig["scene"] as FilmScene)._win), drawn])
	# The decoder keeps playing while the probe waits, so it is AHEAD of `want` by the
	# settle - never behind it, and never back at the start.
	var lead := decoder - want
	if lead < -0.4 or lead > 2.0:
		_fails.append("show_t %.1f: the decoder is at %.2fs, %.2fs from the %.2fs asked for"
			% [show_t, decoder, lead, want])
	# WHAT IS ON SCREEN is the frame the decoder is on. This is the claim `stream_position`
	# cannot make by itself. The tolerance carries the fixture's own quantization: a
	# hundred seconds of ramp across 8 bits of red is 0.4s per step before theora rounds it.
	if absf(drawn - decoder) > 1.8:
		_fails.append("show_t %.1f: the panel draws %.2fs while the decoder is at %.2fs"
			% [show_t, drawn, decoder])
	# ...AND THE POINT OF THE WHOLE FEATURE: a clip sampled at 13s starts at 13s.
	if want > 2.0 and drawn < 1.0:
		_fails.append("show_t %.1f: the clip started from the beginning (%.2fs) instead of %.2fs"
			% [show_t, drawn, want])
	_close(rig)


## A FROZEN PANEL CATCHES BACK UP. The comic stops a panel's viewport when the camera
## looks away, which stops the decoder while the show's clock runs on; on the way back the
## clip would otherwise play on at a permanent lag, drifting further at every freeze.
## [FilmScene] watches for that gap and corrects it.
func _check_catch_up() -> void:
	var rig := _open(2.0)
	for i in SETTLE_FRAMES:
		(rig["scene"] as GhostScene).update(null, 0.0)
		await get_tree().process_frame
	# The show's clock jumps 9 seconds while the decoder's stands still - exactly what a
	# freeze does, handed to the scene as one big delta.
	(rig["scene"] as GhostScene).update(null, 9.0)
	for i in SETTLE_FRAMES:
		(rig["scene"] as GhostScene).update(null, 0.0)
		await get_tree().process_frame
	await RenderingServer.frame_post_draw
	var drawn := await _read_position(rig)
	print("  after a 9.0s freeze  ->  drawn %5.2fs (expected near 11)" % drawn)
	if absf(drawn - 11.0) > 2.0:
		_fails.append("after a 9s freeze the clip is at %.2fs, not near 11s - it did not "
			% drawn + "catch up")
	_close(rig)


func _open(show_t: float) -> Dictionary:
	var sv := SubViewport.new()
	sv.size = Vector2i(W, H)
	sv.disable_3d = true
	sv.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(sv)
	var sc: GhostScene = FilmScene.new()
	sc.set_clip(_clip, show_t)
	sc.init_with_seed(1, "static")
	sv.add_child(sc)
	return {"vp": sv, "scene": sc, "player": _player_of(sc)}


## Where the panel's decoder is IN THE FILM: its window's start plus how far into that
## window it has played.
func _film_time_of(rig: Dictionary) -> float:
	var sc: FilmScene = rig["scene"]
	var player := _player_of(sc)
	if player == null:
		return -1.0
	return float(maxi(sc._win, 0)) * Films.WINDOW + float(player.stream_position)


## The player actually being drawn. Across a handover there are two, so "the first
## VideoStreamPlayer child" is not the same question.
func _player_of(sc: Node) -> VideoStreamPlayer:
	return (sc as FilmScene)._player


func _close(rig: Dictionary) -> void:
	(rig["vp"] as SubViewport).queue_free()


## The clip's timestamp, read out of the drawn panel. Red ramps 0..1 across the clip, so
## the mean red over the middle of the frame IS the position, in units of its length.
func _read_position(rig: Dictionary) -> float:
	var img := (rig["vp"] as SubViewport).get_texture().get_image()
	if img == null:
		_fails.append("no pixels came back - is this running with GHOST_PROBE_GPU=1?")
		return -1.0
	var r := 0.0
	var n := 0
	for y in range(H / 4, H * 3 / 4, 4):
		for x in range(W / 4, W * 3 / 4, 4):
			r += img.get_pixel(x, y).r
			n += 1
	return (r / maxf(1.0, float(n))) * DUR


## Build the colour-ramp fixture if it is not already there. It is a SOURCE file, the kind
## a viewer would import - windows are cut from it by the code under test, which is the
## point. Cached, since it is the same hundred seconds every run.
func _ensure_fixture() -> bool:
	var abs := ProjectSettings.globalize_path(FIXTURE)
	if FileAccess.file_exists(abs):
		return true
	DirAccess.make_dir_recursive_absolute(abs.get_base_dir())
	var out: Array = []
	# geq writes the colour from T directly, so the file's own timestamps and its picture
	# cannot disagree - which matters, because disagreement is what this gate looks for.
	var rc := Deps.execute("ffmpeg", ["-y", "-loglevel", "error",
		"-f", "lavfi", "-i", "color=c=black:s=192x144:d=%d:r=12" % int(DUR),
		"-vf", "geq=r='clip(255*T/%d,0,255)':g='64':b='clip(255-255*T/%d,0,255)'"
			% [int(DUR), int(DUR)],
		"-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
		"-g", "25", abs], out, true)
	if rc != 0:
		printerr("film_clock_check: ffmpeg failed (%d): %s" % [rc, str(out)])
		return false
	return FileAccess.file_exists(abs)
