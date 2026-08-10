extends Node

## Smoke gate for the WHOLE scene catalogue - build, update, draw, free, every entry.
##
## Nothing in tests/ referenced [constant Director.SCENES] before this, which meant the
## catalogue's only real check was somebody watching the show for long enough to see the
## broken one come up. With 45 entries and a novelty-weighted scheduler, "long enough"
## can be a very long time, and the failure surfaces as a crash mid-render.
##
## What it exercises, per entry, at several seeds:
##   build_params  - the seeded definition, including every branch a seed can take
##   update        - a fixed number of frames against SYNTHETIC audio features, chosen to
##                   sweep silence, a steady middle, and a loud transient, because scenes
##                   branch on all three (an activation threshold, a beat gate, a lull)
##   _draw         - forced, because most of the interesting arithmetic (projection,
##                   polygon winding, colour construction) only runs there
##   free          - a scene that leaks a thread or a job shows up as a hang, not an error
##
## Run inside a REAL boot, because scenes touch the Spectrum and Director autoloads:
##   tests/run_boot_probe.sh tests/scene_smoke.gd 300
##
## Exit status is non-zero if any entry failed, so this is a gate and not just a report.

const SEEDS := [1, 7, 999983]
const FRAMES := 24
## Frames after which a real redraw is forced. Not every frame - a draw costs a yielded
## engine frame each - but spread across the run so an early, a settled and a
## post-transient state are all rendered.
const DRAW_AT := [4, 13, 23]

var _fails: Array = []
var _checked := 0
var _drawn := 0


func _ready() -> void:
	# The engine must own the draw call (see _run_one), so this is a coroutine.
	_run.call_deferred()


func _run() -> void:
	var entries: Array = Director.SCENES
	print("scene_smoke: %d registry entries x %d seeds x %d frames"
		% [entries.size(), SEEDS.size(), FRAMES])
	var by_script := {}
	for e in entries:
		var path := String((e["script"] as Script).resource_path)
		by_script[path] = true
		for sv in SEEDS:
			await _run_one(path, e["script"], String(e.get("behavior", "drift")), int(sv))
	print("scene_smoke: %d distinct scripts, %d instantiations, %d rendered frames"
		% [by_script.size(), _checked, _drawn])
	if _fails.is_empty():
		print("scene_smoke: ALL OK")
	else:
		print("scene_smoke: %d FAILURE(S)" % _fails.size())
		for f in _fails:
			print("   ", f)
	# SETTLE BEFORE QUITTING. Freeing 162 Node2Ds hands the RenderingServer that many
	# canvas items to retire, and quitting in the same frame as the last free tears the
	# server down while that queue is still draining - which exits on a fatal
	# out-of-bounds inside the engine rather than on our verdict. The verdict is already
	# printed above, so these frames cost nothing but they make the exit STATUS
	# trustworthy, and the status is the whole point of a gate.
	for _i in 6:
		await get_tree().process_frame
	get_tree().quit(1 if not _fails.is_empty() else 0)


func _run_one(path: String, script: Script, behavior: String, sv: int) -> void:
	_checked += 1
	var short := path.get_file().get_basename()
	var sc = script.new()
	if sc == null:
		_fails.append("%s: script.new() returned null" % short)
		return
	# Parented BEFORE init, because a scene may reach for the viewport size in
	# build_params (framing = "field" sizes off it) and an orphan reports zero.
	add_child(sc)
	sc.init_with_seed(sv, behavior)
	sc.size = Vector2(1920, 1080)
	for i in FRAMES:
		sc.update(_features(i), 1.0 / 30.0)
		if DRAW_AT.has(i):
			# THE ENGINE MUST MAKE THIS CALL. Godot rejects draw_* issued from
			# anywhere but a real NOTIFICATION_DRAW ("Drawing is only allowed
			# inside this node's _draw()"), so calling sc._draw() directly proves
			# nothing - every draw command is refused and the geometry after the
			# first one never runs. Marking it dirty and yielding a frame is what
			# actually renders it.
			sc.queue_redraw()
			await get_tree().process_frame
			_drawn += 1
		if sc.lifecycle == "oneshot" and sc.finished():
			break
	# Tear down the way the Director actually does - queue_free, then let the frame
	# end - and WAIT for that frame. Both halves are load-bearing and each was
	# learned the hard way here:
	#   without the yield, nothing is ever collected and all 162 instantiations
	#   stay alive at once, which hangs the run;
	#   without the deferral, an immediate free() can pull a scene out from under a
	#   FrameForge job still building on a worker thread, which segfaults at exit.
	sc.queue_free()
	await get_tree().process_frame


## A synthetic frame that sweeps the three regimes scenes actually branch on. Frame 0-5
## is silence (an intro, a lull - where a scene must not divide by a zero energy), 6-17
## is a steady middle, and 18+ carries a hard beat and high flux (a drop - where the
## activation gates and cut triggers fire).
func _features(i: int) -> AudioFeatures:
	var f := AudioFeatures.new()
	f.time = float(i) / 30.0
	var loud := 0.0
	if i >= 18:
		loud = 0.85
	elif i >= 6:
		loud = 0.35
	var bands := PackedFloat32Array()
	bands.resize(Spectrum.BAND_COUNT)
	for b in Spectrum.BAND_COUNT:
		# a tilted spectrum with a moving peak, so per-band code sees structure
		# rather than one flat value repeated
		var t := float(b) / float(maxi(1, Spectrum.BAND_COUNT - 1))
		bands[b] = clampf(loud * (1.0 - 0.6 * t) * (0.6 + 0.4 * sin(t * 9.0 + f.time * 3.0)), 0.0, 1.0)
	f.bands = bands
	f.energy = loud
	f.bass = loud * 0.9
	f.low_mid = loud * 0.7
	f.mid = loud * 0.6
	f.high = loud * 0.4
	f.treble = loud * 0.3
	f.beat = 1.0 if i == 18 or i == 22 else 0.0
	f.flux = 0.4 if i == 18 else 0.02
	f.movement = 0.8 if i == 18 else 0.1
	f.beat_period = 0.5
	return f
