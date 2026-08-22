extends Node

## Gate for the one thing a fractal zoom must not do: turn round while you are watching it.
##
##   tests/run_boot_probe.sh tests/fractal_travel_check.gd 240
##
## No GPU needed - this is the CPU half of scenes/fractal_zoom.gd, and the claim is about the
## sign of a number rather than about pixels. fractal_depth_check.gd is the one that renders.
##
## WHAT WENT WRONG. The scene had three directions - push, pull, and a `breathe` that turned
## round on its own clock - and the clock that drove `breathe` was armed in `build_params`
## BEFORE the direction was chosen, so it ran for all three. Every push and every pull
## reversed after 11-26 seconds. Reported as "it rolled forward for a bit, before slowing and
## rolling backwards again", and measured over 240 seeds at the settings it was reported on:
## 100% of scenes reversed inside thirty seconds, the first turn at 18 s on average, and the
## net travel across a whole scene was under two e-folds because the picture kept coming back
## to where it had been. `breathe` went with the bug, on the reporter's argument, which is
## also the argument for gating this: "reversing direction just reveals the same patterns
## we've already seen".
##
## THE CONTROL IS THE RETIRED LAW. "Nothing reversed" is exactly what a scene that never moves
## reports, and it is also what a broken probe reports. So the same instances, with the same
## sampled parameters, are ALSO stepped through the law this replaced - the breathe clock and
## the bounce off each bound - and that has to reverse, or this file is asserting nothing.
## Travel is checked at the same time, for the other half of the same trap.

const DT := 1.0 / 30.0
## A scene holds for `Director.max_hold` x `pace_calm_scale` x `pacing`. 28 x 1.2 x 1.8 is
## about 60 s at the pacing this was reported on; 45 covers the ordinary case and 60 the long
## one, and both are checked because the interesting question is what happens as a scene runs.
const HOLDS := [15.0, 30.0, 45.0, 60.0]
## Loudness scales the rate (0.55 at silence, 1.45 at full drive), so the fast end has to be
## checked or the gate only ever sees the slow one.
const ENERGIES := [0.15, 0.55, 0.95]
const SEEDS := 160
## The retired law's turning clock, at the middle of the range it used to sample (11-26 s).
## The control only has to demonstrate that a timer-and-bounce law reverses; it does at any
## value in that range, so the midpoint is the honest one to state.
const OLD_BREATHE := 18.5
## Held ten minutes at full drive - far past any hold - every instance should have come to
## rest against its bound, and none should have stepped past it or backed away from it.
const LONG := 600.0

var _fails: Array = []


func _ready() -> void:
	_run.call_deferred()


func _ok(cond: bool, msg: String) -> void:
	if not cond:
		_fails.append(msg)


## The law this file replaced, run on one instance's own numbers. Returns the number of times
## the direction changed sign inside `hold` seconds.
func _old_rule_turns(zl: float, zl_min: float, zl_max: float, speed: float, dir: float,
		energy: float, hold: float) -> int:
	var dir_now := dir
	var breathe := OLD_BREATHE
	var turns := 0
	var t := 0.0
	while t < hold:
		dir_now = lerpf(dir_now, dir, 1.0 - exp(-1.4 * DT))
		zl += dir_now * speed * (0.55 + 0.9 * energy) * DT
		if zl <= zl_min:
			zl = zl_min
			if dir < 0.0:
				turns += 1
			dir = 1.0
		elif zl >= zl_max:
			zl = zl_max
			if dir > 0.0:
				turns += 1
			dir = -1.0
		else:
			breathe -= DT
			if breathe <= 0.0:
				breathe = OLD_BREATHE
				dir = -dir
				turns += 1
		t += DT
	return turns


func _run() -> void:
	var vp := SubViewport.new()
	vp.size = Vector2i(320, 180)
	vp.disable_3d = true
	add_child(vp)

	var dirs := {}
	for hold in HOLDS:
		for energy in ENERGIES:
			var reversed := 0
			var control := 0
			var travel: Array = []
			for sv in range(1, SEEDS + 1):
				var sc = load("res://scripts/scenes/fractal_zoom.gd").new()
				vp.add_child(sc)
				sc.init_with_seed(sv, "drift")
				var name := String(sc.params.get("direction", "?"))
				dirs[name] = int(dirs.get(name, 0)) + 1
				sc._energy = float(energy)
				var want := signf(sc._dir)
				var z0: float = sc._zl
				var prev: float = sc._zl
				var back := false
				var t := 0.0
				control += _old_rule_turns(sc._zl, sc._zl_min, sc._zl_max, sc._speed,
					sc._dir, float(energy), float(hold))
				while t < float(hold):
					sc._step_zoom(DT)
					t += DT
					if signf(sc._zl - prev) == -want and absf(sc._zl - prev) > 1e-9:
						back = true
					prev = sc._zl
					_ok(sc._zl >= sc._zl_min - 1e-6 and sc._zl <= sc._zl_max + 1e-6,
						"seed %d: the zoom left its own range at %s" % [sv, sc._zl])
				if back:
					reversed += 1
				travel.append(absf(sc._zl - z0))
				vp.remove_child(sc)
				sc.free()
			travel.sort()
			var mid: float = travel[travel.size() / 2]
			print("  hold %2.0fs energy %.2f | %d reverse | control turns %3d | travel p10 %.2f p50 %.2f p90 %.2f e-folds"
				% [hold, energy, reversed, control, travel[travel.size() / 10], mid,
					travel[travel.size() * 9 / 10]])
			_ok(reversed == 0, "hold %.0fs energy %.2f: %d of %d scenes reversed direction"
				% [hold, energy, reversed, SEEDS])
			# THE CONTROL, and the reason a green run here means anything. Only asked of a
			# hold long enough for the retired clock to have fired: at 15 s against an 18.5 s
			# timer the old law would not have turned either, and demanding that it did would
			# be gating the control's own arithmetic rather than the scene. Those rows still
			# carry the shipped claim and the travel claim - they just cannot carry this one.
			_ok(float(hold) <= OLD_BREATHE or control >= SEEDS / 2,
				"hold %.0fs energy %.2f: the RETIRED law only turned %d times in %d scenes, so this check cannot see a reversal"
				% [hold, energy, control, SEEDS])
			# ...and the other half of the trap: a scene that never moves never reverses.
			_ok(travel[travel.size() / 10] > 0.10 * (float(hold) / 30.0),
				"hold %.0fs energy %.2f: the slowest tenth travelled only %.2f e-folds - the zoom is not moving"
				% [hold, energy, travel[travel.size() / 10]])

	print("  directions over %d seeds: %s" % [SEEDS, dirs])
	_ok(not dirs.has("breathe"), "the breathe direction is still being rolled")
	_ok(int(dirs.get("push", 0)) > 0 and int(dirs.get("pull", 0)) > 0,
		"one of the two directions never came up: %s" % dirs)

	# --- held far past any scene: arrive, and stay arrived ---
	var arrived := 0
	var over := 0
	for sv in range(1, SEEDS + 1):
		var sc = load("res://scripts/scenes/fractal_zoom.gd").new()
		vp.add_child(sc)
		sc.init_with_seed(sv, "drift")
		sc._energy = 1.0
		var prev: float = sc._zl
		var want := signf(sc._dir)
		for _i in int(LONG / DT):
			sc._step_zoom(DT)
			if signf(sc._zl - prev) == -want and absf(sc._zl - prev) > 1e-9:
				over += 1
				break
			prev = sc._zl
		var bound: float = sc._zl_max if sc._dir > 0.0 else sc._zl_min
		if absf(sc._zl - bound) < 0.25:
			arrived += 1
		vp.remove_child(sc)
		sc.free()
	print("  %.0fs at full drive: %d of %d reached their bound, %d ever stepped back"
		% [LONG, arrived, SEEDS, over])
	_ok(over == 0, "%d scenes stepped backwards when held to their bound" % over)
	_ok(arrived >= SEEDS * 9 / 10,
		"only %d of %d reached the bound in %.0fs - the ease-to-rest is stalling the zoom short of it"
		% [arrived, SEEDS, LONG])

	vp.queue_free()
	await get_tree().process_frame
	print("")
	if _fails.is_empty():
		print("fractal_travel_check: ALL OK - the zoom goes one way and keeps going.")
		get_tree().quit()
		return
	print("fractal_travel_check: %d FAILURE(S)" % _fails.size())
	for f in _fails:
		print("   ", f)
	get_tree().quit(1)
