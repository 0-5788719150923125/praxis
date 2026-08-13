extends SceneTree

## sand_settle_check - that a SETTLED [Grains] chamber actually stops.
##
## The bug this gates: the fluid sideways search tried its two directions in the order
## `rb, -rb`, and `rb` flips with the generation - so a fluid cell with passable space on
## both sides took the first option every single tick and walked right, left, right, left
## for ever. Each of those swaps empties the cell it leaves, so a settled pool read as
## grains blinking between their colour and the background: "almost every single grain is
## flickering between black and color, as if the pixels were vibrating".
##
## MEASURED, not eyeballed. Before the fix, with nothing pouring in: a water-only pool had
## 40 cells changing occupancy 40 times over 40 ticks, every one a clean two-tick
## oscillation; salt+water 25 such cells, sand+water 20, sand+oil 30. Powder-only chambers
## had exactly zero, which is what named the branch - powders have no `spread` and never
## enter it. After: 0, 0, 0 and 1 respectively.
##
## The test therefore asserts the ping-pong count, NOT the total change count. A powder
## pile keeps creeping slowly for a long time (about 140 cells still moving in the same
## window, before and after), and that is settling, not flicker - the thing that has to be
## zero is the population of cells that reverse themselves every tick.
##
## Run: godot --headless --path axis/ghost --script tests/sand_settle_check.gd

## A cell that reverses this many times inside [constant WINDOW] ticks is oscillating, not
## settling: the window is 60 ticks, so 20 is a reversal every third tick at worst.
const PINGPONG := 20
const WINDOW := 60

## What a chamber may have. Two is the honest ceiling: one lone cell in an awkward pocket is
## not what the report was about, and holding the number at zero would gate on noise.
const ALLOWED := 2

## The matter subsets to try. Every one of these is reachable from falling_sand's own
## POURABLE bag, and the liquid ones are the whole point - they are what oscillated.
const SETS := [["sand"], ["sand", "water"], ["sand", "dust"], ["water"], ["sand", "oil"],
	["salt", "water"], ["water", "oil"], ["dust", "water"]]

## The chamber's weather, run over every subset. Both are live in falling_sand - a draft in 65%
## of sessions (falling_sand.gd's `_wind`) and a gravity shear on every section change - and both
## keep powder moving after it would otherwise have stopped, so a settle test that leaves them at
## zero is not testing the world the scene actually runs.
const WEATHER := [{"wind": 0.0, "tilt": 0.0}, {"wind": 0.30, "tilt": 0.0},
	{"wind": 0.0, "tilt": 0.36}]

var _fails: Array = []


func _init() -> void:
	for w in WEATHER.size():
		for trial in SETS.size():
			_run(trial, WEATHER[w])
	if _fails.is_empty():
		print("sand_settle_check: ALL OK")
		quit()
		return
	print("sand_settle_check: %d FAILURE(S)" % _fails.size())
	for f in _fails:
		print("   ", f)
	quit(1)


func _run(trial: int, weather: Dictionary) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = 4000 + trial
	var w := 120
	var h := 70
	var g := Grains.new(w, h, rng.randi())
	var stone := g.add("stone", rng)
	g.wall_id = stone
	g.floor_h = 3
	var mats: Array = SETS[trial]
	var ids: Array = []
	for m in mats:
		ids.append(g.add(String(m), rng))
	g.finalize()
	g.fill_rect(0, 0, w - 1, 2, stone)
	g.fill_rect(0, h - 3, w - 1, h - 1, stone)
	g.fill_rect(0, 0, 2, h - 1, stone)
	g.fill_rect(w - 3, 0, w - 1, h - 1, stone)
	g.fill_rect(56, 0, 62, 2, 0)            # a pipe through the ceiling for the pour
	g.wind = float(weather["wind"])
	g.wind_dir = 1
	g.tilt = float(weather["tilt"])
	g.ignite = 0.0

	# Fill the chamber, then STOP pouring and let it settle. The measurement window has to be
	# a world nothing is being added to, or the pour's own churn hides what is being measured.
	for _t in 1200:
		for k in ids.size():
			g.emit(56, 62, 0, int(ids[k]), 5, k)
		g.step()
	for _t in 400:
		g.step()

	var n := w * h
	var frames: Array = []
	for _t in WINDOW:
		g.step()
		var occ := PackedByteArray()
		occ.resize(n)
		for y in h:
			var row := y * w
			for x in w:
				occ[row + x] = 1 if g.at(x, y) != 0 else 0
		frames.append(occ)

	var changed := 0
	var flippy := 0
	for i in n:
		var c := 0
		for t in range(1, frames.size()):
			if (frames[t] as PackedByteArray)[i] != (frames[t - 1] as PackedByteArray)[i]:
				c += 1
		if c > 0:
			changed += 1
		if c >= PINGPONG:
			flippy += 1
	var tag := "%s wind=%.2f tilt=%.2f" % [str(mats), float(weather["wind"]), float(weather["tilt"])]
	print("sand_settle_check: %-46s changed=%4d  oscillating=%3d" % [tag, changed, flippy])
	if flippy > ALLOWED:
		_fails.append("%s: %d cells reverse themselves at least %d times in %d ticks with "
			% [tag, flippy, PINGPONG, WINDOW]
			+ "nothing pouring in - a settled chamber must stop, not vibrate")
