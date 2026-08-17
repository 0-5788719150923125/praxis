extends SceneTree

## sand_warm_check - that a falling-sand chamber opens IN PROGRESS, and that the state it
## opens in is one the automaton would have produced.
##   godot --headless --path . --script res://tests/sand_warm_check.gd
##
## THE REPORT. "Every scene starts exactly the same way: with several lanes of sand, falling
## from the very top of the screen." It did - the chamber was built out of stone and nothing
## else, so the first several seconds of every session this scene was ever cut to were the
## same empty room filling up.
##
## WHY IT IS NOT A PRE-ROLL, measured rather than assumed: a tick on a real chamber costs
## 5-15 ms, so the couple of hundred ticks it takes to raise a cone is well over a second of
## CPU - a stall at the cut on the main thread, or a second of empty chamber before the first
## packet on the worker. [method Grains.prefill] deposits the piles instead, in one pass.
##
## WHICH IS ONLY WORTH ANYTHING IF THE DEPOSIT IS PLAUSIBLE. A pile drawn by a rule the sim
## does not share collapses the moment the sim touches it, and the first frames of the scene
## are then visibly it falling over - worse than the blank slate. So the gate measures the
## deposit against the automaton itself: settle it with nothing pouring in and ask how much of
## it STAYED. Beside it runs a SCATTER of the same amount of matter through the same chamber,
## which is what "some material, arranged by nothing" scores. The margin between them is the
## measurement; the absolute number alone would just be a threshold nobody could argue with.
##
## It also checks the two failures on either side of the one reported. An empty chamber is the
## bug. A chamber packed to the ceiling is the overshoot, and it is not hypothetical - the
## first cut of the deposit walk restarted its budget at every shelf, so a ledged chamber came
## out solid (occupancy 0.78 against 0.48 now) and every build cost 42 ms.

## Seeds to sweep. Enough to see every chamber shape and a spread of fills.
const SEEDS := 14
## Ticks of settling, with nothing pouring in. About two seconds of world time - long enough
## for anything unsupported to fall and anything oversteep to avalanche.
const SETTLE := 60

## Matter (excluding the chamber itself) as a fraction of the grid. Below the floor it is the
## blank slate that was reported; above the ceiling the sweep and the batching cost more than
## the worker's budget, and there is nothing left to fall through anyway.
##
## Measured over 60 seeds: 0.059 min, 0.168 median, 0.340 max. The ceiling sits just above
## that spread ON PURPOSE - the shelf-budget bug it is there to catch put a ledged chamber at
## 0.406, so a ceiling with a comfortable margin would not have caught it.
const MIN_MATTER := 0.015
const MAX_MATTER := 0.40
## How much of the deposit must still be exactly where it was put, after settling.
const MIN_KEPT := 0.60
## ...and how far it must beat an arrangement made by nothing at all.
const MIN_MARGIN := 0.12

var _fails: Array = []
var _fills: Array = []
var _live_counts := {}
var _shapes := {}
var _in_air := 0


func _initialize() -> void:
	for i in SEEDS:
		_run(2200 + i * 7)
	print("sand_warm_check: shapes %s" % [_shapes])
	print("sand_warm_check: lanes pouring at entry %s" % [_live_counts])

	# NOT THE SAME EVERY TIME, which is the other half of the report. Both the depth and how
	# many lanes are running have to differ between sessions, or the scene has simply swapped
	# one fixed opening for another.
	if _live_counts.size() < 2:
		_fails.append("every seed opened with the same number of lanes pouring (%s) - the "
			% [_live_counts] + "spouts' duty phases are not spread")
	var lo := 1.0
	var hi := 0.0
	for f in _fills:
		lo = minf(lo, float(f))
		hi = maxf(hi, float(f))
	print("sand_warm_check: matter fraction %.3f..%.3f over %d seeds, %d opened with material "
		% [lo, hi, SEEDS, _in_air] + "still in the air")
	if hi - lo < 0.02:
		_fails.append("every seed opened at the same depth (%.3f..%.3f)" % [lo, hi])
	if _in_air == 0:
		_fails.append("no seed opened with anything falling - the streams are never seeded")

	if _fails.is_empty():
		print("sand_warm_check: ALL OK")
		quit()
		return
	print("sand_warm_check: %d FAILURE(S)" % _fails.size())
	for f in _fails:
		print("   ", f)
	quit(1)


func _run(seed_value: int) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = seed_value
	var sc: Object = load("res://scripts/scenes/falling_sand.gd").new()
	var p: Dictionary = sc.build_params(rng)
	var g = sc._job.g
	var shape := String(p["chamber"])
	_shapes[shape] = int(_shapes.get(shape, 0)) + 1

	var before := _matter(g)
	var frac := float(before.size()) / float(g.w * g.h)
	_fills.append(frac)
	if _air_borne(g) > 0:
		_in_air += 1

	var live := 0
	for sp in sc._spouts:
		var d: Dictionary = sp
		if fposmod(float(sc._job.t_sim) / maxf(0.5, float(d["period"])) + float(d["phase"]),
				1.0) < float(d["duty"]):
			live += 1
	_live_counts[live] = int(_live_counts.get(live, 0)) + 1

	# SETTLE IT WITH NOTHING POURING IN. The pour would refill whatever fell over and hide
	# exactly what is being measured.
	for _t in SETTLE:
		g.step()
	var kept := _kept(g, before)

	# The shadow: the same chamber, the same amount of matter, arranged by nothing.
	var scatter := _scatter_score(seed_value, before.size())

	print("sand_warm_check: seed %-5d %-9s fill=%.2f matter=%.3f live=%d  kept %.2f "
		% [seed_value, shape, float(p["prefill"]), frac, live, kept]
		+ "(scattered: %.2f)" % scatter)
	if frac < MIN_MATTER:
		_fails.append("seed %d opened with %.3f of the grid holding matter - that is the blank "
			% [seed_value, frac] + "slate the scene was reported for")
	if frac > MAX_MATTER:
		_fails.append("seed %d opened %.3f full - past what the worker can sweep, and nothing "
			% [seed_value, frac] + "is left to fall through")
	if kept < MIN_KEPT:
		_fails.append("seed %d: only %.2f of the deposit was still in place after %d ticks - "
			% [seed_value, kept, SETTLE] + "the piles are not shaped like anything the sim "
			+ "would have built")
	if kept - scatter < MIN_MARGIN:
		_fails.append("seed %d: the deposit held %.2f against %.2f for the same matter thrown "
			% [seed_value, kept, scatter] + "in at random - it is no better arranged than noise")
	sc.free()


## Every cell holding something that is not the chamber itself.
func _matter(g) -> PackedInt32Array:
	var out := PackedInt32Array()
	for y in g.h:
		for x in g.w:
			var m: int = g.at(x, y)
			if m != 0 and m != g.wall_id:
				out.append(y * g.w + x)
	return out


## Matter with nothing under it: the streams still falling.
func _air_borne(g) -> int:
	var n := 0
	for y in g.h - 1:
		for x in g.w:
			var m: int = g.at(x, y)
			if m != 0 and m != g.wall_id and g.at(x, y + 1) == 0:
				n += 1
	return n


## What share of the cells that held matter still do.
func _kept(g, before: PackedInt32Array) -> float:
	if before.is_empty():
		return 0.0
	var same := 0
	for i in before:
		var m: int = g.at(i % g.w, i / g.w)
		if m != 0 and m != g.wall_id:
			same += 1
	return float(same) / float(before.size())


## The same chamber and the same amount of matter, dropped in with no arrangement at all.
## Rebuilt from the seed rather than copied, because the scene owns its grid and there is no
## way to fork one.
func _scatter_score(seed_value: int, count: int) -> float:
	var rng := RandomNumberGenerator.new()
	rng.seed = seed_value
	var sc: Object = load("res://scripts/scenes/falling_sand.gd").new()
	sc.build_params(rng)
	var g = sc._job.g
	# Clear the deposit, keep the chamber.
	for y in g.h:
		for x in g.w:
			var m: int = g.at(x, y)
			if m != 0 and m != g.wall_id:
				g.set_cell(x, y, 0)
	var mat := int((sc._spouts[0] as Dictionary)["mat"])
	var placed := PackedInt32Array()
	var tries := 0
	while placed.size() < count and tries < count * 40:
		tries += 1
		var x := rng.randi_range(0, g.w - 1)
		var y := rng.randi_range(0, g.h - 1)
		if g.at(x, y) != 0:
			continue
		g.set_cell(x, y, mat)
		placed.append(y * g.w + x)
	for _t in SETTLE:
		g.step()
	var score := _kept(g, placed)
	sc.free()
	return score
