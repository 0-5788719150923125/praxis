extends SceneTree

## Gate for the WEATHER rework - does the music actually move the snow and rain now.
##
## The complaint this answers was "the rain and snow scenes seem rather static", and the
## measurement behind it was stark: across the seed's presets the flake count varied
## 8.7x, the fall speed 11.3x and the size 5.7x, while across the MUSIC everything varied
## 1.3-1.5x and only brightness and speed. The seed owned the drizzle-to-downpour knob;
## the music owned a brightness knob. Particle count was fixed at build and never changed.
##
## So the assertions here are about RANGE, not about any particular look:
##   DENSITY   - the visible flake count must differ substantially between silence and a
##               loud passage, and must settle rather than strobe.
##   VARIETY   - the crystal bank must contain genuinely different geometries, and the
##               weathers must produce different habits from each other.
##   REACH     - the crystal gate must actually admit crystals. The old one multiplied the
##               configured fraction by 0.064 through two hidden extra conditions, so a
##               preset asking for 20% delivered about one flake on screen.
##   MOTION    - flakes must not fall down fixed columns forever, which made the field
##               exactly periodic (a blizzard repeated every 2.9-16.7 s).
##
## Run: godot --headless --path axis/ghost --script tests/weather_check.gd

var _fails: Array = []


func _init() -> void:
	_check_density()
	_check_bank_variety()
	_check_habits_differ()
	_check_crystal_reach()
	_check_column_break()
	if _fails.is_empty():
		print("weather_check: ALL OK")
		quit()
		return
	print("weather_check: %d FAILURE(S)" % _fails.size())
	for f in _fails:
		print("   ", f)
	quit(1)


func _ok(cond: bool, msg: String) -> void:
	if not cond:
		_fails.append(msg)


func _rng(sv: int) -> RandomNumberGenerator:
	var r := RandomNumberGenerator.new()
	r.seed = sv
	return r


## Silence versus a loud passage, run to steady state, counting how many flakes are
## actually visible. This is the headline number.
func _check_density() -> void:
	for kind in ["snow", "rain"]:
		var quiet := _visible(kind, _quiet(), 8.0)
		var loud := _visible(kind, _loud(), 8.0)
		var ratio := float(loud) / maxf(1.0, float(quiet))
		print("weather_check: %-4s visible  quiet=%3d  loud=%3d  (%.2fx)" % [kind, quiet, loud, ratio])
		_ok(ratio >= 1.8,
			"%s: the music must move the density by a lot more than the old 1.3x - got %.2fx"
			% [kind, ratio])
		_ok(quiet > 0, "%s: a quiet passage must still show SOME precipitation, not none" % kind)


## How many particles the layer would actually draw, after `secs` of settling.
func _visible(kind: String, f: AudioFeatures, secs: float) -> int:
	var l := Layer.make(kind, _rng(4242), {"count": 200, "crystal_frac": 0.4})
	var dt := 1.0 / 30.0
	var n := int(secs / dt)
	for i in n:
		l.update(f, dt, Vector2(0.9, 0.55))
	var shown := 0
	var arr: Array = l._flakes if kind == "snow" else l._drops
	for p in arr:
		if smoothstep(0.0, 0.14, l._density - float(p["rank"])) >= 0.02:
			shown += 1
	return shown


## The bank must hold genuinely distinct crystals, not one shape repeated. Compared on
## segment count and total drawn length, which differ for any real change of morphology.
func _check_bank_variety() -> void:
	var bank := Crystal.bank(_rng(9), 20,
		{"stellar": 0.4, "fern": 0.2, "plate": 0.2, "needle": 0.1, "column": 0.1}, 0.7, 0.2)
	var sigs := {}
	for sh in bank:
		var total := 0.0
		for i in sh.size():
			total += sh.a[i].distance_to(sh.b[i])
		sigs["%d:%.2f" % [sh.size(), total]] = true
	print("weather_check: bank of %d crystals -> %d distinct geometries" % [bank.size(), sigs.size()])
	_ok(sigs.size() >= 16,
		"bank: at least 16 of 20 crystals should be measurably different, got %d" % sigs.size())
	# and the old single-scalar generator could not produce a flake with NO arms at all
	var habits := {}
	for sh in bank:
		habits[sh.habit] = true
	_ok(habits.size() >= 3, "bank: expected several habits in the mix, got %s" % str(habits.keys()))


## Two different weathers must not produce the same crystals. Habit follows temperature
## and complexity follows supersaturation, so a powder and a hoar frost should share
## almost nothing.
func _check_habits_differ() -> void:
	var powder := Crystal.bank(_rng(3), 24, {"column": 0.45, "needle": 0.30, "plate": 0.25}, 0.1, 0.05)
	var hoar := Crystal.bank(_rng(3), 24, {"fern": 0.40, "stellar": 0.35, "sectored": 0.25}, 0.9, 0.0)
	var pset := {}
	for sh in powder:
		pset[sh.habit] = true
	var overlap := 0
	for sh in hoar:
		if pset.has(sh.habit):
			overlap += 1
	_ok(overlap == 0,
		"habits: powder and hoar must share no habit at all - %d hoar crystals used a powder habit"
		% overlap)
	# and hoar's dendrites must genuinely be more complex than powder's columns
	var pseg := 0
	var hseg := 0
	for sh in powder:
		pseg += sh.size()
	for sh in hoar:
		hseg += sh.size()
	print("weather_check: mean segments  powder=%.1f  hoar=%.1f" % [pseg / 24.0, hseg / 24.0])
	_ok(hseg > pseg * 1.5,
		"habits: a high-supersaturation frost should be far more intricate than dry powder "
		+ "(%d vs %d segments)" % [hseg, pseg])


## The gate must admit roughly the fraction it is given. The old one silently multiplied
## it by 0.064.
func _check_crystal_reach() -> void:
	var l := Layer.make("snow", _rng(11), {"count": 600, "crystal_frac": 0.5})
	var n := 0
	for fl in l._flakes:
		if bool(fl["is_crystal"]):
			n += 1
	var frac := float(n) / 600.0
	print("weather_check: crystal_frac 0.50 -> %.3f actually flagged" % frac)
	_ok(frac > 0.42 and frac < 0.58,
		"crystal gate: asked for 0.50, got %.3f - the hidden depth/size conditions are back" % frac)


## A flake must not fall down the same column forever. Run past a full traversal and
## check that the horizontal positions have genuinely changed.
func _check_column_break() -> void:
	var l := Layer.make("snow", _rng(5), {"count": 120, "fall": 0.5})
	var before: Array = []
	for fl in l._flakes:
		before.append(float(fl["x"]))
	var dt := 1.0 / 30.0
	for i in int(12.0 / dt):
		l.update(_loud(), dt, Vector2(0.9, 0.55))
	var moved := 0
	for i in l._flakes.size():
		if absf(float(l._flakes[i]["x"]) - float(before[i])) > 0.001:
			moved += 1
	print("weather_check: %d/%d flakes re-sited after 12s" % [moved, l._flakes.size()])
	_ok(moved > l._flakes.size() / 2,
		"columns: most flakes should have been re-sited at a wrap by now, only %d were" % moved)


func _quiet() -> AudioFeatures:
	return AudioFeatures.new()


func _loud() -> AudioFeatures:
	var f := AudioFeatures.new()
	var b := PackedFloat32Array()
	b.resize(64)
	for i in 64:
		b[i] = 0.55
	f.bands = b
	f.energy = 0.45          # a realistic loud passage: energy is a MEAN over 64 bands
	f.bass = 0.7
	f.low_mid = 0.5
	f.mid = 0.45
	f.high = 0.3
	f.treble = 0.25
	f.beat = 0.5
	f.flux = 0.04
	return f
