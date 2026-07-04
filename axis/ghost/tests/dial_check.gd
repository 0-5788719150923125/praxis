extends SceneTree

## One-shot Dial engine self-check (run headless):
##   godot --path axis/ghost --headless --script res://tests/dial_check.gd
## Exercises the performance-dial model: wedge structure, surge/decay/persist,
## additivity, turn-counter shape changes, boundedness, determinism.

var fails := 0


func check(name: String, ok: bool, detail := "") -> void:
	if ok:
		print("PASS  " + name)
	else:
		fails += 1
		print("FAIL  " + name + ("  -- " + detail if detail != "" else ""))


func _initialize() -> void:
	# Structure: 5 or 6 wedges per revolution, re-rolled per turn.
	var d := Dial.new(12345)
	var w0 := d.wedges_of(0)
	check("wedges are 5 or 6", w0 == 5 or w0 == 6, str(w0))
	var varies := false
	for t in 12:
		if d.wedges_of(t) != w0:
			varies = true
	check("wedge count re-rolls across turns", varies)

	# Untouched dial is perfectly silent.
	var silent := true
	for slot in Dial.SLOTS:
		if absf(Dial.new(3).value(slot)) > 0.0:
			silent = false
	check("untouched dial is silent", silent)

	# Determinism: same seed + same gesture sequence = same modulation, and a
	# different seed diverges.
	var a := Dial.new(777)
	var b := Dial.new(777)
	var c := Dial.new(778)
	for k in 240:
		var step := 0.05 if (k % 60) < 40 else 0.0       # turn, rest, turn, rest
		a.turn(step)
		b.turn(step)
		c.turn(step)
		a.advance(1.0 / 60.0)
		b.advance(1.0 / 60.0)
		c.advance(1.0 / 60.0)
	var same := true
	var diff := 0.0
	for slot in Dial.SLOTS:
		for i in 3:
			if absf(a.value(slot, i) - b.value(slot, i)) > 1e-6:
				same = false
			diff += absf(a.value(slot, i) - c.value(slot, i))
	check("deterministic per seed + gesture", same)
	check("different seed, different signature", diff > 0.05, str(diff))

	# Surge -> decay -> standing pattern -> additive on further turns.
	var e := Dial.new(42)
	for k in 40:
		e.turn(0.12)
		e.advance(0.016)
	var g0 := e.glow()
	check("turning deposits energy", g0 > 0.05, str(g0))
	for k in 900:
		e.advance(0.016)                                  # ~14s untouched
	var g1 := e.glow()
	check("transient decays at rest", g1 < g0 * 0.85, "%f -> %f" % [g0, g1])
	check("a standing pattern persists", g1 > 0.02, str(g1))
	var moving := absf(e.value("scale", 0)) > 0.0 or absf(e.value("hue", 0)) > 0.0 \
		or absf(e.value("drive", 0)) > 0.0 or absf(e.value("tempo", 0)) > 0.0 \
		or absf(e.value("off_x", 0)) > 0.0 or absf(e.value("off_y", 0)) > 0.0
	check("standing pattern still modulates", moving)
	for k in 40:
		e.turn(0.12)
		e.advance(0.016)
	check("further turns are additive", e.glow() > g1 + 0.02, "%f -> %f" % [g1, e.glow()])

	# Crossing a revolution increments the counter and re-rolls signatures.
	var f2 := Dial.new(9)
	f2.turn(TAU + 0.1)
	check("turn counter increments", f2.turn_count() == 1, str(f2.turn_count()))
	var shapes_differ := false
	for w in 5:
		if f2._sig_hash(0, w) != f2._sig_hash(1, w):
			shapes_differ = true
	check("each turn's signatures differ", shapes_differ)

	# Output is bounded whatever the abuse.
	var g := Dial.new(5)
	for k in 3000:
		g.turn(0.3)
		g.advance(0.002)
	var bounded := true
	for slot in Dial.SLOTS:
		for i in 5:
			if absf(g.value(slot, i)) > 1.0:
				bounded = false
	check("output bounded to (-1, 1)", bounded)

	# Phase diversity: element indices don't move in lockstep.
	var spread := 0.0
	for slot in Dial.SLOTS:
		spread += absf(g.value(slot, 0) - g.value(slot, 1))
	check("element indices decorrelate", spread > 0.01, str(spread))

	print("----")
	print("dial_check: %s (%d failures)" % ["OK" if fails == 0 else "FAILED", fails])
	quit(1 if fails > 0 else 0)
