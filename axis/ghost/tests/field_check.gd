extends SceneTree

## Gate for [VoiceField]. The fishing loop has never had a test of any kind, so
## this is the first guard on it.
##
## The two failure modes are opposite and both silent:
##   COLLAPSE   - the field becomes a fourth attractor toward the belt's centre,
##                every candidate near the belt converges on the same voice, and
##                the game stops producing novelty. Caught-trait spread from a
##                fixed belt was already measured falling 0.152 -> 0.012 per axis
##                as the belt fills, so this is not hypothetical.
##   INVISIBLE  - the field is too weak, or depends only on distance, and the
##                mechanic may as well not exist.
## A test that only checked "the numbers changed" would pass in both cases.
##
## Run: godot --headless --path axis/ghost --script tests/field_check.gd

const VoiceField_ := preload("res://scripts/voice_field.gd")

const N_BELT := 9


func _init() -> void:
	var fails := 0
	var belt := _make_belt(N_BELT)
	fails += _check_determinism(belt)
	fails += _check_falloff(belt)
	fails += _check_angle(belt)
	fails += _check_range_separation(belt)
	fails += _check_polarity(belt)
	fails += _check_budget()
	fails += _check_modes(belt)
	if fails == 0:
		print("field_check: ALL OK")
	else:
		print("field_check: %d FAILURE(S)" % fails)
	quit(fails)


## A synthetic belt: seeded voices with a spread of acceptances, so the sources
## span the polarity range and the gene-derived ranges span theirs.
func _make_belt(n: int) -> Array:
	var out: Array = []
	var rng := RandomNumberGenerator.new()
	rng.seed = 424242
	var reach: Array = []
	var vecs: Array = []
	var gens: Array = []
	var lins: Array = []
	for i in n:
		var lineage: Array = [1000 + i * 37]
		var g: Dictionary = Voice.ProsodyWalk._lineage_genome(lineage)
		var v := PackedFloat32Array()
		for _k in Voice.TRAIT_KEYS.size():
			v.append(rng.randf_range(-1.0, 1.0))
		for key in Voice.ProsodyWalk.PRIOR:
			var b: Array = Voice.ProsodyWalk.G_BOUNDS.get(key, [])
			var spread: float = (float(b[1]) - float(b[0])) if b.size() == 2 else 1.0
			v.append((float(g[key]) - float(Voice.ProsodyWalk.PRIOR[key])) / maxf(spread, 0.001))
		reach.append(maxf(float(g.ring) * (1.0 - float(g.damp)), 0.02))
		vecs.append(v)
		gens.append(g)
		lins.append(lineage)
	reach.sort()
	var med: float = reach[reach.size() / 2]
	for i in n:
		# acceptance spread around 1.0 so roughly half the belt repels
		out.append(VoiceField_.source({}, gens[i], lins[i],
			0.35 + 1.4 * float(i) / float(n - 1), vecs[i], med, 1.0))
	return out


func _check_determinism(belt: Array) -> int:
	var p := Vector3(0.3, -0.2, 0.15)
	var a: Dictionary = VoiceField_.evaluate(belt, p)
	var b: Dictionary = VoiceField_.evaluate(belt, p)
	var shuffled := belt.duplicate()
	shuffled.reverse()
	var c: Dictionary = VoiceField_.evaluate(shuffled, p)
	var same: bool = (a.vec as Vector3).is_equal_approx(b.vec) \
		and (a.vec as Vector3).is_equal_approx(c.vec)
	print("field_check: determinism %s (belt order independent)" % ["ok" if same else "FAIL"])
	return 0 if same else 1


## Far from the belt the field must vanish, and it must do so MONOTONICALLY.
## This is the whole user-facing claim: drift out and the voice settles.
func _check_falloff(belt: Array) -> int:
	# measured from OUTSIDE the belt's own extent. Inside it the energy rightly
	# rises and falls as you pass near individual sources - the field is not
	# centred on the origin, and a monotonic-from-zero assumption was wrong
	# about the model rather than finding a bug in it.
	var extent := 0.0
	for s in belt:
		extent = maxf(extent, (s.pos as Vector3).length() + float(s.range))
	var last := INF
	var mono := true
	var peak := 0.0
	var line := ""
	for step in 9:
		var r: float = extent + float(step) * extent * 0.5
		var e := 0.0
		# average over directions so this measures distance, not one bearing
		for d in _dirs():
			e += float(VoiceField_.evaluate(belt, (d as Vector3) * r).energy)
		e /= float(_dirs().size())
		line += " %.1f:%.4f" % [r, e]
		peak = maxf(peak, e)
		if e > last + 1e-6:
			mono = false
		last = e
	# "far = stable" is a claim about the RATIO, not an absolute: the field must
	# fall to a small fraction of what it is at the belt's edge.
	var vanished: bool = last < peak * 0.05
	print("field_check: falloff beyond the belt's extent (%.2f)%s" % [extent, line])
	print("           monotonic %s, decays to %.1f%% of edge %s" % [
		"ok" if mono else "FAIL", 100.0 * last / maxf(peak, 1e-9),
		"ok" if vanished else "FAIL"])
	return (0 if mono else 1) + (0 if vanished else 1)


## ANGLE must matter. Two probes the same distance out on different bearings
## have to differ by a lot more than numerical noise - otherwise the field is
## pure distance falloff wearing a wave equation, which is the thing the whole
## design exists to avoid.
func _check_angle(belt: Array) -> int:
	var r := 0.9
	var vals: Array = []
	for d in _dirs():
		vals.append(float(VoiceField_.evaluate(belt, (d as Vector3) * r).energy))
	var lo: float = vals[0]
	var hi: float = vals[0]
	var mean := 0.0
	for v in vals:
		lo = minf(lo, v)
		hi = maxf(hi, v)
		mean += v
	mean /= float(vals.size())
	var spread: float = (hi - lo) / maxf(mean, 1e-6)
	var ok: bool = spread > 0.5
	print("field_check: angle spread %.2f at r=%.1f (need > 0.50) %s" % [
		spread, r, "ok" if ok else "FAIL"])
	return 0 if ok else 1


## Seeds must have DIFFERENT reach. Removing the shortest-range source should
## barely change the field at a distance where it has already died, while
## removing the longest-range one should still move it.
func _check_range_separation(belt: Array) -> int:
	var shortest := 0
	var longest := 0
	for i in belt.size():
		if float(belt[i].range) < float(belt[shortest].range):
			shortest = i
		if float(belt[i].range) > float(belt[longest].range):
			longest = i
	var ratio: float = float(belt[longest].range) / maxf(float(belt[shortest].range), 1e-6)
	var p := Vector3(0.55, 0.35, -0.4).normalized() * 1.6
	var base: Vector3 = VoiceField_.evaluate(belt, p).vec
	var no_short := belt.duplicate()
	no_short.remove_at(shortest)
	var no_long := belt.duplicate()
	no_long.remove_at(longest)
	var d_short: float = (base - (VoiceField_.evaluate(no_short, p).vec as Vector3)).length()
	var d_long: float = (base - (VoiceField_.evaluate(no_long, p).vec as Vector3)).length()
	var ok: bool = ratio > 3.0 and d_long > d_short
	print("field_check: range span %.1fx; at r=1.6 dropping the longest-range seed moves the field %.4f vs %.4f for the shortest %s" % [
		ratio, d_long, d_short, "ok" if ok else "FAIL"])
	return 0 if ok else 1


## NEGATIVE influence has to actually exist and actually subtract: with a mixed
## belt, some probes must see the field pushed the opposite way from the
## all-positive case. If every amplitude were positive this is unreachable.
func _check_polarity(belt: Array) -> int:
	var pos := belt.duplicate()
	var flipped: Array = []
	for s in belt:
		var t: Dictionary = (s as Dictionary).duplicate()
		t.amp = absf(float(t.amp))
		flipped.append(t)
	var opposed := 0
	var n := 0
	for d in _dirs():
		var p: Vector3 = (d as Vector3) * 0.7
		var a: Vector3 = VoiceField_.evaluate(pos, p).vec
		var b: Vector3 = VoiceField_.evaluate(flipped, p).vec
		if a.length() > 1e-5 and b.length() > 1e-5:
			n += 1
			if a.normalized().dot(b.normalized()) < 0.0:
				opposed += 1
	var ok: bool = opposed > 0
	print("field_check: polarity - %d/%d probes point OPPOSITE the all-positive belt %s" % [
		opposed, n, "ok" if ok else "FAIL"])
	return 0 if ok else 1


## The shared budget: a belt three times as big must not radiate three times as
## hard, or catches homogenize as the game goes on.
func _check_budget() -> int:
	var small := _make_belt(4)
	var big := _make_belt(22)
	var p := Vector3(0.2, 0.1, -0.1)
	var es: float = float(VoiceField_.evaluate(small, p).prox)
	var eb: float = float(VoiceField_.evaluate(big, p).prox)
	var ok: bool = eb < es * 2.0 + 0.05
	print("field_check: budget - prox %.3f at 4 seeds vs %.3f at 22 (need under 2x) %s" % [
		es, eb, "ok" if ok else "FAIL"])
	return 0 if ok else 1


## The four combination rules must be genuinely different instruments, not four
## spellings of a sum.
func _check_modes(belt: Array) -> int:
	var p := Vector3(0.4, -0.25, 0.3)
	var out := {}
	for m in VoiceField_.COMBINE:
		out[m] = VoiceField_.evaluate(belt, p, m).vec
	var distinct := true
	var line := ""
	for m in VoiceField_.COMBINE:
		line += "  %s %.4f" % [m, (out[m] as Vector3).length()]
	for a in VoiceField_.COMBINE:
		for b in VoiceField_.COMBINE:
			if a != b and (out[a] as Vector3).is_equal_approx(out[b]):
				distinct = false
	print("field_check: modes%s %s" % [line, "ok" if distinct else "FAIL (two modes identical)"])
	return 0 if distinct else 1


func _dirs() -> Array:
	return [
		Vector3(1, 0, 0), Vector3(-1, 0, 0), Vector3(0, 1, 0), Vector3(0, -1, 0),
		Vector3(0, 0, 1), Vector3(0, 0, -1),
		Vector3(1, 1, 1).normalized(), Vector3(-1, 1, -1).normalized(),
		Vector3(1, -1, 1).normalized(), Vector3(-1, -1, 1).normalized(),
	]
