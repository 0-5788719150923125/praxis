extends SceneTree

## One-shot Echo (content re-localization) self-check (run headless):
##   godot --path axis/ghost --headless --script res://tests/echo_check.gd
## Feeds synthetic harmonic-signature streams through the map/localize loop and
## checks the properties the Director depends on: a looped song is recognized from
## the tail, a perturbed replay still matches, novel audio never fires, a repeated
## section that the cursor already explains stays put, and corrections cool down.

var fails := 0


func check(name: String, ok: bool, detail := "") -> void:
	if ok:
		print("PASS  " + name)
	else:
		fails += 1
		print("FAIL  " + name + ("  -- " + detail if detail != "" else ""))


# A smooth, non-negative, music-like 16-dim signature trajectory; `piece` switches
# to unrelated content (different per-dim FREQUENCIES, not just phases - two pieces
# must be different processes, or a time-shift alignment between them exists and
# "recognizing" it is correct behavior, not a false positive). Squared so cells
# differ enough to discriminate.
func sig_at(t: float, piece := 0) -> PackedFloat32Array:
	var v := PackedFloat32Array()
	v.resize(16)
	for d in 16:
		var ph := float(d) * 0.7 + float(piece) * 13.7
		var w := 0.21 + 0.043 * float((d + 5 * piece) % 16) + 0.17 * float(piece)
		var s := 0.5 + 0.5 * sin(t * w + ph)
		v[d] = s * s
	return v


func _initialize() -> void:
	var dt := 0.05

	# --- A. a looping song: pass 1 maps the schedule, then the audio restarts while
	# the cursor sits in the tail (beyond the map). ------------------------------
	var e := Echo.new()
	var heard := 0.0
	var false_fire := -1.0
	var t := 0.0
	while t < 16.0:                       # pass 1: cursor tracks the audio exactly
		e.record(t, sig_at(t))
		var r := e.listen(heard, t, sig_at(t), dt)
		if r >= 0.0:
			false_fire = r
		t += dt
		heard += dt
	check("map records the first pass", e.frontier() > 14.0, str(e.frontier()))
	check("no correction while the cursor explains the audio", false_fire < 0.0, str(false_fire))
	var fired := -1.0
	var fired_after := -1.0
	var s := 0.0
	while s < 8.0:                        # the song loops; the cursor does not
		var r := e.listen(heard, 1e9, sig_at(s), dt)
		if r >= 0.0 and fired < 0.0:
			fired = r
			fired_after = s
		s += dt
		heard += dt
	check("a loop is recognized from the tail", fired >= 0.0)
	check("the correction lands near the top of the schedule", fired >= 0.0 and fired < 3.5, str(fired))
	check("recognition takes seconds, not the whole loop", fired_after >= 0.0 and fired_after < 5.0, str(fired_after))

	# After the correction the cursor tracks the corrected position: explained -> quiet.
	var refire := -1.0
	while s < 14.0:
		var r := e.listen(heard, s, sig_at(s), dt)
		if r >= 0.0:
			refire = r
		s += dt
		heard += dt
	check("a corrected cursor is left alone", refire < 0.0, str(refire))

	# --- B. a PERTURBED replay (noise on every dim) still re-localizes. ---------
	var e2 := Echo.new()
	heard = 0.0
	t = 0.0
	while t < 16.0:
		e2.record(t, sig_at(t))
		e2.listen(heard, t, sig_at(t), dt)
		t += dt
		heard += dt
	var rng := RandomNumberGenerator.new()
	rng.seed = 71
	var fired2 := -1.0
	s = 0.0
	while s < 8.0:
		var v := sig_at(s)
		for d in v.size():
			v[d] = maxf(0.0, v[d] + rng.randfn(0.0, 0.08))
		var r := e2.listen(heard, 1e9, v, dt)
		if r >= 0.0 and fired2 < 0.0:
			fired2 = r
		s += dt
		heard += dt
	check("a perturbed replay still re-localizes", fired2 >= 0.0 and fired2 < 3.5, str(fired2))

	# --- C. NOVEL audio never fires (coherence + floor hold the line). ----------
	var e3 := Echo.new()
	heard = 0.0
	t = 0.0
	while t < 16.0:
		e3.record(t, sig_at(t))
		e3.listen(heard, t, sig_at(t), dt)
		t += dt
		heard += dt
	var fired3 := -1.0
	s = 0.0
	while s < 12.0:
		var r := e3.listen(heard, 1e9, sig_at(s, 1), dt)
		if r >= 0.0:
			fired3 = r
		s += dt
		heard += dt
	check("novel audio never fires a correction", fired3 < 0.0, str(fired3))

	# --- D. a repeated section the cursor already EXPLAINS stays put: the map holds
	# two identical passes; the cursor walks the second while hearing music that
	# matches the first equally well. ---------------------------------------------
	var e4 := Echo.new()
	heard = 0.0
	t = 0.0
	while t < 16.0:
		e4.record(t, sig_at(fmod(t, 8.0)))   # 0..8 repeated twice in the schedule
		t += dt
	var fired4 := -1.0
	var c := 8.0
	while c < 15.0:                          # cursor in the second copy, tracking
		var r := e4.listen(heard, c, sig_at(fmod(c, 8.0)), dt)
		if r >= 0.0:
			fired4 = r
		c += dt
		heard += dt
	check("an explained repeat (chorus) never moves the cursor", fired4 < 0.0, str(fired4))

	# --- E. degenerate input is safe. -------------------------------------------
	var e5 := Echo.new()
	e5.record(0.0, PackedFloat32Array())
	var r5 := e5.listen(0.0, 0.0, PackedFloat32Array(), dt)
	check("empty signatures are inert", r5 < 0.0 and e5.frontier() == 0.0)

	print("----")
	print("echo_check: %s (%d failures)" % ["OK" if fails == 0 else "FAILED", fails])
	quit(1 if fails > 0 else 0)
