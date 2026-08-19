extends Node

## Gate for the contour map's RE-PRINT - the property that the sheet is rebuilt a couple of
## times a second and the viewer cannot tell.
##
## The sheet is extracted inside a worker on a [0.3, 1.0] s cadence and the finished packet
## is re-submitted every frame until the next one lands. That is only invisible if two
## consecutive builds are the SAME DRAWING. They were not, and it read exactly as reported:
## "the entire scene shifts every 0.5 to 1.0 seconds, jumpy rather than continuous, which
## turns an otherwise peaceful scene into something distracting."
##
## Three things differed between builds, and all three are asserted here:
##
##   THE SAMPLING PHASE. The land is static and only the window moves, so a re-print at the
##   same lattice phase reads the same points of the field and needs only translating - and
##   _draw translates it exactly. Sampled a fraction of a cell along, marching squares on
##   cells a dozen pixels across answers with a wobble of PIXELS on every line at once. The
##   window is snapped to its own lattice now, so a drift under one cell must produce a
##   byte-identical packet, and a drift of exactly one cell must produce the same drawing
##   moved by exactly one cell. The unsnapped case is measured alongside as the control: if
##   it ever matched as well, this gate would be asserting nothing.
##
##   THE HATCH PITCH. `f.flux` used to set the hatch SPACING. The ruling is anchored, so a
##   change of pitch moves every line by its own multiple of the change and the whole band
##   slides. Spacing is the seed's now and no audio feature may reach it.
##
##   THE INK. The highlighted contour's colour is baked into the packet, so the tonal
##   centre it comes from has to be eased or it arrives as a step every re-print.
##
## Run inside a real boot (the scene reaches the Spectrum autoload):
##   tests/run_boot_probe.sh tests/contour_flow_check.gd 180

const SEEDS := [3, 11, 404]
const QUANT := 50.0            # point-match resolution: a fiftieth of a pixel

var _fails: Array = []


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	for sv in SEEDS:
		await _run_seed(sv)
	print("")
	if _fails.is_empty():
		print("contour_flow_check: ALL OK - a re-print is the same drawing.")
	else:
		print("contour_flow_check: %d FAILURE(S)" % _fails.size())
		for f in _fails:
			print("   ", f)
	for _i in 6:
		await get_tree().process_frame
	get_tree().quit(1 if not _fails.is_empty() else 0)


func _run_seed(sv: int) -> void:
	var sc = load("res://scripts/scenes/contour_map.gd").new()
	add_child(sc)
	sc.init_with_seed(sv, "drift")
	sc.size = Vector2(1920, 1080)
	# The elevation ladder and the summit table, without going through update() - which
	# would also start the cadence and hand a real extraction to a worker thread, and a
	# worker still building when the tree tears down faults at exit. Every job here is run
	# on this thread, deliberately, so the exit status means something.
	sc._ensure_sites()
	print("")
	print("--- seed %d: grid %d, hatch %s ---"
		% [sv, sc._res, "on" if sc._hatch_on else "off"])

	# The window is pinned by hand: _warp_now() returns _warp_b once _warp_t has arrived.
	sc._warp_t = 1.0
	sc._warp_a = Vector2.ZERO
	sc._warp_b = Vector2.ZERO
	var a = sc._make_job()
	if a == null:
		_fails.append("seed %d: no job - the ladder never got solved" % sv)
		sc.queue_free()
		await get_tree().process_frame
		return
	var cw: float = 1.0 / float(a.nx - 1)           # one grid cell, in warp (sheet) units
	var cpx: float = 2.0 * a.half.x / float(a.nx - 1)   # ...and in pixels
	var pa := _points(a.run({}))

	# --- a drift of less than one cell: nothing may change at all ---
	var chy: float = 1.0 / float(a.ny - 1)
	sc._warp_b = Vector2(cw * 0.4, chy * 0.4)
	var b = sc._make_job()
	var same_off: bool = a.off.distance_to(b.off) < 1e-6
	var pb := _points(b.run({}))
	var identical := pa == pb
	print("  sub-cell drift: window %s, packet %s (%d pts)"
		% ["snapped to the same lattice point" if same_off else "MOVED",
			"identical" if identical else "DIFFERENT", pb.size()])
	_ok(same_off, "seed %d: a 0.4-cell drift moved the sampling window - it must snap" % sv)
	_ok(identical,
		"seed %d: a 0.4-cell drift changed the packet - the sheet re-wobbles on every "
		% sv + "re-print, which is the jumpiness this gate exists for")

	# --- a drift of exactly one cell: the same drawing, moved by exactly one cell ---
	sc._warp_b = Vector2(cw, 0.0)
	var c = sc._make_job()
	var pc := _points(c.run({}))
	var hit_snap := _match(pa, pc, -cpx)
	# ...against the same drift taken OFF the lattice, which is what used to happen.
	sc._warp_b = Vector2.ZERO
	var d = sc._make_job()
	var half_cell: float = cw * 0.5 * a.fspan.x
	d.off = a.off + Vector2(half_cell, 0.0)
	var pd := _points(d.run({}))
	var hit_raw := _match(pa, pd, -cpx * 0.5)
	print("  one-cell drift:  %5.1f%% of the sheet is the same drawing translated"
		% (hit_snap * 100.0))
	print("  half-cell, OFF the lattice (the control): %5.1f%%" % (hit_raw * 100.0))
	_ok(hit_snap > 0.9,
		"seed %d: only %.1f%% of a one-cell re-print is the previous sheet translated - "
		% [sv, hit_snap * 100.0] + "the lattice snap is not holding")
	_ok(hit_raw < hit_snap - 0.2,
		"seed %d: sampling off the lattice matched %.1f%% - as well as sampling on it, so "
		% [sv, hit_raw * 100.0] + "this gate is measuring nothing")

	# --- the hatch pitch belongs to the seed, not to the music ---
	sc._warp_b = Vector2.ZERO
	sc._flux_e = 0.005
	var quiet = sc._make_job()
	sc._flux_e = 0.9
	var busy = sc._make_job()
	print("  hatch: %.2f px at flux 0.005, %.2f px at flux 0.9 (ink %.2f -> %.2f)"
		% [quiet.hatch_sp, busy.hatch_sp, quiet.hatch_col.a, busy.hatch_col.a])
	_ok(absf(quiet.hatch_sp - busy.hatch_sp) < 1e-6,
		"seed %d: flux moved the hatch spacing %.2f -> %.2f px - the ruling is anchored, so "
		% [sv, quiet.hatch_sp, busy.hatch_sp] + "a change of pitch slides the whole band")
	if sc._hatch_on:
		_ok(busy.hatch_col.a > quiet.hatch_col.a + 0.05,
			"seed %d: flux does not reach the hatching at all any more" % sv)

	_ease_test(sc, sv)
	# Freed the way the Director does, and WAITED FOR - a scene pulled out from under a
	# FrameForge job still building on a worker faults at exit, which turns a printed
	# verdict into a core dump and an exit status nobody can trust.
	sc.queue_free()
	await get_tree().process_frame


## The ink of the highlighted contour is baked into a packet rebuilt at most every 0.3 s, so
## what matters is how far its colour can move ACROSS ONE RE-PRINT under the most hostile
## input there is: a tonal centre sweeping continuously through the hue wrap, which has no
## settled answer at all. Unsmoothed this is a full-saturation colour arriving in one step.
func _ease_test(sc, sv: int) -> void:
	var dt := 1.0 / 60.0
	var hist: Array = []
	var worst := 0.0
	for i in 1800:
		sc._ch_raw = Vector2(fposmod(float(i) * dt * 0.7, 1.0), 0.9)
		sc._ease_audio(dt)
		hist.append(sc._pick_color())
		if hist.size() > 18:                   # 0.3 s, the fastest cadence there is
			var then: Color = hist[hist.size() - 19]
			var now: Color = hist[hist.size() - 1]
			worst = maxf(worst, Vector3(now.r - then.r, now.g - then.g, now.b - then.b).length())
	print("  inked contour: worst colour step across one re-print %.3f" % worst)
	_ok(worst < 0.06,
		"seed %d: the inked contour's colour can move %.3f in one re-print - that is a "
		% [sv, worst] + "flash on every loop at that elevation, all over the sheet at once")


# Every vertex of a packet, as one flat array.
func _points(chunks: Array) -> PackedVector2Array:
	var out := PackedVector2Array()
	for ch in chunks:
		out.append_array((ch as Dictionary)["pts"] as PackedVector2Array)
	return out


# What fraction of `b` lands on a vertex of `a` shifted by `dx`. Quantized rather than
# compared exactly, because the two builds reach the same land through different arithmetic
# and agree only to within a float ulp of the field sample.
func _match(a: PackedVector2Array, b: PackedVector2Array, dx: float) -> float:
	if b.is_empty():
		return 0.0
	var set := {}
	for p in a:
		set[Vector2i(roundi((p.x + dx) * QUANT), roundi(p.y * QUANT))] = true
	var hit := 0
	for p in b:
		if set.has(Vector2i(roundi(p.x * QUANT), roundi(p.y * QUANT))):
			hit += 1
	return float(hit) / float(b.size())


func _ok(cond: bool, msg: String) -> void:
	if not cond:
		_fails.append(msg)
