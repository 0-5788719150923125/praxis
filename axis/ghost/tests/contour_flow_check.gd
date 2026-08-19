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
## AND THEN THE OPPOSITE REPORT, which is the other half of the same subject: "I just ran into a
## contour map scene that zoomed, but no longer moves at all... I would expect the map to slowly
## evolve, not in large jumps, but in slow morphing over time... I would also not expect the
## entire map to change simultaneously; I would expect the evolution to be localized to certain
## regions". Making a re-print invisible had made the SHEET static: the only thing that ever moved
## the land was a window warp gated on a `f.movement` rising edge, which a spoken chapter never
## produces. So the EVOLUTION half below asserts the three properties that request names - that it
## changes at all, that one print changes only a little, and that what changes is a REGION - with
## a window warp as the control, because a window warp is what "the entire map at once" looks like
## and the measure has to be able to tell them apart.
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

	_evolution_test(sc, sv, a)
	_ease_test(sc, sv)
	# Freed the way the Director does, and WAITED FOR - a scene pulled out from under a
	# FrameForge job still building on a worker faults at exit, which turns a printed
	# verdict into a core dump and an exit status nobody can trust.
	sc.queue_free()
	await get_tree().process_frame


## THE EVOLUTION: does the land change, slowly, and in one region rather than everywhere?
##
## Driven through the scene's own swell integrator (_step_swells) rather than through update() or
## _step, both of which submit a FrameForge job to a worker - and a scene pulled out from under a
## running worker faults at exit, which turns a printed verdict into a core dump.
func _evolution_test(sc, sv: int, ref) -> void:
	var base := _points(ref.run({}))
	if base.is_empty():
		_ok(false, "seed %d: the reference sheet is empty" % sv)
		return
	# ONE PRINT'S WORTH of evolution: whatever the cadence is, this is the step the viewer sees.
	for _i in int(round(sc._cadence * 60.0)):
		sc._step_swells(1.0 / 60.0)
	var one := _points(sc._make_job().run({}))
	var moved_print := _moved(base, one)
	# ...and eight seconds of it, which is the timescale the request asks for.
	for _i in 480:
		sc._step_swells(1.0 / 60.0)
	var later := _points(sc._make_job().run({}))
	var moved_long := _moved(base, later)
	var spread := _spread(base, later)
	# THE CONTROL: one step of the window ring, which is the global mechanism. If the measure
	# cannot tell this apart from the swells, it is not measuring "localized" at all.
	var keep_a: Vector2 = sc._warp_a
	var keep_b: Vector2 = sc._warp_b
	var keep_t: float = sc._warp_t
	sc._warp_a = keep_b
	sc._warp_b = keep_b + Vector2(0.06, 0.04)
	sc._warp_t = 1.0
	var warped := _points(sc._make_job().run({}))
	var moved_warp := _moved(later, warped)
	var spread_warp := _spread(later, warped)
	sc._warp_a = keep_a
	sc._warp_b = keep_b
	sc._warp_t = keep_t
	print("  evolution: %4.1f%% of the sheet redrawn in one print, %4.1f%% over 8 s (over %4.1f%% "
		% [moved_print * 100.0, moved_long * 100.0, spread * 100.0]
		+ "of its area) | window warp control: %4.1f%% redrawn over %4.1f%% of the area"
		% [moved_warp * 100.0, spread_warp * 100.0])
	_ok(moved_long > 0.02,
		"seed %d: 8 seconds of evolution redrew %.2f%% of the sheet - the map does not evolve"
		% [sv, moved_long * 100.0])
	_ok(moved_print < 0.10,
		"seed %d: ONE print redrew %.1f%% of the sheet - that is a jump, not a morph"
		% [sv, moved_print * 100.0])
	_ok(spread < 0.60,
		"seed %d: the change over 8 s is spread over %.1f%% of the sheet's area - the whole map "
		% [sv, spread * 100.0] + "is evolving at once, which is what was asked against")
	_ok(spread_warp > spread + 0.15,
		"seed %d: a window warp spread over %.1f%% against the swells' %.1f%% - this measure "
		% [sv, spread_warp * 100.0, spread * 100.0] + "cannot tell local from global")


## MEASURED WITH A TOLERANCE, and it has to be. Exact vertex matching is right for the
## re-print half of this gate (a translation preserves every point exactly) and completely wrong
## here: a contour re-routed inside a swell is re-traced and re-simplified along its WHOLE length,
## so its vertices land in different places by fractions of a pixel a metre away from the change.
## Hashed exactly, a local morph reads as "97% of the sheet redrawn". What the reader can see is
## ink moving by a pixel or more, so that is what is counted.
const TOL_PX := 1.25

## Fraction of `a`'s vertices with no vertex of `b` within TOL_PX - "how much of this drawing
## actually moved". A coarse spatial hash at the tolerance, checked over its 3x3 neighbourhood.
func _moved(a: PackedVector2Array, b: PackedVector2Array) -> float:
	if a.is_empty():
		return 0.0
	var grid := {}
	for p in b:
		var key := Vector2i(floori(p.x / TOL_PX), floori(p.y / TOL_PX))
		if not grid.has(key):
			grid[key] = PackedVector2Array()
		grid[key].append(p)
	var moved := 0
	var tol2 := TOL_PX * TOL_PX
	for p in a:
		var cx := floori(p.x / TOL_PX)
		var cy := floori(p.y / TOL_PX)
		var near := false
		for oy in range(-1, 2):
			for ox in range(-1, 2):
				var cell: Variant = grid.get(Vector2i(cx + ox, cy + oy))
				if cell == null:
					continue
				for q: Vector2 in cell:
					if p.distance_squared_to(q) <= tol2:
						near = true
						break
				if near:
					break
			if near:
				break
		if not near:
			moved += 1
	return float(moved) / float(a.size())


## How much of the SHEET'S AREA a change touches: the fraction of a coarse grid of cells that
## hold a vertex present in one packet and absent from the other. Area, not vertex count, because
## "localized" is a claim about where on the paper the reader sees ink move.
func _spread(a: PackedVector2Array, b: PackedVector2Array) -> float:
	var cells := 24
	var lo := Vector2(INF, INF)
	var hi := Vector2(-INF, -INF)
	for p in a:
		lo = Vector2(minf(lo.x, p.x), minf(lo.y, p.y))
		hi = Vector2(maxf(hi.x, p.x), maxf(hi.y, p.y))
	var span := hi - lo
	if span.x <= 0.0 or span.y <= 0.0:
		return 0.0
	var grid := {}
	for p in b:
		var key := Vector2i(floori(p.x / TOL_PX), floori(p.y / TOL_PX))
		if not grid.has(key):
			grid[key] = PackedVector2Array()
		grid[key].append(p)
	var tol2 := TOL_PX * TOL_PX
	var touched := {}
	var occupied := {}
	for p in a:
		var cx := clampi(int((p.x - lo.x) / span.x * float(cells)), 0, cells - 1)
		var cy := clampi(int((p.y - lo.y) / span.y * float(cells)), 0, cells - 1)
		var key := Vector2i(cx, cy)
		occupied[key] = true
		if touched.has(key):
			continue
		var hx := floori(p.x / TOL_PX)
		var hy := floori(p.y / TOL_PX)
		var near := false
		for oy in range(-1, 2):
			for ox in range(-1, 2):
				var cell: Variant = grid.get(Vector2i(hx + ox, hy + oy))
				if cell == null:
					continue
				for q: Vector2 in cell:
					if p.distance_squared_to(q) <= tol2:
						near = true
						break
				if near:
					break
			if near:
				break
		if not near:
			touched[key] = true
	return float(touched.size()) / float(maxi(1, occupied.size()))


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
