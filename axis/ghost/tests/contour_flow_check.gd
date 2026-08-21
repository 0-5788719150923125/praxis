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
## AND THEN THE OPPOSITE REPORT, twice, which is the other half of the same subject.
##
##   "I just ran into a contour map scene that zoomed, but no longer moves at all... I would
##   expect the map to slowly evolve, not in large jumps, but in slow morphing over time."
##   Making a re-print invisible had made the SHEET static: the only thing that ever moved the
##   land was a window warp gated on a `f.movement` rising edge, which a spoken chapter never
##   produces.
##
##   That was answered by localizing the evolution into a few compact swells, and the answer was
##   wrong: "if that feature works at all, it is not very good... multiple contour maps where they
##   barely have any movement at all... the masking seems to localize to certain regions, then
##   NEVER moves to other regions, which isn't how erosion and plate tectonics work. The goal was
##   not to limit the ability of the entire map to evolve; the goal was to stop shifting lines
##   uniformly, every ~0.5s, in large jumps."
##
## So the EVOLUTION half below asserts what that actually asks for, and the second and third are
## the ones the swells failed:
##
##   IT MOVES. Eight seconds visibly redraws the sheet, and over a minute and a half every part
##   of it has had a turn - nothing is permanently masked out.
##
##   IT IS NOT UNIFORM. At any moment some of the sheet is clearly reorganising and some of it is
##   clearly holding still. Measured per CELL of a coarse grid over the paper, because that is
##   the difference between "the land is evolving" and "the whole frame shifted": a window warp,
##   which is the control here, leaves no cell still at all.
##
##   NO PRINT IS A JUMP. One re-extraction's worth of it moves almost nothing, whatever the
##   cadence - the sheet is rebuilt one to three times a second and every difference between two
##   builds arrives as a step.
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


## THE EVOLUTION. Four questions, and the middle two are the ones the swells failed.
##
##   Does it move at all, and does every part of the paper get a turn?
##   Is one print's worth of it small, measured in PIXELS OF INK?
##   Is the change a UNIFORM SHIFT - the whole drawing translated - or is it the ground?
##
## THE FIRST THREE ARE MEASURED ON THE LAND, NOT ON THE DRAWING, and that is not a shortcut.
## A contour is smoothed and then simplified, so when it moves anywhere it is re-traced along
## its whole length, and a vertex dropped as collinear in one print and kept in the next lands
## the better part of a grid cell from where it was ON A LINE THAT DID NOT MOVE. Measured by
## nearest vertex, that reads as a 3 px step at the 95th percentile and it is insensitive to the
## scene's own budget - which is how it was caught. `SheetJob.sample()` hands back the elevation
## grid the drawing is a function of, and the distance a contour moves for a change in it is
## exactly `dv / |grad v|`, so the question can be answered in pixels with no drawing involved.
##
## The UNIFORMITY question stays on the drawn packets, because it has to: a window warp does not
## change the field at all (the window is snapped to the sampling lattice precisely so that it
## does not), it moves the picture. It is a property of the drawing and is measured there.
##
## Driven through the scene's own clock (_step_tectonics) rather than through update() or _step,
## both of which submit a FrameForge job to a worker - and a scene pulled out from under a
## running worker faults at exit, which turns a printed verdict into a core dump.
func _evolution_test(sc, sv: int, ref) -> void:
	# Explicitly typed: `ref` is an untyped local (a job loaded through the scene), so nothing
	# it returns carries a type for inference to work from.
	var g0: PackedFloat32Array = ref.sample()
	if g0.is_empty():
		_ok(false, "seed %d: the reference sheet is empty" % sv)
		return
	var cpx: float = 2.0 * ref.half.x / float(ref.nx - 1)
	var grad := _grad(g0, ref.nx, ref.ny, ref.relief)
	# ONE PRINT'S WORTH, in pixels of ink movement: whatever the cadence, this is the step the
	# viewer is handed, and "jumpy" is a statement about its SIZE rather than about how much of
	# the sheet it touched.
	var prints := int(round(sc._cadence * 60.0))
	for _i in prints:
		sc._step_tectonics(1.0 / 60.0)
	var dp := _sorted(_disp(g0, sc._make_job().sample(), grad, cpx))
	var p50 := _pct(dp, 0.50)
	var p95 := _pct(dp, 0.95)
	# ...and eight seconds of it, which is the timescale "slow morphing over time" asks for.
	for _i in 480 - prints:
		sc._step_tectonics(1.0 / 60.0)
	var d8 := _disp(g0, sc._make_job().sample(), grad, cpx)
	var blk8 := _sorted(_blocks(d8, ref.nx, ref.ny))
	var quiet := _pct(blk8, 0.10)
	var busy := _pct(blk8, 0.90)
	# NOTHING IS PERMANENTLY MASKED. Sampled at checkpoints and UNIONED rather than compared once
	# at the end, because the land oscillates: a region can be mid-reorganisation at forty
	# seconds and back where it started at ninety, and a single late comparison would score that
	# as a region that never moved.
	var reach := _blocks(d8, ref.nx, ref.ny)
	for _c in 5:
		for _i in 1080:
			sc._step_tectonics(1.0 / 60.0)
		var snap := _blocks(_disp(g0, sc._make_job().sample(), grad, cpx), ref.nx, ref.ny)
		for i in reach.size():
			reach[i] = maxf(reach[i], snap[i])
	var still := 0
	for v in reach:
		if v < 2.0:
			still += 1
	var reached := 1.0 - float(still) / float(maxi(1, reach.size()))
	# THE CONTROL, and it is the whole reason the per-region measure is here: THE SAME LAND MOVED
	# ONE GRID CELL, which is what "shifting lines uniformly" actually is - the entire drawing
	# translated by one vector. Put through the identical measure, so the two numbers sit beside
	# each other: a translation moves every region of the paper by the same distance, and ground
	# that is rising in one place and subsiding in another does not.
	#
	# It has to be synthesized rather than taken from the scene's own window warp. A window move
	# does not change this grid AT ALL - the window is snapped to the sampling lattice precisely
	# so that it does not - and the drawing it produces cannot be measured against the one before
	# it either: a contour sheet is a dense web of lines, so a web slid by twenty pixels still has
	# ink within a pixel of most of where the old ink was, and every nearest-point estimate of the
	# offset comes back as zero. That was tried first and it reported a rigid translation as
	# unexplainable, which reads as a pass.
	var blk_w := _sorted(_blocks(_disp(g0, _shifted(g0, ref.nx, ref.ny), grad, cpx),
		ref.nx, ref.ny))
	var quiet_w := _pct(blk_w, 0.10)
	var busy_w := _pct(blk_w, 0.90)
	print("  evolution: one print moves the ink %.2f px (median), %.2f px (95th)" % [p50, p95])
	print("             8 s moves it %.1f px; per region of the paper, the quietest tenth "
		% _pct(_sorted(d8), 0.50) + "%.1f px and the busiest tenth %.1f px" % [quiet, busy])
	print("             a rigid one-cell shift of the same land, for comparison: quietest tenth "
		+ "%.1f px, busiest %.1f px" % [quiet_w, busy_w])
	print("             over 90 s, %4.1f%% of the paper has had a turn" % (reached * 100.0))
	# THE LINE IS ON THE TAIL'S SIZE, NOT ON THE SPREAD BELOW IT, and that distinction was
	# measured rather than assumed. The median step is 0.2 to 0.4 px and the 95th percentile 1.0
	# to 1.7 px across these seeds, and the ratio between them is not a defect to be tuned out -
	# it IS the non-uniformity this whole half exists to demand, so a threshold tight enough to
	# close it would be asserting against the feature. What matters is that the tail stays under
	# a step the eye reads as a step. For scale, the rigid-shift control below moves the same
	# sheet 20 to 30 px in ONE go, which is what the original report was about.
	_ok(p95 < 2.0,
		"seed %d: one print moves the ink %.2f px at the 95th percentile - the sheet is rebuilt "
		% [sv, p95] + "one to three times a second, so that arrives as a visible step")
	_ok(p50 > 0.05,
		"seed %d: one print moves the ink %.3f px at the median - the land is standing still"
		% [sv, p50])
	_ok(busy > 2.5,
		"seed %d: over 8 s even the busiest tenth of the paper moved only %.1f px - the map "
		% [sv, busy] + "barely evolves")
	_ok(busy > quiet * 2.0,
		"seed %d: the busiest tenth of the paper moved %.1f px over 8 s against the quietest "
		% [sv, busy] + "tenth's %.1f px - the whole sheet is evolving at one rate, which is not "
		% quiet + "what ground does")
	_ok(busy_w < quiet_w * 1.6,
		"seed %d: a RIGID SHIFT of the same land measured %.1f px in its busiest tenth against "
		% [sv, busy_w] + "%.1f px in its quietest - a translation moves the whole sheet by one "
		% quiet_w + "distance, so if this measure sees a spread there it sees nothing anywhere")
	_ok(reached > 0.70,
		"seed %d: over 90 s only %.1f%% of the paper ever moved - the rest is masked out of the "
		% [sv, reached * 100.0] + "evolution permanently, which is not how ground behaves")


## The land's gradient magnitude per grid cell, floored. The floor is what stops the displacement
## measure dividing by zero at every summit and every basin floor, where a contour ring genuinely
## does expand a long way for a small change - true, but a handful of points, and left unfloored
## they would be the only thing any percentile above the median could see.
const GRAD_FLOOR := 0.25

func _grad(h: PackedFloat32Array, nx: int, ny: int, relief: float) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(nx * ny)
	var floor_g := GRAD_FLOOR * relief
	for y in ny:
		var row := y * nx
		var yn := maxi(0, y - 1) * nx
		var yp := mini(ny - 1, y + 1) * nx
		for x in nx:
			var gx := (h[row + mini(nx - 1, x + 1)] - h[row + maxi(0, x - 1)]) * 0.5
			var gy := (h[yp + x] - h[yn + x]) * 0.5
			out[row + x] = maxf(sqrt(gx * gx + gy * gy), floor_g)
	return out


## How far the ink at each grid point moved, in PIXELS. A contour is a level set, so a change of
## `dv` in the elevation slides it by `dv / |grad v|` across the ground - the one conversion this
## whole half rests on, and the reason the measure needs no drawing.
func _disp(a: PackedFloat32Array, b: PackedFloat32Array, grad: PackedFloat32Array,
		cell_px: float) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(a.size())
	for i in a.size():
		out[i] = absf(b[i] - a[i]) / grad[i] * cell_px
	return out


## The same, averaged over a coarse grid of REGIONS of the paper - which is the unit "some of the
## sheet is reorganising and some of it is holding still" is a claim about.
const BLK := 10

func _blocks(d: PackedFloat32Array, nx: int, ny: int) -> PackedFloat32Array:
	var acc := PackedFloat32Array()
	acc.resize(BLK * BLK)
	var cnt := PackedInt32Array()
	cnt.resize(BLK * BLK)
	for y in ny:
		var by := clampi(y * BLK / ny, 0, BLK - 1)
		for x in nx:
			var bi := by * BLK + clampi(x * BLK / nx, 0, BLK - 1)
			acc[bi] += d[y * nx + x]
			cnt[bi] += 1
	for i in BLK * BLK:
		acc[i] = acc[i] / float(maxi(1, cnt[i]))
	return acc


## The same land, moved one grid cell diagonally - a rigid translation of the drawing, built on
## the grid so it can go through the same measure as the evolution.
func _shifted(h: PackedFloat32Array, nx: int, ny: int) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(nx * ny)
	for y in ny:
		var sy := maxi(0, y - 1) * nx
		for x in nx:
			out[y * nx + x] = h[sy + maxi(0, x - 1)]
	return out


func _sorted(v: PackedFloat32Array) -> PackedFloat32Array:
	var out := v.duplicate()
	out.sort()
	return out


## The displacement measures work on a SUBSAMPLE - a sheet carries half a million vertices and
## a nearest-point search over all of them, several times a seed, is minutes of wall clock for a
## statistic that is settled by twenty thousand.
## The percentile of an ALREADY SORTED array.
func _pct(v: PackedFloat32Array, q: float) -> float:
	if v.is_empty():
		return 0.0
	return v[clampi(int(float(v.size() - 1) * q), 0, v.size() - 1)]


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
