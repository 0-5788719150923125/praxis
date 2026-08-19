extends Node

## Gate for the ONE thing that can make a strata band disappear: the engine refusing to
## triangulate it.
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/strata_band_check.gd 240
##
## Needs a real renderer, and that is exactly why this shipped broken - a dummy driver never
## triangulates anything, so `run_scene_smoke` rendered these bands hundreds of times and could
## not have seen it. It surfaced in a live session instead:
##   ERROR: Invalid polygon data, triangulation failed. at: _draw (strata.gd:122)
##
## WHY IT HAPPENS. Godot ear-clips a closed polygon and its snip test needs a strictly positive
## cross product, so a zero-area ear can never be clipped. The `ridge` profile is straight flanks
## - exactly collinear runs of vertices - and the crest clamp lays flat plateaus on top of that.
## Measured below: it is about one band in twenty thousand, always `ridge`, and when it lands the
## band does not glitch, it VANISHES for that frame.
##
## The fix is to stop asking: a band is x-monotone, so the scene fills it as an explicit strip of
## quads in one batched call ([TriBatch]). Two claims, and the first one has to SEARCH.
##
##   FOUND STATE. A captured fixture is no good here: the condition is a knife edge, and a state
##   captured from arithmetic that differs in the fifth decimal (a probe's own copy of the
##   formula, say) triangulates perfectly well through the scene. So the gate sweeps real scene
##   states, asking [method Strata.band_polygon] until the engine refuses one - then renders that
##   state twice, through the scene and through the OLD closed-polygon path over the scene's own
##   points, and requires the two pictures to DIFFER far more than they do in a state the engine
##   handles. That comparison is relative for a reason: the bands are translucent and each fills
##   down to the foot of the frame, so a vanished band mostly reveals the band behind it. Measured
##   as "fraction of the frame painted at all" it moves by four tenths of a percent and proves
##   nothing; measured as "fraction of pixels that changed" it is the size of the missing band.
##
##   EQUIVALENCE. The strip must be the same SHAPE as the polygon it replaced, or this is a
##   redesign rather than a fill. Asserted as an area identity (the strip's trapezoids against
##   the closed polygon's shoelace area) over every state swept, which is exact and needs no
##   pixels - so the crest lines and the rasterizer stay out of it.

const COLS := 72          # matches Strata.COLS; the gate reads the scene's points, not its math
## Bands to sweep looking for a refusal. About one in twenty thousand is refused, so this is
## sized to expect several: at 120k the chance of finding none is under a percent, and if it ever
## does find none it says so and passes rather than crying wolf. The sweep is RIDGE-weighted
## because every refusal ever measured here has been that profile - straight flanks are what
## produce the exactly-collinear triples the ear clipper cannot snip.
const SEARCH := 120000

var _fails: Array = []
var _sc = null
var _vp: SubViewport = null


func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	_vp = SubViewport.new()
	_vp.size = Vector2i(1920, 1080)
	_vp.disable_3d = true
	_vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(_vp)
	_sc = load("res://scripts/scenes/strata.gd").new()
	_vp.add_child(_sc)
	_sc.init_with_seed(404, "static")
	_sc.size = Vector2(1920, 1080)

	# --- the sweep: find a refusal, and check the area identity on everything it touches ---
	var rng := RandomNumberGenerator.new()
	rng.seed = 20260819
	var refused := 0
	var worst_area := 0.0
	var tested := 0
	var found := {}
	while tested < SEARCH:
		var st := {
			"planes": rng.randi_range(3, 17),
			"profile": "ridge" if rng.randf() < 0.8 else (
				"terrace" if rng.randf() < 0.5 else "smooth"),
			"steps": rng.randi_range(4, 11),
			"wave_k": rng.randf_range(1.0, 5.5),
			"amp": rng.randf_range(0.04, 0.22),
			"t": rng.randf_range(0.0, 200.0),
			"beat": 1.0 if rng.randf() < 0.3 else 0.0,
			"spec": [0.0, rng.randf(), 0.95][rng.randi() % 3],
		}
		_apply(st)
		for i in int(st["planes"]):
			tested += 1
			# One call, not two: band_polygon IS band_points plus the two feet, so the strip's
			# points are the polygon minus its last pair.
			var poly: PackedVector2Array = _sc.band_polygon(i)
			var tops := poly.slice(0, poly.size() - 2)
			# EQUIVALENCE, on every band swept: same shape, to float precision.
			worst_area = maxf(worst_area, _area_gap(tops, poly))
			if Geometry2D.triangulate_polygon(poly).size() < 3:
				refused += 1
				if found.is_empty():
					found = st.duplicate()
					found["i"] = i
	print("strata_band_check: swept %d bands; the engine refused %d of them as closed polygons"
		% [tested, refused])
	print("strata_band_check: worst strip-vs-polygon AREA gap %.9f (relative)" % worst_area)
	if worst_area > 1e-4:
		_fails.append("equivalence: the strip's area differs from the closed polygon's by %.9f - "
			% worst_area + "it is not filling the same shape")

	# --- the found state, rendered both ways ---
	if found.is_empty():
		print("strata_band_check: no untriangulable band in %d - nothing to regress against, "
			% SEARCH + "which is itself worth knowing (the strip makes the question moot)")
	else:
		print("strata_band_check: refusal at planes %d plane %d %s steps %d wave_k %.6f amp %.6f "
			% [int(found["planes"]), int(found["i"]), found["profile"], int(found["steps"]),
				float(found["wave_k"]), float(found["amp"])]
			+ "t %.6f beat %.0f spec %.3f" % [float(found["t"]), float(found["beat"]),
				float(found["spec"])])
		var d_bad := await _two_paths(found)
		# The same state at a phase the engine is happy with, as the baseline: whatever the two
		# paths differ by when nothing is dropped (the crest lines the control does not draw, and
		# rasterizer edges) is the floor this measurement sits on.
		var okst := found.duplicate()
		var moved := false
		for k in 400:
			okst["t"] = float(found["t"]) + 0.013 * float(k + 1)
			_apply(okst)
			if Geometry2D.triangulate_polygon(_sc.band_polygon(int(found["i"]))).size() >= 3:
				moved = true
				break
		var d_ok := await _two_paths(okst) if moved else -1.0
		print("strata_band_check: pixels changed between the two paths - %.2f%% in the refused "
			% (100.0 * d_bad) + "state, %.2f%% in a state the engine handles" % (100.0 * d_ok))
		if d_ok < 0.0:
			_fails.append("baseline: could not find a nearby state the engine accepts")
		else:
			# The paths must agree pixel for pixel where the engine copes - the strip is a fill,
			# not a redesign - and disagree materially where it does not, because there the old
			# path is missing a whole band. The bands are translucent and each reaches the foot of
			# the frame, so a dropped band changes the composite over its entire footprint even
			# when nearer planes lie over it; that is why this reads clearly whichever plane the
			# refusal lands on.
			if d_ok > 0.005:
				_fails.append("equivalence: %.4f of pixels differ between the strip and the "
					% d_ok + "polygon path in a state the engine handles - the strip is not "
					+ "drawing the same picture")
			if d_bad < 0.02:
				_fails.append("regression: only %.4f of pixels differ in a state the engine "
					% d_bad + "REFUSES - the dropped band is not coming back")

	print("")
	if _fails.is_empty():
		print("strata_band_check: ALL OK")
	else:
		print("strata_band_check: %d FAILURE(S)" % _fails.size())
		for f in _fails:
			print("   ", f)
	for _i in 6:
		await get_tree().process_frame
	get_tree().quit(1 if not _fails.is_empty() else 0)


func _apply(st: Dictionary) -> void:
	_sc.params["planes"] = int(st["planes"])
	_sc.params["profile"] = String(st["profile"])
	_sc.params["steps"] = int(st["steps"])
	_sc.params["wave_k"] = float(st["wave_k"])
	_sc.params["amp"] = float(st["amp"])
	_sc._t = float(st["t"])
	_sc._f = _features(float(st["spec"]), float(st["beat"]))
	_sc.queue_redraw()


## |strip area - polygon area| / polygon area. The strip is one trapezoid per column down to the
## foot; the polygon is the same outline closed. They are the same region, so this is 0 up to
## float error - and if the strip ever stops being that region, this is what says so.
func _area_gap(tops: PackedVector2Array, poly: PackedVector2Array) -> float:
	var foot: float = poly[poly.size() - 1].y
	var strip := 0.0
	for c in tops.size() - 1:
		strip += (tops[c + 1].x - tops[c].x) \
			* ((foot - tops[c].y) + (foot - tops[c + 1].y)) * 0.5
	var sh := 0.0
	for k in poly.size():
		var a := poly[k]
		var b := poly[(k + 1) % poly.size()]
		sh += a.x * b.y - b.x * a.y
	sh = absf(sh) * 0.5
	return absf(absf(strip) - sh) / maxf(sh, 1.0)


## The OLD fill path, faithfully: every band as one closed polygon through the engine's
## triangulator, over the scene's OWN points, so the difference isolates the change.
class OldPath:
	extends Node2D
	var scene = null

	func _draw() -> void:
		if scene == null:
			return
		draw_set_transform_matrix(scene.view.matrix(scene.size))
		var planes := int(scene.params["planes"])
		for i in planes:
			var depth := float(i) / float(maxi(2, planes) - 1)
			var loud: float = scene._f.sample(1.0 - depth)
			var h: float = fposmod(float(scene.params["hue"])
				+ float(scene.params["hue_span"]) * depth, 1.0)
			var fill := Color.from_hsv(h, clampf(float(scene.params["sat"]) * 0.85, 0.0, 1.0),
				clampf((0.25 + 0.6 * (0.3 + loud) * (0.4 + depth))
					* float(scene.params["val"]), 0.0, 1.0), float(scene.params["alpha"]))
			var poly: PackedVector2Array = scene.band_polygon(i)
			draw_colored_polygon(poly, fill)
			# The crest line too, so the only difference between this render and the scene's is
			# the FILL PATH. The two colour formulas are copied here and could drift from the
			# scene's - which is precisely what the "accepted state" comparison below would
			# report, since it requires the two pictures to agree pixel for pixel.
			var lcol := Color.from_hsv(h, clampf(float(scene.params["sat"]) * 0.55, 0.0, 1.0),
				clampf((0.7 + 0.3 * loud) * float(scene.params["val"]), 0.0, 1.0), 0.7)
			draw_polyline(poly.slice(0, poly.size() - 2), lcol, 1.5 + 2.0 * depth, true)


func _features(level: float, beat: float) -> AudioFeatures:
	var f := AudioFeatures.new()
	var bands := PackedFloat32Array()
	bands.resize(Spectrum.BAND_COUNT)
	for b in Spectrum.BAND_COUNT:
		bands[b] = level
	f.bands = bands
	f.energy = level
	f.beat = beat
	f.beat_period = 0.5
	return f


func _grab() -> Image:
	for _i in 3:
		await get_tree().process_frame
	return _vp.get_texture().get_image()


## Render `st` through the scene and through the old closed-polygon path, and return the fraction
## of pixels that differ.
func _two_paths(st: Dictionary) -> float:
	_apply(st)
	var strip := await _grab()
	var ctrl := OldPath.new()
	ctrl.scene = _sc
	_vp.add_child(ctrl)
	_sc.visible = false
	var poly := await _grab()
	_sc.visible = true
	_vp.remove_child(ctrl)
	ctrl.free()
	await get_tree().process_frame
	return _diff_frac(strip, poly)


## Fraction of sampled pixels whose colour differs between two renders.
func _diff_frac(a: Image, b: Image) -> float:
	var hit := 0
	var n := 0
	for y in range(0, a.get_height(), 3):
		for x in range(0, a.get_width(), 3):
			var p := a.get_pixel(x, y)
			var q := b.get_pixel(x, y)
			if absf(p.r - q.r) + absf(p.g - q.g) + absf(p.b - q.b) > 0.02:
				hit += 1
			n += 1
	return float(hit) / float(maxi(1, n))
