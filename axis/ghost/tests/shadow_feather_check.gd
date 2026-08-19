extends SceneTree

## The landscape scenes' cast shadows: are their EDGES resolvable, or do they land in one step?
##
## Written for a report that the shadows over landscape geometry "are blocky". They were, in two
## independent places, and each needs its own measurement because each fails differently:
##
##   TERRAIN (Terrain._cast) - a per-vertex ray march over the heightfield, drawn Gouraud across
##   quads. A shadow edge that changes by its whole range between two adjacent vertices cannot be
##   drawn as anything but a staircase whose step is one quad, and at this map's scale a near quad
##   is tens of pixels. So the claim under test is a SPATIAL one about the map itself: no single
##   lattice step may carry a large fraction of the map's whole lit-to-shadowed range.
##
##   PROPS (ShadowField) - a light-space grid. The rasterizer used to snap an occluder's silhouette
##   to whole cells, so the edge was hard however smoothly `factor()` then interpolated: four cells
##   that are each fully lit or fully shadowed can only ever produce a one-cell ramp. The claim is
##   therefore about the WIDTH of the lit-to-shadowed transition, measured in texels along a line
##   crossing it, and about a round occluder casting a round shadow rather than its bounding square.
##
## Every check is TWO-SIDED: a shadow that has been feathered into nothing would pass a softness
## test trivially, so each one is paired with an assertion that the shadow is still there and still
## dark where it should be.
##   godot --headless --path axis/ghost --script tests/shadow_feather_check.gd

var _fails: Array = []


func _init() -> void:
	_terrain_edges()
	_field_edges()
	_round_silhouette()
	if _fails.is_empty():
		print("shadow_feather_check: ALL OK")
		quit(0)
	else:
		print("shadow_feather_check: %d FAILURE(S)" % _fails.size())
		for f in _fails:
			print("   ", f)
		quit(1)


func _check(ok: bool, msg: String) -> void:
	print(("   ok   " if ok else "   FAIL ") + msg)
	if not ok:
		_fails.append(msg)


# --- the terrain's own cast-shadow map -------------------------------------------------------
func _terrain_edges() -> void:
	print("TERRAIN cast-shadow map (Terrain._cast, %d x %d lattice)" % [Terrain.RES, Terrain.RES])
	var rng := RandomNumberGenerator.new()
	rng.seed = 11
	var t := Terrain.new()
	t.build(rng, "mountains", 4.0, 1.7, null, "temperate")
	t.set_light(0.7, deg_to_rad(16.0))            # a LOW sun: long shadows, hard edges
	for _i in 240:                                 # settle: the map refreshes a few rows per call
		t.step_light(1.0 / 60.0)
	var cast: PackedFloat32Array = t.get("_cast")
	var res: int = t.res
	var lo := 1e9
	var hi := -1e9
	for v in cast:
		lo = minf(lo, v)
		hi = maxf(hi, v)
	var span := hi - lo
	# The biggest jump between neighbouring vertices, as a fraction of the whole range. This is
	# exactly "how much of the edge lands in one quad".
	var worst := 0.0
	var steep := 0
	for gy in res:
		for gx in res - 1:
			var d: float = absf(cast[gy * res + gx + 1] - cast[gy * res + gx])
			worst = maxf(worst, d)
			if d > 0.25 * span:
				steep += 1
	for gy in res - 1:
		for gx in res:
			var d: float = absf(cast[(gy + 1) * res + gx] - cast[gy * res + gx])
			worst = maxf(worst, d)
			if d > 0.25 * span:
				steep += 1
	var frac := worst / maxf(1e-6, span)
	print("   range %.3f .. %.3f (span %.3f); worst neighbour step %.3f = %.0f%% of the span; "
		% [lo, hi, span, worst, frac * 100.0] + "%d steps over 25%% of it" % steep)
	_check(frac < 0.42, "no single lattice step carries most of the edge (%.0f%% < 42%%)" % (frac * 100.0))
	_check(steep < 60, "few near-vertical steps across the whole map (%d < 60)" % steep)
	# ...and the shadows are still SHADOWS: a low sun over relief must darken a real share of the map.
	var dark := 0
	for v in cast:
		if v < 1.0 - 0.25 * (1.0 - Terrain.SHADOW_MIN):
			dark += 1
	var dfrac := float(dark) / float(cast.size())
	print("   %.1f%% of the map is meaningfully shadowed" % (dfrac * 100.0))
	_check(dfrac > 0.05, "the feather did not erase the shadows (%.1f%% > 5%%)" % (dfrac * 100.0))


# --- the prop shadow map ----------------------------------------------------------------------
func _field_edges() -> void:
	print("PROP shadow map (ShadowField, 128 cells over the canopy span)")
	var half := 4.25
	var relief := 1.5
	var el := deg_to_rad(20.0)
	var az := 0.7
	var l := Vector3(cos(el) * cos(az), sin(el), cos(el) * sin(az)).normalized()
	var sf := ShadowField.new()
	sf.build(l, Vector3(-half, -relief, -half), Vector3(half, relief + 1.4, half), 128)
	sf.add_box(Vector3.ZERO, Vector3.UP, Vector3.RIGHT, Vector3.BACK, 0.42, 0.9)
	var texel := 2.0 * ((Vector3(half, relief + 1.4, half) - Vector3(-half, -relief, -half)).length()
		* 0.5 + 0.5) / 128.0
	# FIND the shadow first - where it lands depends on the sun's azimuth and elevation - then
	# walk a line out of it and measure the 10%-90% transition width.
	var lit := 1.0
	var dark := ShadowField.SHADOW_MIN
	var best := Vector3.ZERO
	var deepest := 1.0
	for gy in 121:
		for gx in 121:
			var q := Vector3(-3.0 + float(gx) * 0.05, 0.0, -3.0 + float(gy) * 0.05)
			var v := sf.factor(q)
			if v < deepest:
				deepest = v
				best = q
	var s10 := -1.0
	var s90 := -1.0
	for k in 2000:
		var s := float(k) * 0.002
		var v := sf.factor(best + Vector3(1.0, 0.0, 0.0) * s)
		var f := (lit - v) / maxf(1e-6, lit - dark)    # 1 fully shadowed .. 0 lit
		if s90 < 0.0 and f <= 0.90:
			s90 = s
		if s10 < 0.0 and f <= 0.10:
			s10 = s
			break
	var width := (s10 - s90) if (s10 >= 0.0 and s90 >= 0.0) else -1.0
	print("   darkest %.3f at %s (floor %.2f); 90%%-10%% edge width %.4f world = %.2f texels"
		% [deepest, best, dark, width, width / texel])
	_check(deepest < dark + 0.06, "the line really crosses the shadow (%.3f)" % deepest)
	_check(width > 1.4 * texel, "the edge is a ramp, not a step (%.2f texels > 1.4)" % (width / texel))


# --- a round occluder must not cast a square ---------------------------------------------------
func _round_silhouette() -> void:
	print("ROUND silhouette (a crown is a ball; its bounding box is a square)")
	var half := 4.25
	var relief := 1.5
	# Sun straight overhead, so the silhouette lands directly under the occluder and its corners
	# are where the square would show.
	var l := Vector3(0.02, 1.0, 0.02).normalized()
	var r := 0.6
	var res := []
	for round_ in [false, true]:
		var sf := ShadowField.new()
		sf.build(l, Vector3(-half, -relief, -half), Vector3(half, relief + 1.4, half), 128)
		sf.add_box(Vector3(0, 1.0, 0), Vector3.UP, Vector3.RIGHT, Vector3.BACK, r, 0.2, round_)
		var mid := sf.factor(Vector3(0.0, 0.0, 0.0))
		var corner := sf.factor(Vector3(r * 0.80, 0.0, r * 0.80))     # inside the square, outside the disc
		res.append([mid, corner])
		print("   round=%s   centre %.3f   corner %.3f" % [round_, mid, corner])
	var sq: Array = res[0]
	var rd: Array = res[1]
	_check(sq[1] < ShadowField.SHADOW_MIN + 0.08, "a SQUARE silhouette does shadow its corner (%.3f)" % sq[1])
	_check(rd[0] < ShadowField.SHADOW_MIN + 0.08, "a ROUND silhouette still shadows its centre (%.3f)" % rd[0])
	_check(rd[1] > sq[1] + 0.15, "a ROUND silhouette leaves the corner lit (%.3f > %.3f)" % [rd[1], sq[1]])
