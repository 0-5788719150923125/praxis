extends Scene3D

## Terrain city - blocks rising as a city over real 3D terrain, growing nonlinearly.
##
## The metropolis idea on the [Terrain] foundation: a [Swarm] development field creeps
## across a landscape (rolling hills / mesa), and where it has grown, blocks stand on the
## surface - **upright** (real buildings are vertical whatever the ground does), with only a
## faint lean toward the terrain normal so the field is not a perfectly rigid grid. Heights are driven
## by development x a per-block spectral band (nonlinear), so the skyline rises with the
## music. Some plots **detach**, their blocks floating a little off the ground. Camera
## orbits under a wide lens; the city grows over time from a few seeds.

const C := 30                    # city grid is C x C plots

var _f: AudioFeatures = AudioFeatures.new()
var _terrain: Terrain
var _dev: Swarm
var _detach := PackedFloat32Array()   # per-plot float-off height (0 = grounded)
var _thresh := PackedFloat32Array()   # per-plot development level needed before a building rises
var _grown := PackedFloat32Array()    # per-plot BUILT height 0..1, eased up over time (starts small)
var _foot := PackedFloat32Array()     # per-plot footprint scale (skewed: many small, a few big anchors)
var _hclass := PackedFloat32Array()   # per-plot MAX height potential (reached only at critical mass)
var _sky := PackedFloat32Array()      # per-plot "tower from the start" propensity (rare, for variety)
var _phase := PackedFloat32Array()    # per-plot phase for the slow rearrange wobble
var _maturity := 0.0                  # 0..1, rises over the scene: thresholds drop -> arms thicken, gaps fill
var _shadow := ShadowField.new()      # light-space cast-shadow map (buildings shadow ground + each other)
var _cores: Array = []                # the 1-2 valley cells the city grows out from (re-pinned each frame)
var _light_az := 0.0
var _light_el := 0.5
var _light_dir := 1.0
var _hue := 0.0
var _glow := 0.0
var _yaw := 0.0
var _dist := 7.5
var _pitch := 0.5
var _beat_prev := 0.0


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "field"
	var ttype := "mesa" if rng.randf() < 0.5 else "hills"
	_terrain = Terrain.new()
	_terrain.build(rng, ttype, 3.0, rng.randf_range(0.35, 0.55), null,
		"verdant" if rng.randf() < 0.5 else "temperate")
	_hue = fposmod(rng.randf() + 0.5, 1.0)
	# The city grows from just ONE or TWO cores, each seeded in a LOW valley (easy ground): sample a
	# handful of central cells and keep the lowest. Development creeps outward from there; the cores
	# are re-pinned every frame so the origin never fades.
	_dev = Swarm.new(C, C, Swarm.GROW, rng, 0)
	for k in rng.randi_range(1, 2):
		var bx := C / 2
		var by := C / 2
		var blo := 1e9
		for _t in 10:
			var cx := rng.randi_range(int(C * 0.28), int(C * 0.72))
			var cy := rng.randi_range(int(C * 0.28), int(C * 0.72))
			var hh: float = _terrain.height_at(_plot_wx(cx), _plot_wz(cy))
			if hh < blo:
				blo = hh
				bx = cx
				by = cy
		_cores.append(Vector2i(bx, by))
		_dev.inject(bx, by, 1.0)
	# Per-plot BUILD THRESHOLD, biased by TERRAIN HEIGHT: valleys (low, easy ground) need only a
	# little development to build, so they fill FIRST; hillsides need more; peaks stay bare. Plus a
	# little noise so the frontier is ragged, not a clean contour.
	_thresh.resize(C * C)
	_grown.resize(C * C)
	_foot.resize(C * C)
	_hclass.resize(C * C)
	_sky.resize(C * C)
	_phase.resize(C * C)
	# A ridged "arm" field: its branching high ridges become the channels the city builds ALONG, so
	# development reads as dendritic ARMS reaching out from the core rather than a filled blob.
	var armf := Field.make("ridged", rng.randi(), rng.randf_range(1.5, 2.4), 3)
	for cy in C:
		for cx in C:
			var i := cy * C + cx
			var p := Vector2(float(cx) / float(C - 1) - 0.5, float(cy) / float(C - 1) - 0.5) * 2.0
			var elev: float = clampf(_terrain.height_at(_plot_wx(cx), _plot_wz(cy)), 0.0, 1.0)
			var arm: float = armf.at(p)
			# Threshold = valley bias + an OFF-ARM penalty. On an arm ridge in a valley: builds first.
			# Between the arms / up the hills: needs far more development (fills only as the city matures).
			# Valley bias + a GENTLE off-arm bias: the arm ridges build a bit sooner so the frontier
			# reaches out in dendritic arms, but off-arm plots still fill in as the base matures (this
			# is a preference, not a hard gate - a hard gate starved a rugged map of any city at all).
			# Threshold is dominated by the ridged ARM field: only cells ON a branching ridge of that
			# field can build (a strong OFF-ridge penalty keeps the rest bare whatever the development),
			# so the city is a sparse DENDRITIC network of arms rather than a solid blob. Elevation adds
			# a gentle bias (lower ground a touch likelier); noise ragged-ifies the frontier.
			_thresh[i] = 0.05 + 0.2 * elev + (1.0 - smoothstep(0.46, 0.64, arm)) * 0.9 + rng.randf_range(-0.03, 0.05)
			_grown[i] = 0.0
			# Footprint + MAX-height potential: a skewed distribution - most plots modest, a rare few are
			# big anchors. But this height is only ever REALISED once the district hits critical mass (see
			# the draw): blocks start SMALL and grow taller as their surroundings develop.
			var big := pow(rng.randf(), 3.2)                             # strongly skewed: MOST plots small
			_foot[i] = 0.5 + 1.0 * big
			_hclass[i] = 0.9 + 2.5 * big + rng.randf_range(-0.1, 0.3)     # tall POTENTIAL (a rare few big)
			# A rare few plots are skyscrapers FROM THE START (variety) - most are 0 (grow up naturally).
			_sky[i] = rng.randf_range(0.55, 1.0) if rng.randf() < 0.06 else 0.0
			_phase[i] = rng.randf() * TAU
	# Per-plot detach: a few districts float off the ground.
	_detach.resize(C * C)
	for i in C * C:
		_detach[i] = rng.randf_range(0.10, 0.30) if rng.randf() < 0.12 else 0.0
	lens.fov = rng.randf_range(56.0, 72.0)
	_dist = rng.randf_range(6.5, 8.5)
	_pitch = rng.randf_range(0.34, 0.55)
	_yaw = rng.randf() * TAU
	# A low key light so the mountains (and the skyline) cast long, gently sweeping shadows.
	_light_az = rng.randf() * TAU
	_light_dir = 1.0 if rng.randf() < 0.5 else -1.0
	_light_el = rng.randf_range(0.34, 0.52)
	_terrain.set_light(_light_az, _light_el)
	return {"type": ttype}


func _plot_wx(cx: int) -> float:
	return (float(cx) / float(C - 1) - 0.5) * 2.0 * _terrain.half


func _plot_wz(cy: int) -> float:
	return (float(cy) / float(C - 1) - 0.5) * 2.0 * _terrain.half


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.012, 0.018)
	# BURST then decay: growth is FAST in the opening seconds and eases to a slow crawl after. A scene's
	# hold is drive-scaled and can be short (a few seconds on an energetic passage), so front-loading the
	# growth means even a brief scene BURSTS into a small city up front; a long one keeps creeping after.
	# `_life` is the scene's age (starts ~0.6 after the pre-warm, which already banks some burst growth).
	# A quick initial POP (a handful of blocks fast, so even a short scene isn't empty), decaying sharply
	# to a SLOW crawl - the city keeps developing gently for the rest of the scene, never filling all at
	# once. Fast decay (~1s) so the burst is a brief opener, not a fill.
	var burst: float = 1.0 + 3.5 * exp(-_life * 0.7)
	# Nonlinear growth drive: beats lunge development outward through a spike curve, bursting at the start.
	var drive := (1.0 + 1.1 * Nonlinear.apply("spike", clampf(0.7 * f.energy + f.beat, 0.0, 1.0), 2.0)) * (0.7 + 0.3 * burst)
	for core in _cores:
		_dev.inject(core.x, core.y, 1.0)          # keep the origin cores alive
	_dev.step(drive, delta, 0.015)
	# The city matures over the scene: the effective thresholds drop, so the arms THICKEN a little - but
	# kept modest so it stays SPARSE and dendritic (bands/arms reaching along the terrain), never a solid
	# filled blob. Slow BASE rate (the burst supplies the opening pop; the rest is a gentle creep).
	_maturity = minf(1.0, _maturity + delta * 0.05 * burst)
	# Ease each plot's BUILT height up toward its current maturity, so buildings START SMALL and grow
	# taller as their district matures - and the densest (most-developed) plots grow tallest.
	var rise := delta * 0.35 * burst
	for cy in C:
		for cx in C:
			var i := cy * C + cx
			var thr: float = maxf(0.04, float(_thresh[i]) - _maturity * 0.1)
			var target: float = clampf((_dev.at(cx, cy) - thr) / maxf(0.05, 1.0 - thr), 0.0, 1.0)
			_grown[i] = move_toward(float(_grown[i]), target, rise)
	_glow = lerpf(_glow, clampf(0.3 * f.energy + 0.5 * f.beat, 0.0, 1.0), 1.0 - exp(-5.0 * delta))
	_yaw += delta * (0.08 + 0.16 * f.energy)
	lens.orbit(Vector3(0.0, _terrain.relief * 0.25, 0.0), _dist, _yaw, _pitch + 0.04 * sin(_life * 0.13))
	# Drift the key light and refresh the terrain's sweeping cast shadows.
	_light_az += delta * 0.035 * _light_dir
	_terrain.set_light(_light_az, _light_el)
	_terrain.step_light(delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	lens.prepare()
	var u := unit()
	var lit := clampf(0.7 + 0.4 * _glow + 0.3 * _f.energy, 0.4, 1.4)
	texture_repeat = CanvasItem.TEXTURE_REPEAT_ENABLED
	var bw := _terrain.half / float(C) * 0.62          # block half-footprint (world)
	var bgain := 0.5 + 0.3 * _f.energy
	# How deep each building is sunk INTO the terrain: its box starts below the surface and the merged
	# land hides that buried part, so the visible base is ragged (cut by the ground), never a clean line.
	var embed: float = 0.35 * _terrain.relief + 0.14
	# The buildings are EMBEDDED below the surface, so while the scene fades IN (partial alpha) the
	# semi-transparent terrain would let the buried geometry show through - reading as blocks under the
	# ground. So the buildings only appear once the terrain is nearly opaque: hidden through the fade,
	# then eased in over its last stretch (`presence` is the scene's transition opacity, 1 when settled).
	var reveal: float = smoothstep(0.8, 1.0, view.presence)
	# Pass A: compute every building and RASTERIZE it into the light-space shadow map. This must finish
	# before anything is shaded, since a building can cast a shadow on the ground and on other buildings.
	_shadow.build(_terrain.light_dir(), Vector3(-_terrain.half, -_terrain.relief, -_terrain.half),
		Vector3(_terrain.half, _terrain.relief + 3.0, _terrain.half))
	var blds: Array = []
	for cy in C:
		for cx in C:
			if reveal < 0.02:                               # still fading the terrain in - no buildings yet
				break
			var i := cy * C + cx
			var grown: float = _grown[i]
			if grown < 0.02:                                # nothing built here yet (a gap / bare peak)
				continue
			var wx := (float(cx) / float(C - 1) - 0.5) * 2.0 * _terrain.half
			var wz := (float(cy) / float(C - 1) - 0.5) * 2.0 * _terrain.half
			var ground := _terrain.height_at(wx, wz) * _terrain.relief
			var float_off: float = _detach[i]
			var dev := _dev.at(cx, cy)
			var react := _f.sample(clampf(_terrain.height_at(wx, wz) + 0.5, 0.0, 1.0))
			# A slow per-plot wobble keeps the built skyline REARRANGING over time.
			var wob := 0.85 + 0.28 * sin(_life * 0.12 + float(_phase[i]))
			# CRITICAL MASS: how developed (local density `dev`) AND mature the district here is. A block
			# is BUILT small, then only grows toward its tall potential as its surroundings fill in and
			# the city matures - so most stay low (many small blocks) and TOWERS emerge later, in the
			# dense core. Nonlinear (holds low, then surges past the threshold) so the transformation
			# reads as natural, not linear. A rare few (`_sky`) tower from the start for variety.
			var crit := clampf(dev * lerpf(0.28, 1.0, _maturity), 0.0, 1.0)
			var realize := clampf(maxf(Nonlinear.apply("spike", crit, 2.4), float(_sky[i])), 0.0, 1.0)
			var tall := lerpf(0.5, float(_hclass[i]), realize)   # small base .. this plot's full potential
			var h := grown * tall * (0.42 + 0.4 * bgain + 0.5 * react) * wob
			var bw_i := bw * float(_foot[i])
			var up := _terrain.normal_world(wx, wz).lerp(Vector3.UP, 0.92).normalized()
			var bx := up.cross(Vector3(1, 0, 0))
			if bx.length() < 1e-3:
				bx = up.cross(Vector3(0, 0, 1))
			bx = bx.normalized()
			var bz := bx.cross(up).normalized()
			# Sink the base BELOW the surface so its bottom is buried; the top stays where it was.
			var base := Vector3(wx, ground + float_off - embed, wz)
			var htot := h + embed
			var hue := fposmod(_hue + 0.12 * _terrain.height_at(wx, wz) + 0.25 * dev, 1.0)
			# The TERRAIN also shadows the building (a block in a hill's cast shadow darkens).
			var tsh: float = _terrain.shadow_at(wx, wz)
			var blit := clampf(0.18 + 0.5 * dev + 0.5 * react + 0.6 * _glow, 0.05, 1.2) * lit * (0.35 + 0.65 * tsh)
			var ext := _shadow.add_box(base, up, bx, bz, bw_i, htot)   # rasterize + get self-shadow bias
			blds.append({"base": base, "up": up, "bx": bx, "bz": bz, "w": bw_i, "h": htot,
				"hue": hue, "lit": blit, "ext": ext})

	# ONE merged list: terrain quads (shadowed by the buildings) + every building face (per-vertex
	# shadowed, so a neighbour's shadow LAYERS onto the block), depth-sorted together so the land also
	# occludes the buried building bases. Then draw per entry type.
	var faces: Array = _terrain.collect_surface(lens, u, lit, _life, _shadow)
	for b in blds:
		_block_faces(faces, b.base, b.up, b.bx, b.bz, b.w, b.h, b.hue, b.lit, float(b.ext), reveal)
	faces.sort_custom(func(a, b): return a.d > b.d)
	var tex := Terrain.detail_texture()
	for fc in faces:
		if fc.has("uvs"):
			Terrain.draw_quad(self, fc.poly, fc.cols, fc.uvs, tex)   # terrain land
		else:
			Terrain.draw_quad(self, fc.poly, fc.cols)                # building face / water


# Append the camera-facing faces of one oriented box to `out` (each {poly, cols, d}), per-VERTEX
# shaded by the key light AND the cast-shadow map - so a neighbour's shadow lands as a real band on
# the wall. `ext` is this box's own light-depth extent (its self-shadow bias).
func _block_faces(out: Array, base: Vector3, up: Vector3, bx: Vector3, bz: Vector3, w: float, h: float,
		hue: float, lit: float, ext: float, alpha := 1.0) -> void:
	var top := base + up * h
	# 4 base + 4 top corners.
	var corners := [
		base - bx * w - bz * w, base + bx * w - bz * w, base + bx * w + bz * w, base - bx * w + bz * w,
		top - bx * w - bz * w, top + bx * w - bz * w, top + bx * w + bz * w, top - bx * w + bz * w]
	var pr: Array = []
	for cwld in corners:
		pr.append(_terrain_proj(cwld))
	# Faces as index quads + outward normal direction; draw only those facing the camera.
	var quads := [[4, 5, 6, 7, up],                     # top
		[0, 1, 5, 4, -bz], [1, 2, 6, 5, bx], [2, 3, 7, 6, bz], [3, 0, 4, 7, -bx]]
	for q in quads:
		var i0: int = q[0]
		var i1: int = q[1]
		var i2: int = q[2]
		var i3: int = q[3]
		var fn: Vector3 = q[4]
		var fc: Vector3 = (corners[i0] + corners[i1] + corners[i2] + corners[i3]) * 0.25
		if fn.dot(lens.eye - fc) <= 0.0:                 # facing away
			continue
		var p0: Vector3 = pr[i0]
		var p1: Vector3 = pr[i1]
		var p2: Vector3 = pr[i2]
		var p3: Vector3 = pr[i3]
		if p0.z <= lens.near or p1.z <= lens.near or p2.z <= lens.near or p3.z <= lens.near:
			continue
		var fpoly := PackedVector2Array([Vector2(p0.x, p0.y), Vector2(p1.x, p1.y),
			Vector2(p2.x, p2.y), Vector2(p3.x, p3.y)])
		if Terrain._quad_area(fpoly) < 1.0:        # edge-on face - skip (else triangulation fails)
			continue
		# Strong directional key light: the sunward faces are bright, the faces turned away fall into
		# real shade - the contrast is what makes a block read as a lit SOLID instead of a flat card.
		var shade := 0.34 + 0.72 * clampf(fn.dot(_terrain.light_dir()), 0.0, 1.0)
		# Per-corner CAST shadow from the field (bias by this box's own depth extent so it doesn't
		# shadow itself, but a taller neighbour's shadow still bands across it).
		var cols := PackedColorArray()
		for idx in [i0, i1, i2, i3]:
			var sf: float = _shadow.factor(corners[idx], ext + 0.06)
			var c := Color.from_hsv(hue, 0.45, clampf(lit * shade * sf, 0.0, 1.0))
			c.a = alpha                                  # fade the buildings in AFTER the terrain (embed reveal)
			cols.append(c)
		out.append({"d": (p0.z + p1.z + p2.z + p3.z) * 0.25, "poly": fpoly, "cols": cols})


# Project a world point to (screen x, screen y, camera depth).
func _terrain_proj(wld: Vector3) -> Vector3:
	var pr := lens.project(wld)
	return Vector3(pr.x * unit(), pr.y * unit(), pr.z)
