extends Scene3D

## Spires - a recursive fractal metropolis whose skyline IS the spectrum.
##
## The prototype for "periodic, monotonic fractal geometry over a landscape": a dense grid of
## ornate towers stands on real [Terrain], each one a self-similar STACK of set-back tiers capped
## by a spire and studded with corner turrets (which are little echoes of the whole - the fractal
## recursion). Every constant is sampled per instance ([method _gen_tower]) with a per-vertex
## jitter, so no two towers - and no two tiers - are identical: the hand-built, irregular masonry
## of the reference, not a grid of clones.
##
## The whole structure is generated ONCE; the HARMONICS then animate it. Each tower is pinned to a
## position in the spectrum (radially: the centre is bass, the rim is treble), and its height grows
## nonlinearly with that band. The fine detail - turrets, spires, ledges - is gated on its OWN,
## higher band and only appears when that band is strong. So as the spectral DISTRIBUTION shifts,
## whole districts rise and tower while others collapse to stubs, and ornament blooms and vanishes -
## the city re-sculpts itself, drastically, to the music. That is the point being proved: extreme
## seeded 3D detail that reorganises with the sound.

const C := 10                    # city is C x C tower slots
const UP := Vector3(0, 1, 0)

var _f: AudioFeatures = AudioFeatures.new()
var _terrain: Terrain
var _towers: Array = []          # each: {base, up, bx, bz, t (spectral pos), nodes, hue, grow}
var _hue := 0.10                 # brass / gold
var _glow := 0.0
var _yaw := 0.0
var _dist := 8.0
var _pitch := 0.42
var _yaw_dir := 1.0
var _light_az := 0.0
var _light_el := 0.5
var _light_dir := 1.0
var _u := 1.0                    # cached unit() scale, for _emit_frustum's screen projection
var _ao_span := 2.6              # height over which the vertical ambient gradient runs (street->sky)
var _fog_near := 6.0             # camera depth where distance fog begins
var _fog_far := 20.0             # camera depth where fog is thickest
var _fog_col := Color(0.05, 0.06, 0.09)   # the deep haze the far spires dissolve into


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "field"
	_terrain = Terrain.new()
	# A gentle, low landscape - the towers are the subject; the ground just grounds them.
	var ttype: String = "hills" if rng.randf() < 0.6 else "mesa"
	_terrain.build(rng, ttype, 3.0, rng.randf_range(0.4, 0.6), null,
		"temperate" if rng.randf() < 0.5 else "tundra")
	_hue = rng.randf_range(0.075, 0.12)                 # warm brass through gold
	# Place a tower on every slot, denser toward the centre. Each is pinned to a spectral position
	# by its RADIUS (centre = bass, rim = treble), so the skyline reads as the spectrum in the round.
	for gy in C:
		for gx in C:
			var wx := _slot(gx) + rng.randf_range(-0.06, 0.06)
			var wz := _slot(gy) + rng.randf_range(-0.06, 0.06)
			var ground: float = _terrain.height_at(wx, wz) * _terrain.relief
			var up: Vector3 = _terrain.normal_world(wx, wz).lerp(UP, 0.94).normalized()
			var bx: Vector3 = up.cross(Vector3(1, 0, 0))
			if bx.length() < 1e-3:
				bx = up.cross(Vector3(0, 0, 1))
			bx = bx.normalized()
			var bz: Vector3 = bx.cross(up).normalized()
			var cxn := float(gx) / float(C - 1) * 2.0 - 1.0
			var cyn := float(gy) / float(C - 1) * 2.0 - 1.0
			var t := clampf(sqrt(cxn * cxn + cyn * cyn) / 1.415 + rng.randf_range(-0.06, 0.06), 0.0, 1.0)
			# Bulk varies a lot per tower: broad palatial masses through slender spires, so the city
			# is a mix of forms (not a field of identical needles).
			var bulk := rng.randf_range(0.4, 1.0)
			var w := _terrain.half / float(C) * (0.42 + 0.7 * bulk)
			var height := rng.randf_range(1.0, 2.3) * (1.3 - 0.4 * bulk)   # broad ones a little shorter
			var nodes: Array = []
			_gen_tower(nodes, Vector3(wx, ground, wz), up, bx, bz, w, height, rng, t)
			_towers.append({"base": Vector3(wx, ground, wz), "up": up, "bx": bx, "bz": bz,
				"t": t, "nodes": nodes, "hue": fposmod(_hue + rng.randf_range(-0.02, 0.02), 1.0),
				"grow": 0.0})
	lens.fov = rng.randf_range(46.0, 56.0)
	_dist = rng.randf_range(10.0, 13.0)                  # back far enough to read the whole skyline
	_pitch = rng.randf_range(0.44, 0.64)                # an elevated 3/4 establishing angle
	_yaw = rng.randf() * TAU
	_yaw_dir = 1.0 if rng.randf() < 0.5 else -1.0
	_light_az = rng.randf() * TAU
	_light_dir = 1.0 if rng.randf() < 0.5 else -1.0
	_light_el = rng.randf_range(0.34, 0.5)
	_terrain.set_light(_light_az, _light_el)
	return {"type": ttype}


func _slot(g: int) -> float:
	return (float(g) / float(C - 1) - 0.5) * 2.0 * _terrain.half * 0.92


# The drive at spectral position `tt`: the LIVE band, or - when the track is quiet / absent - a slow
# travelling wave across the spectrum, so the city is always alive (districts swelling and ebbing in
# sequence) and the harmonic-driven morph reads even in the idle preview. Live audio overrides it.
func _band(tt: float) -> float:
	var live: float = _f.sample(tt)
	var idle := 0.30 + 0.34 * sin(_life * 0.4 + tt * 8.0)
	return clampf(maxf(live, idle * 0.85), 0.0, 1.0)


# Generate one tower's fractal node list (positions/sizes are FULL-height; the draw scales them by
# the live harmonics). A stack of set-back tiers, each a frustum; upper tiers grow corner turrets
# (small echoes of the tower, each a tapered shaft + a spire) and thin ledges; a tall spire caps it.
# Tiers are STRUCTURAL (present whenever the tower grows); ornament is a LEAF gated on a higher band.
func _gen_tower(nodes: Array, base: Vector3, up: Vector3, bx: Vector3, bz: Vector3,
		w: float, height: float, rng: RandomNumberGenerator, t: float) -> void:
	var tiers := rng.randi_range(3, 5)
	var cur := base
	var cw := w
	var th := height / float(tiers)
	var twist := rng.randf_range(-0.35, 0.35)
	var setback := rng.randf_range(0.80, 0.94)                       # per-tower: how fast it narrows
	for i in tiers:
		var tw := cw * setback * rng.randf_range(0.96, 1.04)          # set-back: each tier narrower
		var dtw := rng.randf_range(-0.28, 0.28)
		nodes.append({"leaf": false, "c": cur, "w0": cw, "w1": tw, "h": th * rng.randf_range(0.9, 1.15),
			"tw": twist, "dtw": dtw, "hoff": rng.randf_range(-0.015, 0.02), "vb": rng.randf_range(0.82, 1.0),
			"band": t, "thr": 0.0, "seed": rng.randf() * TAU})
		# A thin overhanging ledge at the tier's shoulder - ornament.
		if rng.randf() < 0.6:
			nodes.append({"leaf": true, "c": cur, "w0": cw * 1.14, "w1": cw * 1.14, "h": th * 0.07,
				"tw": twist, "dtw": 0.0, "hoff": 0.0, "vb": 0.7, "band": clampf(t + 0.05, 0.0, 1.0),
				"thr": 0.14, "seed": rng.randf() * TAU})
		# Corner turrets on the upper tiers - little self-similar echoes, each a shaft + a spire.
		if i >= tiers - 2 and rng.randf() < 0.72:
			var top := cur + up * th
			for sx in [-1.0, 1.0]:
				for sz in [-1.0, 1.0]:
					var off: Vector3 = bx * (tw * 0.86 * float(sx)) + bz * (tw * 0.86 * float(sz))
					var tcw := cw * rng.randf_range(0.15, 0.26)
					var tch := th * rng.randf_range(1.0, 1.9)
					var band := clampf(t + rng.randf_range(0.08, 0.22), 0.0, 1.0)   # finer detail = higher band
					var thr := rng.randf_range(0.22, 0.48)
					nodes.append({"leaf": true, "c": top + off, "w0": tcw, "w1": tcw * 0.82, "h": tch,
						"tw": rng.randf() * TAU, "dtw": rng.randf_range(-0.2, 0.2), "hoff": rng.randf_range(-0.02, 0.03),
						"vb": 0.92, "band": band, "thr": thr, "seed": rng.randf() * TAU})
					nodes.append({"leaf": true, "c": top + off + up * tch, "w0": tcw * 0.82, "w1": tcw * 0.05,
						"h": tch * rng.randf_range(0.8, 1.4), "tw": 0.0, "dtw": 0.0, "hoff": rng.randf_range(0.0, 0.04),
						"vb": 1.0, "band": band, "thr": clampf(thr + 0.05, 0.0, 1.0), "seed": rng.randf() * TAU})
		cur = cur + up * th
		cw = tw
		twist += dtw
	# The crowning spire (short enough that the tower reads as a building with a peak, not a needle).
	nodes.append({"leaf": true, "c": cur, "w0": cw, "w1": cw * 0.04, "h": height * rng.randf_range(0.22, 0.5),
		"tw": twist, "dtw": 0.0, "hoff": rng.randf_range(0.0, 0.05), "vb": 1.0,
		"band": clampf(t + 0.12, 0.0, 1.0), "thr": 0.18, "seed": rng.randf() * TAU})


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.01, 0.015)
	_glow = Nonlinear.flare(_glow, clampf(0.3 * f.energy + 0.7 * f.beat, 0.0, 1.0), delta, 9.0, 1.6)
	_yaw += delta * (0.05 + 0.12 * f.energy) * _yaw_dir
	lens.orbit(Vector3(0.0, 0.9, 0.0), _dist, _yaw, _pitch + 0.03 * sin(_life * 0.11))
	# Each tower eases toward the height its OWN spectral band commands (nonlinear): weak band -> a
	# stub, strong band -> a soaring tower, so the skyline is the spectrum in the round and re-sculpts
	# itself as the distribution shifts.
	var ease := 1.0 - exp(-3.5 * delta)
	for tower in _towers:
		var target: float = clampf(0.14 + 1.35 * Nonlinear.apply("spike", _band(float(tower.t)), 1.4), 0.08, 1.5)
		tower.grow = lerpf(float(tower.grow), target, ease)
	_light_az += delta * 0.03 * _light_dir
	_terrain.set_light(_light_az, _light_el)
	_terrain.step_light(delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	lens.prepare()
	var u := unit()
	_u = u
	var lit := clampf(0.6 + 0.4 * _glow + 0.3 * _f.energy, 0.4, 1.4)
	_terrain.draw_surface(self, lens, u, lit, _life)

	var ldir: Vector3 = _terrain.light_dir()
	var wob := _life * 1.3                                # the per-vertex jitter animates slowly
	# Distance-fog range from the camera, so far spires dissolve into haze (depth + atmosphere).
	_fog_near = _dist * 0.45
	_fog_far = _dist * 2.3
	var faces: Array = []
	for tower in _towers:
		var vg: float = float(tower.grow)
		if vg < 0.05:
			continue
		var base: Vector3 = tower.base
		var up: Vector3 = tower.up
		var bx: Vector3 = tower.bx
		var bz: Vector3 = tower.bz
		var thue: float = float(tower.hue)
		for nd in tower.nodes:
			# Ornament grows in only when its own (higher) band is strong; tiers ride the tower grow.
			var bandv: float = _band(float(nd.band))
			var ng := 1.0
			if bool(nd.leaf):
				ng = smoothstep(float(nd.thr), float(nd.thr) + 0.12, bandv)
				if ng < 0.03:
					continue
			var cc: Vector3 = nd.c
			var by: float = base.y + (cc.y - base.y) * vg          # vertical scale about the tower base
			var bc := Vector3(cc.x, by, cc.z)
			var hh: float = float(nd.h) * vg * ng
			if hh < 0.004:
				continue
			var wsc: float = (0.35 + 0.65 * ng)                    # ornament also thins as it grows in
			var jit := 0.05 + 0.05 * sin(wob + float(nd.seed))     # per-node vertex wobble (irregular masonry)
			# Per-node colour with real VARIATION (not one flat gold): hue and saturation both wander a
			# little per node, and weathered nodes read duller - so the material breaks up.
			var seedf := float(nd.seed)
			var val: float = clampf(float(nd.vb) * (0.8 + 0.5 * bandv + 0.5 * _glow) * lit, 0.2, 1.5)
			var sat: float = clampf(0.44 + 0.16 * sin(seedf * 2.3) + 0.1 * bandv, 0.2, 0.7)
			var col := Color.from_hsv(fposmod(thue + float(nd.hoff) + 0.03 * sin(seedf), 1.0), sat, val)
			_emit_frustum(faces, bc, up, bx, bz, float(nd.w0) * wsc, float(nd.w1) * wsc, hh,
				float(nd.tw), float(nd.dtw), jit, seedf, col, ldir, base.y)
	faces.sort_custom(func(a, b): return a.d > b.d)          # far first
	for fc in faces:
		Terrain.draw_quad(self, fc.poly, fc.cols)


# Four corners of a square (half-width w) in the up-perpendicular plane, twisted by `ang`, with a
# small per-corner radial jitter so the shaft is subtly irregular (not a machined box).
func _square(center: Vector3, bx: Vector3, bz: Vector3, w: float, ang: float, jit: float, seed: float) -> Array:
	var ca := cos(ang)
	var sa := sin(ang)
	var ax := bx * ca + bz * sa
	var az := bz * ca - bx * sa
	var out: Array = []
	var dirs := [Vector2(-1, -1), Vector2(1, -1), Vector2(1, 1), Vector2(-1, 1)]
	for k in 4:
		var jw: float = w * (1.0 + jit * sin(seed + float(k) * 1.7))
		out.append(center + ax * (dirs[k].x * jw) + az * (dirs[k].y * jw))
	return out


# Emit the camera-facing side + top faces of one frustum (a tapering box: base half-width w0, top
# w1, height h, base twist `tw`, top twist tw+dtw) into `out`, shaded by the key light.
func _emit_frustum(out: Array, base_c: Vector3, up: Vector3, bx: Vector3, bz: Vector3,
		w0: float, w1: float, h: float, tw: float, dtw: float, jit: float, seed: float,
		col: Color, ldir: Vector3, ao_lo: float) -> void:
	var top_c := base_c + up * h
	var b := _square(base_c, bx, bz, w0, tw, jit, seed)
	var tp := _square(top_c, bx, bz, maxf(w1, 0.0008), tw + dtw, jit, seed + 1.3)
	var quads := [[b[0], b[1], tp[1], tp[0]], [b[1], b[2], tp[2], tp[1]],
		[b[2], b[3], tp[3], tp[2]], [b[3], b[0], tp[0], tp[3]]]
	if w1 > 0.02 * w0:
		quads.append([tp[0], tp[1], tp[2], tp[3]])           # a flat roof (spires taper to a point -> skip)
	for q in quads:
		var fc: Vector3 = (q[0] + q[1] + q[2] + q[3]) * 0.25
		var nrm: Vector3 = (q[1] - q[0]).cross(q[3] - q[0])
		if nrm.dot(lens.eye - fc) <= 0.0:                    # facing away from the camera
			continue
		var pr := [lens.project(q[0]), lens.project(q[1]), lens.project(q[2]), lens.project(q[3])]
		if pr[0].z <= lens.near or pr[1].z <= lens.near or pr[2].z <= lens.near or pr[3].z <= lens.near:
			continue
		var poly := PackedVector2Array([Vector2(pr[0].x, pr[0].y) * _u, Vector2(pr[1].x, pr[1].y) * _u,
			Vector2(pr[2].x, pr[2].y) * _u, Vector2(pr[3].x, pr[3].y) * _u])
		if Terrain._quad_area(poly) < 0.6:
			continue
		# Directional key light on the face, plus a per-vertex VERTICAL ambient gradient (dark down at
		# street level, brighter up toward the sky) and DISTANCE FOG (far spires dissolve into haze) -
		# so the towers gain form and depth instead of one flat wall of gold.
		var shade := 0.42 + 0.6 * clampf(nrm.normalized().dot(ldir), 0.0, 1.0)
		var cols := PackedColorArray()
		for k in 4:
			var ao := clampf((q[k].y - ao_lo) / _ao_span, 0.0, 1.0)
			var vb := shade * (0.5 + 0.55 * ao)
			var cc := Color(col.r * vb, col.g * vb, col.b * vb, 1.0)
			var fogt := smoothstep(_fog_near, _fog_far, pr[k].z) * 0.9
			cols.append(cc.lerp(_fog_col, fogt))
		out.append({"d": (pr[0].z + pr[1].z + pr[2].z + pr[3].z) * 0.25, "poly": poly, "cols": cols})
