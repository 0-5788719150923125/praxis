extends Scene3D

## Spires - a fractal metropolis of harmonic spires over a landscape.
##
## Each spire is a lofted PROFILE, not a stack of jittered boxes: its radius-as-a-function-of-height
## is a short series of cosine MODES. A GLOBAL set of mode amplitudes/frequencies is the city's shared
## architectural "style"; each spire carries a per-spire DEVIATION (amplitude scale + phase) so the
## towers are a family, not clones. The live SPECTRUM drives the modes (each mode is pinned to a band),
## so the silhouettes - tapers, bulges, tiers - re-sculpt with the music: the point being proved.
##
## Spires are lofted ring by ring from the GROUND UP and grow in over the intro (from the centre
## outward), so the city assembles itself rather than starting fully formed. The key light gives each
## face directional + vertical-ambient shading, drops a contact shadow on the ground, and distance fog
## dissolves the far skyline into haze.
##
## RENDERING follows the [FrameForge] contract: update() snapshots the frame's
## inputs into a [SpireJob] (a plain RefCounted - the worker's Callable keeps
## it alive, and it never touches the scene node), the job builds the whole
## frame - shadow pass, terrain merge, loft, sort, batch - off the main
## thread, and _draw() only submits the finished packet. The UI cannot block
## on this scene, whatever the city costs.

const C := 10                    # city is C x C spire slots
const UP := Vector3(0, 1, 0)
const SIDES := 6                 # facets around each spire
const LEVELS := 16               # profile samples up the height (rings are lofted between them)
const MODES := 4                 # cosine modes that define the radius-vs-height profile

var _f: AudioFeatures = AudioFeatures.new()
var _terrain: Terrain
var _spires: Array = []
# CITY PALETTES. The hue was a single hardcoded band (0.07-0.13), so every city
# ever generated was brass. Each entry is a whole material mood: the stone hue,
# how far individual towers drift from it, a saturation bias, and the colour of
# a lit window - which is what actually sells the material, since a window is
# the only self-lit thing in the scene.
const PALETTES := [
	{"name": "brass",     "hue": 0.10, "spread": 0.05, "sat": 0.00, "win": Color(0.95, 0.74, 0.36)},
	{"name": "verdigris", "hue": 0.44, "spread": 0.05, "sat": -0.08, "win": Color(0.98, 0.86, 0.52)},
	{"name": "sodium",    "hue": 0.06, "spread": 0.03, "sat": 0.10, "win": Color(1.00, 0.62, 0.24)},
	{"name": "glacier",   "hue": 0.55, "spread": 0.06, "sat": -0.12, "win": Color(0.78, 0.90, 1.00)},
	{"name": "ember",     "hue": 0.98, "spread": 0.04, "sat": 0.08, "win": Color(1.00, 0.52, 0.30)},
	{"name": "bone",      "hue": 0.09, "spread": 0.03, "sat": -0.22, "win": Color(0.92, 0.88, 0.74)},
	{"name": "violet",    "hue": 0.74, "spread": 0.06, "sat": -0.04, "win": Color(0.86, 0.72, 1.00)},
]

var _hue := 0.10                 # the chosen palette's stone hue
var _sat_bias := 0.0
var _win_col := Color(0.95, 0.74, 0.36)
var _palette_name := "brass"
var _glow := 0.0
var _yaw := 0.0
var _dist := 8.0
var _pitch := 0.42
var _yaw_dir := 1.0
var _light_az := 0.0
var _light_el := 0.5
var _light_dir := 1.0
var _maturity := 0.0             # 0..1, the slow structural growth of the whole city
var _forge := FrameForge.new()   # off-thread frame builder (see the class doc)
# The GLOBAL profile style, shared by every spire (so the city speaks one architectural language).
var _mode_freq: Array = []       # half-wave count up the height, per mode
var _mode_amp: Array = []        # base amplitude, per mode
var _mode_pos: Array = []        # spectral position each mode is driven by


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "field"
	_terrain = Terrain.new()
	# A gentle, low landscape - the spires are the subject; the ground just grounds them.
	var ttype: String = "hills" if rng.randf() < 0.6 else "mesa"
	_terrain.build(rng, ttype, 3.0, rng.randf_range(0.4, 0.6), null,
		"temperate" if rng.randf() < 0.5 else "tundra")
	var pal: Dictionary = PALETTES[rng.randi() % PALETTES.size()]
	_palette_name = String(pal["name"])
	_hue = fposmod(float(pal["hue"]) + rng.randf_range(-0.02, 0.02), 1.0)
	_sat_bias = float(pal["sat"])
	_win_col = pal["win"]
	var spread := float(pal["spread"])

	# LAYOUT VARIETY. Every slot always got a spire and the height law was the
	# same each time, so the footprint and the skyline were identical from seed
	# to seed - only the noise under them changed. `density` opens plazas and
	# gaps; `core` decides whether the tall towers cluster downtown (positive),
	# ring the edge (negative), or spread evenly (near zero).
	var density := rng.randf_range(0.62, 1.0)
	var core := rng.randf_range(-0.55, 0.9)
	# The shared profile modes: low modes shape the broad form, higher modes add fine tiers. Each is
	# pinned to a spectral band, so the whole city's silhouette morphs coherently as the sound shifts.
	for k in MODES:
		_mode_freq.append(float(k + 1) * rng.randf_range(1.1, 1.7))
		_mode_amp.append(rng.randf_range(0.16, 0.40) / sqrt(float(k + 1)))
		_mode_pos.append(clampf(0.12 + 0.72 * float(k) / float(MODES - 1) + rng.randf_range(-0.05, 0.05), 0.0, 1.0))
	# Place a spire on every slot, pinned to a spectral position by its radius (centre = bass, rim =
	# treble). Growth is staggered by that radius, so the city rises from the centre outward.
	for gy in C:
		for gx in C:
			var wx := _slot(gx) + rng.randf_range(-0.06, 0.06)
			var wz := _slot(gy) + rng.randf_range(-0.06, 0.06)
			var ground: float = _terrain.height_at(wx, wz) * _terrain.relief
			var up: Vector3 = _terrain.normal_world(wx, wz).lerp(UP, 0.96).normalized()
			var bx: Vector3 = up.cross(Vector3(1, 0, 0))
			if bx.length() < 1e-3:
				bx = up.cross(Vector3(0, 0, 1))
			bx = bx.normalized()
			var bz: Vector3 = bx.cross(up).normalized()
			var cxn := float(gx) / float(C - 1) * 2.0 - 1.0
			var cyn := float(gy) / float(C - 1) * 2.0 - 1.0
			var t := clampf(sqrt(cxn * cxn + cyn * cyn) / 1.415 + rng.randf_range(-0.06, 0.06), 0.0, 1.0)
			if rng.randf() > density:
				continue                     # a plaza, a park, a hole in the block
			var bulk := rng.randf_range(0.4, 1.0)
			var base_r := _terrain.half / float(C) * (0.40 + 0.7 * bulk)
			# `core` bends the skyline: >0 puts the towers downtown, <0 rings them
			# around a low middle, ~0 is an even sprawl. t is 0 at centre, 1 at rim.
			var law := 1.0 + core * (0.5 - t) * 1.6
			var height := rng.randf_range(1.0, 2.1) * (1.2 - 0.3 * bulk) * clampf(law, 0.35, 2.0)
			# Per-spire mode deviations + phases: a family resemblance to the global style, each unique.
			var mamp: Array = []
			var mph: Array = []
			for k in MODES:
				mamp.append(rng.randf_range(0.5, 1.5))
				mph.append(rng.randf() * TAU)
			# `body` = the fraction that stays a COLUMNAR shaft before the taper begins; small = a pure
			# spire/needle, large = a fat tower with a short spire cap. The mix gives varied architecture.
			_spires.append({
				"base": Vector3(wx, ground, wz), "up": up, "bx": bx, "bz": bz, "t": t,
				"base_r": base_r, "height": height, "taper": rng.randf_range(0.6, 1.4),
				"body": rng.randf_range(0.12, 0.68), "tiers": rng.randi_range(3, 6),
				"twist": rng.randf_range(-0.6, 0.6), "mamp": mamp, "mph": mph,
				"hue": fposmod(_hue + rng.randf_range(-spread, spread), 1.0),
				"delay": clampf(t * 0.45 + rng.randf_range(0.0, 0.22), 0.0, 0.62),
				"grow": 0.0})
	lens.fov = rng.randf_range(46.0, 56.0)
	_dist = rng.randf_range(12.5, 16.0)                  # back far enough to read the whole skyline
	_pitch = rng.randf_range(0.48, 0.68)
	_yaw = rng.randf() * TAU
	_yaw_dir = 1.0 if rng.randf() < 0.5 else -1.0
	_light_az = rng.randf() * TAU
	_light_dir = 1.0 if rng.randf() < 0.5 else -1.0
	_light_el = rng.randf_range(0.34, 0.5)
	_terrain.set_light(_light_az, _light_el)
	return {"type": ttype, "palette": _palette_name}


func _slot(g: int) -> float:
	return (float(g) / float(C - 1) - 0.5) * 2.0 * _terrain.half * 0.92


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.01, 0.015)
	_glow = Nonlinear.flare(_glow, clampf(0.3 * f.energy + 0.7 * f.beat, 0.0, 1.0), delta, 9.0, 1.6)
	_yaw += delta * (0.05 + 0.12 * f.energy) * _yaw_dir
	lens.orbit(Vector3(0.0, 1.15, 0.0), _dist, _yaw, _pitch + 0.03 * sin(_life * 0.11))
	# The city assembles itself: maturity rises (BURST fast at the start, so a short scene still fills in,
	# then easing to a crawl) and each spire eases toward its own maturity-gated completion (staggered by
	# radius, so growth spreads from the centre out).
	var burst: float = 1.0 + 3.5 * exp(-_life * 0.7)
	_maturity = minf(1.0, _maturity + delta * 0.05 * burst)
	var ease := 1.0 - exp(-3.0 * delta * burst)
	for spire in _spires:
		var target: float = clampf((_maturity - float(spire.delay)) / 0.35, 0.0, 1.0)
		spire.grow = lerpf(float(spire.grow), target, ease)
	_light_az += delta * 0.03 * _light_dir
	_terrain.set_light(_light_az, _light_el)
	_terrain.step_light(delta)
	# Snapshot the frame into a job for the worker. Spire dicts are DUPLICATED
	# (main keeps easing `grow`); the terrain rides by reference (its per-frame
	# light writes are in-place float updates - a concurrently-read cell is at
	# worst a frame stale, never unsafe); the lens is copied (main re-orbits it).
	var job := SpireJob.new()
	job.f = f
	job.life = _life
	job.reveal = view.reveal
	job.glow = _glow
	job.u = unit()
	job.terrain = _terrain
	job.tex_rid = Terrain.detail_texture().get_rid()
	job.mode_freq = _mode_freq
	job.mode_amp = _mode_amp
	job.mode_pos = _mode_pos
	job.dist = _dist
	var sp: Array = []
	for s in _spires:
		sp.append((s as Dictionary).duplicate())
	job.spires = sp
	job.win_col = _win_col
	job.sat_bias = _sat_bias
	job.lens = Lens3D.new()
	job.lens.eye = lens.eye
	job.lens.look = lens.look
	job.lens.up = lens.up
	job.lens.fov = lens.fov
	job.lens.near = lens.near
	_forge.kick(job.run, {}, self, job)   # retain: a Callable alone will not keep the job alive


func _draw() -> void:
	begin_draw()
	texture_repeat = CanvasItem.TEXTURE_REPEAT_ENABLED
	_forge.submit(self)


## The whole frame, built off the main thread (the FrameForge job): shadow
## pass, terrain merge, spire loft, painter sort, batch runs. Reads ONLY its
## own members - never the scene node - so a mid-job Director cut is harmless.
class SpireJob:
	extends RefCounted

	const SIDES := 6
	const LEVELS := 16
	const MODES := 4

	var f: AudioFeatures
	var life := 0.0
	var reveal := 1.0
	var glow := 0.0
	var u := 1.0
	var dist := 8.0
	var terrain: Terrain
	var tex_rid := RID()
	var lens: Lens3D
	var spires: Array = []
	var mode_freq: Array = []
	var mode_amp: Array = []
	var mode_pos: Array = []
	var ao_span := 3.0               # height over which the vertical ambient gradient runs
	var fog_near := 6.0
	var fog_far := 20.0
	var fog_col := Color(0.05, 0.06, 0.09)
	var win_col := Color(0.95, 0.74, 0.36)
	var sat_bias := 0.0

	func run(_s: Dictionary) -> Array:
		lens.prepare()
		var lit := clampf(0.6 + 0.4 * glow + 0.3 * f.energy, 0.4, 1.4)
		var ldir: Vector3 = terrain.light_dir()
		fog_near = dist * 0.45
		fog_far = dist * 2.3
		var embed: float = 0.3 * terrain.relief + 0.14
		# Pass A: rasterize every spire (as tapering segments) into a light-space shadow map, so a
		# spire casts a real volumetric shadow onto the ground AND onto other spires.
		var shadow := ShadowField.new()
		shadow.build(ldir, Vector3(-terrain.half, -terrain.relief, -terrain.half),
			Vector3(terrain.half, terrain.relief + 3.5, terrain.half))
		if reveal >= 0.02:
			for spire in spires:
				var g: float = clampf(float(spire.grow), 0.0, 1.0)
				var height: float = float(spire.height) * (0.78 + 0.34 * Nonlinear.apply("spike", _band(float(spire.t)), 1.4))
				# Only the part ABOVE the ground casts a shadow (the base is sunk `embed` into the
				# terrain), so the shadow GROWS with the visible height, as tapering stacked segments.
				var vis_h: float = g * height - embed
				if vis_h < 0.05:
					spire["ext"] = 0.0
					continue
				var base: Vector3 = spire.base
				var up: Vector3 = spire.up
				var br: float = float(spire.base_r)
				var segs := 3
				var seg_h := vis_h / float(segs)
				for si in segs:
					var f0 := float(si) / float(segs)
					# ROUND: a spire is a tapering shaft, not a box, and its bounding square cast a
					# rectangular shadow with hard corners. See ShadowField.add_box.
					shadow.add_box(base + up * (float(si) * seg_h), up, spire.bx, spire.bz,
						br * (1.0 - 0.72 * f0), seg_h, true)
				# Self-shadow bias = the spire's own light-depth reach, so a taller neighbour still shadows it.
				spire["ext"] = absf(shadow.light_depth(base + up * vis_h) - shadow.light_depth(base)) + br * 1.4

		# ONE merged list: terrain quads (shadowed by the spires) + spire faces/windows, depth-sorted
		# together, so the LAND occludes the buried spire bases and spires occlude one another correctly.
		var faces: Array = terrain.collect_surface(lens, u, lit, life, shadow)
		if reveal >= 0.02:
			for spire in spires:
				if float(spire.grow) >= 0.05:
					_emit_spire(faces, spire, ldir, lit, shadow, embed)
		faces = TriBatch.painter_sort(faces)
		var tb := TriBatch.new()
		for fc in faces:
			if fc.has("flat"):                               # window / cornice detail poly (flat colour)
				tb.mark_run(false, RID())
				tb.poly(fc.poly, fc.flat)
			elif fc.has("uvs"):                              # terrain land (textured run)
				tb.mark_run(true, tex_rid)
				tb.quad_textured(fc.poly, fc.cols, fc.uvs)
			else:                                            # spire wall / water (both carry cols)
				tb.mark_run(false, RID())
				tb.quad_colored(fc.poly, fc.cols)
		return tb.take_chunks()

	# The drive at spectral position `tt`: the LIVE band, or - when quiet/absent - a slow travelling
	# wave across the spectrum, so the city is always alive even in idle preview.
	func _band(tt: float) -> float:
		var live: float = f.sample(tt)
		var idle := 0.30 + 0.34 * sin(life * 0.4 + tt * 8.0)
		return clampf(maxf(live, idle * 0.85), 0.0, 1.0)

	# The spire's radius at normalized height u (0 = base, 1 = tip): a taper envelope modulated by the
	# harmonic mode series (the tiers / bulges the spectrum sculpts), stepped into set-back tiers with
	# cornice lips.
	func _profile_r(uu: float, spire: Dictionary) -> float:
		var body := float(spire.body)
		var env: float
		if uu <= body:
			env = 1.0
		else:
			env = pow(clampf((1.0 - uu) / maxf(1e-3, 1.0 - body), 0.0, 1.0), float(spire.taper))
		var m := 1.0
		var mamp: Array = spire.mamp
		var mph: Array = spire.mph
		for k in MODES:
			var drive := 0.4 + 0.95 * _band(float(mode_pos[k]))
			m += float(mode_amp[k]) * float(mamp[k]) * drive * cos(float(mode_freq[k]) * uu * PI + float(mph[k]))
		var nt := float(spire.tiers)
		var tf := uu * nt
		var setback: float = 1.0 - 0.14 * floor(tf) / nt
		var cornice: float = 0.17 * exp(-pow((tf - floor(tf) - 0.9) / 0.07, 2.0))
		return maxf(float(spire.base_r) * 0.02, float(spire.base_r) * env * (clampf(m, 0.2, 1.9) * setback + cornice))

	# Loft one spire: sample its profile bottom-up into rings (only as high as it has GROWN), close the
	# current top to a point, then emit the side faces.
	func _emit_spire(faces: Array, spire: Dictionary, ldir: Vector3, lit: float,
			shadow: ShadowField, embed: float) -> void:
		var g: float = clampf(float(spire.grow), 0.0, 1.0)
		var band := _band(float(spire.t))
		var height: float = float(spire.height) * (0.78 + 0.34 * Nonlinear.apply("spike", band, 1.4))
		var base: Vector3 = spire.base
		# The TERRAIN shadows the spire: standing in a hill's cast shadow darkens the whole tower.
		lit = lit * (0.42 + 0.58 * terrain.shadow_at(base.x, base.z))
		# Sink the base BELOW the surface so the loft starts underground; the merged terrain hides the
		# buried shaft, so the spire rises out of the land with a ragged base.
		base.y -= embed
		var up: Vector3 = spire.up
		var bx: Vector3 = spire.bx
		var bz: Vector3 = spire.bz
		var twist: float = float(spire.twist)
		var top_l := int(ceil(g * float(LEVELS)))
		if top_l < 1:
			return
		var rings: Array = []
		var us: Array = []
		for l in top_l + 1:
			var uu := minf(float(l) / float(LEVELS), g)
			var r := _profile_r(uu, spire)
			var ang0 := twist * uu * TAU * 0.15
			var c := base + up * (uu * height)
			var ring := PackedVector3Array()
			for s in SIDES:
				var a := ang0 + float(s) / float(SIDES) * TAU
				ring.append(c + bx * (cos(a) * r) + bz * (sin(a) * r))
			rings.append(ring)
			us.append(uu)
		# Collapsed tip ring (all corners at the top-centre) so the loft closes to a point.
		var tip_u: float = float(us[us.size() - 1])
		var tip := base + up * (tip_u * height + float(spire.base_r) * 0.05)
		var tip_ring := PackedVector3Array()
		for s in SIDES:
			tip_ring.append(tip)
		rings.append(tip_ring)
		us.append(tip_u)

		var thue := float(spire.hue)
		var sext: float = float(spire.get("ext", 0.0))
		for l in range(1, rings.size()):
			var lo: PackedVector3Array = rings[l - 1]
			var hi: PackedVector3Array = rings[l]
			# Windows go on the tower BODY (below the roof taper), never on the collapsed tip rings.
			var body_face: bool = float(us[l]) < 0.78 and l < rings.size() - 1
			for s in SIDES:
				var s2 := (s + 1) % SIDES
				var wseed := float(l) * 3.31 + float(s) * 1.73 + thue * 40.0
				_emit_face(faces, lo[s], lo[s2], hi[s2], hi[s], thue, band, lit, ldir,
					base.y, float(us[l]), shadow, body_face, wseed, sext)

	# Emit one lofted quad face, shaded by the key light with a vertical ambient gradient and distance
	# fog. Every wall face draws (no back-face cull - the towers are hollow lofted tubes; far-to-near
	# fill closes them); windows go only on front-facing walls.
	func _emit_face(faces: Array, p0: Vector3, p1: Vector3, p2: Vector3, p3: Vector3,
			thue: float, band: float, lit: float, ldir: Vector3, ao_lo: float, uu: float,
			shadow: ShadowField, body_face := false, wseed := 0.0, sext := 0.0) -> void:
		var fc := (p0 + p1 + p2 + p3) * 0.25
		var nrm := (p3 - p0).cross(p1 - p0)
		if nrm.length_squared() < 1e-12:
			return
		var front := nrm.dot(lens.eye - fc) > 0.0
		var q := [p0, p1, p2, p3]
		var pr := [lens.project(p0), lens.project(p1), lens.project(p2), lens.project(p3)]
		if pr[0].z <= lens.near or pr[1].z <= lens.near or pr[2].z <= lens.near or pr[3].z <= lens.near:
			return
		var poly := PackedVector2Array([Vector2(pr[0].x, pr[0].y) * u, Vector2(pr[1].x, pr[1].y) * u,
			Vector2(pr[2].x, pr[2].y) * u, Vector2(pr[3].x, pr[3].y) * u])
		var area := Terrain._quad_area(poly)
		if area < 0.5:
			return
		var val := clampf((0.66 + 0.5 * band + 0.5 * glow) * lit, 0.28, 1.5)
		var sat := clampf(0.42 + sat_bias + 0.14 * sin((thue + uu) * 12.0) + 0.1 * band, 0.12, 0.78)
		var col := Color.from_hsv(fposmod(thue + 0.05 * uu, 1.0), sat, val)
		var shade := 0.40 + 0.66 * clampf(nrm.normalized().dot(ldir), 0.0, 1.0)
		var cols := PackedColorArray()
		for k in 4:
			var ao := clampf((q[k].y - ao_lo) / ao_span, 0.0, 1.0)
			var sf: float = shadow.factor(q[k], sext + 0.06)
			var vb := shade * (0.62 + 0.44 * ao) * sf
			var cc := Color(col.r * vb, col.g * vb, col.b * vb, 1.0)
			var fogt := smoothstep(fog_near, fog_far, pr[k].z) * 0.9
			var fc2 := cc.lerp(fog_col, fogt)
			fc2.a = reveal                        # fade the spires in after the terrain (embed reveal)
			cols.append(fc2)
		var depth: float = (pr[0].z + pr[1].z + pr[2].z + pr[3].z) * 0.25
		faces.append({"d": depth, "poly": poly, "cols": cols})
		# Intricate detail: arched windows, only on faces big enough on screen to read (automatic LOD).
		if body_face and front and area > 60.0:
			_add_windows(faces, poly[0], poly[1], poly[2], poly[3], depth, col, band, wseed)

	# Bilinear point inside a projected quad (corners lo-left, lo-right, hi-right, hi-left).
	func _fp(a: Vector2, b: Vector2, c: Vector2, d: Vector2, s: float, v: float) -> Vector2:
		return a.lerp(b, s).lerp(d.lerp(c, s), v)

	# Two columns of a pointed (gothic) arched window per wall face: a lit stone frame with a recessed
	# dark (or warmly lit) pane, drawn as flat depth-sorted polys just in front of the wall.
	func _add_windows(faces: Array, a: Vector2, b: Vector2, c: Vector2, d: Vector2,
			depth: float, wall: Color, band: float, wseed: float) -> void:
		var frame_col := Color(clampf(wall.r * 1.5 + 0.06, 0.0, 1.0), clampf(wall.g * 1.5 + 0.05, 0.0, 1.0),
			clampf(wall.b * 1.4 + 0.05, 0.0, 1.0), reveal)
		for c_i in 2:
			var uc := 0.30 + 0.40 * float(c_i)
			var hw := 0.13
			# Stable per-window lottery: some panes glow warm (lit interior), most are dark recesses.
			var h := fposmod(sin((wseed + float(c_i) * 5.7) * 12.9898) * 43758.5453, 1.0)
			var pane: Color
			if h > 0.78:
				# the palette's own lit-window colour, brightened by the band
				pane = Color(win_col.r, clampf(win_col.g + 0.2 * band, 0.0, 1.0),
					win_col.b, reveal)                                     # a lit window (minority)
			else:
				pane = Color(wall.r * 0.14, wall.g * 0.12, wall.b * 0.18, reveal)  # a dark recess (most)
			var frame := PackedVector2Array([
				_fp(a, b, c, d, uc - hw - 0.03, 0.16), _fp(a, b, c, d, uc + hw + 0.03, 0.16),
				_fp(a, b, c, d, uc + hw + 0.03, 0.60), _fp(a, b, c, d, uc, 0.90), _fp(a, b, c, d, uc - hw - 0.03, 0.60)])
			var reccess := PackedVector2Array([
				_fp(a, b, c, d, uc - hw, 0.22), _fp(a, b, c, d, uc + hw, 0.22),
				_fp(a, b, c, d, uc + hw, 0.58), _fp(a, b, c, d, uc, 0.82), _fp(a, b, c, d, uc - hw, 0.58)])
			faces.append({"d": depth - 0.006, "poly": frame, "flat": frame_col})
			faces.append({"d": depth - 0.010, "poly": reccess, "flat": pane})
