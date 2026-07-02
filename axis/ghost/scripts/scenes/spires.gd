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

const C := 10                    # city is C x C spire slots
const UP := Vector3(0, 1, 0)
const SIDES := 6                 # facets around each spire
const LEVELS := 16               # profile samples up the height (rings are lofted between them)
const MODES := 4                 # cosine modes that define the radius-vs-height profile

var _f: AudioFeatures = AudioFeatures.new()
var _terrain: Terrain
var _spires: Array = []
var _hue := 0.10                 # brass / gold
var _glow := 0.0
var _yaw := 0.0
var _dist := 8.0
var _pitch := 0.42
var _yaw_dir := 1.0
var _light_az := 0.0
var _light_el := 0.5
var _light_dir := 1.0
var _maturity := 0.0             # 0..1, the slow structural growth of the whole city
var _shadow := ShadowField.new() # light-space cast-shadow map (spires shadow ground + each other)
var _reveal := 1.0               # spires fade in AFTER the terrain (embed reveal); 1 = fully shown
var _u := 1.0                    # cached unit() scale for screen projection
var _ao_span := 3.0              # height over which the vertical ambient gradient runs (street -> sky)
var _fog_near := 6.0
var _fog_far := 20.0
var _fog_col := Color(0.05, 0.06, 0.09)
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
	_hue = rng.randf_range(0.07, 0.13)
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
			var bulk := rng.randf_range(0.4, 1.0)
			var base_r := _terrain.half / float(C) * (0.40 + 0.7 * bulk)
			var height := rng.randf_range(1.0, 2.1) * (1.2 - 0.3 * bulk)
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
				"hue": fposmod(_hue + rng.randf_range(-0.05, 0.06), 1.0),
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
	return {"type": ttype}


func _slot(g: int) -> float:
	return (float(g) / float(C - 1) - 0.5) * 2.0 * _terrain.half * 0.92


# The drive at spectral position `tt`: the LIVE band, or - when quiet/absent - a slow travelling wave
# across the spectrum, so the city is always alive (districts swelling in sequence) even in idle preview.
func _band(tt: float) -> float:
	var live: float = _f.sample(tt)
	var idle := 0.30 + 0.34 * sin(_life * 0.4 + tt * 8.0)
	return clampf(maxf(live, idle * 0.85), 0.0, 1.0)


# The spire's radius at normalized height u (0 = base, 1 = tip): a taper envelope (narrows to a point,
# so it reads as a spire) modulated by the harmonic mode series (the tiers / bulges that the spectrum
# sculpts). Shared global mode amp/freq x per-spire deviation x live band drive.
func _profile_r(u: float, spire: Dictionary) -> float:
	# Envelope: a COLUMNAR shaft up to `body`, then a taper to a point (the spire cap). So spires read
	# as towers-with-a-peak, not pure cones - and the mode bulges have a shaft to sit on.
	var body := float(spire.body)
	var env: float
	if u <= body:
		env = 1.0
	else:
		env = pow(clampf((1.0 - u) / maxf(1e-3, 1.0 - body), 0.0, 1.0), float(spire.taper))
	var m := 1.0
	var mamp: Array = spire.mamp
	var mph: Array = spire.mph
	for k in MODES:
		var drive := 0.4 + 0.95 * _band(float(_mode_pos[k]))
		m += float(_mode_amp[k]) * float(mamp[k]) * drive * cos(float(_mode_freq[k]) * u * PI + float(mph[k]))
	# Stepped TIERS + a CORNICE lip: quantize the height into `tiers` bands. Each band is set back a
	# little narrower than the one below, and just under each boundary the radius juts out into an
	# overhanging cornice - the stacked, ledged silhouette of the reference towers.
	var nt := float(spire.tiers)
	var tf := u * nt
	var setback: float = 1.0 - 0.14 * floor(tf) / nt
	var cornice: float = 0.17 * exp(-pow((tf - floor(tf) - 0.9) / 0.07, 2.0))
	return maxf(float(spire.base_r) * 0.02, float(spire.base_r) * env * (clampf(m, 0.2, 1.9) * setback + cornice))


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
	queue_redraw()


func _draw() -> void:
	begin_draw()
	lens.prepare()
	var u := unit()
	_u = u
	var lit := clampf(0.6 + 0.4 * _glow + 0.3 * _f.energy, 0.4, 1.4)
	texture_repeat = CanvasItem.TEXTURE_REPEAT_ENABLED

	var ldir: Vector3 = _terrain.light_dir()
	_fog_near = _dist * 0.45
	_fog_far = _dist * 2.3
	var embed: float = 0.3 * _terrain.relief + 0.14
	# The spires are EMBEDDED below the surface, so during the fade-IN the semi-transparent terrain would
	# reveal their buried bases. Hold them hidden until the terrain is nearly opaque, then ease in.
	_reveal = smoothstep(0.8, 1.0, view.presence)

	# Pass A: rasterize every spire (as its bounding box) into the light-space shadow map, so a spire
	# casts a real volumetric shadow onto the ground AND onto other spires (block-on-block).
	_shadow.build(_terrain.light_dir(), Vector3(-_terrain.half, -_terrain.relief, -_terrain.half),
		Vector3(_terrain.half, _terrain.relief + 3.5, _terrain.half))
	if _reveal >= 0.02:
		for spire in _spires:
			var g: float = clampf(float(spire.grow), 0.0, 1.0)
			var height: float = float(spire.height) * (0.78 + 0.34 * Nonlinear.apply("spike", _band(float(spire.t)), 1.4))
			# Only the part ABOVE the ground casts a shadow (the base is sunk `embed` into the terrain),
			# so a spire throws NO shadow until its shape has actually emerged, and the shadow then GROWS
			# with the visible height. Rasterize it as a few stacked, TAPERING segments so the cast shadow
			# matches the spire's cone-ish silhouette instead of a full-width column.
			var vis_h: float = g * height - embed
			if vis_h < 0.05:
				spire["ext"] = 0.0
				continue
			var base: Vector3 = spire.base                  # at the GROUND (not the embedded base)
			var up: Vector3 = spire.up
			var br: float = float(spire.base_r)
			var segs := 3
			var seg_h := vis_h / float(segs)
			for si in segs:
				var f0 := float(si) / float(segs)
				_shadow.add_box(base + up * (float(si) * seg_h), up, spire.bx, spire.bz, br * (1.0 - 0.72 * f0), seg_h)
			# Self-shadow bias = the spire's own light-depth reach, so a taller neighbour still shadows it.
			spire["ext"] = absf(_shadow.light_depth(base + up * vis_h) - _shadow.light_depth(base)) + br * 1.4

	# ONE merged list: terrain quads (shadowed by the spires) + spire faces/windows, depth-sorted
	# together, so the LAND occludes the buried spire bases and spires occlude one another correctly.
	var faces: Array = _terrain.collect_surface(lens, u, lit, _life, _shadow)
	if _reveal >= 0.02:
		for spire in _spires:
			if float(spire.grow) >= 0.05:
				_emit_spire(faces, spire, ldir, lit)
	faces.sort_custom(func(a, b): return a.d > b.d)          # far first
	var tex := Terrain.detail_texture()
	for fc in faces:
		if fc.has("flat"):                                   # window / cornice detail poly (flat colour)
			draw_colored_polygon(fc.poly, fc.flat)
		elif fc.has("uvs"):                                  # terrain land
			Terrain.draw_quad(self, fc.poly, fc.cols, fc.uvs, tex)
		else:                                                # spire wall / water (both carry cols)
			Terrain.draw_quad(self, fc.poly, fc.cols)


# Loft one spire: sample its profile bottom-up into rings (only as high as it has GROWN, so it builds
# from the ground), close the current top to a point, then emit the side faces.
func _emit_spire(faces: Array, spire: Dictionary, ldir: Vector3, lit: float) -> void:
	var g: float = clampf(float(spire.grow), 0.0, 1.0)
	var band := _band(float(spire.t))
	var height: float = float(spire.height) * (0.78 + 0.34 * Nonlinear.apply("spike", band, 1.4))
	var base: Vector3 = spire.base
	# The TERRAIN shadows the spire: one standing where the ground is in a hill's cast shadow is
	# darkened by the same moving shadow, so it sits IN the light rather than lit flat.
	lit = lit * (0.42 + 0.58 * _terrain.shadow_at(base.x, base.z))
	# Sink the base BELOW the surface so the loft starts underground; the merged terrain hides that
	# buried shaft, so the spire rises out of the land with a ragged base instead of off a flat disc.
	base.y -= 0.3 * _terrain.relief + 0.14
	var up: Vector3 = spire.bx.cross(spire.bz)      # = spire.up (kept orthonormal)
	up = spire.up
	var bx: Vector3 = spire.bx
	var bz: Vector3 = spire.bz
	var twist: float = float(spire.twist)
	var top_l := int(ceil(g * float(LEVELS)))
	if top_l < 1:
		return
	# Build the rings (bottom-up). The last ring is clamped to the exact grown height; a collapsed TIP
	# ring above it caps the spire to a point wherever it currently reaches.
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
			_emit_face(faces, lo[s], lo[s2], hi[s2], hi[s], thue, band, lit, ldir, base.y, float(us[l]), body_face, wseed, sext)


# Emit one lofted quad face (camera-facing only), shaded by the key light with a vertical ambient
# gradient and distance fog. `uu` is the face's height fraction (for a subtle hue drift up the spire).
func _emit_face(faces: Array, p0: Vector3, p1: Vector3, p2: Vector3, p3: Vector3,
		thue: float, band: float, lit: float, ldir: Vector3, ao_lo: float, uu: float,
		body_face := false, wseed := 0.0, sext := 0.0) -> void:
	var fc := (p0 + p1 + p2 + p3) * 0.25
	# Outward normal (the loft frame is left-handed, so cross the diagonal-then-tangent to face OUT).
	var nrm := (p3 - p0).cross(p1 - p0)
	if nrm.length_squared() < 1e-12:
		return
	# Draw EVERY wall face - do NOT back-face cull. Each tower is a hollow lofted tube; culling the far
	# wall (and, with an inward normal, wrongly the NEAR wall) let you see into the shell like a lantern.
	# Drawing all faces far-to-near fills the solid, so a tower reads as CLOSED. Windows still go only on
	# the front-facing walls (the back ones are covered anyway), so the detail cost doesn't double.
	var front := nrm.dot(lens.eye - fc) > 0.0
	var q := [p0, p1, p2, p3]
	var pr := [lens.project(p0), lens.project(p1), lens.project(p2), lens.project(p3)]
	if pr[0].z <= lens.near or pr[1].z <= lens.near or pr[2].z <= lens.near or pr[3].z <= lens.near:
		return
	var poly := PackedVector2Array([Vector2(pr[0].x, pr[0].y) * _u, Vector2(pr[1].x, pr[1].y) * _u,
		Vector2(pr[2].x, pr[2].y) * _u, Vector2(pr[3].x, pr[3].y) * _u])
	var area := Terrain._quad_area(poly)
	if area < 0.5:
		return
	# Base material: warm, with a little hue/sat wander and value from the driving band + glow.
	var val := clampf((0.66 + 0.5 * band + 0.5 * _glow) * lit, 0.28, 1.5)
	var sat := clampf(0.42 + 0.14 * sin((thue + uu) * 12.0) + 0.1 * band, 0.2, 0.68)
	var col := Color.from_hsv(fposmod(thue + 0.05 * uu, 1.0), sat, val)
	var shade := 0.40 + 0.66 * clampf(nrm.normalized().dot(ldir), 0.0, 1.0)
	var cols := PackedColorArray()
	for k in 4:
		var ao := clampf((q[k].y - ao_lo) / _ao_span, 0.0, 1.0)
		# Cast shadow from the field (biased by this spire's own depth extent, so a taller neighbour's
		# shadow bands across it without the spire shadowing itself).
		var sf: float = _shadow.factor(q[k], sext + 0.06)
		var vb := shade * (0.62 + 0.44 * ao) * sf
		var cc := Color(col.r * vb, col.g * vb, col.b * vb, 1.0)
		var fogt := smoothstep(_fog_near, _fog_far, pr[k].z) * 0.9
		var fc2 := cc.lerp(_fog_col, fogt)
		fc2.a = _reveal                                # fade the spires in after the terrain (embed reveal)
		cols.append(fc2)
	var depth: float = (pr[0].z + pr[1].z + pr[2].z + pr[3].z) * 0.25
	faces.append({"d": depth, "poly": poly, "cols": cols})
	# Intricate detail: rows of arched windows recessed into the wall, but only on faces big enough on
	# screen to read (an automatic LOD - distant spires stay clean, near ones gain the fine masonry).
	if body_face and front and area > 60.0:
		_add_windows(faces, poly[0], poly[1], poly[2], poly[3], depth, col, shade, band, wseed)


# Bilinear point inside a projected quad (corners lo-left, lo-right, hi-right, hi-left).
func _fp(a: Vector2, b: Vector2, c: Vector2, d: Vector2, s: float, v: float) -> Vector2:
	return a.lerp(b, s).lerp(d.lerp(c, s), v)


# Two columns of a pointed (gothic) arched window per wall face: a lit stone frame with a recessed
# dark (or warmly lit) pane, drawn as flat depth-sorted polys just in front of the wall.
func _add_windows(faces: Array, a: Vector2, b: Vector2, c: Vector2, d: Vector2,
		depth: float, wall: Color, shade: float, band: float, wseed: float) -> void:
	var frame_col := Color(clampf(wall.r * 1.5 + 0.06, 0.0, 1.0), clampf(wall.g * 1.5 + 0.05, 0.0, 1.0),
		clampf(wall.b * 1.4 + 0.05, 0.0, 1.0), _reveal)
	for c_i in 2:
		var uc := 0.30 + 0.40 * float(c_i)
		var hw := 0.13
		# Stable per-window lottery: some panes glow warm (lit interior), most are dark recesses.
		var h := fposmod(sin((wseed + float(c_i) * 5.7) * 12.9898) * 43758.5453, 1.0)
		var pane: Color
		if h > 0.78:
			pane = Color(0.95, 0.68 + 0.2 * band, 0.32, _reveal)              # a lit window (minority)
		else:
			pane = Color(wall.r * 0.14, wall.g * 0.12, wall.b * 0.18, _reveal) # a dark recess (most)
		var frame := PackedVector2Array([
			_fp(a, b, c, d, uc - hw - 0.03, 0.16), _fp(a, b, c, d, uc + hw + 0.03, 0.16),
			_fp(a, b, c, d, uc + hw + 0.03, 0.60), _fp(a, b, c, d, uc, 0.90), _fp(a, b, c, d, uc - hw - 0.03, 0.60)])
		var reccess := PackedVector2Array([
			_fp(a, b, c, d, uc - hw, 0.22), _fp(a, b, c, d, uc + hw, 0.22),
			_fp(a, b, c, d, uc + hw, 0.58), _fp(a, b, c, d, uc, 0.82), _fp(a, b, c, d, uc - hw, 0.58)])
		faces.append({"d": depth - 0.006, "poly": frame, "flat": frame_col})
		faces.append({"d": depth - 0.010, "poly": reccess, "flat": pane})


