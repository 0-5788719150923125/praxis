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
				"body": rng.randf_range(0.12, 0.68),
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
	return maxf(float(spire.base_r) * 0.02, float(spire.base_r) * env * clampf(m, 0.2, 1.9))


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.01, 0.015)
	_glow = Nonlinear.flare(_glow, clampf(0.3 * f.energy + 0.7 * f.beat, 0.0, 1.0), delta, 9.0, 1.6)
	_yaw += delta * (0.05 + 0.12 * f.energy) * _yaw_dir
	lens.orbit(Vector3(0.0, 1.15, 0.0), _dist, _yaw, _pitch + 0.03 * sin(_life * 0.11))
	# The city assembles itself: maturity rises slowly, and each spire eases toward its own maturity-
	# gated completion (staggered by radius, so growth spreads from the centre out).
	_maturity = minf(1.0, _maturity + delta * 0.075)
	var ease := 1.0 - exp(-3.0 * delta)
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
	_terrain.draw_surface(self, lens, u, lit, _life)

	var ldir: Vector3 = _terrain.light_dir()
	# The horizontal direction the contact shadows fall (away from the light).
	var shadow_off := Vector3(cos(_light_az), 0.0, sin(_light_az)) * -1.0
	_fog_near = _dist * 0.45
	_fog_far = _dist * 2.3

	# Pass 1: contact shadows on the ground (over the terrain, under the spires) - grounds each spire.
	for spire in _spires:
		if float(spire.grow) >= 0.06:
			_ground_shadow(spire, shadow_off)

	# Pass 2: the lofted spires, all faces depth-sorted together so they occlude correctly.
	var faces: Array = []
	for spire in _spires:
		if float(spire.grow) >= 0.05:
			_emit_spire(faces, spire, ldir, lit)
	faces.sort_custom(func(a, b): return a.d > b.d)          # far first
	for fc in faces:
		Terrain.draw_quad(self, fc.poly, fc.cols)


# Loft one spire: sample its profile bottom-up into rings (only as high as it has GROWN, so it builds
# from the ground), close the current top to a point, then emit the side faces.
func _emit_spire(faces: Array, spire: Dictionary, ldir: Vector3, lit: float) -> void:
	var g: float = clampf(float(spire.grow), 0.0, 1.0)
	var band := _band(float(spire.t))
	var height: float = float(spire.height) * (0.78 + 0.34 * Nonlinear.apply("spike", band, 1.4))
	var base: Vector3 = spire.base
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
	for l in range(1, rings.size()):
		var lo: PackedVector3Array = rings[l - 1]
		var hi: PackedVector3Array = rings[l]
		for s in SIDES:
			var s2 := (s + 1) % SIDES
			_emit_face(faces, lo[s], lo[s2], hi[s2], hi[s], thue, band, lit, ldir, base.y, float(us[l]))


# Emit one lofted quad face (camera-facing only), shaded by the key light with a vertical ambient
# gradient and distance fog. `uu` is the face's height fraction (for a subtle hue drift up the spire).
func _emit_face(faces: Array, p0: Vector3, p1: Vector3, p2: Vector3, p3: Vector3,
		thue: float, band: float, lit: float, ldir: Vector3, ao_lo: float, uu: float) -> void:
	var fc := (p0 + p1 + p2 + p3) * 0.25
	var nrm := (p1 - p0).cross(p3 - p0)
	if nrm.length_squared() < 1e-12:
		return
	if nrm.dot(lens.eye - fc) <= 0.0:                        # facing away
		return
	var q := [p0, p1, p2, p3]
	var pr := [lens.project(p0), lens.project(p1), lens.project(p2), lens.project(p3)]
	if pr[0].z <= lens.near or pr[1].z <= lens.near or pr[2].z <= lens.near or pr[3].z <= lens.near:
		return
	var poly := PackedVector2Array([Vector2(pr[0].x, pr[0].y) * _u, Vector2(pr[1].x, pr[1].y) * _u,
		Vector2(pr[2].x, pr[2].y) * _u, Vector2(pr[3].x, pr[3].y) * _u])
	if Terrain._quad_area(poly) < 0.5:
		return
	# Base material: warm, with a little hue/sat wander and value from the driving band + glow.
	var val := clampf((0.66 + 0.5 * band + 0.5 * _glow) * lit, 0.28, 1.5)
	var sat := clampf(0.42 + 0.14 * sin((thue + uu) * 12.0) + 0.1 * band, 0.2, 0.68)
	var col := Color.from_hsv(fposmod(thue + 0.05 * uu, 1.0), sat, val)
	var shade := 0.52 + 0.52 * clampf(nrm.normalized().dot(ldir), 0.0, 1.0)
	var cols := PackedColorArray()
	for k in 4:
		var ao := clampf((q[k].y - ao_lo) / _ao_span, 0.0, 1.0)
		var vb := shade * (0.62 + 0.44 * ao)
		var cc := Color(col.r * vb, col.g * vb, col.b * vb, 1.0)
		var fogt := smoothstep(_fog_near, _fog_far, pr[k].z) * 0.9
		cols.append(cc.lerp(_fog_col, fogt))
	faces.append({"d": (pr[0].z + pr[1].z + pr[2].z + pr[3].z) * 0.25, "poly": poly, "cols": cols})


# A soft dark contact shadow on the ground under a spire, offset away from the light and sized by how
# grown/tall the spire is - so the spires sit ON the terrain instead of floating.
func _ground_shadow(spire: Dictionary, shadow_off: Vector3) -> void:
	var g := float(spire.grow)
	var base: Vector3 = spire.base
	var bx: Vector3 = spire.bx
	var bz: Vector3 = spire.bz
	var r: float = float(spire.base_r) * (1.7 + 0.5 * g)
	var off := shadow_off * float(spire.height) * g * 0.5
	var poly := PackedVector2Array()
	for s in SIDES:
		var a := float(s) / float(SIDES) * TAU
		var p := base + bx * (cos(a) * r) + bz * (sin(a) * r) + off
		var pr := lens.project(p)
		if pr.z <= lens.near:
			return
		poly.append(Vector2(pr.x, pr.y) * _u)
	draw_colored_polygon(poly, Color(0.0, 0.0, 0.0, clampf(0.10 + 0.15 * g, 0.0, 0.32)))
