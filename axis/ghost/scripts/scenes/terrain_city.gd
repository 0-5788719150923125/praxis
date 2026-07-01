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
var _hclass := PackedFloat32Array()   # per-plot height multiplier (big footprints tend taller)
var _phase := PackedFloat32Array()    # per-plot phase for the slow rearrange wobble
var _maturity := 0.0                  # 0..1, rises over the scene: thresholds drop -> arms thicken, gaps fill
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
	_phase.resize(C * C)
	# A ridged "arm" field: its branching high ridges become the channels the city builds ALONG, so
	# development reads as dendritic ARMS reaching out from the core rather than a filled blob.
	var armf := Field.make("ridged", rng.randi(), rng.randf_range(2.0, 3.4), 3)
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
			_thresh[i] = 0.05 + 0.18 * elev + (1.0 - smoothstep(0.4, 0.7, arm)) * 0.2 + rng.randf_range(-0.03, 0.06)
			_grown[i] = 0.0
			# Footprint + height class: a skewed distribution - most plots small, a rare few large
			# (the anchors that become the towers of a cluster). Big footprints trend taller.
			var big := pow(rng.randf(), 2.3)
			_foot[i] = 0.5 + 1.0 * big
			_hclass[i] = 0.6 + 1.7 * big + rng.randf_range(-0.15, 0.35)
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
	# Nonlinear growth drive: beats lunge development outward through a spike curve. Kept slow, so
	# the city fills in gradually (districts spread and the gaps close over time), not all at once.
	var drive := 0.9 + 1.2 * Nonlinear.apply("spike", clampf(0.7 * f.energy + f.beat, 0.0, 1.0), 2.0)
	for core in _cores:
		_dev.inject(core.x, core.y, 1.0)          # keep the origin cores alive
	_dev.step(drive, delta, 0.015)
	# The city keeps maturing: over the scene the effective thresholds drop, so the arms THICKEN and
	# the gaps between them fill in - the base keeps growing and getting denser, not a fixed footprint.
	_maturity = minf(1.0, _maturity + delta * 0.045)
	# Ease each plot's BUILT height up toward its current maturity, so buildings START SMALL and grow
	# taller as their district matures - and the densest (most-developed) plots grow tallest, so height
	# reads as a property of the cluster, not of a single plot popping up full-formed.
	var rise := delta * 0.28
	for cy in C:
		for cx in C:
			var i := cy * C + cx
			var thr: float = maxf(0.04, float(_thresh[i]) - _maturity * 0.4)
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
	_terrain.draw_surface(self, lens, u, lit, _life)
	texture_repeat = CanvasItem.TEXTURE_REPEAT_DISABLED   # terrain left it enabled; blocks are untextured

	# Collect every visible block face, depth-sort the lot, then draw - so blocks occlude
	# one another correctly over the terrain.
	var faces: Array = []
	var bw := _terrain.half / float(C) * 0.62          # block half-footprint (world)
	var bgain := 0.5 + 0.3 * _f.energy
	for cy in C:
		for cx in C:
			var i := cy * C + cx
			var dev := _dev.at(cx, cy)
			var grown: float = _grown[i]
			if grown < 0.02:                                # nothing built here yet (a gap / bare peak)
				continue
			var wx := (float(cx) / float(C - 1) - 0.5) * 2.0 * _terrain.half
			var wz := (float(cy) / float(C - 1) - 0.5) * 2.0 * _terrain.half
			var ground := _terrain.height_at(wx, wz) * _terrain.relief
			var float_off: float = _detach[i]
			# Each block bounces with its own spectral band - a responsive skyline.
			var react := _f.sample(clampf(_terrain.height_at(wx, wz) + 0.5, 0.0, 1.0))
			# A slow per-plot wobble keeps the built skyline REARRANGING over time (buildings edge up
			# and down at their own pace) rather than freezing once grown.
			var wob := 0.85 + 0.28 * sin(_life * 0.12 + float(_phase[i]))
			# Height = built level x this plot's height CLASS (big-footprint anchors tower) x a base
			# height (so buildings read even in quiet passages) plus spectral reaction, x the slow
			# rearrange - a real mix of tall and small, not uniform, and tall enough to see.
			var h := grown * float(_hclass[i]) * (0.42 + 0.45 * bgain + 0.6 * react) * wob
			var bw_i := bw * float(_foot[i])                # varied footprint (few big, many small)
			# Buildings stand UPRIGHT - real ones are vertical whatever the ground does. Keep
			# only a faint lean toward the terrain normal so the field isn't perfectly rigid.
			var up := _terrain.normal_world(wx, wz).lerp(Vector3.UP, 0.92).normalized()
			var base := Vector3(wx, ground + float_off, wz)
			var hue := fposmod(_hue + 0.12 * _terrain.height_at(wx, wz) + 0.25 * dev, 1.0)
			var blit := clampf(0.18 + 0.5 * dev + 0.5 * react + 0.6 * _glow, 0.05, 1.2) * lit
			_block_faces(faces, base, up, bw_i, h, hue, blit)
	faces.sort_custom(func(a, b): return a.d > b.d)
	for fc in faces:
		var c: Color = fc.col
		Terrain.draw_quad(self, fc.poly, PackedColorArray([c, c, c, c]))


# Append the camera-facing faces of one oriented box to `out` (each {poly, col, d}).
func _block_faces(out: Array, base: Vector3, up: Vector3, w: float, h: float,
		hue: float, lit: float) -> void:
	var bx := up.cross(Vector3(1, 0, 0))
	if bx.length() < 1e-3:
		bx = up.cross(Vector3(0, 0, 1))
	bx = bx.normalized()
	var bz := bx.cross(up).normalized()                 # the two tangent axes; `up` is height
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
		var shade := 0.55 + 0.45 * clampf(fn.dot(_terrain.light_dir()), 0.0, 1.0)
		out.append({"d": (p0.z + p1.z + p2.z + p3.z) * 0.25, "poly": fpoly,
			"col": Color.from_hsv(hue, 0.45, clampf(lit * shade, 0.0, 1.0))})


# Project a world point to (screen x, screen y, camera depth).
func _terrain_proj(wld: Vector3) -> Vector3:
	var pr := lens.project(wld)
	return Vector3(pr.x * unit(), pr.y * unit(), pr.z)
