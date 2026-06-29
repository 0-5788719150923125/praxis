extends Scene3D

## Terrain city - blocks rising as a city over real 3D terrain, growing nonlinearly.
##
## The metropolis idea on the [Terrain] foundation: a [Swarm] development field creeps
## across a landscape (rolling hills / mesa), and where it has grown, blocks stand on the
## surface - **oriented to the terrain normal**, so they lean with the gentle curvature of
## the land like real buildings on a hillside rather than a flat grid. Heights are driven
## by development x a per-block spectral band (nonlinear), so the skyline rises with the
## music. Some plots **detach**, their blocks floating a little off the ground. Camera
## orbits under a wide lens; the city grows over time from a few seeds.

const C := 30                    # city grid is C x C plots

var _f: AudioFeatures = AudioFeatures.new()
var _terrain: Terrain
var _dev: Swarm
var _detach := PackedFloat32Array()   # per-plot float-off height (0 = grounded)
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
	_terrain.build(rng, ttype, 3.0, rng.randf_range(0.7, 1.1), Palette.named("earth", rng))
	_hue = fposmod(rng.randf() + 0.5, 1.0)
	_dev = Swarm.new(C, C, Swarm.GROW, rng, rng.randi_range(2, 4))
	# Per-plot detach: a few districts float off the ground.
	_detach.resize(C * C)
	for i in C * C:
		_detach[i] = rng.randf_range(0.10, 0.30) if rng.randf() < 0.12 else 0.0
	lens.fov = rng.randf_range(56.0, 72.0)
	_dist = rng.randf_range(6.5, 8.5)
	_pitch = rng.randf_range(0.34, 0.55)
	_yaw = rng.randf() * TAU
	return {"type": ttype}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.012, 0.018)
	# Nonlinear growth drive: beats lunge development outward through a spike curve.
	var drive := 0.4 + 1.4 * Nonlinear.apply("spike", clampf(0.7 * f.energy + f.beat, 0.0, 1.0), 2.0)
	_dev.step(drive, delta, 0.015)
	_glow = lerpf(_glow, clampf(0.3 * f.energy + 0.5 * f.beat, 0.0, 1.0), 1.0 - exp(-5.0 * delta))
	_yaw += delta * (0.08 + 0.16 * f.energy)
	lens.orbit(Vector3(0.0, _terrain.relief * 0.25, 0.0), _dist, _yaw, _pitch + 0.04 * sin(_life * 0.13))
	queue_redraw()


func _draw() -> void:
	begin_draw()
	lens.prepare()
	var u := unit()
	var lit := clampf(0.7 + 0.4 * _glow + 0.3 * _f.energy, 0.4, 1.4)
	_terrain.draw_surface(self, lens, u, lit, _life)

	# Collect every visible block face, depth-sort the lot, then draw - so blocks occlude
	# one another correctly over the terrain.
	var faces: Array = []
	var bw := _terrain.half / float(C) * 0.62          # block half-footprint (world)
	var bgain := 0.5 + 0.3 * _f.energy
	for cy in C:
		for cx in C:
			var dev := _dev.at(cx, cy)
			if dev < 0.12:
				continue
			var wx := (float(cx) / float(C - 1) - 0.5) * 2.0 * _terrain.half
			var wz := (float(cy) / float(C - 1) - 0.5) * 2.0 * _terrain.half
			var ground := _terrain.height_at(wx, wz) * _terrain.relief
			var float_off: float = _detach[cy * C + cx]
			# Each block bounces with its own spectral band - a responsive skyline.
			var react := _f.sample(clampf(_terrain.height_at(wx, wz) + 0.5, 0.0, 1.0))
			var h := dev * (bgain + 0.7 * react) * _terrain.relief * 0.9
			# Orient to the terrain, but only halfway to the normal - a gentle lean, not a
			# wild tilt - so blocks follow the land's curvature like buildings on a hillside.
			var up := _terrain.normal_world(wx, wz).lerp(Vector3.UP, 0.5).normalized()
			var base := Vector3(wx, ground + float_off, wz)
			var hue := fposmod(_hue + 0.12 * _terrain.height_at(wx, wz) + 0.25 * dev, 1.0)
			var blit := clampf(0.18 + 0.5 * dev + 0.5 * react + 0.6 * _glow, 0.05, 1.2) * lit
			_block_faces(faces, base, up, bw, h, hue, blit)
	faces.sort_custom(func(a, b): return a.d > b.d)
	for fc in faces:
		Terrain.draw_quad(self, fc.poly, fc.col)


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
		var shade := 0.55 + 0.45 * clampf(fn.dot(Mesh3D.LIGHT.normalized()), 0.0, 1.0)
		out.append({"d": (p0.z + p1.z + p2.z + p3.z) * 0.25, "poly": fpoly,
			"col": Color.from_hsv(hue, 0.45, clampf(lit * shade, 0.0, 1.0))})


# Project a world point to (screen x, screen y, camera depth).
func _terrain_proj(wld: Vector3) -> Vector3:
	var pr := lens.project(wld)
	return Vector3(pr.x * unit(), pr.y * unit(), pr.z)
