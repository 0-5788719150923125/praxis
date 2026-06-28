extends Scene3D

## Shatter glass - a real pane of glass, shattering in true 3D.
##
## A flat pane stands in space and is seen at a three-quarter angle through the
## [Lens3D] camera (not pinned flat to the screen). On a beat it fractures into
## irregular angular shards - [Geo.fracture] cracks that radiate from an impact, not
## pizza slices - and the shards burst off the plane and **tumble through space**,
## each spiralling on its own 3D axis, before easing back together (loop) or
## drifting to rest (oneshot). Real depth: a shard in front occludes one behind, and
## the glass catches the light by how each shard faces the camera. The old version
## was flat shards sliding in the 2D plane; this one is dimensional.

var _f: AudioFeatures = AudioFeatures.new()
var _rng := RandomNumberGenerator.new()
var _shards: Array = []
var _hue := 0.0
var _oneshot := false
var _fired := false
var _moved := false
var _beat_prev := 0.0
var _angle := 0.5

const PANE := 1.25       # pane half-size, world units


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "plane"
	_rng.seed = rng.randi()
	_hue = rng.randf()
	_oneshot = rng.randf() < 0.4
	lifecycle = "oneshot" if _oneshot else "loop"
	lens.fov = rng.randf_range(40.0, 52.0)
	_angle = rng.randf_range(0.35, 0.7)

	# The intact pane (a disc or a rectangle), fractured into shards.
	var base := PackedVector2Array()
	if rng.randf() < 0.5:
		var n := rng.randi_range(12, 18)
		for i in n:
			var a := TAU * float(i) / float(n)
			base.append(Vector2(cos(a), sin(a)) * PANE)
	else:
		var hw := PANE * rng.randf_range(0.85, 1.1)
		var hh := PANE * rng.randf_range(0.6, 1.0)
		base = PackedVector2Array([Vector2(-hw, -hh), Vector2(hw, -hh), Vector2(hw, hh), Vector2(-hw, hh)])
	var impact := Vector2(rng.randf_range(-1, 1), rng.randf_range(-1, 1)) * PANE * 0.4
	var shards := Geo.fracture(base, rng.randi_range(18, 34), impact, PANE * 0.12, rng)
	for poly: PackedVector2Array in shards:
		var cen := Geo.centroid(poly)
		var local := PackedVector2Array()
		for v in poly:
			local.append(v - cen)
		_shards.append({
			"poly": local,
			"home": Vector3(cen.x, cen.y, 0.0),
			"pos": Vector3(cen.x, cen.y, 0.0),
			"basis": Basis.IDENTITY,
			"vel": Vector3.ZERO,
			"spin": Vector3.ZERO,
			"hue": fposmod(_hue + 0.22 * cen.length() / PANE, 1.0),
			"noise": Vector3(rng.randf_range(-1, 1), rng.randf_range(-1, 1), rng.randf_range(-1, 1)),
		})
	return {}


func finished() -> bool:
	return _oneshot and _moved and _settled()


func _settled() -> bool:
	for s in _shards:
		if s.vel.length() > 0.04:
			return false
	return true


# Burst every shard off the plane: outward in-plane + toward the viewer + a random
# kick, with a hard random spin so they tumble and spiral rather than slide flat.
func _burst(strength: float) -> void:
	for s in _shards:
		var radial: Vector3 = s.home
		var dir: Vector3 = radial.normalized() if radial.length() > 0.01 else Vector3(0, 0, 1)
		s.vel += (dir + s.noise * 0.7 + Vector3(0, 0, _rng.randf_range(0.4, 1.1))) * strength
		s.spin += s.noise * _rng.randf_range(2.5, 6.0)


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	# A three-quarter view that drifts slowly, so the pane reads as a plane in space.
	var ang := _angle + 0.18 * sin(_life * 0.12)
	lens.eye = Vector3(sin(ang) * 1.7, 0.7 + 0.2 * sin(_life * 0.1), cos(ang) * 3.6)
	lens.look = Vector3.ZERO

	var beat_edge: bool = f.beat > 0.55 and _beat_prev <= 0.55
	_beat_prev = f.beat
	if _oneshot:
		if not _fired and (beat_edge or _life > 1.2):
			_burst(1.0)
			_fired = true
	elif beat_edge:
		_burst(0.5)

	for s in _shards:
		s.pos += s.vel * delta
		# Re-orthonormalize: repeated incremental rotation accumulates float drift,
		# which would otherwise make slerp/quaternion conversion below fail.
		s.basis = (s.basis * Basis.from_euler(s.spin * delta)).orthonormalized()
		s.vel *= maxf(0.0, 1.0 - 1.1 * delta)        # drag
		s.spin *= maxf(0.0, 1.0 - 0.4 * delta)
		if not _oneshot:                              # pull back home and re-knit
			s.vel += (s.home - s.pos) * 2.6 * delta
			if (s.pos - s.home).length() < 0.02:
				s.basis = s.basis.slerp(Basis.IDENTITY, 1.0 - exp(-3.0 * delta))
	if not _settled():
		_moved = true
	queue_redraw()


func _draw() -> void:
	begin_draw()
	lens.prepare()
	var u := unit()
	# Depth-sort shards back-to-front so nearer ones occlude farther ones.
	var order := range(_shards.size())
	order.sort_custom(func(a, b): return lens.depth(_shards[a].pos) > lens.depth(_shards[b].pos))
	for idx in order:
		var s: Dictionary = _shards[idx]
		var poly := PackedVector2Array()
		var ok := true
		for v: Vector2 in s.poly:
			var world: Vector3 = s.basis * Vector3(v.x, v.y, 0.0) + s.pos
			var pr := lens.project(world)
			if pr.z <= lens.near:
				ok = false
				break
			poly.append(Vector2(pr.x, pr.y) * u)
		if not ok or poly.size() < 3 or Geo.area(poly) < 1.0:   # skip edge-on (degenerate) shards
			continue
		# Glassy shading: how squarely the shard faces the camera sets its glint.
		var nrm: Vector3 = (s.basis * Vector3(0, 0, 1)).normalized()
		var face := absf(nrm.dot((s.pos - lens.eye).normalized()))
		var scatter := clampf((s.pos - s.home).length() * 1.4, 0.0, 1.0)
		var val := clampf(0.28 + 0.45 * _f.energy + 0.35 * face, 0.1, 1.0)
		draw_colored_polygon(poly, Color.from_hsv(s.hue, 0.4, val, 0.45 + 0.35 * face))
		var edge := poly.duplicate()
		edge.append(poly[0])
		draw_polyline(edge, Color.from_hsv(s.hue, 0.15, 1.0, 0.35 + 0.5 * scatter), 1.0, true)
