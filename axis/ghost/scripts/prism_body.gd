extends RefCounted
class_name PrismBody

## PrismBody - the browser Prism, ported (praxis web, dark mode).
##
## A see-through wireframe tetrahedron: only the four corner points and the thin
## glowing edges are drawn - NO faces, NO fill. At the centre floats a living core:
## fine neural tendrils that flow and branch outward from the middle, staying inside
## the shell, growing and fading, surging brighter with the audio. The shell's edges
## light up where the tendrils approach them (so the wireframe flickers with the
## core's activity), and the whole thing hovers and slowly rotates "looking around".
## Reusable: the `prism` and `prism_split` scenes both draw these. Faithful to the
## browser's geometry (same vertices) and projection (`2/(2 - z·0.3)`, y squashed).

const APEX := Vector3(0.0, -0.5, 0.0)
const BACK := Vector3(0.0, 0.5, -0.577)
const LEFT := Vector3(-0.5, 0.5, 0.289)
const RIGHT := Vector3(0.5, 0.5, 0.289)
const VERTS := [APEX, BACK, LEFT, RIGHT]
# Each edge: [a, b, is_apex]. Apex edges read brighter than the base triangle.
const EDGES := [
	[APEX, BACK, true], [APEX, LEFT, true], [APEX, RIGHT, true],
	[BACK, LEFT, false], [LEFT, RIGHT, false], [RIGHT, BACK, false]]

const TENDRIL_MAX_LEN := 0.46     # keep tendrils inside the shell

var rot := Vector3.ZERO
var _vel := Vector3.ZERO
var _rng := RandomNumberGenerator.new()
var _tendrils: Array = []
var _energy := 0.0
var _t := 0.0


func _init(seed_value := 0) -> void:
	_rng.seed = seed_value
	_vel = Vector3(_rng.randf_range(0.18, 0.42), _rng.randf_range(0.28, 0.55), _rng.randf_range(0.08, 0.22))
	rot = Vector3(_rng.randf() * TAU, _rng.randf() * TAU, _rng.randf() * TAU)
	for i in 12:
		_tendrils.append(_new_tendril())


func _new_tendril() -> Dictionary:
	var dir: Vector3
	if _rng.randf() < 0.30:                                  # aim some at the corners/edges
		dir = (VERTS[_rng.randi() % VERTS.size()]
			+ Vector3(_rng.randf_range(-0.3, 0.3), _rng.randf_range(-0.3, 0.3), _rng.randf_range(-0.3, 0.3)))
	else:
		var phi := _rng.randf() * TAU
		var theta := (_rng.randf() - 0.5) * PI
		dir = Vector3(cos(theta) * cos(phi), sin(theta), cos(theta) * sin(phi))
	if dir.length() < 1e-4:
		dir = Vector3.UP
	return {
		"dir": dir.normalized(),
		"len": _rng.randf_range(0.30, TENDRIL_MAX_LEN),
		"life": 0.0, "maxlife": _rng.randf_range(0.7, 1.6),
		"op": 0.0, "grow": true, "prog": 0.0,
		"wave": _rng.randf_range(0.015, 0.035), "phase": _rng.randf() * TAU,
		"thick": _rng.randf_range(0.8, 2.0)}


## Advance the core and rotation. `drive` (0..1) is the audio energy - it surges the
## tendril count and brightens the core (the core "comes to life" on the music).
func update(dt: float, drive: float) -> void:
	_t += dt
	_energy = lerpf(_energy, clampf(drive, 0.0, 1.0), 1.0 - exp(-5.0 * dt))
	rot += _vel * dt * (0.4 + 0.7 * _energy)               # looks around faster when lively
	var want := int(lerpf(10.0, 42.0, _energy))            # surge with energy
	while _tendrils.size() < want:
		_tendrils.append(_new_tendril())
	for td in _tendrils:
		td.life += dt
		td.prog = minf(1.0, td.prog + dt * 3.0)
		if td.grow:
			td.op = minf(0.7, td.op + dt * 2.0)
			if td.life > td.maxlife * 0.6:
				td.grow = false
		else:
			td.op = maxf(0.0, td.op - dt * 1.4)
	for i in range(_tendrils.size() - 1, -1, -1):
		if _tendrils[i].op <= 0.0 and not _tendrils[i].grow:
			if _tendrils.size() > want:
				_tendrils.remove_at(i)
			else:
				_tendrils[i] = _new_tendril()


# Browser projection: weak perspective on z, y squashed to 0.8.
func _project(p: Vector3, center: Vector2, scale: float) -> Vector2:
	var persp := 2.0 / (2.0 - p.z * 0.3)
	return center + Vector2(p.x * persp, p.y * persp * 0.8) * scale


## Draw the prism at [param center], [param scale] px, in [param hue] (blue ≈ 0.6,
## red ≈ 0.0). [param fade] (0..1) scales overall opacity for transitions.
func draw(ci: CanvasItem, center: Vector2, scale: float, hue: float, fade := 1.0) -> void:
	var basis := Basis.from_euler(rot)
	# Project tendril tips once, for edge illumination.
	var tips: Array = []
	for td in _tendrils:
		tips.append((td.dir as Vector3) * (float(td.len) * float(td.prog)))

	# Edges, lit where tendrils pass near them (the wireframe flickers with the core).
	for e in EDGES:
		var a: Vector3 = basis * e[0]
		var b: Vector3 = basis * e[1]
		var illum := 0.0
		for j in tips.size():
			var d := _point_seg_dist(basis * tips[j], a, b)
			if d < 0.4:
				illum += (1.0 - d / 0.4) * float(_tendrils[j].op) * 0.5
		illum = clampf(0.12 + illum, 0.0, 1.0)            # a faint base + tendril boost
		var pa := _project(a, center, scale)
		var pb := _project(b, center, scale)
		var es := 0.45 if e[2] else 0.30                  # apex edges brighter
		var col := Color.from_hsv(hue, 0.35 if e[2] else 0.6, 1.0, illum * fade)
		_glow_line(ci, pa, pb, col, (1.0 + illum * 2.0), es * fade)

	# Tendrils: a wavy line from the centre outward, white-hot core -> coloured tip.
	for td in _tendrils:
		if td.op <= 0.01:
			continue
		var pts := PackedVector2Array()
		var segs := 9
		var ln := float(td.len)
		var pr := float(td.prog)
		for i in segs + 1:
			var t := float(i) / float(segs)
			var w := sin(_t * 4.0 + t * 5.0 + float(td.phase)) * float(td.wave) * (1.0 - t * 0.4)
			var local: Vector3 = (td.dir as Vector3) * (t * ln * pr) + Vector3(w, w * 0.6, w * 0.3)
			pts.append(_project(basis * local, center, scale))
		var op: float = float(td.op) * fade
		var thick := float(td.thick)
		ci.draw_polyline(pts, Color.from_hsv(hue, 0.5, 1.0, op), thick, true)
		# bright inner strand + tip spark
		ci.draw_polyline(pts, Color(1, 1, 1, op * 0.5), thick * 0.4, true)
		if pr > 0.85:
			ci.draw_circle(pts[pts.size() - 1], thick * 0.9, Color(1, 1, 1, op * 0.6))

	# The bright living nucleus at the very centre, pulsing with energy.
	var core := 0.5 + 0.5 * _energy
	for layer in 4:
		var r := scale * (0.02 + 0.05 * float(layer)) * (0.7 + 0.6 * core)
		ci.draw_circle(center, r, Color.from_hsv(hue, 0.3, 1.0, (0.10 + 0.25 * core) * fade / float(layer + 1)))


# A cheap bloom: the same line drawn wide+faint then thin+bright.
func _glow_line(ci: CanvasItem, a: Vector2, b: Vector2, col: Color, width: float, glow: float) -> void:
	ci.draw_line(a, b, Color(col.r, col.g, col.b, col.a * 0.25 * glow), width * 3.0, true)
	ci.draw_line(a, b, col, width, true)


# Distance from point p to segment a-b (3D).
func _point_seg_dist(p: Vector3, a: Vector3, b: Vector3) -> float:
	var ab := b - a
	var len2 := ab.length_squared()
	if len2 < 1e-9:
		return p.distance_to(a)
	var t := clampf((p - a).dot(ab) / len2, 0.0, 1.0)
	return p.distance_to(a + ab * t)
