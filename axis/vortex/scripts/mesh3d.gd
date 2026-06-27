extends RefCounted
class_name Mesh3D

## Mesh3D - a tiny software 3D mesh for the 2D canvas.
##
## Real 3D where flat polygons sheared by the 2D view read as cardboard: an
## icosphere (optionally displaced into a lump) drawn as depth-sorted, flat-shaded
## triangles under perspective. Rotation is a genuine [Basis], so tumbling looks
## dimensional. Faces can be pushed out along their normals for an exploded view.
## Reusable - any scene that wants a solid spinning body uses this, not its own.

const LIGHT := Vector3(0.35, -0.55, 0.75)   # upper-front key light (normalized below)

var verts := PackedVector3Array()
var faces: Array = []   # each face is a PackedInt32Array of 3 vertex indices


## A subdivided icosphere (subdiv 0 = 20-face icosahedron, 1 = 80, 2 = 320).
static func icosphere(subdiv := 1) -> Mesh3D:
	var m := Mesh3D.new()
	var t := (1.0 + sqrt(5.0)) / 2.0
	var raw := [
		Vector3(-1, t, 0), Vector3(1, t, 0), Vector3(-1, -t, 0), Vector3(1, -t, 0),
		Vector3(0, -1, t), Vector3(0, 1, t), Vector3(0, -1, -t), Vector3(0, 1, -t),
		Vector3(t, 0, -1), Vector3(t, 0, 1), Vector3(-t, 0, -1), Vector3(-t, 0, 1)]
	for v: Vector3 in raw:
		m.verts.append(v.normalized())
	m.faces = [
		PackedInt32Array([0, 11, 5]), PackedInt32Array([0, 5, 1]), PackedInt32Array([0, 1, 7]),
		PackedInt32Array([0, 7, 10]), PackedInt32Array([0, 10, 11]), PackedInt32Array([1, 5, 9]),
		PackedInt32Array([5, 11, 4]), PackedInt32Array([11, 10, 2]), PackedInt32Array([10, 7, 6]),
		PackedInt32Array([7, 1, 8]), PackedInt32Array([3, 9, 4]), PackedInt32Array([3, 4, 2]),
		PackedInt32Array([3, 2, 6]), PackedInt32Array([3, 6, 8]), PackedInt32Array([3, 8, 9]),
		PackedInt32Array([4, 9, 5]), PackedInt32Array([2, 4, 11]), PackedInt32Array([6, 2, 10]),
		PackedInt32Array([8, 6, 7]), PackedInt32Array([9, 8, 1])]
	for s in subdiv:
		m._subdivide()
	return m


# Split every triangle into four, projecting new midpoints back onto the sphere.
func _subdivide() -> void:
	var cache := {}
	var new_faces: Array = []
	for f: PackedInt32Array in faces:
		var a := f[0]
		var b := f[1]
		var c := f[2]
		var ab := _midpoint(a, b, cache)
		var bc := _midpoint(b, c, cache)
		var ca := _midpoint(c, a, cache)
		new_faces.append(PackedInt32Array([a, ab, ca]))
		new_faces.append(PackedInt32Array([b, bc, ab]))
		new_faces.append(PackedInt32Array([c, ca, bc]))
		new_faces.append(PackedInt32Array([ab, bc, ca]))
	faces = new_faces


func _midpoint(i: int, j: int, cache: Dictionary) -> int:
	var key := "%d_%d" % [mini(i, j), maxi(i, j)]
	if cache.has(key):
		return cache[key]
	var mid := ((verts[i] + verts[j]) * 0.5).normalized()
	var idx := verts.size()
	verts.append(mid)
	cache[key] = idx
	return idx


## A unit cube (12 triangles).
static func cube() -> Mesh3D:
	var m := Mesh3D.new()
	m.verts = PackedVector3Array([
		Vector3(-1, -1, -1), Vector3(1, -1, -1), Vector3(1, 1, -1), Vector3(-1, 1, -1),
		Vector3(-1, -1, 1), Vector3(1, -1, 1), Vector3(1, 1, 1), Vector3(-1, 1, 1)])
	m.faces = [
		PackedInt32Array([0, 1, 2]), PackedInt32Array([0, 2, 3]),   # -z
		PackedInt32Array([4, 5, 6]), PackedInt32Array([4, 6, 7]),   # +z
		PackedInt32Array([0, 1, 5]), PackedInt32Array([0, 5, 4]),   # -y
		PackedInt32Array([3, 2, 6]), PackedInt32Array([3, 6, 7]),   # +y
		PackedInt32Array([0, 3, 7]), PackedInt32Array([0, 7, 4]),   # -x
		PackedInt32Array([1, 2, 6]), PackedInt32Array([1, 6, 5])]   # +x
	return m


## A regular octahedron (8 triangles).
static func octahedron() -> Mesh3D:
	var m := Mesh3D.new()
	m.verts = PackedVector3Array([
		Vector3(1, 0, 0), Vector3(-1, 0, 0), Vector3(0, 1, 0),
		Vector3(0, -1, 0), Vector3(0, 0, 1), Vector3(0, 0, -1)])
	m.faces = [
		PackedInt32Array([0, 2, 4]), PackedInt32Array([0, 4, 3]),
		PackedInt32Array([0, 3, 5]), PackedInt32Array([0, 5, 2]),
		PackedInt32Array([1, 2, 5]), PackedInt32Array([1, 5, 3]),
		PackedInt32Array([1, 3, 4]), PackedInt32Array([1, 4, 2])]
	return m


## A regular tetrahedron (4 triangles).
static func tetrahedron() -> Mesh3D:
	var m := Mesh3D.new()
	m.verts = PackedVector3Array([
		Vector3(1, 1, 1), Vector3(1, -1, -1), Vector3(-1, 1, -1), Vector3(-1, -1, 1)])
	m.faces = [
		PackedInt32Array([0, 1, 2]), PackedInt32Array([0, 2, 3]),
		PackedInt32Array([0, 3, 1]), PackedInt32Array([1, 3, 2])]
	return m


## Push each vertex out/in along its direction by seeded noise - turns the sphere
## into an irregular lump. Shared vertices keep the mesh watertight.
func displace(amp: float, rng: RandomNumberGenerator) -> void:
	for i in verts.size():
		verts[i] = verts[i] * (1.0 + rng.randf_range(-amp, amp))


## Draw the mesh, flat-shaded and depth sorted. `explode` pushes each face out
## along its normal; `edge` 0 none / 1 dark / 2 bright outlines; `face_alpha` < 1
## makes the faces translucent (a see-through solid - real 3D, not a wireframe).
func draw_shaded(ci: CanvasItem, basis: Basis, center: Vector2, scale: float,
		hue: float, sat: float, explode: float, edge: int, face_alpha := 1.0,
		glow := 0.0) -> void:
	var light := LIGHT.normalized()
	var focal := 3.2
	var rv := []                      # rotated vertices
	rv.resize(verts.size())
	for i in verts.size():
		rv[i] = basis * verts[i]

	var depth := []                   # per-face mean z, for painter sort
	depth.resize(faces.size())
	for fi in faces.size():
		var f: PackedInt32Array = faces[fi]
		var z := 0.0
		for idx in f:
			z += rv[idx].z
		depth[fi] = z / float(f.size())
	var order := range(faces.size())
	order.sort_custom(func(a, b): return depth[a] < depth[b])   # far first

	for fi in order:
		var f: PackedInt32Array = faces[fi]
		var v0: Vector3 = rv[f[0]]
		var v1: Vector3 = rv[f[1]]
		var v2: Vector3 = rv[f[2]]
		var n := (v1 - v0).cross(v2 - v0).normalized()
		var cz := (v0 + v1 + v2) / 3.0
		if n.dot(cz) < 0.0:
			n = -n
		var bright := clampf(0.22 + 0.78 * maxf(0.0, n.dot(light)) + glow, 0.0, 1.0)
		var push := n * explode
		var poly := PackedVector2Array()
		for idx in f:
			var p: Vector3 = rv[idx] + push
			var pr := focal / (focal - p.z)
			poly.append(center + Vector2(p.x, p.y) * scale * pr)
		ci.draw_colored_polygon(poly, Color.from_hsv(hue, sat, bright, face_alpha))
		if edge == 1:
			var e := poly.duplicate(); e.append(poly[0])
			ci.draw_polyline(e, Color(0, 0, 0, 0.5), 1.0, true)
		elif edge == 2:
			var e2 := poly.duplicate(); e2.append(poly[0])
			ci.draw_polyline(e2, Color.from_hsv(hue, 0.15, 1.0, 0.7), 1.0, true)
