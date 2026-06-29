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
const VIEW := Vector3(0.0, 0.0, 1.0)        # viewer direction, for specular highlights


# A projected triangle whose 2D area is sub-pixel (an edge-on or collapsed face)
# can't be triangulated by the canvas - draw_colored_polygon rejects it. Skip those
# faces; edge-on, they contribute nothing visible anyway.
static func _degenerate(poly: PackedVector2Array) -> bool:
	if poly.size() < 3:
		return true
	var a := poly[0]
	var b := poly[1]
	var c := poly[2]
	return absf((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) < 1.0

var verts := PackedVector3Array()
var faces: Array = []   # each face is a PackedInt32Array of 3 vertex indices
## Optional per-face surface variation in roughly -amp..amp (see [method texturize]),
## applied as a brightness/saturation mottle when drawn - the "texture" layer.
var face_tint := PackedFloat32Array()
## Optional per-face additive glow (0..1), added to each face's brightness when drawn -
## so faces can light up independently (e.g. an async per-face flicker). Empty = off.
var face_glow := PackedFloat32Array()
## Optional per-vertex normals (see [method compute_normals]) for smooth (Gouraud)
## shading - the colour is interpolated across each triangle instead of flat, so a
## subdivided sphere reads as a smooth ball, not faceted.
var vertex_normals := PackedVector3Array()


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


# Split every triangle into four. With `project` (the icosphere case) new midpoints
# are pushed back onto the sphere; without it (flat bases like a cube) they stay put,
# so subdivision just adds vertices to deform while keeping the geometric form.
func _subdivide(project := true) -> void:
	var cache := {}
	var new_faces: Array = []
	for f: PackedInt32Array in faces:
		var a := f[0]
		var b := f[1]
		var c := f[2]
		var ab := _midpoint(a, b, cache, project)
		var bc := _midpoint(b, c, cache, project)
		var ca := _midpoint(c, a, cache, project)
		new_faces.append(PackedInt32Array([a, ab, ca]))
		new_faces.append(PackedInt32Array([b, bc, ab]))
		new_faces.append(PackedInt32Array([c, ca, bc]))
		new_faces.append(PackedInt32Array([ab, bc, ca]))
	faces = new_faces


func _midpoint(i: int, j: int, cache: Dictionary, project := true) -> int:
	var key := "%d_%d" % [mini(i, j), maxi(i, j)]
	if cache.has(key):
		return cache[key]
	var mid := (verts[i] + verts[j]) * 0.5
	if project:
		mid = mid.normalized()
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


## A circular cap of `rings` concentric rings and `segments` around, radius 1 at the
## rim (z=0), with a spherical-cap profile to `height` at the centre: height > 0 is a
## dome bulging toward +z (a cornea/lens), height < 0 a funnel/recess (an iris bowl),
## 0 a flat disc. Built facing +z; orient it with the draw basis.
static func dome(rings: int, segments: int, height: float) -> Mesh3D:
	var m := Mesh3D.new()
	m.verts.append(Vector3(0, 0, height))                       # centre
	for ri in range(1, rings + 1):
		var rr := float(ri) / float(rings)
		var z := height * sqrt(maxf(0.0, 1.0 - rr * rr))       # spherical-cap profile
		for si in segments:
			var a := TAU * float(si) / float(segments)
			m.verts.append(Vector3(cos(a) * rr, sin(a) * rr, z))
	for si in segments:                                         # centre fan (first ring)
		m.faces.append(PackedInt32Array([0, 1 + si, 1 + (si + 1) % segments]))
	for ri in range(1, rings):                                  # strips between rings
		var b0 := 1 + (ri - 1) * segments
		var b1 := 1 + ri * segments
		for si in segments:
			var sn := (si + 1) % segments
			m.faces.append(PackedInt32Array([b0 + si, b1 + si, b1 + sn]))
			m.faces.append(PackedInt32Array([b0 + si, b1 + sn, b0 + sn]))
	return m


## Push each vertex out/in along its direction by seeded *uncorrelated* noise - a
## quick fuzzy lump. Neighbours are independent, so the surface reads as a noisy
## ball; for a believable solid prefer [method warp] (coherent), which is what the
## rock factory uses. Kept as a cheap primitive. Shared vertices stay watertight.
func displace(amp: float, rng: RandomNumberGenerator) -> void:
	for i in verts.size():
		verts[i] = verts[i] * (1.0 + rng.randf_range(-amp, amp))


## Coherent fractal displacement: move each vertex along its own direction by a
## sum of noise octaves sampled in 3D over the *position*, so neighbouring vertices
## move together. The surface deforms in smooth correlated lumps - a real mass with
## bulges and hollows - instead of the per-vertex fuzz [method displace] gives.
## `amp` is the peak fraction of radius, `detail` the octave count (finer crag),
## `gain` how much each finer octave contributes, `freq` the base lump scale.
func warp(amp: float, detail: int, gain: float, freq: float, rng: RandomNumberGenerator) -> void:
	var noise := FastNoiseLite.new()
	noise.noise_type = FastNoiseLite.TYPE_SIMPLEX
	noise.seed = rng.randi()
	noise.frequency = freq
	noise.fractal_type = FastNoiseLite.FRACTAL_FBM
	noise.fractal_octaves = detail
	noise.fractal_gain = gain
	for i in verts.size():
		var dir: Vector3 = verts[i].normalized()
		var n := noise.get_noise_3dv(dir)            # coherent, ~ -1..1
		verts[i] = dir * (verts[i].length() * (1.0 + amp * n))


## Shave flat faces by cutting against `count` random interior planes: any vertex
## poking past a plane is pushed back onto it. Turns a rounded lump into an angular
## stone with flat fracture faces (the conchoidal flats of broken rock, the facets
## of a crystal). Watertight: a shared vertex is only ever moved inward. `depth` is
## the closest a cut sits to the centre (smaller = deeper bites, blockier result).
func facet(count: int, depth: float, rng: RandomNumberGenerator) -> void:
	for c in count:
		var nrm := Vector3(rng.randf_range(-1, 1), rng.randf_range(-1, 1),
			rng.randf_range(-1, 1)).normalized()
		var d := rng.randf_range(depth, 1.0)
		for i in verts.size():
			var proj := verts[i].dot(nrm)
			if proj > d:
				verts[i] -= nrm * (proj - d)


## Non-uniform scale along the axes - rocks are rarely round, so each gets a
## seeded squashed / elongated proportion. Apply after warp/facet.
func stretch(s: Vector3) -> void:
	for i in verts.size():
		verts[i] = Vector3(verts[i].x * s.x, verts[i].y * s.y, verts[i].z * s.z)


## Compute per-vertex normals (area-weighted average of adjacent face normals) for
## smooth shading. Call once after the geometry is final; pass smooth = true to
## [method draw_through] to use them.
func compute_normals() -> void:
	vertex_normals.resize(verts.size())
	for i in verts.size():
		vertex_normals[i] = Vector3.ZERO
	for f: PackedInt32Array in faces:
		var fn := (verts[f[1]] - verts[f[0]]).cross(verts[f[2]] - verts[f[0]])
		for idx in f:
			vertex_normals[idx] = vertex_normals[idx] + fn
	for i in verts.size():
		var n: Vector3 = vertex_normals[i]
		vertex_normals[i] = n.normalized() if n.length() > 1e-6 else Vector3.UP


## Bake a coherent surface texture: a value-noise mottle sampled at each face
## centroid, stored in [member face_tint] and applied as a per-face brightness /
## saturation variation when drawn - patches of light and dark across the surface,
## the way real stone is never one flat colour. `amount` is the depth of the mottle.
func texturize(amount: float, freq: float, rng: RandomNumberGenerator) -> void:
	var noise := FastNoiseLite.new()
	noise.noise_type = FastNoiseLite.TYPE_SIMPLEX
	noise.seed = rng.randi()
	noise.frequency = freq
	noise.fractal_type = FastNoiseLite.FRACTAL_FBM
	noise.fractal_octaves = 3
	face_tint.resize(faces.size())
	for fi in faces.size():
		var f: PackedInt32Array = faces[fi]
		var c := (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
		face_tint[fi] = noise.get_noise_3dv(c) * amount


## Coherent fractal displacement *masked by a second noise field*, so the growth is
## patchy: where the mask is high the surface erupts into rock, where it is low it
## stays the underlying (geometric) form. The basis for "rock growing on a structure"
## - run it on a subdivided cube/octahedron and only parts crust over.
func warp_masked(amp: float, detail: int, gain: float, freq: float, mask_freq: float,
		rng: RandomNumberGenerator) -> void:
	var noise := FastNoiseLite.new()
	noise.noise_type = FastNoiseLite.TYPE_SIMPLEX
	noise.seed = rng.randi()
	noise.frequency = freq
	noise.fractal_type = FastNoiseLite.FRACTAL_FBM
	noise.fractal_octaves = detail
	noise.fractal_gain = gain
	var mask := FastNoiseLite.new()
	mask.noise_type = FastNoiseLite.TYPE_SIMPLEX
	mask.seed = rng.randi()
	mask.frequency = mask_freq
	for i in verts.size():
		var v := verts[i]
		var dir: Vector3 = v.normalized() if v.length() > 1e-6 else Vector3.UP
		var m := smoothstep(-0.1, 0.5, mask.get_noise_3dv(v))   # 0 = bare geometry, 1 = full rock
		var n := noise.get_noise_3dv(dir)
		verts[i] = v + dir * (v.length() * amp * n * m)


## A hybrid body: a crisp geometric base (cube / octahedron / tetrahedron) with rock
## crusting over part of it, where a noise mask says so ([method warp_masked]). Part
## machined, part grown - "rock growing upon the geometric structure."
static func hybrid(rng: RandomNumberGenerator) -> Mesh3D:
	var m: Mesh3D
	match rng.randi_range(0, 2):
		0: m = cube()
		1: m = octahedron()
		_: m = tetrahedron()
	for s in 3:
		m._subdivide(false)        # add vertices to grow on, keep the flat geometric form
	m.warp_masked(rng.randf_range(0.35, 0.55), 4, 0.55,
		rng.randf_range(1.6, 2.6), rng.randf_range(0.9, 1.7), rng)
	m.facet(rng.randi_range(2, 5), 0.6, rng)
	m.stretch(Vector3(
		rng.randf_range(0.85, 1.15), rng.randf_range(0.8, 1.1), rng.randf_range(0.85, 1.15)))
	m.texturize(rng.randf_range(0.15, 0.30), rng.randf_range(2.0, 4.0), rng)
	return m


## A believable stone, built from data rather than displayed as a sphere: a
## subdivided icosphere given coherent fractal mass ([method warp]), shaved into
## angular fracture faces ([method facet]), then stretched into a natural, non-round
## proportion. Style sets the surface character:
##   "plain"   - smooth, rounded mass, a couple of broad flats.
##   "rough"   - heavy lumps and many shallow facets, a craggy boulder.
##   "crystal" - lightly warped, many deep flats - a faceted gem.
static func rock(style: String, rng: RandomNumberGenerator) -> Mesh3D:
	var subdiv := 3 if style == "plain" else 2
	var m := icosphere(subdiv)
	match style:
		"rough":
			m.warp(0.34, 5, 0.55, 1.4, rng)
			m.facet(rng.randi_range(5, 9), 0.52, rng)
		"crystal":
			m.warp(0.14, 3, 0.5, 1.1, rng)
			m.facet(rng.randi_range(7, 12), 0.42, rng)
		_:
			m.warp(0.24, 4, 0.5, 1.3, rng)
			m.facet(rng.randi_range(2, 4), 0.7, rng)
	m.stretch(Vector3(
		rng.randf_range(0.82, 1.18), rng.randf_range(0.70, 1.05), rng.randf_range(0.82, 1.18)))
	var tex := 0.20 if style == "rough" else (0.10 if style == "crystal" else 0.15)
	m.texturize(tex, rng.randf_range(2.0, 4.0), rng)
	return m


## Draw the mesh, flat-shaded and depth sorted. `explode` pushes each face out
## along its normal; `edge` 0 none / 1 dark / 2 bright outlines; `face_alpha` < 1
## makes the faces translucent. `gloss` adds a specular highlight (a wet/polished
## look) whose tightness is set by `rough` (low rough = a sharp glossy glint, high
## rough = matte); per-face [member face_tint] mottles the surface like real texture.
func draw_shaded(ci: CanvasItem, basis: Basis, center: Vector2, scale: float,
		hue: float, sat: float, explode: float, edge: int, face_alpha := 1.0,
		glow := 0.0, gloss := 0.0, rough := 0.6) -> void:
	var light := LIGHT.normalized()
	var half := (light + VIEW).normalized()       # for the specular highlight
	var shininess := lerpf(48.0, 4.0, clampf(rough, 0.0, 1.0))
	var textured := face_tint.size() == faces.size()
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
		var spec := gloss * pow(maxf(0.0, n.dot(half)), shininess) if gloss > 0.0 else 0.0
		var tint := face_tint[fi] if textured else 0.0
		var bright := clampf((0.22 + 0.78 * maxf(0.0, n.dot(light)) + glow + spec) * (1.0 + tint), 0.0, 1.0)
		# Specular washes the colour toward white; texture pulls saturation around.
		var fsat := clampf(sat * (1.0 - 0.6 * spec) * (1.0 - 0.25 * absf(tint)), 0.0, 1.0)
		var push := n * explode
		var poly := PackedVector2Array()
		var ok := true
		for idx in f:
			var p: Vector3 = rv[idx] + push
			var denom := focal - p.z
			if denom < 0.05:                 # at/behind the focal plane: skip this face
				ok = false
				break
			poly.append(center + Vector2(p.x, p.y) * scale * (focal / denom))
		if not ok or _degenerate(poly):      # edge-on faces project to a line - skip
			continue
		ci.draw_colored_polygon(poly, Color.from_hsv(hue, fsat, bright, face_alpha))
		if edge == 1:
			var e := poly.duplicate(); e.append(poly[0])
			ci.draw_polyline(e, Color(0, 0, 0, 0.5), 1.0, true)
		elif edge == 2:
			var e2 := poly.duplicate(); e2.append(poly[0])
			ci.draw_polyline(e2, Color.from_hsv(hue, 0.15, 1.0, 0.7), 1.0, true)


## Build a gaussian alpha mask for the partial-reveal look: a coherent noise field
## thresholded into a sparse pattern of solid coat (alpha 1) and holes (alpha 0), with
## a soft edge. RGB stays white so the face colour shows through where the coat is
## present. `threshold` is the sampleable masking knob - low leaves mostly coat with a
## few bare patches, near 0 makes it roughly half coat / half holes. `soft` sets how
## feathered the hole edges are; `freq` the patch scale. One texture per rock, built
## once; the holes reveal the wireframe drawn beneath (see [method draw_revealed]).
static func reveal_texture(rng: RandomNumberGenerator, threshold := -0.15, soft := 0.16,
		freq := 0.05, size := 128) -> ImageTexture:
	var noise := FastNoiseLite.new()
	noise.noise_type = FastNoiseLite.TYPE_SIMPLEX
	noise.seed = rng.randi()
	noise.frequency = freq
	noise.fractal_type = FastNoiseLite.FRACTAL_FBM
	noise.fractal_octaves = 4
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	for y in size:
		for x in size:
			var n := noise.get_noise_2d(float(x), float(y))            # -1..1
			var a := smoothstep(threshold - soft, threshold + soft, n)  # coat where n > threshold
			img.set_pixel(x, y, Color(1.0, 1.0, 1.0, a))
	return ImageTexture.create_from_image(img)


## Draw the mesh as a *partially revealed* body: the solid shaded coat is punched
## through by a gaussian alpha mask ([param reveal_tex]) to expose the wireframe lattice
## beneath - patches of bare geometry, patches of crusted surface, uneven even across a
## single face (the mask is sampled per-pixel via the texture, not per-face). Back faces
## are culled, so the holes reveal only the near shell's wireframe, never the interior.
## The mask UVs come from each face's dominant object-space axis, so the pattern sticks
## to the rock as it tumbles instead of swimming in screen space.
func draw_revealed(ci: CanvasItem, basis: Basis, center: Vector2, scale: float,
		hue: float, sat: float, glow: float, wire_col: Color, reveal_tex: Texture2D) -> void:
	var light := LIGHT.normalized()
	var focal := 3.2
	var textured := face_tint.size() == faces.size()
	# Object-space bounding box, for normalizing planar UVs into 0..1 (no tiling needed).
	var bbmin := verts[0]
	var bbmax := verts[0]
	for v in verts:
		bbmin = Vector3(minf(bbmin.x, v.x), minf(bbmin.y, v.y), minf(bbmin.z, v.z))
		bbmax = Vector3(maxf(bbmax.x, v.x), maxf(bbmax.y, v.y), maxf(bbmax.z, v.z))
	var ext := Vector3(maxf(bbmax.x - bbmin.x, 1e-3), maxf(bbmax.y - bbmin.y, 1e-3),
		maxf(bbmax.z - bbmin.z, 1e-3))

	var rv := []
	rv.resize(verts.size())
	for i in verts.size():
		rv[i] = basis * verts[i]

	# Collect the front-facing faces (back-face culled) with their projected polygon,
	# coat colour, UVs, and depth.
	var fronts: Array = []
	for fi in faces.size():
		var f: PackedInt32Array = faces[fi]
		var v0: Vector3 = rv[f[0]]
		var v1: Vector3 = rv[f[1]]
		var v2: Vector3 = rv[f[2]]
		var n := (v1 - v0).cross(v2 - v0).normalized()
		var cz := (v0 + v1 + v2) / 3.0
		if n.dot(cz) < 0.0:
			n = -n
		if n.z <= 0.02:                       # facing away from the viewer (+z) - cull
			continue
		var poly := PackedVector2Array()
		var ok := true
		for idx in f:
			var p: Vector3 = rv[idx]
			var denom := focal - p.z
			if denom < 0.05:
				ok = false
				break
			poly.append(center + Vector2(p.x, p.y) * scale * (focal / denom))
		if not ok or _degenerate(poly):
			continue
		# Per-face planar UVs from the dominant OBJECT-space axis (so the mask sticks).
		var no := (verts[f[1]] - verts[f[0]]).cross(verts[f[2]] - verts[f[0]]).normalized()
		var ax := absf(no.x)
		var ay := absf(no.y)
		var az := absf(no.z)
		var uvs := PackedVector2Array()
		for idx in f:
			var vo: Vector3 = verts[idx]
			var uv: Vector2
			if ax >= ay and ax >= az:
				uv = Vector2((vo.y - bbmin.y) / ext.y, (vo.z - bbmin.z) / ext.z)
			elif ay >= az:
				uv = Vector2((vo.x - bbmin.x) / ext.x, (vo.z - bbmin.z) / ext.z)
			else:
				uv = Vector2((vo.x - bbmin.x) / ext.x, (vo.y - bbmin.y) / ext.y)
			uvs.append(uv)
		var tint := face_tint[fi] if textured else 0.0
		var bright := clampf((0.22 + 0.78 * maxf(0.0, n.dot(light)) + glow) * (1.0 + tint), 0.0, 1.0)
		var fsat := clampf(sat * (1.0 - 0.25 * absf(tint)), 0.0, 1.0)
		fronts.append({"poly": poly, "uvs": uvs,
			"col": Color.from_hsv(hue, fsat, bright), "z": cz.z})

	# Pass 1: the wireframe lattice (every front face's edges), drawn first so the coat
	# can cover it where present and the holes reveal it where absent.
	for fr in fronts:
		var e: PackedVector2Array = (fr.poly as PackedVector2Array).duplicate()
		e.append(fr.poly[0])
		ci.draw_polyline(e, wire_col, 1.0, true)

	# Pass 2: the alpha-masked coat, far to near, so nearer coat correctly occludes.
	fronts.sort_custom(func(a, b): return a.z < b.z)
	for fr in fronts:
		ci.draw_colored_polygon(fr.poly, fr.col, fr.uvs, reveal_tex)


## Draw the mesh as a body in a [Lens3D] world: rotate by `basis`, scale, place at
## `pos` (world units), then project every vertex through the camera - true forced
## perspective, not the fixed centred projector of [method draw_shaded]. Faces are
## depth-sorted against the camera and flat-shaded; normals face the eye. `u_px` is
## the pixel unit (projection returns centred unit-fractions). This is the [Scene3D]
## counterpart used by the unified path; draw_shaded stays for the legacy centred
## look. A face is skipped if any of its vertices fall behind the camera.
func draw_through(ci: CanvasItem, lens: Lens3D, u_px: float, basis: Basis, pos: Vector3,
		scale: float, hue: float, sat: float, edge: int, face_alpha := 1.0,
		glow := 0.0, explode := 0.0, gloss := 0.0, rough := 0.6,
		unlit := Color(0, 0, 0, 0), smooth := false) -> void:
	var light := LIGHT.normalized()
	var shininess := lerpf(48.0, 4.0, clampf(rough, 0.0, 1.0))
	var textured := face_tint.size() == faces.size()
	var wv := []                      # world-space vertices
	wv.resize(verts.size())
	for i in verts.size():
		wv[i] = basis * (verts[i] * scale) + pos

	var depth := []                   # per-face camera depth, for painter sort
	depth.resize(faces.size())
	for fi in faces.size():
		var f: PackedInt32Array = faces[fi]
		var z := 0.0
		for idx in f:
			z += lens.depth(wv[idx])
		depth[fi] = z / float(f.size())
	var order := range(faces.size())
	order.sort_custom(func(a, b): return depth[a] > depth[b])   # far first

	# Smooth (Gouraud) path: per-vertex normals -> per-vertex colour interpolated
	# across each triangle, so a sphere reads smooth, not faceted. Back-face culled.
	if smooth and vertex_normals.size() == verts.size():
		for fi in order:
			var f: PackedInt32Array = faces[fi]
			var a0: Vector3 = wv[f[0]]
			var a1: Vector3 = wv[f[1]]
			var a2: Vector3 = wv[f[2]]
			var cen := (a0 + a1 + a2) / 3.0
			if (a1 - a0).cross(a2 - a0).dot(cen - lens.eye) > 0.0:
				continue                       # back-facing
			var poly := PackedVector2Array()
			var cols := PackedColorArray()
			var ok := true
			for idx in f:
				var wp: Vector3 = wv[idx]
				var pr := lens.project(wp)
				if pr.z <= lens.near:
					ok = false
					break
				poly.append(Vector2(pr.x, pr.y) * u_px)
				var nw: Vector3 = (basis * vertex_normals[idx]).normalized()
				var view: Vector3 = (lens.eye - wp).normalized()
				var sp := gloss * pow(maxf(0.0, nw.dot((light + view).normalized())), shininess) if gloss > 0.0 else 0.0
				var fgv := face_glow[fi] if face_glow.size() == faces.size() else 0.0
				var b := clampf(0.22 + 0.78 * maxf(0.0, nw.dot(light)) + glow + fgv + sp, 0.0, 1.0)
				cols.append(Color.from_hsv(hue, clampf(sat * (1.0 - 0.6 * sp), 0.0, 1.0), b, face_alpha))
			if ok and not _degenerate(poly):
				ci.draw_polygon(poly, cols)
		return

	for fi in order:
		var f: PackedInt32Array = faces[fi]
		var v0: Vector3 = wv[f[0]]
		var v1: Vector3 = wv[f[1]]
		var v2: Vector3 = wv[f[2]]
		var n := (v1 - v0).cross(v2 - v0).normalized()
		var cen := (v0 + v1 + v2) / 3.0
		if n.dot(cen - lens.eye) > 0.0:        # facing away from the camera -> flip
			n = -n
		var col: Color
		if unlit.a > 0.0:                       # flat, unlit (e.g. a black pupil)
			col = Color(unlit.r, unlit.g, unlit.b, unlit.a * face_alpha)
		else:
			var view := (lens.eye - cen).normalized()
			var spec := gloss * pow(maxf(0.0, n.dot((light + view).normalized())), shininess) if gloss > 0.0 else 0.0
			var tint := face_tint[fi] if textured else 0.0
			var fg := face_glow[fi] if face_glow.size() == faces.size() else 0.0
			var bright := clampf((0.22 + 0.78 * maxf(0.0, n.dot(light)) + glow + fg + spec) * (1.0 + tint), 0.0, 1.0)
			var fsat := clampf(sat * (1.0 - 0.6 * spec) * (1.0 - 0.25 * absf(tint)), 0.0, 1.0)
			col = Color.from_hsv(hue, fsat, bright, face_alpha)
		var push := n * explode
		var poly := PackedVector2Array()
		var ok := true
		for idx in f:
			var pr := lens.project(wv[idx] + push)
			if pr.z <= lens.near:
				ok = false
				break
			poly.append(Vector2(pr.x, pr.y) * u_px)
		if not ok or _degenerate(poly):
			continue
		ci.draw_colored_polygon(poly, col)
		if edge == 1:
			var e := poly.duplicate(); e.append(poly[0])
			ci.draw_polyline(e, Color(0, 0, 0, 0.5), 1.0, true)
		elif edge == 2:
			var e2 := poly.duplicate(); e2.append(poly[0])
			ci.draw_polyline(e2, Color.from_hsv(hue, 0.15, 1.0, 0.7), 1.0, true)
