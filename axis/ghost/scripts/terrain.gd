extends RefCounted
class_name Terrain

## Terrain - a heightfield assembled from [Field]s and drawn through a [Lens3D].
##
## The composable landscape: a type (rolling hills, mountains, valleys, fissured canyon,
## islands, banded mesa) selects a recipe of [Field]s that becomes a height grid, sampled
## ONCE at build (terrain is static) into world vertices and per-vertex colours from a
## [Palette] plus a fine surface-texture field and slope shading. Thereafter the grid is
## just projected and depth-sorted each frame - cheap. It exposes height(wx, wz) and a
## surface normal so other scenes can stand things on it (blocks, water, growth).

const RES := 64                  # grid resolution (RES x RES vertices)

var res := RES
var half := 3.0                  # world half-extent in x and z
var relief := 1.4                # vertical world scale
var water := 0.0                 # water plane world height (0 = none)
var type := "hills"
var palette: Palette
var hgrid := PackedFloat32Array()   # heights 0..1
var _world := PackedVector3Array()  # world-space vertices
var _vcol: PackedColorArray         # base per-vertex colour (palette + texture + slope)


func build(rng: RandomNumberGenerator, type_: String, world_half := 3.0,
		relief_ := 1.4, pal: Palette = null) -> void:
	type = type_
	half = world_half
	relief = relief_
	palette = pal if pal != null else Palette.named("earth", rng)
	var height := _recipe(rng)
	var detail := Field.make("fbm", rng.randi(), 9.0, 4)     # fine surface texture
	hgrid.resize(res * res)
	_world.resize(res * res)
	_vcol = PackedColorArray()
	_vcol.resize(res * res)
	# Sample the field into the grid (the only expensive pass; done once).
	for gy in res:
		for gx in res:
			var p := Vector2(float(gx) / float(res - 1) - 0.5, float(gy) / float(res - 1) - 0.5) * 2.0
			hgrid[gy * res + gx] = height.at(p)
	_smooth(2)        # kill grid-scale aliasing so the land rolls instead of spiking
	for gy in res:
		for gx in res:
			var i := gy * res + gx
			var p := Vector2(float(gx) / float(res - 1) - 0.5, float(gy) / float(res - 1) - 0.5) * 2.0
			_world[i] = Vector3(p.x * half, (hgrid[i] - water) * relief, p.y * half)
	# Per-vertex colour: palette by height, mottled by the detail field, shaded by slope.
	for gy in res:
		for gx in res:
			var i := gy * res + gx
			var p := Vector2(float(gx) / float(res - 1) - 0.5, float(gy) / float(res - 1) - 0.5) * 2.0
			var n := _normal(gx, gy)
			var slope := clampf(n.dot(Vector3(0, 1, 0)), 0.0, 1.0)     # 1 flat .. 0 cliff
			var tex := detail.at(p)
			var c := palette.at(clampf(hgrid[i] + 0.12 * (tex - 0.5), 0.0, 1.0))
			var shade := 0.45 + 0.55 * slope + 0.18 * (tex - 0.5)      # cliffs darker
			_vcol[i] = Color(c.r * shade, c.g * shade, c.b * shade, 1.0)


# A few box-blur passes over the height grid - removes high-frequency aliasing so the
# surface reads as rolling land, while peaks and valleys keep their large-scale shape.
func _smooth(passes: int) -> void:
	var tmp := PackedFloat32Array()
	tmp.resize(res * res)
	for _p in passes:
		for gy in res:
			for gx in res:
				var s := 0.0
				var c := 0
				for dy in [-1, 0, 1]:
					for dx in [-1, 0, 1]:
						var nx: int = gx + dx
						var ny: int = gy + dy
						if nx >= 0 and nx < res and ny >= 0 and ny < res:
							s += hgrid[ny * res + nx]
							c += 1
				tmp[gy * res + gx] = s / float(c)
		for i in res * res:
			hgrid[i] = tmp[i]


# Recipe: compose the height [Field] for each terrain type.
func _recipe(rng: RandomNumberGenerator) -> Field:
	var f := rng.randf_range(0.7, 1.05)
	match type:
		"mountains":
			water = 0.18
			relief = maxf(relief, 2.0)
			var base := Field.make("fbm", rng.randi(), f * 0.7).scale(0.45)
			var peaks := Field.make("ridged", rng.randi(), f * 1.5, 5).curve("smoothstep")
			var where := Field.make("fbm", rng.randi(), f * 0.5).curve("sigmoid", 5.0)
			return Field.combine(base.offset(0.18), "add", Field.combine(peaks, "mask", where, 1.0), 0.85)
		"valleys":
			water = 0.20
			var rolling := Field.make("fbm", rng.randi(), f).offset(0.30).scale(0.6)
			var rivers := Field.make("ridged", rng.randi(), f * 0.9, 4).inverted().curve("spike", 2.0)
			return Field.combine(rolling, "sub", rivers, 0.5)
		"canyon":
			water = 0.10
			relief = maxf(relief, 1.8)
			var plateau := Field.make("fbm", rng.randi(), f * 0.6).curve("smoothstep").scale(0.4).offset(0.5)
			var cracks := Field.make("cells", rng.randi(), f * 0.9).curve("spike", 3.0)
			return Field.combine(plateau, "sub", cracks, 0.7)
		"islands":
			water = 0.42
			var land := Field.make("fbm", rng.randi(), f * 1.1, 5)
			var bowl := Field.make("gradient", 0).scale(1.4).curve("smoothstep")   # centre high
			return Field.combine(land, "mask", bowl, 1.0)
		"mesa":
			water = 0.12
			var bands := Field.make("fbm", rng.randi(), f * 0.8).strata(rng.randf_range(4.0, 7.0))
			var base := Field.make("fbm", rng.randi(), f * 0.5).scale(0.5).offset(0.25)
			return Field.combine(base, "add", bands, 0.4)
		_:  # rolling hills
			water = 0.0
			var warpf := Field.make("fbm", rng.randi(), f * 0.5)
			return Field.make("fbm", rng.randi(), f, 4).warp(warpf, 0.4).offset(0.05)


# World height (relief units, water-relative) at continuous world (wx, wz), bilinear.
func height_at(wx: float, wz: float) -> float:
	var gx := (wx / half * 0.5 + 0.5) * float(res - 1)
	var gy := (wz / half * 0.5 + 0.5) * float(res - 1)
	var x0 := clampi(int(floor(gx)), 0, res - 1)
	var y0 := clampi(int(floor(gy)), 0, res - 1)
	var x1 := mini(x0 + 1, res - 1)
	var y1 := mini(y0 + 1, res - 1)
	var fx := clampf(gx - float(x0), 0.0, 1.0)
	var fy := clampf(gy - float(y0), 0.0, 1.0)
	var h00 := hgrid[y0 * res + x0]
	var h10 := hgrid[y0 * res + x1]
	var h01 := hgrid[y1 * res + x0]
	var h11 := hgrid[y1 * res + x1]
	return lerpf(lerpf(h00, h10, fx), lerpf(h01, h11, fx), fy) - water


# Surface normal at grid cell (gx, gy), from central differences on the height grid.
func _normal(gx: int, gy: int) -> Vector3:
	var l := hgrid[gy * res + maxi(gx - 1, 0)]
	var r := hgrid[gy * res + mini(gx + 1, res - 1)]
	var d := hgrid[maxi(gy - 1, 0) * res + gx]
	var u := hgrid[mini(gy + 1, res - 1) * res + gx]
	var sx := (r - l) * relief * float(res) / (4.0 * half)
	var sz := (u - d) * relief * float(res) / (4.0 * half)
	return Vector3(-sx, 1.0, -sz).normalized()


func normal_world(wx: float, wz: float) -> Vector3:
	var gx := clampi(int((wx / half * 0.5 + 0.5) * float(res - 1)), 0, res - 1)
	var gy := clampi(int((wz / half * 0.5 + 0.5) * float(res - 1)), 0, res - 1)
	return _normal(gx, gy)


## Project + depth-sort + draw the terrain surface (and water) through the lens. `lit`
## scales brightness (audio); `shimmer` (time) animates the water.
func draw_surface(ci: CanvasItem, lens: Lens3D, u: float, lit: float, shimmer: float) -> void:
	var n := res * res
	var sv := PackedVector2Array()
	sv.resize(n)
	var dep := PackedFloat32Array()
	dep.resize(n)
	for i in n:
		var pr := lens.project(_world[i])
		sv[i] = Vector2(pr.x, pr.y) * u
		dep[i] = pr.z
	var quads: Array = []
	for gy in res - 1:
		for gx in res - 1:
			var i0 := gy * res + gx
			var i1 := i0 + 1
			var i2 := i0 + res
			var i3 := i2 + 1
			if dep[i0] <= lens.near or dep[i1] <= lens.near or dep[i2] <= lens.near or dep[i3] <= lens.near:
				continue
			var poly := PackedVector2Array([sv[i0], sv[i1], sv[i3], sv[i2]])
			if _quad_area(poly) < 2.0:        # edge-on / collapsed quad - skip (else triangulation fails)
				continue
			var col: Color = _vcol[i0]
			quads.append({"d": (dep[i0] + dep[i1] + dep[i2] + dep[i3]) * 0.25,
				"poly": poly, "col": Color(col.r * lit, col.g * lit, col.b * lit, 1.0)})
	# Water plane (a coarse translucent grid at y=0), shimmering, depth-sorted with the land.
	if water > 0.0:
		var wc := palette.at(0.0)
		var wr := 12
		for gy in wr:
			for gx in wr:
				var c0 := _wpt(gx, gy, wr, lens, u)
				var c1 := _wpt(gx + 1, gy, wr, lens, u)
				var c2 := _wpt(gx, gy + 1, wr, lens, u)
				var c3 := _wpt(gx + 1, gy + 1, wr, lens, u)
				if c0.z <= lens.near or c1.z <= lens.near or c2.z <= lens.near or c3.z <= lens.near:
					continue
				var wpoly := PackedVector2Array([Vector2(c0.x, c0.y), Vector2(c1.x, c1.y),
					Vector2(c3.x, c3.y), Vector2(c2.x, c2.y)])
				if _quad_area(wpoly) < 2.0:
					continue
				var sh := 0.85 + 0.15 * sin(shimmer * 1.3 + float(gx) * 0.7 + float(gy) * 0.5)
				quads.append({"d": (c0.z + c1.z + c2.z + c3.z) * 0.25,
					"poly": wpoly, "col": Color(wc.r * sh * lit, wc.g * sh * lit, wc.b * sh * lit, 0.62)})
	quads.sort_custom(func(a, b): return a.d > b.d)        # far first
	for q in quads:
		draw_quad(ci, q.poly, q.col)


# Screen-space area of a quad (shoelace). Near-zero => the quad is edge-on / collapsed /
# folded, which makes draw_colored_polygon's triangulation fail - so we skip those.
static func _quad_area(p: PackedVector2Array) -> float:
	var a := 0.0
	for i in p.size():
		var j := (i + 1) % p.size()
		a += p[i].x * p[j].y - p[j].x * p[i].y
	return absf(a) * 0.5


## Draw a 4-point quad as its two triangles (split on the 0-2 diagonal). A projected
## heightfield quad can fold into a bowtie that draw_colored_polygon can't triangulate;
## two triangles never can, and degenerate (collinear) triangles are skipped.
static func draw_quad(ci: CanvasItem, poly: PackedVector2Array, col: Color) -> void:
	if poly.size() < 4:
		return
	var t1 := PackedVector2Array([poly[0], poly[1], poly[2]])
	if _quad_area(t1) > 0.3:
		ci.draw_colored_polygon(t1, col)
	var t2 := PackedVector2Array([poly[0], poly[2], poly[3]])
	if _quad_area(t2) > 0.3:
		ci.draw_colored_polygon(t2, col)


# A water-plane grid point (y = 0), projected to screen-pixels + depth.
func _wpt(gx: int, gy: int, wr: int, lens: Lens3D, u: float) -> Vector3:
	var wx := (float(gx) / float(wr) - 0.5) * 2.0 * half
	var wz := (float(gy) / float(wr) - 0.5) * 2.0 * half
	var pr := lens.project(Vector3(wx, 0.0, wz))
	return Vector3(pr.x * u, pr.y * u, pr.z)
