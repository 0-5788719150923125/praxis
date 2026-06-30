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

# Biome colour sets [h, s, v] for grass / dirt / low rock / high rock / snow / sand /
# water. A climate gives the natural look (green lowland, brown+grey rock, snow peaks,
# blue water) chosen by height + slope + moisture, instead of one height ramp.
const CLIMATES := {
	"temperate": {"grass": [0.28, 0.55, 0.50], "dirt": [0.09, 0.50, 0.42],
		"rock_lo": [0.07, 0.22, 0.42], "rock_hi": [0.0, 0.05, 0.55], "snow": [0.58, 0.05, 0.86],
		"sand": [0.11, 0.35, 0.66], "water": [0.58, 0.62, 0.46]},
	"arid": {"grass": [0.18, 0.45, 0.46], "dirt": [0.08, 0.55, 0.50],
		"rock_lo": [0.06, 0.52, 0.50], "rock_hi": [0.05, 0.22, 0.60], "snow": [0.10, 0.10, 0.80],
		"sand": [0.11, 0.45, 0.74], "water": [0.50, 0.45, 0.50]},
	"tundra": {"grass": [0.26, 0.24, 0.42], "dirt": [0.08, 0.30, 0.38],
		"rock_lo": [0.60, 0.08, 0.42], "rock_hi": [0.0, 0.03, 0.60], "snow": [0.60, 0.03, 0.92],
		"sand": [0.10, 0.16, 0.60], "water": [0.55, 0.40, 0.55]},
	"verdant": {"grass": [0.32, 0.62, 0.46], "dirt": [0.10, 0.48, 0.36],
		"rock_lo": [0.10, 0.28, 0.40], "rock_hi": [0.0, 0.05, 0.52], "snow": [0.55, 0.06, 0.84],
		"sand": [0.13, 0.40, 0.64], "water": [0.50, 0.60, 0.44]},
}

var _biome_on := false
var _c_grass := Color.WHITE
var _c_dirt := Color.WHITE
var _c_rock_lo := Color.WHITE
var _c_rock_hi := Color.WHITE
var _c_snow := Color.WHITE
var _c_sand := Color.WHITE
var _water_col := Color(0.1, 0.3, 0.5)


func build(rng: RandomNumberGenerator, type_: String, world_half := 3.0,
		relief_ := 1.4, pal: Palette = null, climate := "") -> void:
	type = type_
	half = world_half
	relief = relief_
	_biome_on = climate != ""
	if _biome_on:
		_setup_climate(climate, rng)
	else:
		palette = pal if pal != null else Palette.named("earth", rng)
		_water_col = palette.at(0.0)
	var height := _recipe(rng)
	# Surface texture: a coarse mottle plus a fine grain and a rocky ridged striation,
	# combined - so the land reads as a textured material, not a bare coloured mesh.
	var mottle := Field.make("fbm", rng.randi(), 14.0, 4)
	var grain := Field.make("fbm", rng.randi(), 34.0, 3)
	var striate := Field.make("ridged", rng.randi(), 22.0, 3)
	var detail := Field.combine(Field.combine(mottle, "add", grain, 0.5), "add", striate, 0.4)
	var moist := Field.make("fbm", rng.randi(), 3.2, 4)        # wet (grass) vs dry (rock) regions
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
			if _biome_on:
				_vcol[i] = _biome(hgrid[i], slope, moist.at(p), tex)
				continue
			# Palette path (surreal climates): take the palette's ACTUAL interpolated colour
			# (smooth in RGB) and just shade it by slope and surface detail. Rebuilding it via
			# from_hsv(c.h, ...) flipped the hue wildly wherever the RGB lerp crossed grey - that
			# was the wrong colours and the hard edges. The detail field still shifts the band a
			# little and grains the brightness, so the surface keeps its striation.
			var c := palette.at(clampf(hgrid[i] + 0.18 * (tex - 0.5), 0.0, 1.0))
			var contour := 0.88 + 0.12 * sin(hgrid[i] * PI * 16.0)
			var shade := clampf((0.50 + 0.42 * slope + 0.40 * (tex - 0.5)) * contour, 0.14, 1.25)
			_vcol[i] = Color(c.r * shade, c.g * shade, c.b * shade, 1.0)


# Resolve a climate's material colours (jittered per seed so no two are identical).
func _setup_climate(name: String, rng: RandomNumberGenerator) -> void:
	var cl: Dictionary = CLIMATES.get(name, CLIMATES["temperate"])
	var jh := rng.randf_range(-0.025, 0.025)
	_c_grass = _hsv(cl.grass, jh, rng)
	_c_dirt = _hsv(cl.dirt, jh, rng)
	_c_rock_lo = _hsv(cl.rock_lo, jh, rng)
	_c_rock_hi = _hsv(cl.rock_hi, jh, rng)
	_c_snow = _hsv(cl.snow, jh, rng)
	_c_sand = _hsv(cl.sand, jh, rng)
	_water_col = _hsv(cl.water, jh, rng)


func _hsv(a: Array, jh: float, rng: RandomNumberGenerator) -> Color:
	return Color.from_hsv(fposmod(float(a[0]) + jh, 1.0),
		clampf(float(a[1]) * rng.randf_range(0.9, 1.1), 0.0, 1.0),
		clampf(float(a[2]) * rng.randf_range(0.92, 1.08), 0.0, 1.0))


# Pick a vertex colour from elevation, slope, moisture and surface detail: green lowland
# grading to brown/grey rock on steeps and heights, snow on high flats, sand at the
# shore - the natural colour variety, all from cheap per-vertex fields.
func _biome(h: float, slope: float, moist: float, det: float) -> Color:
	var t := clampf((h - water) / maxf(0.25, 1.0 - water), 0.0, 1.0)    # 0 shore .. 1 peak
	var ground := _c_grass.lerp(_c_dirt, clampf(0.45 - 0.7 * (moist - 0.5) + 0.5 * (det - 0.5), 0.0, 1.0))
	var rock := _c_rock_lo.lerp(_c_rock_hi, smoothstep(0.15, 0.9, t))
	var rocky := clampf((1.0 - slope) * 1.7 + smoothstep(0.5, 0.88, t) * 0.6, 0.0, 1.0)
	var c := ground.lerp(rock, rocky)
	c = c.lerp(_c_snow, smoothstep(0.80, 0.97, t) * clampf(slope * 1.3, 0.0, 1.0))
	c = c.lerp(_c_sand, (1.0 - smoothstep(0.0, 0.07, t)) * 0.7)
	var contour := 0.84 + 0.16 * sin(h * PI * 22.0)
	var sh := clampf((0.55 + 0.42 * slope + 0.32 * (det - 0.5)) * contour, 0.15, 1.25)
	return Color(c.r * sh, c.g * sh, c.b * sh, 1.0)


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
			quads.append({"d": (dep[i0] + dep[i1] + dep[i2] + dep[i3]) * 0.25, "poly": poly,
				"cols": PackedColorArray([_lit(_vcol[i0], lit), _lit(_vcol[i1], lit),
					_lit(_vcol[i3], lit), _lit(_vcol[i2], lit)])})
	# Water plane (a coarse translucent grid at y=0), shimmering, depth-sorted with the land.
	if water > 0.0:
		var wc := _water_col
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
				var wcol := Color(wc.r * sh * lit, wc.g * sh * lit, wc.b * sh * lit, 0.62)
				quads.append({"d": (c0.z + c1.z + c2.z + c3.z) * 0.25, "poly": wpoly,
					"cols": PackedColorArray([wcol, wcol, wcol, wcol])})
	quads.sort_custom(func(a, b): return a.d > b.d)        # far first
	for q in quads:
		draw_quad(ci, q.poly, q.cols)


# Screen-space area of a quad (shoelace). Near-zero => the quad is edge-on / collapsed /
# folded, which makes draw_colored_polygon's triangulation fail - so we skip those.
static func _quad_area(p: PackedVector2Array) -> float:
	var a := 0.0
	for i in p.size():
		var j := (i + 1) % p.size()
		a += p[i].x * p[j].y - p[j].x * p[i].y
	return absf(a) * 0.5


static func _lit(c: Color, k: float) -> Color:
	return Color(c.r * k, c.g * k, c.b * k, c.a)


## Draw a 4-point quad as its two Gouraud (per-vertex-coloured) triangles, split on the
## 0-2 diagonal. A projected heightfield quad can fold into a bowtie that a single
## polygon can't triangulate; two triangles never can, and degenerate ones are skipped.
## Per-vertex colour is what makes the surface texture read instead of flat facets.
static func draw_quad(ci: CanvasItem, poly: PackedVector2Array, cols: PackedColorArray) -> void:
	if poly.size() < 4:
		return
	var t1 := PackedVector2Array([poly[0], poly[1], poly[2]])
	if _quad_area(t1) > 0.3:
		ci.draw_polygon(t1, PackedColorArray([cols[0], cols[1], cols[2]]))
	var t2 := PackedVector2Array([poly[0], poly[2], poly[3]])
	if _quad_area(t2) > 0.3:
		ci.draw_polygon(t2, PackedColorArray([cols[0], cols[2], cols[3]]))


# A water-plane grid point (y = 0), projected to screen-pixels + depth.
func _wpt(gx: int, gy: int, wr: int, lens: Lens3D, u: float) -> Vector3:
	var wx := (float(gx) / float(wr) - 0.5) * 2.0 * half
	var wz := (float(gy) / float(wr) - 0.5) * 2.0 * half
	var pr := lens.project(Vector3(wx, 0.0, wz))
	return Vector3(pr.x * u, pr.y * u, pr.z)
