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

const RES := 112                 # grid resolution (RES x RES vertices)

var res := RES
var half := 3.0                  # world half-extent in x and z
var relief := 1.4                # vertical world scale
var water := 0.0                 # water plane world height (0 = none)
var type := "hills"
var palette: Palette
var hgrid := PackedFloat32Array()   # heights 0..1
var _world := PackedVector3Array()  # world-space vertices
var _vcol: PackedColorArray         # base per-vertex colour (palette + texture + slope)
var _vnorm := PackedVector3Array()  # per-vertex surface normal (for the moving directional light)

# Cinematic area light + cast shadows: a low directional key light whose azimuth drifts, so the
# mountains cast long shadows that gently sweep as it moves. The shadow map is per-vertex and
# refreshed a few rows per frame (never a full-grid recompute), so it stays hitch-free.
var _light_dir := Vector3(0.55, 0.5, 0.35).normalized()   # world direction TOWARD the key light
var _cast := PackedFloat32Array()   # per-vertex cast-shadow factor SHOWN (eased toward _cast_target)
var _cast_target := PackedFloat32Array()   # the raw ray-march result, refreshed a few rows per frame
var _shadow_row := 0                 # incremental shadow-refresh cursor (row being recomputed)
const SHADOW_MIN := 0.42            # ground brightness where fully in a mountain's cast shadow
var _fog_level := -1.0               # world height below which valley fog pools (< min = no fog)
var _fog_col := Color(0.62, 0.66, 0.72)

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
	_vnorm.resize(res * res)
	_cast.resize(res * res)
	_cast.fill(1.0)
	_vcol = PackedColorArray()
	_vcol.resize(res * res)
	# Sample the field into the grid (the only expensive pass; done once).
	for gy in res:
		for gx in res:
			var p := Vector2(float(gx) / float(res - 1) - 0.5, float(gy) / float(res - 1) - 0.5) * 2.0
			hgrid[gy * res + gx] = height.at(p)
	_smooth(1)        # a single pass: knock down grid-scale aliasing but KEEP the recipe's detail
	# Micro-relief: overlay fine GEOMETRIC detail over the whole surface so the land reads as
	# textured ground, not low-res smooth blobs. Three octaves - a coarse roll, a ridged grain,
	# and a fine crinkle - tiled across the terrain and added AFTER the smoothing pass (so it
	# survives), turned into real height (not just colour). This catches the slope shading and the
	# per-vertex normals, so the surface keeps crisp bumps and creases even under a close camera,
	# instead of the over-smoothed sheen that read as blur when the push-in magnified it.
	var micro := Field.combine(
		Field.combine(Field.make("fbm", rng.randi(), 9.0, 5), "add",
			Field.make("ridged", rng.randi(), 19.0, 4), 0.7),
		"add", Field.make("fbm", rng.randi(), 42.0, 3), 0.4)
	var micro_amp := 0.09
	for gy in res:
		for gx in res:
			var i := gy * res + gx
			var p := Vector2(float(gx) / float(res - 1) - 0.5, float(gy) / float(res - 1) - 0.5) * 2.0
			hgrid[i] = clampf(hgrid[i] + (micro.at(p) - 0.5) * micro_amp, 0.0, 1.0)
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
			_vnorm[i] = n
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
	# Valley fog pools a little above the lowest ground (or the water line), so mist gathers in
	# the low valleys and clears off the ridges. A tundra/arid palette gets a cooler, thinner fog.
	var lo := 1.0
	var hi := 0.0
	for hv in hgrid:
		lo = minf(lo, hv)
		hi = maxf(hi, hv)
	_fog_level = maxf(water, lo) + 0.11 * (hi - lo)
	_fog_col = Color(0.66, 0.70, 0.76) if _biome_on else Color(0.60, 0.62, 0.70)


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


## Aim the key light. `az` is the azimuth (radians, drifts over time); `el` the elevation
## (kept low for long dramatic shadows). Call each frame from the scene with a slowly moving az.
func set_light(az: float, el := 0.5) -> void:
	_light_dir = Vector3(cos(el) * cos(az), sin(el), cos(el) * sin(az)).normalized()


## The world direction toward the key light (so a scene can shade its own props - city blocks -
## with the SAME light as the terrain).
func light_dir() -> Vector3:
	return _light_dir


## Refresh a few rows of the cast-shadow TARGET each frame, then ease the SHOWN shadow toward it.
## The easing is the anti-flicker: a cell never hard-flips lit<->shadowed as the light drifts (which
## popped and shimmered), it glides; and the incremental refresh no longer shows a moving seam.
func step_light(delta: float) -> void:
	var n := res * res
	if _cast.size() != n:
		_cast.resize(n)
		_cast.fill(1.0)
	if _cast_target.size() != n:
		_cast_target.resize(n)
		_cast_target.fill(1.0)
	var rows := maxi(1, int(res / 16))            # a few rows per frame; whole target refreshed ~every 16
	for _r in rows:
		var gy := _shadow_row
		for gx in res:
			_cast_target[gy * res + gx] = _cast_at(gx, gy)
		_shadow_row = (_shadow_row + 1) % res
	var ease := 1.0 - exp(-3.0 * delta)           # smooth glide toward the target - no pop, no seam
	for i in n:
		_cast[i] = lerpf(_cast[i], _cast_target[i], ease)


# March from a vertex toward the light through the heightfield and return a SOFT shadow factor
# (SHADOW_MIN fully shadowed .. 1 lit). Instead of a hard hit/miss, it tracks how far the terrain
# rises ABOVE the light ray along the way and maps that penetration through a smoothstep, so shadow
# edges get a penumbra and don't harshly flip on/off as the light sweeps - killing the flicker.
func _cast_at(gx: int, gy: int) -> float:
	if _light_dir.y <= 0.02:
		return 1.0
	var p: Vector3 = _world[gy * res + gx]
	var ds := (2.0 * half / float(res)) * 2.1
	var bias := 0.03 * relief
	var occ := 0.0
	for s in range(1, 18):
		var d := ds * float(s)
		var wx := p.x + _light_dir.x * d
		var wz := p.z + _light_dir.z * d
		if absf(wx) > half or absf(wz) > half:
			break                                 # ray left the terrain - nothing more can occlude
		var margin := height_at(wx, wz) * relief - (p.y + _light_dir.y * d + bias)
		if margin > occ:
			occ = margin
	var shade := smoothstep(0.0, 0.14 * relief, occ)   # 0 lit .. 1 fully shadowed, with a penumbra
	return lerpf(1.0, SHADOW_MIN, shade)


## Project + depth-sort + draw the terrain surface (and water) through the lens. `lit`
## scales brightness (audio); `shimmer` (time) animates the water.
func draw_surface(ci: CanvasItem, lens: Lens3D, u: float, lit: float, shimmer: float) -> void:
	var n := res * res
	var sv := PackedVector2Array()
	sv.resize(n)
	var dep := PackedFloat32Array()
	dep.resize(n)
	# Drifting CLOUD SHADOWS: soft bands moving across the land over time (per-vertex, so they
	# follow the real 3D surface), darkening the ground where a cloud passes and brightening the
	# sunlit gaps - layered under the directional key light and the mountains' cast shadows.
	var sxd := shimmer * 0.06
	var szd := shimmer * 0.045
	# The final lit colour per vertex: base colour x audio brightness x cloud shadow x directional
	# key light (n.l) x mountain cast shadow, then valley fog blended over the low ground. Computed
	# once here so both triangles of every quad reuse it.
	var vc := PackedColorArray()
	vc.resize(n)
	# World-space UVs for the tiling detail texture (grain follows the real surface, not the screen).
	var uvg := PackedVector2Array()
	uvg.resize(n)
	var tex := detail_texture()
	ci.texture_repeat = CanvasItem.TEXTURE_REPEAT_ENABLED   # so the UVs > 1 tile
	var tile := 1.7 / maxf(0.5, half)                        # ~a dozen repeats across the land
	for i in n:
		var pr := lens.project(_world[i])
		sv[i] = Vector2(pr.x, pr.y) * u
		dep[i] = pr.z
		var p: Vector3 = _world[i]
		uvg[i] = Vector2(p.x, p.z) * tile
		var cv := sin(p.x * 0.7 + sxd) + sin(p.z * 0.55 - szd) + 0.6 * sin((p.x + p.z) * 1.1 + sxd * 1.5)
		var cloud := clampf(0.55 + 0.5 * smoothstep(-0.7, 0.9, cv * 0.5), 0.5, 1.0)
		# Directional key light: sunny slopes brighten, slopes facing away fall into shade.
		var ndotl := clampf(_vnorm[i].dot(_light_dir), 0.0, 1.0)
		var key := 0.55 + 0.6 * ndotl                 # ambient floor + directional term
		var col := _lit(_vcol[i], lit * cloud * key * _cast[i])
		# Valley fog: a thin drifting haze pooling in the deepest LAND hollows (never over water,
		# never on the ridges), thickest at the very bottom and feathering out quickly upward.
		if hgrid[i] > water and hgrid[i] < _fog_level:
			var hw := (hgrid[i] - water) / maxf(0.02, _fog_level - water)   # 0 valley floor .. 1 fog line
			var drift := 0.7 + 0.3 * sin(shimmer * 0.15 + p.x * 0.5 + p.z * 0.4)
			var fog := clampf((1.0 - hw) * (1.0 - hw), 0.0, 1.0) * 0.42 * drift
			col = col.lerp(Color(_fog_col.r * lit, _fog_col.g * lit, _fog_col.b * lit), fog)
		vc[i] = col
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
			if _quad_area(poly) < 0.25:       # only truly collapsed quads (lowered: the old 2.0 left black holes)
				continue
			quads.append({"d": (dep[i0] + dep[i1] + dep[i2] + dep[i3]) * 0.25, "poly": poly,
				"cols": PackedColorArray([vc[i0], vc[i1], vc[i3], vc[i2]]),
				"uvs": PackedVector2Array([uvg[i0], uvg[i1], uvg[i3], uvg[i2]])})
	# Water plane (a coarse translucent grid at y=0), shimmering, depth-sorted with the land.
	if water > 0.0:
		var wc := _water_col
		var wr := 22
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
				if _quad_area(wpoly) < 0.25:
					continue
				var sh := 0.85 + 0.15 * sin(shimmer * 1.3 + float(gx) * 0.7 + float(gy) * 0.5)
				var wcol := Color(wc.r * sh * lit, wc.g * sh * lit, wc.b * sh * lit, 0.62)
				quads.append({"d": (c0.z + c1.z + c2.z + c3.z) * 0.25, "poly": wpoly,
					"cols": PackedColorArray([wcol, wcol, wcol, wcol])})
	quads.sort_custom(func(a, b): return a.d > b.d)        # far first
	for q in quads:
		if q.has("uvs"):
			draw_quad(ci, q.poly, q.cols, q.uvs, tex)      # land: modulated by the detail texture
		else:
			draw_quad(ci, q.poly, q.cols)                  # water: flat translucent


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
## Per-vertex colour is what makes the surface texture read instead of flat facets. If `uvs`
## + `tex` are given, the vertex colours are MODULATED by a tiling detail texture (world-space
## UVs) - genuine sub-vertex surface grain (a value/bump texture), not just interpolated colour.
static func draw_quad(ci: CanvasItem, poly: PackedVector2Array, cols: PackedColorArray,
		uvs := PackedVector2Array(), tex: Texture2D = null) -> void:
	if poly.size() < 4:
		return
	var textured := tex != null and uvs.size() >= 4
	var t1 := PackedVector2Array([poly[0], poly[1], poly[2]])
	if _quad_area(t1) > 0.04:
		if textured:
			ci.draw_polygon(t1, PackedColorArray([cols[0], cols[1], cols[2]]),
				PackedVector2Array([uvs[0], uvs[1], uvs[2]]), tex)
		else:
			ci.draw_polygon(t1, PackedColorArray([cols[0], cols[1], cols[2]]))
	var t2 := PackedVector2Array([poly[0], poly[2], poly[3]])
	if _quad_area(t2) > 0.04:
		if textured:
			ci.draw_polygon(t2, PackedColorArray([cols[0], cols[2], cols[3]]),
				PackedVector2Array([uvs[0], uvs[2], uvs[3]]), tex)
		else:
			ci.draw_polygon(t2, PackedColorArray([cols[0], cols[2], cols[3]]))


# A tiling grayscale DETAIL texture (built once): fbm value-noise crossed with a ridged streak, so
# terrain quads carry fine sub-vertex grain when this modulates their colour. Tiled finely across
# the land via world-space UVs, so the seams (it is not perfectly seamless) fall well below a pixel.
static var _dtex: Texture2D = null
static func detail_texture() -> Texture2D:
	if _dtex == null:
		var s := 128
		var img := Image.create(s, s, false, Image.FORMAT_RGBA8)
		var nf := FastNoiseLite.new()
		nf.seed = 1337
		nf.frequency = 0.045
		nf.fractal_octaves = 4
		var nr := FastNoiseLite.new()
		nr.seed = 4242
		nr.noise_type = FastNoiseLite.TYPE_SIMPLEX
		nr.fractal_type = FastNoiseLite.FRACTAL_RIDGED
		nr.frequency = 0.09
		nr.fractal_octaves = 3
		for y in s:
			for x in s:
				var a := nf.get_noise_2d(float(x), float(y)) * 0.5 + 0.5
				var b := nr.get_noise_2d(float(x) + 33.0, float(y) - 12.0) * 0.5 + 0.5
				var v := clampf(0.72 + 0.5 * (a - 0.5) + 0.34 * (b - 0.5), 0.4, 1.18)
				img.set_pixel(x, y, Color(v, v, v, 1.0))
		_dtex = ImageTexture.create_from_image(img)
	return _dtex


# A water-plane grid point (y = 0), projected to screen-pixels + depth.
func _wpt(gx: int, gy: int, wr: int, lens: Lens3D, u: float) -> Vector3:
	var wx := (float(gx) / float(wr) - 0.5) * 2.0 * half
	var wz := (float(gy) / float(wr) - 0.5) * 2.0 * half
	var pr := lens.project(Vector3(wx, 0.0, wz))
	return Vector3(pr.x * u, pr.y * u, pr.z)
