extends Scene3D

## Terrain - real 3D landscapes built from the composable [Field] / [Terrain] foundation.
##
## By seed a different world: rolling hills, ridged mountains, river valleys, a fissured
## canyon, ocean islands, or a banded mesa - each a recipe of [Field]s sampled into a
## heightfield, coloured by a [Palette] plus a fine surface-texture field and slope
## shading, with water pooling below its level. The camera orbits under a wide lens for
## forced perspective; audio drives the light, not the land (the terrain holds its form).

var _f: AudioFeatures = AudioFeatures.new()
var _terrain: Terrain
var _glow := 0.0
var _yaw := 0.0
var _dist := 7.0
var _pitch := 0.45
var _yaw_dir := 1.0
var _yaw_speed := 0.10
var _light_az := 0.0
var _light_dir := 1.0
var _light_el := 0.5
var _fog: Array = []             # sparse volumetric mist pooling in the valleys
var _clouds: Array = []          # a broken cloud layer well above the peaks
var _fog_col := Color(0.66, 0.70, 0.78)

const TYPES := ["hills", "mountains", "valleys", "canyon", "islands", "mesa"]


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "field"
	var ttype: String = TYPES[rng.randi() % TYPES.size()]
	var relief := rng.randf_range(1.1, 1.9)
	# Mostly a natural biome (green lowland, brown/grey rock, snow peaks, blue water),
	# biased by terrain type; sometimes a surreal palette (volcanic / alien) for variety.
	var climate := ""
	var pal: Palette = null
	if rng.randf() < 0.22:
		pal = Palette.named(["volcanic", "alien", "ocean"][rng.randi() % 3], rng)
	elif ttype == "canyon" or ttype == "mesa":
		climate = "arid"
	elif ttype == "islands":
		climate = "verdant" if rng.randf() < 0.6 else "temperate"
	else:
		climate = ["temperate", "tundra", "verdant"][rng.randi() % 3]
	_terrain = Terrain.new()
	# Bigger extent (4.0) + a closer camera so the land runs off every edge and fills the
	# frame, rather than floating as a small island of geometry in a black field.
	_terrain.build(rng, ttype, 4.0, relief, pal, climate)
	# Sampled camera (never the same flyover twice).
	lens.fov = rng.randf_range(50.0, 62.0)
	_dist = rng.randf_range(5.4, 7.2)
	_pitch = rng.randf_range(0.30, 0.52)
	_yaw = rng.randf() * TAU
	_yaw_dir = 1.0 if rng.randf() < 0.5 else -1.0
	_yaw_speed = rng.randf_range(0.05, 0.14)
	# A low key light (long shadows) whose azimuth drifts slowly, so the mountains' cast shadows
	# gently sweep across the land as it moves.
	_light_az = rng.randf() * TAU
	_light_dir = 1.0 if rng.randf() < 0.5 else -1.0
	_light_el = rng.randf_range(0.32, 0.52)
	_terrain.set_light(_light_az, _light_el)
	_seed_atmosphere(rng, climate)
	return {"type": ttype, "climate": climate, "surreal": pal != null}


# Sparse VALLEY FOG (soft mist puffs pooled just above the low ground, clearing off the ridges) and a
# broken CLOUD layer high above the peaks - real 3D billboards, projected through the same orbiting lens.
func _seed_atmosphere(rng: RandomNumberGenerator, climate: String) -> void:
	var half: float = _terrain.half
	var relief: float = _terrain.relief
	_fog_col = Color(0.60, 0.63, 0.72) if climate == "arid" or climate == "tundra" else Color(0.68, 0.72, 0.80)
	# Valley fog: sample many spots, keep the LOW ones, pool a soft puff just above the surface there.
	for _i in 170:
		var wx := rng.randf_range(-half * 0.98, half * 0.98)
		var wz := rng.randf_range(-half * 0.98, half * 0.98)
		var hn: float = _terrain.height_at(wx, wz)
		if hn < rng.randf_range(0.14, 0.38):                       # only the low ground gathers mist
			var y := hn * relief + rng.randf_range(0.02, 0.16)
			_fog.append({"pos": Vector3(wx, y, wz), "r": rng.randf_range(0.35, 0.85),
				"dens": rng.randf_range(0.035, 0.10), "ph": rng.randf() * TAU})
	# A high, broken cloud layer (spread wider than the land so it fills the sky as the camera orbits).
	var ch := relief * 0.5 + rng.randf_range(0.9, 1.7)
	for _i in rng.randi_range(26, 40):
		_clouds.append({"pos": Vector3(rng.randf_range(-half * 1.6, half * 1.6), ch + rng.randf_range(-0.25, 0.5),
			rng.randf_range(-half * 1.6, half * 1.6)), "r": rng.randf_range(0.7, 1.5),
			"dens": rng.randf_range(0.05, 0.12), "ph": rng.randf() * TAU})


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.012, 0.018)
	_glow = lerpf(_glow, clampf(0.3 * f.energy + 0.5 * f.beat, 0.0, 1.0), 1.0 - exp(-5.0 * delta))
	_yaw += delta * (_yaw_speed + 0.18 * f.energy) * _yaw_dir
	var pitch := _pitch + 0.05 * sin(_life * 0.15)
	lens.orbit(Vector3(0.0, _terrain.relief * 0.15, 0.0), _dist, _yaw, pitch)
	# Drift the key light slowly and refresh the sweeping cast shadows.
	_light_az += delta * 0.035 * _light_dir
	_terrain.set_light(_light_az, _light_el)
	_terrain.step_light(delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	lens.prepare()
	var lit := clampf(0.7 + 0.4 * _glow + 0.3 * _f.energy, 0.4, 1.4)
	_terrain.draw_surface(self, lens, unit(), lit, _life)
	# Atmosphere over the land: the high clouds first (they sit farthest), then the valley mist above
	# the surface. Both brighten a touch with the light.
	var b := clampf(0.55 + 0.5 * lit, 0.4, 1.1)
	_draw_puffs(_clouds, Color(_fog_col.r * b, _fog_col.g * b, _fog_col.b * b, 1.0), 0.12)
	_draw_puffs(_fog, Color(_fog_col.r * b * 0.95, _fog_col.g * b * 0.95, _fog_col.b * b, 1.0), 0.06)


# Draw a set of soft billboard puffs (fog / clouds) through the orbiting lens: project each, size it by
# depth (via the lens right-vector), depth-sort far-to-near, and stack the gaussians into soft banks.
func _draw_puffs(items: Array, col: Color, drift: float) -> void:
	if items.is_empty():
		return
	var u := unit()
	var vis: Array = []
	for it in items:
		var p: Vector3 = it.pos + Vector3(sin(_life * 0.08 + float(it.ph)) * drift, 0.0,
			cos(_life * 0.06 + float(it.ph)) * drift)
		var z: float = lens.depth(p)
		if z <= lens.near:
			continue
		var pr := lens.project(p)
		var pe := lens.project(p + lens._r * float(it.r))
		var sp := Vector2(pr.x, pr.y) * u
		var scr_r: float = (Vector2(pe.x, pe.y) * u - sp).length()
		if scr_r < 1.5:
			continue
		vis.append({"sp": sp, "z": z, "r": scr_r, "d": float(it.dens)})
	vis.sort_custom(func(a, b): return a.z > b.z)               # far first
	for v in vis:
		Layer.puff(self, v.sp, v.r, Color(col.r, col.g, col.b, clampf(col.a * float(v.d), 0.0, 0.5)))
