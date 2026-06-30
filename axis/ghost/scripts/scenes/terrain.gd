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
	return {"type": ttype, "climate": climate, "surreal": pal != null}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.012, 0.018)
	_glow = lerpf(_glow, clampf(0.3 * f.energy + 0.5 * f.beat, 0.0, 1.0), 1.0 - exp(-5.0 * delta))
	_yaw += delta * (_yaw_speed + 0.18 * f.energy) * _yaw_dir
	var pitch := _pitch + 0.05 * sin(_life * 0.15)
	lens.orbit(Vector3(0.0, _terrain.relief * 0.15, 0.0), _dist, _yaw, pitch)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	lens.prepare()
	var lit := clampf(0.7 + 0.4 * _glow + 0.3 * _f.energy, 0.4, 1.4)
	_terrain.draw_surface(self, lens, unit(), lit, _life)
