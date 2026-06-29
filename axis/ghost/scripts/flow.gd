extends RefCounted
class_name Flow2D

## Flow2D - a divergence-free curl-noise vector field.
##
## The shared source of organic *meander*. A plain noise field, used as a velocity,
## has sources and sinks - things pool and stall. The curl of a scalar noise
## potential is divergence-free: it only ever swirls, so anything that follows it
## (a filament growing, a particle drifting) wanders and curls the way something
## alive moves through space, never just sliding to a stop.
##
## Coordinates are in ghost's centred unit-fraction space (roughly -0.7..0.7). The
## field slowly evolves over time so the flow itself breathes. Filaments follow it
## ([Filament]); it is equally usable as a wind/drift field for particles.

var _noise := FastNoiseLite.new()
var _freq: float
var _t := 0.0
var _evolve: float


func _init(seed_value: int, freq := 2.0, evolve := 0.05) -> void:
	_noise.noise_type = FastNoiseLite.TYPE_SIMPLEX
	_noise.seed = seed_value
	_noise.frequency = 1.0
	_noise.fractal_type = FastNoiseLite.FRACTAL_FBM
	_noise.fractal_octaves = 3
	_freq = freq
	_evolve = evolve


# Scalar potential at p (3rd noise axis = slow time, so the field drifts).
func _psi(p: Vector2) -> float:
	return _noise.get_noise_3d(p.x * _freq, p.y * _freq, _t)


## The flow vector at [param p]: the curl of the potential, (dψ/dy, -dψ/dx),
## by central differences.
func at(p: Vector2) -> Vector2:
	var e := 0.0015 / _freq
	var dpsi_dx := (_psi(p + Vector2(e, 0.0)) - _psi(p - Vector2(e, 0.0))) / (2.0 * e)
	var dpsi_dy := (_psi(p + Vector2(0.0, e)) - _psi(p - Vector2(0.0, e))) / (2.0 * e)
	return Vector2(dpsi_dy, -dpsi_dx)


## The flow *direction* (radians) at [param p], or [param fallback] where the field
## is effectively flat.
func angle_at(p: Vector2, fallback := 0.0) -> float:
	var v := at(p)
	return v.angle() if v.length() > 1e-6 else fallback


## Drift the field forward in time so the flow slowly reshapes.
func advance(dt: float) -> void:
	_t += dt * _evolve
