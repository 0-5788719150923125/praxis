extends RefCounted
class_name ParticleSystem

## ParticleSystem - a bag of [Particle]s and a composed list of [Primitives]
## forces. The shared substrate that lets scenes be assembled from physics
## instead of hand-rolling it.
##
## Usage: add particles (geometry), add forces by key (behavior), then call
## step() each frame and draw the particles yourself. settled() lets a oneshot
## scene know its sequence has come to rest.

var particles: Array = []
var forces: Array = []
var _dynamic := false


func add(p: Particle) -> void:
	particles.append(p)


## Compose in a force by registry key (see [Primitives]).
func add_force(key: String, cfg := {}) -> void:
	var force := Primitives.make(key, cfg)
	forces.append(force)
	if force.is_dynamic():
		_dynamic = true


func step(f: AudioFeatures, dt: float) -> void:
	for p in particles:
		p.acc = Vector2.ZERO
	for force in forces:
		force.accumulate(particles, f, dt)
	if _dynamic:
		for p in particles:
			p.vel += p.acc * dt
			p.off += p.vel * dt
			p.ang += p.angvel * dt
	for force in forces:
		force.resolve(particles, f, dt)


## True when every particle has effectively stopped - used by oneshot scenes.
func settled(thresh := 0.02) -> bool:
	for p in particles:
		if p.vel.length() > thresh:
			return false
	return true


## The particle's local polygon transformed into pixel space, for draw_*.
func world_poly(p: Particle, u: float) -> PackedVector2Array:
	var c := p.pos() * u
	var out := PackedVector2Array()
	for v in p.poly:
		out.append(c + (Vector2(v) * p.scale).rotated(p.ang) * u)
	return out
