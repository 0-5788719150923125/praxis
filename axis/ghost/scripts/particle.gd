extends RefCounted
class_name Particle

## Particle - one transformable element a [ParticleSystem] moves.
##
## Geometry-agnostic: it carries a rest position and a current rigid transform
## (offset / angle / scale), plus either a local polygon (drawn as a shard/facet)
## or a radius (drawn as a point). Forces from [Primitives] read and write these
## fields; scenes own the drawing and colour. All positions are in unit-fraction
## space (multiply by [method GhostScene.unit] at draw time) so the physics is
## resolution independent.

var home := Vector2.ZERO     ## rest position
var off := Vector2.ZERO      ## displacement from home
var vel := Vector2.ZERO
var acc := Vector2.ZERO       ## reset and re-summed each step by accumulate forces
var ang := 0.0
var angvel := 0.0
var scale := 1.0
var poly := PackedVector2Array()   ## local shape, centered; empty => a point
var radius := 0.0                   ## point radius (when poly is empty)
var hue := 0.0
var noise := Vector2.ZERO     ## per-particle random in -1..1 (seeded), for force jitter
var nspin := 0.0              ## per-particle random spin in -1..1
var data := {}                ## scratch for forces (e.g. "center" of a sub-group)


func pos() -> Vector2:
	return home + off
