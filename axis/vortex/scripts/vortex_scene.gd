extends Node2D
class_name VortexScene

## VortexScene - one visualizer.
##
## The base of every scene. Mirrors the Arena / business-card generators: a
## seeded [RandomNumberGenerator] rolls a typed *definition* ([member params]),
## and that definition is then pushed through transforms that are modulated by
## the audio. Subclasses override exactly two things:
##
##   build_params(rng) -> Dictionary   # the seeded definition (geometry, colors)
##   update(features, delta)           # modulate by audio, then queue_redraw()
##
## ...and draw in [method _draw] using [member params] plus whatever state
## [method update] stashed. [Director] handles seeding, sizing, and lifetime, so
## a scene file is just "definition + how it reacts."

## The typed definition for this instance, produced by [method build_params].
var params: Dictionary = {}

## Viewport size, kept current so scenes can lay out in pixels.
var size: Vector2 = Vector2.ZERO


## Called once by [Director] before the scene enters the tree. Seeds the
## definition deterministically. Do not override - override [method build_params].
func init_with_seed(seed_value: int) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = seed_value
	params = build_params(rng)


## Override: roll the typed definition from a seeded RNG. Same seed -> same scene.
func build_params(_rng: RandomNumberGenerator) -> Dictionary:
	return {}


## Override: read [param f], update internal state, then call queue_redraw().
func update(_f: AudioFeatures, _delta: float) -> void:
	pass


func _ready() -> void:
	size = get_viewport_rect().size
	get_viewport().size_changed.connect(_on_resize)


func _on_resize() -> void:
	size = get_viewport_rect().size
	queue_redraw()


## Center of the screen - the usual anchor for radial scenes.
func center() -> Vector2:
	return size * 0.5
