extends RefCounted
class_name Shots

## Shots - the registry of camera framings (composition + slow moves).
##
## A typed axis for *where the camera sits*, separate from what a scene draws or
## how its elements move. Instead of every subject sitting dead center at zoom 1,
## a scene is assigned a shot: pushed off to a third, zoomed in so it runs partly
## off-frame, slowly pushing in, panning across, or canted. Moves are deliberately
## slow (tens of seconds) so they read as cinematic, not restless.
##
## Each shot sets the scene's [SceneView] base each frame from the scene's life
## time `t`; the scene's own organic drift is added on top. Field-filling scenes
## get a gentler pool (see FIELD_BAG) so their edges never swing into view.

class Shot:
	var rng := RandomNumberGenerator.new()
	func _init(seed_value := 0) -> void:
		rng.seed = seed_value
	func apply(_view, _f, _t) -> void:
		pass


class Centered extends Shot:
	func apply(view, _f, _t) -> void:
		view.zoom = 1.0
		view.offset = Vector2.ZERO
		view.rotation = 0.0


class Offset extends Shot:
	var ox := 0.0
	var oy := 0.0
	var z := 1.0
	var rot := 0.0
	func _init(seed_value := 0) -> void:
		super(seed_value)
		ox = rng.randf_range(-0.20, 0.20)
		oy = rng.randf_range(-0.13, 0.13)
		z = rng.randf_range(1.08, 1.24)
		rot = rng.randf_range(-0.03, 0.03)
	func apply(view, _f, _t) -> void:
		view.zoom = z
		view.offset = Vector2(ox, oy)
		view.rotation = rot


class PushIn extends Shot:
	var z0 := 1.0
	var z1 := 1.35
	var ox := 0.0
	var oy := 0.0
	func _init(seed_value := 0) -> void:
		super(seed_value)
		z0 = rng.randf_range(0.95, 1.05)
		z1 = rng.randf_range(1.25, 1.5)
		ox = rng.randf_range(-0.1, 0.1)
		oy = rng.randf_range(-0.08, 0.08)
	func apply(view, _f, t) -> void:
		var k := 1.0 - exp(-t * 0.06)
		view.zoom = lerpf(z0, z1, k)
		view.offset = Vector2(ox, oy) * k
		view.rotation = 0.0


class PullBack extends Shot:
	var z0 := 1.4
	var z1 := 1.0
	var ox := 0.0
	var oy := 0.0
	func _init(seed_value := 0) -> void:
		super(seed_value)
		z0 = rng.randf_range(1.3, 1.55)
		z1 = rng.randf_range(0.98, 1.08)
		ox = rng.randf_range(-0.12, 0.12)
		oy = rng.randf_range(-0.1, 0.1)
	func apply(view, _f, t) -> void:
		var k := 1.0 - exp(-t * 0.06)
		view.zoom = lerpf(z0, z1, k)
		view.offset = Vector2(ox, oy) * (1.0 - k)
		view.rotation = 0.0


class Pan extends Shot:
	var z := 1.4
	var oy := 0.0
	var amp := 0.24
	var phase := 0.0
	func _init(seed_value := 0) -> void:
		super(seed_value)
		z = rng.randf_range(1.25, 1.45)
		oy = rng.randf_range(-0.1, 0.1)
		amp = rng.randf_range(0.10, 0.18)
		phase = rng.randf() * TAU
	func apply(view, _f, t) -> void:
		view.zoom = z
		view.offset = Vector2(sin(t * 0.04 + phase) * amp, oy)
		view.rotation = 0.0


class Canted extends Shot:
	var ox := 0.0
	var oy := 0.0
	var z := 1.2
	var rot := 0.0
	func _init(seed_value := 0) -> void:
		super(seed_value)
		ox = rng.randf_range(-0.18, 0.18)
		oy = rng.randf_range(-0.12, 0.12)
		z = rng.randf_range(1.12, 1.28)
		rot = rng.randf_range(-0.07, 0.07)
	func apply(view, _f, _t) -> void:
		view.zoom = z
		view.offset = Vector2(ox, oy)
		view.rotation = rot


const REGISTRY := {
	"centered": Centered,
	"offset": Offset,
	"push_in": PushIn,
	"pull_back": PullBack,
	"pan": Pan,
	"canted": Canted,
}

# Weighted pools. Subjects get the expressive shots; field-filling scenes get
# only gentle ones so panning/offset never exposes their overscan edges.
const SUBJECT_BAG := ["centered", "offset", "offset", "push_in", "pull_back", "pan", "canted"]
const FIELD_BAG := ["centered", "centered", "push_in", "pull_back"]
# Single flat planes (a lone snowflake / glass pane): keep the camera square-on so
# the plane never reads as a tumbling card. Mostly centered, occasional slow push.
const PLANE_BAG := ["centered", "centered", "centered", "push_in", "offset"]


static func make(name: String, seed_value := 0) -> Shot:
	return REGISTRY[name].new(seed_value)
