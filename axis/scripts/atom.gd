extends RigidBody3D

var base_emission_color := Color(0.2, 0.4, 1.0)
var highlight_emission_color := Color(1.0, 0.4, 0.2)
var is_highlighted := false

@onready var mesh_instance: MeshInstance3D = $MeshInstance3D
@onready var material: StandardMaterial3D = mesh_instance.get_surface_override_material(0)

func _ready() -> void:
	# Initialize any starting properties
	set_highlight(false)

func set_highlight(enabled: bool) -> void:
	is_highlighted = enabled
	if material:
		material.emission = highlight_emission_color if enabled else base_emission_color

func get_radius() -> float:
	return $CollisionShape3D.shape.radius

func set_radius(value: float) -> void:
	# Update both visual and collision shapes
	var mesh: SphereMesh = mesh_instance.mesh
	mesh.radius = value
	mesh.height = value * 2
	$CollisionShape3D.shape.radius = value
