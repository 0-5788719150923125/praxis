extends RigidBody3D

var base_emission_color := Color(0.2, 0.4, 1.0)
var highlight_emission_color := Color(1.0, 0.4, 0.2)
var is_highlighted := false

@onready var mesh_instance: MeshInstance3D = $MeshInstance3D
@onready var material: StandardMaterial3D = mesh_instance.get_surface_override_material(0)

# Signal for when the atom is selected/clicked
signal atom_selected(atom: Node3D)

func _ready() -> void:
	set_highlight(false)
	
	# Create a duplicate of the material to prevent sharing
	material = material.duplicate()
	mesh_instance.set_surface_override_material(0, material)

func _input_event(_camera: Node, event: InputEvent, _position: Vector3, _normal: Vector3, _shape_idx: int) -> void:
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
			emit_signal("atom_selected", self)
	elif event is InputEventScreenTouch:
		if event.pressed:
			emit_signal("atom_selected", self)

func set_highlight(enabled: bool) -> void:
	is_highlighted = enabled
	if material:
		material.emission = highlight_emission_color if enabled else base_emission_color

func get_radius() -> float:
	return $CollisionShape3D.shape.radius

func set_radius(value: float) -> void:
	var mesh: SphereMesh = mesh_instance.mesh
	mesh.radius = value
	mesh.height = value * 2
	$CollisionShape3D.shape.radius = value
