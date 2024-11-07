extends Area3D

const HIGHLIGHT_TRANSITION_TIME = 0.3

var base_emission_color := Color(0.2, 0.4, 1.0)
var highlight_emission_color := Color(1.0, 0.4, 0.2)
var is_highlighted := false

@onready var mesh_instance: MeshInstance3D = $MeshInstance3D
@onready var material: StandardMaterial3D = mesh_instance.get_surface_override_material(0)

signal atom_selected(atom: Area3D)

func _ready() -> void:
	# Create a unique material instance
	material = material.duplicate()
	mesh_instance.set_surface_override_material(0, material)
	material.emission = base_emission_color
	
	# Enable input
	input_ray_pickable = true
	collision_layer = 1
	collision_mask = 1

func _input_event(_camera: Node, event: InputEvent, _position: Vector3, _normal: Vector3, _shape_idx: int) -> void:
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT:
			print("Atom clicked: ", name, " - Emitting signal")
			atom_selected.emit(self)  # Changed to emit(self)
			get_viewport().set_input_as_handled()

func set_highlight(enabled: bool) -> void:
	print("Setting highlight for ", name, " to: ", enabled)  # Debug print
	is_highlighted = enabled
	
	if material:
		# Create smooth color transition
		var tween = create_tween()
		tween.set_trans(Tween.TRANS_CUBIC)
		tween.set_ease(Tween.EASE_OUT)
		
		var target_color = highlight_emission_color if enabled else base_emission_color
		tween.tween_property(material, "emission", target_color, HIGHLIGHT_TRANSITION_TIME)
	else:
		print("ERROR: No material found for atom: ", name)


func get_radius() -> float:
	return $CollisionShape3D.shape.radius

func set_radius(value: float) -> void:
	# Update both visual and collision shapes
	var mesh: SphereMesh = mesh_instance.mesh.duplicate()
	mesh.radius = value
	mesh.height = value * 2
	mesh_instance.mesh = mesh
	
	# Update collision shape
	var collision_shape: SphereShape3D = $CollisionShape3D.shape.duplicate()
	collision_shape.radius = value
	$CollisionShape3D.shape = collision_shape
