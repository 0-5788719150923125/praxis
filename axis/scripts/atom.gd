extends Area3D

const HIGHLIGHT_TRANSITION_TIME = 0.3
const INTERIOR_TRANSITION_TIME = 0.5

var base_emission_color = Color(0.2, 0.4, 1.0)
var highlight_emission_color = Color(1.0, 0.4, 0.2)
var interior_color = Color(0, 0, 0, 1)  # Pure black for interior view
var is_highlighted = false
var original_material_state = {}  # Store original material properties

@onready var mesh_instance: MeshInstance3D = $MeshInstance3D
@onready var material: StandardMaterial3D = mesh_instance.get_surface_override_material(0)
@onready var nucleus: MeshInstance3D = $Nucleus

signal atom_selected(atom: Area3D)

func _ready() -> void:
	# Create a unique material instance
	material = material.duplicate()
	mesh_instance.set_surface_override_material(0, material)
	material.emission = base_emission_color
	
	# Store original material properties
	_store_original_material_state()
	
	# Enable input
	input_ray_pickable = true
	collision_layer = 1
	collision_mask = 1

func _store_original_material_state() -> void:
	original_material_state = {
		"albedo_color": material.albedo_color,
		"emission_enabled": material.emission_enabled,
		"emission": material.emission,
		"emission_energy_multiplier": material.emission_energy_multiplier,
		"metallic": material.metallic,
		"roughness": material.roughness,
		"transparency": material.transparency
	}

func set_interior_view(enabled: bool, is_current: bool = false) -> void:
	if is_current:
		# For the current atom, hide mesh but keep nucleus
		mesh_instance.visible = false
		nucleus.visible = true
		input_ray_pickable = true  # Ensure we can still click
		return
		
	# Always ensure input handling is enabled
	input_ray_pickable = true
	mesh_instance.visible = true
	nucleus.visible = true
	
	var tween = create_tween()
	tween.set_trans(Tween.TRANS_CUBIC)
	tween.set_ease(Tween.EASE_OUT)
	
	if enabled:
		# Configure material for interior view (black)
		tween.tween_property(material, "albedo_color", interior_color, INTERIOR_TRANSITION_TIME)
		tween.parallel().tween_property(material, "emission_enabled", false, INTERIOR_TRANSITION_TIME)
		tween.parallel().tween_property(material, "metallic", 0.0, INTERIOR_TRANSITION_TIME)
		tween.parallel().tween_property(material, "roughness", 1.0, INTERIOR_TRANSITION_TIME)
		material.transparency = BaseMaterial3D.TRANSPARENCY_DISABLED
	else:
		# Restore original material properties
		var emission_color = highlight_emission_color if is_highlighted else base_emission_color
		tween.tween_property(material, "albedo_color", original_material_state["albedo_color"], INTERIOR_TRANSITION_TIME)
		tween.parallel().tween_property(material, "emission_enabled", original_material_state["emission_enabled"], INTERIOR_TRANSITION_TIME)
		tween.parallel().tween_property(material, "emission", emission_color, INTERIOR_TRANSITION_TIME)
		tween.parallel().tween_property(material, "emission_energy_multiplier", original_material_state["emission_energy_multiplier"], INTERIOR_TRANSITION_TIME)
		tween.parallel().tween_property(material, "metallic", original_material_state["metallic"], INTERIOR_TRANSITION_TIME)
		tween.parallel().tween_property(material, "roughness", original_material_state["roughness"], INTERIOR_TRANSITION_TIME)
		material.transparency = original_material_state["transparency"]

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

func _input_event(_camera: Node, event: InputEvent, _position: Vector3, _normal: Vector3, _shape_idx: int) -> void:
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT:
			print("Atom clicked: ", name, " - Emitting signal")
			atom_selected.emit(self)  # Changed to emit(self)
			get_viewport().set_input_as_handled()

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
