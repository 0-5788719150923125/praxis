extends Node3D

# Line properties
var line_color: Color = Color.WHITE  # Changed from const to var
const EXTERIOR_COLOR = Color.WHITE
const INTERIOR_COLOR = Color.BLACK
const LINE_WIDTH = 2.0

# Timing properties
const MIN_INTERVAL = 0.5
const MAX_INTERVAL = 3.0
const NUM_CONNECTIONS = 3

var next_update_time: float = 0.0
var current_lines: Array[MeshInstance3D] = []
var atoms: Array = []
var is_interior: bool = false

func _ready() -> void:
	_schedule_next_update()
	# Find the AtomInteriorSystem and connect to it
	var root = get_tree().root
	var main = root.get_node("Main")
	if main:
		var interior_system = main.find_child("InteriorAtomSystem", true, false)
		if interior_system:
			# Connect to transition progress property
			interior_system.connect("is_inside_changed", _on_interior_state_changed)

func initialize(atom_list: Array) -> void:
	atoms = atom_list

func _on_interior_state_changed(is_inside: bool) -> void:
	is_interior = is_inside
	line_color = INTERIOR_COLOR if is_inside else EXTERIOR_COLOR
	_update_line_colors()

func _update_line_colors() -> void:
	for line in current_lines:
		var material = line.material_override as StandardMaterial3D
		if material:
			material.albedo_color = line_color

func _process(_delta: float) -> void:
	if Time.get_unix_time_from_system() >= next_update_time:
		_update_connections()
		_schedule_next_update()

func _schedule_next_update() -> void:
	var interval = randf_range(MIN_INTERVAL, MAX_INTERVAL)
	next_update_time = Time.get_unix_time_from_system() + interval

func _update_connections() -> void:
	# Clear existing lines
	for line in current_lines:
		line.queue_free()
	current_lines.clear()
	
	# Create new random connections
	for i in range(NUM_CONNECTIONS):
		if atoms.size() < 2:  # Safety check
			return
			
		# Get two random atoms
		var from_atom = atoms.pick_random()
		var to_atom = atoms.pick_random()
		
		# Ensure we don't connect an atom to itself
		while to_atom == from_atom:
			to_atom = atoms.pick_random()
		
		_create_line(from_atom.global_position, to_atom.global_position)

func _create_line(start_pos: Vector3, end_pos: Vector3) -> void:
	var mesh_instance = MeshInstance3D.new()
	add_child(mesh_instance)
	
	# Create the line mesh
	var mesh = ImmediateMesh.new()
	mesh_instance.mesh = mesh
	
	# Create the material
	var material = StandardMaterial3D.new()
	material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	material.albedo_color = line_color  # Use current line_color
	material.vertex_color_use_as_albedo = true
	material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	
	# Enable double-sided rendering and disable depth testing
	material.cull_mode = BaseMaterial3D.CULL_DISABLED
	material.depth_draw_mode = BaseMaterial3D.DEPTH_DRAW_ALWAYS
	material.render_priority = 1
	
	mesh_instance.material_override = material
	
	# Draw the line
	mesh.clear_surfaces()
	mesh.surface_begin(Mesh.PRIMITIVE_LINES)
	
	# Add vertices for the line
	mesh.surface_add_vertex(start_pos)
	mesh.surface_add_vertex(end_pos)
	
	mesh.surface_end()
	
	current_lines.append(mesh_instance)
