extends Node3D

var atom_scene = preload("res://scenes/atom.tscn")
@onready var atoms_container = $Atoms
@onready var camera = get_node("../Camera3D")

const NUM_ATOMS = 7

# Planetary scale constants
const MIN_RADIUS = 0.3      # Smallest atom size
const MAX_RADIUS = 1.2      # Largest atom size
const CENTER_RADIUS = 0.9   # Size of central atom
const MIN_SEPARATION_FACTOR = 3.0  # Added this constant

# Distance ranges (like planetary orbits)
const INNER_ORBIT = Vector2(2.0, 4.0)     # Close atoms
const MIDDLE_ORBIT = Vector2(6.0, 10.0)   # Medium distance atoms
const OUTER_ORBIT = Vector2(15.0, 25.0)   # Far atoms

# Define orbit distributions for the atoms
var orbit_distributions = [
	INNER_ORBIT,   # 2 atoms in inner orbit
	INNER_ORBIT,
	MIDDLE_ORBIT,  # 2 atoms in middle orbit
	MIDDLE_ORBIT,
	OUTER_ORBIT,   # 2 atoms in outer orbit
	OUTER_ORBIT
]

# Store all atoms for later reference
var atoms: Array = []
var current_focused_atom: Node3D = null  # Track currently focused atom

func _ready() -> void:
	print("Neural Network initializing...")
	
	if camera:
		print("Camera found successfully!")
	else:
		print("ERROR: Camera not found! Current path: ", get_node("../Camera3D"))
	
	# Create initial central atom
	var central_atom = _create_atom(Vector3.ZERO)
	central_atom.set_radius(CENTER_RADIUS)
	central_atom.set_highlight(true)  # Set initial highlight
	atoms.append(central_atom)
	current_focused_atom = central_atom
	
	# Create surrounding atoms
	for orbit_range in orbit_distributions:
		var pos = _get_random_position_in_orbit(orbit_range)
		var atom = _create_atom(pos)
		
		# Randomize size based on distance
		var distance_factor = pos.length() / OUTER_ORBIT.y
		var size_variation = randf_range(-0.2, 0.2)
		var radius = lerp(MIN_RADIUS, MAX_RADIUS, distance_factor + size_variation)
		atom.set_radius(radius)
		atom.set_highlight(false)  # Ensure it starts unhighlighted
		
		atoms.append(atom)
	
	# Set camera focus
	if camera:
		camera.set_focus_target(current_focused_atom)
		camera.min_zoom = INNER_ORBIT.x * 0.5
		camera.max_zoom = OUTER_ORBIT.y * 1.2
		camera.initial_distance = MIDDLE_ORBIT.x
	
	# Initialize synapse manager if it exists
	if $SynapseManager:
		$SynapseManager.initialize(atoms)
	
	print("Neural network initialized with ", atoms.size(), " atoms")

func _create_atom(pos: Vector3) -> Node3D:
	var atom = atom_scene.instantiate()
	atoms_container.add_child(atom)
	atom.global_position = pos
	
	# Connect the signal with error checking
	var connect_result = atom.connect("atom_selected", _on_atom_selected)
	if connect_result == OK:
		print("Successfully connected signal for atom: ", atom.name)
	else:
		print("Failed to connect signal for atom: ", atom.name, " Error: ", connect_result)
	
	return atom

func _on_atom_selected(selected_atom: Area3D) -> void:
	print("Signal received! Atom selected: ", selected_atom.name)
	
	if selected_atom == current_focused_atom:
		print("Atom already focused")
		return
	
	# Update highlights
	if current_focused_atom:
		print("Un-highlighting current atom: ", current_focused_atom.name)
		current_focused_atom.set_highlight(false)
	
	print("Highlighting new atom: ", selected_atom.name)
	selected_atom.set_highlight(true)
	current_focused_atom = selected_atom
	
	# Update camera
	if camera:
		print("Moving camera to focus on: ", selected_atom.name)
		camera.set_focus_target(selected_atom)
	else:
		print("ERROR: Camera not found in _on_atom_selected")

func _update_atom_highlights(focused_atom: Node3D) -> void:
	print("Updating atom highlights, focused atom: ", focused_atom.name)
	for atom in atoms:
		atom.set_highlight(atom == focused_atom)

func _get_random_position_in_orbit(orbit_range: Vector2) -> Vector3:
	var max_attempts = 50
	var attempts = 0
	
	while attempts < max_attempts:
		# Generate random spherical coordinates
		var phi = randf() * TAU                    # Random angle around Y axis
		var theta = acos(randf_range(-1.0, 1.0))   # Random angle from Y axis
		var distance = randf_range(orbit_range.x, orbit_range.y)
		
		# Convert to Cartesian coordinates
		var pos = Vector3(
			distance * sin(theta) * cos(phi),
			distance * cos(theta),
			distance * sin(theta) * sin(phi)
		)
		
		# Check if position is valid
		if _is_position_valid(pos):
			return pos
		
		attempts += 1
	
	# Fallback: Return a position on the orbit's middle radius
	var fallback_distance = (orbit_range.x + orbit_range.y) * 0.5
	return Vector3(fallback_distance, 0, 0)

func _is_position_valid(pos: Vector3) -> bool:
	for atom in atoms:
		var min_distance = (atom.get_radius() + MIN_RADIUS) * MIN_SEPARATION_FACTOR
		if atom.global_position.distance_to(pos) < min_distance:
			return false
	return true
