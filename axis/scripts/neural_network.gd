extends Node3D

var atom_scene = preload("res://scenes/atom.tscn")
@onready var atoms_container = $Atoms
@onready var camera = $"../Camera3D"
@onready var synapse_manager = $SynapseManager

const NUM_ATOMS = 7

# Planetary scale constants
const MIN_RADIUS = 0.3      # Smallest atom size
const MAX_RADIUS = 1.2      # Largest atom size
const CENTER_RADIUS = 0.9   # Size of central atom

# Distance ranges (like planetary orbits)
const INNER_ORBIT = Vector2(2.0, 4.0)     # Close atoms
const MIDDLE_ORBIT = Vector2(6.0, 10.0)   # Medium distance atoms
const OUTER_ORBIT = Vector2(15.0, 25.0)   # Far atoms

# Minimum separation between atoms (scaled by their combined radii)
const MIN_SEPARATION_FACTOR = 3.0

var atoms: Array = []

func _ready() -> void:
	print("Neural Network initializing...")
	
	# Create central atom
	var central_atom = _create_atom(Vector3.ZERO)
	central_atom.set_radius(CENTER_RADIUS)
	central_atom.set_highlight(true)
	atoms.append(central_atom)
	
	# Distribute remaining atoms across different orbits
	var orbit_distributions = [
		INNER_ORBIT,   # 2 atoms in inner orbit
		INNER_ORBIT,
		MIDDLE_ORBIT,  # 2 atoms in middle orbit
		MIDDLE_ORBIT,
		OUTER_ORBIT,   # 2 atoms in outer orbit
		OUTER_ORBIT
	]
	
	# Create surrounding atoms
	for orbit_range in orbit_distributions:
		var position = _get_random_position_in_orbit(orbit_range)
		var atom = _create_atom(position)
		
		# Randomize size based on distance (further atoms can be larger)
		var distance_factor = position.length() / OUTER_ORBIT.y  # 0 to 1 based on distance
		var size_variation = randf_range(-0.2, 0.2)  # Add some randomness
		var radius = lerp(MIN_RADIUS, MAX_RADIUS, distance_factor + size_variation)
		atom.set_radius(radius)
		
		atoms.append(atom)
	
	# Set camera focus to central atom
	if camera:
		camera.set_focus_target(atoms[0])
		# Adjust camera's zoom limits based on our scale
		camera.min_zoom = INNER_ORBIT.x * 0.5  # Close enough to see inner atoms
		camera.max_zoom = OUTER_ORBIT.y * 1.2   # Far enough to see outer atoms
		camera.initial_distance = MIDDLE_ORBIT.x # Start at middle distance
	
	# Initialize synapse manager with our atoms
	synapse_manager.initialize(atoms)
	
	print("Neural network initialized with ", atoms.size(), " atoms")

func _create_atom(position: Vector3) -> Node3D:
	var atom = atom_scene.instantiate()
	atoms_container.add_child(atom)
	atom.global_position = position
	return atom

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

# Optional: Add method to handle atom selection
func _on_atom_selected(atom: Node3D) -> void:
	# Unhighlight all atoms
	for a in atoms:
		a.set_highlight(false)
	
	# Highlight and focus the selected atom
	atom.set_highlight(true)
	camera.set_focus_target(atom)
