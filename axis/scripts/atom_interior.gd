extends Node3D
class_name AtomInteriorSystem

# Constants for transition and scaling
const TRANSITION_DURATION = 1.0
const INTERIOR_ENTRY_THRESHOLD = 1.2  # When camera distance is this times atom radius
const INTERIOR_EXIT_THRESHOLD = 3.0   # When interior camera distance exceeds this
const INTERIOR_SCALE_FACTOR = 200.0   # How much bigger the interior feels
const MIN_INTERIOR_DISTANCE = 0.001   # Allow closer zoom
const INVERSE_ZOOM_FACTOR = 4.0       # Controls how "infinite" the interior zoom feels

# Add these new constants
const DISTANCE_SCALE_FACTOR = 10.0    # Controls how quickly distance affects zoom
const BASE_ZOOM_SPEED = 0.25          # Original zoom speed
const MIN_ZOOM_SPEED = 0.001          # Minimum zoom speed 

# Initial camera values to restore
const INITIAL_MIN_ZOOM = 1.0
const INITIAL_MAX_ZOOM = 30.0
const INITIAL_ZOOM_SPEED = 0.25

# State tracking
var is_inside_atom = false
var is_transitioning = false
var current_atom: Node3D = null
var camera: Camera3D = null
var transition_tween: Tween
var transition_progress: float = 0.0
var neural_network: Node3D = null
var world_environment: WorldEnvironment = null
var starfield: Node3D = null

signal is_inside_changed(is_inside: bool)

func _ready() -> void:
	camera = get_node("../Camera3D")
	neural_network = get_node("../NeuralNetwork")
	world_environment = get_node("../WorldEnvironment")
	starfield = get_node("../Skybox")

func _process(_delta: float) -> void:
	if not camera or not world_environment:
		return
		
	# Don't check for entry conditions if we're transitioning
	if is_transitioning:
		return
		
	# Check all atoms for proximity
	if neural_network:
		var atoms_node = neural_network.get_node("Atoms")
		if atoms_node:
			for atom in atoms_node.get_children():
				var distance_factor = _get_atom_distance_factor(atom)
				
				# If we're inside an atom
				if is_inside_atom:
					if atom == current_atom and distance_factor > INTERIOR_EXIT_THRESHOLD:
						print("Distance factor:", distance_factor, " - Exiting atom")
						_exit_atom()
				# If we're outside atoms
				else:
					if distance_factor < INTERIOR_ENTRY_THRESHOLD:
						print("Distance factor:", distance_factor, " - Entering atom")
						_enter_atom(atom)

func _enter_atom(atom: Node3D) -> void:
	if is_transitioning:
		return
		
	print("Entering atom interior...")
	is_transitioning = true
	is_inside_atom = true
	current_atom = atom
	
	if neural_network:
		for other_atom in neural_network.atoms:
			other_atom.set_interior_view(true, other_atom == atom)
			
		var highlighted_atom = neural_network.current_focused_atom
		if highlighted_atom:
			highlighted_atom.set_highlight(true)
	
	# Toggle starfield to inverse mode
	if starfield:
		starfield.toggle_inverse_mode(true)
	
	# Update camera settings
	var target_max_zoom = current_atom.get_radius() * INTERIOR_SCALE_FACTOR
	camera.min_zoom = MIN_INTERIOR_DISTANCE
	camera.max_zoom = target_max_zoom
	_modify_camera_for_interior()
	
	is_inside_changed.emit(true)
	
	if transition_tween and transition_tween.is_valid():
		transition_tween.kill()
	
	transition_tween = create_tween()
	transition_tween.tween_property(
		self, 
		"transition_progress", 
		1.0, 
		TRANSITION_DURATION
	).set_trans(Tween.TRANS_CUBIC).set_ease(Tween.EASE_IN_OUT)
	
	transition_tween.connect("finished", _on_enter_transition_complete)

func _exit_atom() -> void:
	if is_transitioning:
		return
		
	print("Exiting atom interior...")
	is_transitioning = true
	
	if neural_network:
		var highlighted_atom = neural_network.current_focused_atom
		
		for atom in neural_network.atoms:
			atom.set_interior_view(false)
		
		if highlighted_atom:
			highlighted_atom.set_highlight(true)
	
	# Toggle starfield back to normal mode
	if starfield:
		starfield.toggle_inverse_mode(false)
	
	is_inside_changed.emit(false)
	
	if transition_tween and transition_tween.is_valid():
		transition_tween.kill()
	
	transition_tween = create_tween()
	transition_tween.tween_property(
		self, 
		"transition_progress", 
		0.0, 
		TRANSITION_DURATION
	).set_trans(Tween.TRANS_CUBIC).set_ease(Tween.EASE_IN_OUT)
	
	transition_tween.connect("finished", _on_exit_transition_complete)
	
	camera.set_interior_mode(false)
	_reset_camera()

func _reset_camera() -> void:
	print("Resetting camera parameters...")
	
	# Get the original zoom speed if stored
	var original_speed = camera.get_meta("original_zoom_speed", INITIAL_ZOOM_SPEED)
	
	# Reset camera parameters but maintain position
	camera.min_zoom = INITIAL_MIN_ZOOM
	camera.max_zoom = INITIAL_MAX_ZOOM
	camera.zoom_speed = original_speed
	
	# Clear any stored metadata
	if camera.has_meta("original_zoom_speed"):
		camera.remove_meta("original_zoom_speed")
	
	print("Camera parameters reset - min_zoom:", camera.min_zoom, " max_zoom:", camera.max_zoom)

# Modify _get_atom_distance_factor to be more lenient:
func _get_atom_distance_factor(atom: Node3D) -> float:
	var distance = camera.global_position.distance_to(atom.global_position)
	var radius = atom.get_radius()
	# Add some padding to make entry easier
	return (distance - 0.1) / (radius + 0.2)  # Added padding to both distance and radius

func _on_enter_transition_complete() -> void:
	is_transitioning = false
	print("Enter transition complete!")

func _on_exit_transition_complete() -> void:
	is_transitioning = false
	is_inside_atom = false
	current_atom = null
	print("Exit transition complete!")

func _modify_camera_for_interior() -> void:
	# Store original zoom speed if not already stored
	if not camera.has_meta("original_zoom_speed"):
		camera.set_meta("original_zoom_speed", camera.zoom_speed)
	
	camera.zoom_speed = BASE_ZOOM_SPEED
	camera.set_interior_mode(true)  # Enable interior mode
