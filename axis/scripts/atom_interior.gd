extends Node3D
class_name AtomInteriorSystem

# Base constants (calibrated for reference atom)
const BASE_ENTRY_THRESHOLD = 1.2
const BASE_EXIT_THRESHOLD = 3.0
const BASE_INTERIOR_SCALE = 200.0
const TRANSITION_DURATION = 1.0
const INVERSE_ZOOM_FACTOR = 4.0

# Camera constants
const INITIAL_MIN_ZOOM = 1.0
const INITIAL_MAX_ZOOM = 30.0
const INITIAL_ZOOM_SPEED = 0.25

# Scaling factors
const MIN_SCALE_FACTOR = 0.1  # Prevent scales from getting too small
const NUCLEUS_RELATIVE_SIZE = 0.01  # Size of nucleus relative to atom

# State tracking
var is_inside_atom = false
var is_transitioning = false
var is_transferring_interior = false
var transition_progress: float = 0.0
var transition_protected = false
var current_atom: Node3D = null
var reference_radius: float = 1.0
var camera: Camera3D = null
var transition_tween: Tween
var neural_network: Node3D = null
var world_environment: WorldEnvironment = null
var starfield: Node3D = null

signal is_inside_changed(is_inside: bool)

func _ready() -> void:
	camera = get_node("../Camera3D")
	neural_network = get_node("../NeuralNetwork")
	world_environment = get_node("../WorldEnvironment")
	starfield = get_node("../Skybox")
	
	# Get reference radius from central atom
	if neural_network and neural_network.atoms.size() > 0:
		reference_radius = neural_network.atoms[0].get_radius()

func _get_scaled_parameters(atom: Node3D) -> Dictionary:
	var atom_radius = atom.get_radius()
	var scale_factor = maxf(atom_radius / reference_radius, MIN_SCALE_FACTOR)
	
	# Calculate nucleus-relative parameters
	var nucleus_size = atom_radius * NUCLEUS_RELATIVE_SIZE
	var distance_to_nucleus = atom_radius * 0.5  # Half the atom radius
	
	# Adjust thresholds based on atom size
	var entry_threshold = BASE_ENTRY_THRESHOLD / scale_factor  # Easier to enter small atoms
	var exit_threshold = BASE_EXIT_THRESHOLD * scale_factor    # Harder to accidentally exit small atoms
	
	# Scale interior parameters
	var interior_scale = BASE_INTERIOR_SCALE / scale_factor   # Smaller atoms feel bigger inside
	var zoom_speed = INITIAL_ZOOM_SPEED * scale_factor        # Slower movement in small atoms
	
	# Calculate minimum approach distance based on nucleus size
	var min_distance = nucleus_size * 2.0  # Keep camera from getting too close to nucleus
	
	return {
		"entry_threshold": entry_threshold,
		"exit_threshold": exit_threshold,
		"interior_scale": interior_scale,
		"zoom_speed": zoom_speed,
		"min_distance": min_distance,
		"nucleus_distance": distance_to_nucleus
	}


func _process(_delta: float) -> void:
	if not camera or not world_environment:
		return
		
	# Don't check distances during any kind of transition
	if is_transitioning or is_transferring_interior:
		return
		
	# Don't check distances during camera movement
	if camera.transition_tween and camera.transition_tween.is_valid():
		return
		
	if neural_network:
		var atoms_node = neural_network.get_node("Atoms")
		if atoms_node:
			for atom in atoms_node.get_children():
				var params = _get_scaled_parameters(atom)
				var distance_factor = _get_atom_distance_factor(atom)
				
				if is_inside_atom:
					if atom == current_atom and distance_factor > params.exit_threshold:
						print("Distance check: ", distance_factor, " vs threshold ", params.exit_threshold)
						_exit_atom()
				else:
					if distance_factor < params.entry_threshold:
						_enter_atom(atom)

func _switch_atom_interior(new_atom: Node3D) -> void:
	if new_atom == current_atom:
		return
		
	print("Starting interior transfer...")
	is_transferring_interior = true
	
	if neural_network:
		# Keep all atoms black
		for atom in neural_network.atoms:
			if atom == current_atom:
				# Current atom stays black but becomes visible
				atom.set_interior_view(true, false)
			elif atom == new_atom:
				# New target atom starts transition
				atom.set_interior_view(true, false)  # Keep visible initially
				atom.start_fade()  # Start fade transition for target
			else:
				# All other atoms stay black
				atom.set_interior_view(true, false)
			
			atom.set_highlight(atom == new_atom)
	
	# Update current atom reference
	current_atom = new_atom
	
	# Update camera settings
	const BASE_MIN_DISTANCE = 0.2
	var atom_radius = new_atom.get_radius()
	camera.min_zoom = BASE_MIN_DISTANCE * atom_radius
	camera.max_zoom = atom_radius * BASE_INTERIOR_SCALE
	
	camera.set_interior_mode(true, {
		"radius": atom_radius,
		"transfer_from_interior": true
	})
	
	# Monitor distance for fade
	var monitor_tween = create_tween()
	monitor_tween.set_loops()
	monitor_tween.tween_interval(0.016)
	
	const START_FADE = 6.0    # Start fading much earlier (6x radius)
	const END_FADE = 1.0     # Complete fade at entry
	const FADE_POWER = 0.3   # Power curve for fade (less than 1 = slower fade initially)
	
	var monitor = func(_step = 0):  # Fixed: Accept step parameter
		var distance = camera.global_position.distance_to(new_atom.global_position)
		var relative_distance = distance / atom_radius
		
		# Calculate base fade progress
		var fade_progress = clamp((relative_distance - END_FADE) / (START_FADE - END_FADE), 0.0, 1.0)
		
		# Apply power curve to make fade more dramatic near end
		var fade = pow(fade_progress, FADE_POWER)
		
		print("Distance: ", relative_distance, " Fade: ", fade)  # Debug info
		new_atom.update_fade(fade)
		
		# Complete transition when fully faded
		if relative_distance <= END_FADE:
			monitor_tween.kill()
			is_transferring_interior = false
			new_atom.set_interior_view(true, true)
			return false
			
		return true
	
	monitor_tween.connect("step_finished", monitor)
	camera.set_focus_target(new_atom)

# Also modify enter_atom to ensure consistent behavior
func _enter_atom(atom: Node3D) -> void:
	if is_transitioning or transition_protected:
		return
	
	print("Entering atom interior...")
	is_transitioning = true
	is_inside_atom = true
	current_atom = atom
	
	if neural_network:
		# Set ALL atoms to interior mode first
		for other_atom in neural_network.atoms:
			other_atom.set_interior_view(true, false)
		
		# Then set the target atom's final state
		current_atom.set_interior_view(true, true)
	
	if starfield:
		starfield.toggle_inverse_mode(true)
	
	# Update camera settings
	var atom_radius = atom.get_radius()
	camera.min_zoom = 0.2 * atom_radius
	camera.max_zoom = atom_radius * BASE_INTERIOR_SCALE
	
	camera.set_interior_mode(true, {
		"radius": atom_radius,
		"transfer_from_interior": false
	})
	
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

func _exit_atom(force_skybox_toggle: bool = true) -> void:
	# Never exit if we're in any kind of protected state
	if is_transitioning or is_transferring_interior or transition_protected:
		print("Preventing exit during protected state...")
		return
	
	print("Exiting atom interior...")
	is_transitioning = true
	
	if neural_network:
		for atom in neural_network.atoms:
			atom.set_interior_view(false)
	
	# Only toggle skybox if actually exiting to exterior
	if starfield and force_skybox_toggle:
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

func _get_atom_distance_factor(atom: Node3D) -> float:
	var distance = camera.global_position.distance_to(atom.global_position)
	var radius = atom.get_radius()
	# Scale padding with atom size
	var padding = radius * 0.1  # 10% of radius as padding
	return (distance - padding) / (radius + padding)

func _modify_camera_for_interior(params: Dictionary) -> void:
	if not camera.has_meta("original_zoom_speed"):
		camera.set_meta("original_zoom_speed", camera.zoom_speed)
	
	camera.zoom_speed = params.zoom_speed
	
	# Configure camera for interior mode with specific parameters
	var nucleus_params = {
		"nucleus_distance": params.nucleus_distance,
		"min_distance": params.min_distance,
		"scale_factor": params.interior_scale
	}
	
	camera.set_interior_mode(true, nucleus_params)

func _reset_camera() -> void:
	camera.min_zoom = INITIAL_MIN_ZOOM
	camera.max_zoom = INITIAL_MAX_ZOOM
	camera.zoom_speed = INITIAL_ZOOM_SPEED
	
	if camera.has_meta("original_zoom_speed"):
		camera.remove_meta("original_zoom_speed")

func _on_enter_transition_complete() -> void:
	is_transitioning = false

func _on_exit_transition_complete() -> void:
	is_transitioning = false
	is_inside_atom = false
	current_atom = null
