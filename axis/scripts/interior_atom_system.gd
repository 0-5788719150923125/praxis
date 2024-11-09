extends Node3D
class_name AtomInteriorSystem

# Constants for transition and scaling
const TRANSITION_DURATION := 1.0
# Modify these constants
#const INTERIOR_ENTRY_THRESHOLD := 0.5  # When camera distance is this times atom radius
#const INTERIOR_EXIT_THRESHOLD := 4.0   # When interior camera distance exceeds this
#const INTERIOR_SCALE_FACTOR := 100.0   # How much bigger the interior feels
#const MIN_INTERIOR_DISTANCE := 0.05    # Allow closer zoom
const INTERIOR_ENTRY_THRESHOLD := 1.2  # Made it easier to enter again
const INTERIOR_EXIT_THRESHOLD := 3.0   # Reduced exit threshold
const INTERIOR_SCALE_FACTOR := 50.0    # Made scaling less extreme
const MIN_INTERIOR_DISTANCE := 0.1     # Increased minimum distance
const INVERSE_ZOOM_FACTOR := 2.0     # Controls how "infinite" the interior zoom feels

# Initial camera values to restore
const INITIAL_MIN_ZOOM := 1.0  # Changed from 2.0 to match your debug output
const INITIAL_MAX_ZOOM := 30.0 # Changed from 8.0 to match your debug output
const INITIAL_ZOOM_SPEED := 0.25

# State tracking
var is_inside_atom := false
var is_transitioning := false
var transition_progress: float = 0.0  # Added this property
var current_atom: Node3D = null
var original_skybox: Node3D = null
var camera: Camera3D = null
var transition_tween: Tween
var neural_network: Node3D = null
var world_environment: WorldEnvironment = null
var base_environment: Environment = null
var interior_environment: Environment = null

func _ready() -> void:
	# Get references
	await get_tree().create_timer(0.1).timeout  # Give other nodes time to initialize
	
	camera = get_node("../Camera3D")
	original_skybox = get_node("../Skybox")
	neural_network = get_node("../NeuralNetwork")
	world_environment = get_node("../WorldEnvironment")
	
	if world_environment:
		# Store the original environment
		base_environment = world_environment.environment.duplicate()
	
	if neural_network:
		# Connect to the neural network's atom creation signal
		neural_network.connect("atom_created", _on_atom_created)
	
	# Create inverse skybox environment
	_setup_inverse_skybox()
	print("AtomInteriorSystem initialized!")
	print("Initial camera settings - min_zoom:", camera.min_zoom, " max_zoom:", camera.max_zoom)

func _setup_inverse_skybox() -> void:
	var image = Image.create(2048, 1024, false, Image.FORMAT_RGBA8)
	image.fill(Color(1, 1, 1, 1))  # White background
	
	var rng = RandomNumberGenerator.new()
	rng.randomize()
	
	# Create black stars
	for _i in range(4000):
		var x = rng.randi() % 2048
		var y = rng.randi() % 1024
		var size = rng.randf_range(0.5, 1.5)
		var color = Color(0, 0, 0, rng.randf_range(0.5, 0.8))
		_draw_star(image, x, y, size, color)
	
	var texture = ImageTexture.create_from_image(image)
	
	# Create the inverse sky setup
	var sky_material = PanoramaSkyMaterial.new()
	sky_material.panorama = texture
	
	var sky = Sky.new()
	sky.sky_material = sky_material
	
	# Create interior environment
	interior_environment = base_environment.duplicate()
	interior_environment.background_mode = Environment.BG_SKY
	interior_environment.sky = sky

func _draw_star(image: Image, center_x: int, center_y: int, size: float, color: Color) -> void:
	var radius = ceil(size)
	for y in range(max(0, center_y - radius), min(1024, center_y + radius + 1)):
		for x in range(max(0, center_x - radius), min(2048, center_x + radius + 1)):
			var dist = Vector2(center_x, center_y).distance_to(Vector2(x, y))
			if dist <= size:
				var alpha = (1.0 - (dist / size)) * color.a
				var pixel_color = Color(color.r, color.g, color.b, alpha)
				image.set_pixel(x, y, pixel_color)

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

# Add this helper function to check if we can enter an atom:
func _can_enter_atom() -> bool:
	return not is_transitioning and not is_inside_atom

# Modify _process to use the new helper:
func _process(_delta: float) -> void:
	if not camera or not world_environment:
		return
		
	# Check all atoms for proximity
	if neural_network:
		var atoms_node = neural_network.get_node("Atoms")
		if atoms_node:
			for atom in atoms_node.get_children():
				var distance_factor = _get_atom_distance_factor(atom)
				
				# Check for entry condition
				if _can_enter_atom() and distance_factor < INTERIOR_ENTRY_THRESHOLD:
					print("Distance factor:", distance_factor, " - Entering atom")
					_enter_atom(atom)
					break
				
				# Check for exit condition if we're already inside
				elif is_inside_atom and atom == current_atom and not is_transitioning:
					if distance_factor > INTERIOR_EXIT_THRESHOLD:
						print("Distance factor:", distance_factor, " - Exiting atom")
						_exit_atom()
					break

# Modify _get_atom_distance_factor to be more lenient:
func _get_atom_distance_factor(atom: Node3D) -> float:
	var distance = camera.global_position.distance_to(atom.global_position)
	var radius = atom.get_radius()
	# Add some padding to make entry easier
	return (distance - 0.1) / (radius + 0.2)  # Added padding to both distance and radius

func _enter_atom(atom: Node3D) -> void:
	if not _can_enter_atom():
		print("Cannot enter atom - system busy or already inside")
		return
		
	print("Entering atom interior...")
	print("Current camera distance:", camera.global_position.distance_to(atom.global_position))
	print("Atom radius:", atom.get_radius())
	print("Distance factor:", _get_atom_distance_factor(atom))
	
	is_transitioning = true
	is_inside_atom = true
	current_atom = atom
	
	# Cancel any existing transition
	if transition_tween and transition_tween.is_valid():
		transition_tween.kill()
	
	# Create transition effect
	transition_tween = create_tween()
	
	# Properly tween the transition_progress property
	transition_tween.tween_property(
		self, 
		"transition_progress", 
		1.0, 
		TRANSITION_DURATION
	).set_trans(Tween.TRANS_CUBIC).set_ease(Tween.EASE_IN_OUT)
	
	# Connect to transition completion
	transition_tween.connect("finished", _on_enter_transition_complete)
	
	# Switch to interior environment
	world_environment.environment = interior_environment
	
	# Scale up the camera's perception of distance
	var target_max_zoom = current_atom.get_radius() * INTERIOR_SCALE_FACTOR
	
	camera.min_zoom = MIN_INTERIOR_DISTANCE
	camera.max_zoom = target_max_zoom
	
	# Modify camera behavior for interior navigation
	_modify_camera_for_interior()

func _exit_atom() -> void:
	if is_transitioning:
		return
		
	print("Exiting atom interior...")
	is_transitioning = true
	
	# Cancel any existing transition
	if transition_tween and transition_tween.is_valid():
		transition_tween.kill()
	
	# Create transition effect
	transition_tween = create_tween()
	
	# Properly tween the transition_progress property
	transition_tween.tween_property(
		self, 
		"transition_progress", 
		0.0, 
		TRANSITION_DURATION
	).set_trans(Tween.TRANS_CUBIC).set_ease(Tween.EASE_IN_OUT)
	
	# Connect to transition completion
	transition_tween.connect("finished", _on_exit_transition_complete)
	
	# Restore original environment
	world_environment.environment = base_environment
	
	# Reset camera parameters without moving the camera
	_reset_camera()


func _on_enter_transition_complete() -> void:
	is_transitioning = false
	print("Enter transition complete!")
	print("Current environment:", "interior" if world_environment.environment == interior_environment else "base")

func _on_exit_transition_complete() -> void:
	is_transitioning = false
	is_inside_atom = false
	current_atom = null
	print("Exit transition complete!")
	print("Current environment:", "interior" if world_environment.environment == interior_environment else "base")

func _modify_camera_for_interior() -> void:
	# Store original zoom speed if not already stored
	if not camera.has_meta("original_zoom_speed"):
		camera.set_meta("original_zoom_speed", camera.zoom_speed)
	
	# Modify zoom behavior to create "infinite" feeling
	camera.zoom_speed = camera.zoom_speed / INVERSE_ZOOM_FACTOR

func _on_atom_created(atom: Node3D) -> void:
	print("New atom registered with interior system:", atom.name)
