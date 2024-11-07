extends Camera3D

@export var orbit_speed := 1.0
@export var min_zoom := 2.0
@export var max_zoom := 10.0
@export var zoom_speed := 0.5
@export var initial_distance := 4.0

var orbit_point := Vector3.ZERO
var camera_distance: float
var polar_angle: float
var azimuth_angle: float
var is_orbiting := false
var touch_start_position := Vector2.ZERO

func _ready() -> void:
	# Initialize camera position
	camera_distance = initial_distance
	polar_angle = 0.0  # Vertical angle
	azimuth_angle = 0.0  # Horizontal angle
	_update_camera_position()

func _input(event: InputEvent) -> void:
	# Handle mouse input
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT:
			is_orbiting = event.pressed
			if is_orbiting:
				touch_start_position = event.position
				get_viewport().set_input_as_handled()
				
	# Handle touch input
	elif event is InputEventScreenTouch:
		is_orbiting = event.pressed
		if is_orbiting:
			touch_start_position = event.position
			get_viewport().set_input_as_handled()
			
	# Handle mouse/touch drag
	elif (event is InputEventMouseMotion or event is InputEventScreenDrag) and is_orbiting:
		var delta = event.position - touch_start_position
		touch_start_position = event.position
		
		# Invert the deltas to make movement more intuitive
		# Now dragging right will move the camera right (object appears to move left)
		# Dragging up will move the camera up (object appears to move down)
		azimuth_angle += delta.x * orbit_speed * 0.01  # Removed the minus sign
		polar_angle += delta.y * orbit_speed * 0.01   # Removed the minus sign
		
		# Clamp polar angle to prevent camera flipping
		polar_angle = clamp(polar_angle, -PI/2 + 0.1, PI/2 - 0.1)
		
		_update_camera_position()
		get_viewport().set_input_as_handled()
	
	# Handle pinch to zoom on mobile
	elif event is InputEventMagnifyGesture:
		camera_distance = clamp(camera_distance / event.factor, min_zoom, max_zoom)
		_update_camera_position()
		get_viewport().set_input_as_handled()
	
	# Handle mouse wheel zoom
	elif event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			camera_distance = clamp(camera_distance - zoom_speed, min_zoom, max_zoom)
			_update_camera_position()
			get_viewport().set_input_as_handled()
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			camera_distance = clamp(camera_distance + zoom_speed, min_zoom, max_zoom)
			_update_camera_position()
			get_viewport().set_input_as_handled()

func _update_camera_position() -> void:
	# Convert spherical coordinates to Cartesian
	var x = camera_distance * cos(polar_angle) * cos(azimuth_angle)
	var y = camera_distance * sin(polar_angle)
	var z = camera_distance * cos(polar_angle) * sin(azimuth_angle)
	
	# Update camera position and look at center
	position = Vector3(x, y, z)
	look_at(orbit_point)
