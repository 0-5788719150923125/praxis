extends Camera3D

@export var orbit_speed := 1.0
@export var min_zoom := 2.0
@export var max_zoom := 10.0
@export var zoom_speed := 0.5
@export var initial_distance := 4.0
@export var touch_zoom_sensitivity := 0.01  # Adjust this to control pinch zoom speed

var orbit_point := Vector3.ZERO
var camera_distance: float
var polar_angle: float
var azimuth_angle: float
var is_orbiting := false
var touch_start_position := Vector2.ZERO

# Touch zoom variables
var touch_points := {}
var previous_touch_distance := 0.0
var is_zooming := false

func _ready() -> void:
	# Initialize camera position
	camera_distance = initial_distance
	polar_angle = 0.0  # Vertical angle
	azimuth_angle = 0.0  # Horizontal angle
	_update_camera_position()

func _input(event: InputEvent) -> void:
	# Handle desktop zoom
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			_zoom_camera(-zoom_speed)
			get_viewport().set_input_as_handled()
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			_zoom_camera(zoom_speed)
			get_viewport().set_input_as_handled()
	
	# Handle touch input
	if event is InputEventScreenTouch:
		if event.pressed:
			touch_points[event.index] = event.position
			if touch_points.size() == 2:
				# Two fingers touching - start zoom
				is_zooming = true
				is_orbiting = false  # Stop orbiting while zooming
				previous_touch_distance = _get_touch_distance()
		else:
			touch_points.erase(event.index)
			if touch_points.size() < 2:
				is_zooming = false
	
	elif event is InputEventScreenDrag:
		if touch_points.has(event.index):
			touch_points[event.index] = event.position
			
			if is_zooming and touch_points.size() == 2:
				# Handle pinch zoom
				var new_touch_distance = _get_touch_distance()
				var zoom_delta = (previous_touch_distance - new_touch_distance) * touch_zoom_sensitivity
				_zoom_camera(zoom_delta)
				previous_touch_distance = new_touch_distance
				get_viewport().set_input_as_handled()
			elif !is_zooming and touch_points.size() == 1:
				# Handle single finger orbit
				if is_orbiting:
					var delta = event.position - touch_start_position
					touch_start_position = event.position
					
					azimuth_angle += delta.x * orbit_speed * 0.01
					polar_angle += delta.y * orbit_speed * 0.01
					polar_angle = clamp(polar_angle, -PI/2 + 0.1, PI/2 - 0.1)
					
					_update_camera_position()
					get_viewport().set_input_as_handled()
	
	# Handle single touch for orbiting
	if event is InputEventScreenTouch and event.index == 0:
		is_orbiting = event.pressed and !is_zooming
		if is_orbiting:
			touch_start_position = event.position
			get_viewport().set_input_as_handled()
	
	# Handle mouse orbit
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT:
			is_orbiting = event.pressed
			if is_orbiting:
				touch_start_position = event.position
				get_viewport().set_input_as_handled()
	elif event is InputEventMouseMotion and is_orbiting:
		var delta = event.position - touch_start_position
		touch_start_position = event.position
		
		azimuth_angle += delta.x * orbit_speed * 0.01
		polar_angle += delta.y * orbit_speed * 0.01
		polar_angle = clamp(polar_angle, -PI/2 + 0.1, PI/2 - 0.1)
		
		_update_camera_position()
		get_viewport().set_input_as_handled()

func _zoom_camera(zoom_delta: float) -> void:
	camera_distance = clamp(camera_distance + zoom_delta, min_zoom, max_zoom)
	_update_camera_position()

func _get_touch_distance() -> float:
	if touch_points.size() < 2:
		return 0.0
	var points = touch_points.values()
	return points[0].distance_to(points[1])

func _update_camera_position() -> void:
	# Convert spherical coordinates to Cartesian
	var x = camera_distance * cos(polar_angle) * cos(azimuth_angle)
	var y = camera_distance * sin(polar_angle)
	var z = camera_distance * cos(polar_angle) * sin(azimuth_angle)
	
	# Update camera position and look at center
	position = Vector3(x, y, z)
	look_at(orbit_point)
