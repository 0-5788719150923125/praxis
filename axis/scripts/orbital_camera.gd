extends Camera3D

@export var orbit_speed := 1.0
@export var min_zoom := 2.0
@export var max_zoom := 10.0
@export var zoom_speed := 0.5
@export var initial_distance := 4.0
@export var touch_zoom_sensitivity := 0.01

var camera_distance: float
var rotation_quaternion: Quaternion
var up_vector := Vector3.UP
var is_orbiting := false
var touch_start_position := Vector2.ZERO
var focus_target: Node3D = null

# Touch zoom variables
var touch_points := {}
var previous_touch_distance := 0.0
var is_zooming := false

func _ready() -> void:
	camera_distance = initial_distance
	rotation_quaternion = Quaternion.IDENTITY
	_update_camera_position()

func set_focus_target(target: Node3D) -> void:
	focus_target = target
	_update_camera_position()

func _process(_delta: float) -> void:
	if focus_target:
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
				is_zooming = true
				is_orbiting = false
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
					_handle_rotation(event.position - touch_start_position)
					touch_start_position = event.position
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
		_handle_rotation(event.position - touch_start_position)
		touch_start_position = event.position
		get_viewport().set_input_as_handled()

func _handle_rotation(delta: Vector2) -> void:
	# Convert screen delta to rotation angles
	var horizontal_rotation = Quaternion(up_vector, -delta.x * orbit_speed * 0.01)
	
	# Create rotation around the right vector (for vertical rotation)
	var right_vector = rotation_quaternion * Vector3.RIGHT
	var vertical_rotation = Quaternion(right_vector, -delta.y * orbit_speed * 0.01)
	
	# Apply rotations
	rotation_quaternion = horizontal_rotation * rotation_quaternion * vertical_rotation
	rotation_quaternion = rotation_quaternion.normalized()  # Prevent accumulated errors

func _zoom_camera(zoom_delta: float) -> void:
	camera_distance = clamp(camera_distance + zoom_delta, min_zoom, max_zoom)
	_update_camera_position()

func _get_touch_distance() -> float:
	if touch_points.size() < 2:
		return 0.0
	var points = touch_points.values()
	return points[0].distance_to(points[1])

func _update_camera_position() -> void:
	var target_pos = focus_target.global_position if focus_target else Vector3.ZERO
	
	# Start with the base offset
	var offset = Vector3(0, 0, camera_distance)
	
	# Apply rotation
	offset = rotation_quaternion * offset
	
	# Set camera position and look at target
	position = target_pos + offset
	look_at(target_pos, up_vector)
