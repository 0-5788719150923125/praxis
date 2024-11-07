extends Camera3D

@export var orbit_speed := 1.0
@export var min_zoom := 2.0
@export var max_zoom := 10.0
@export var zoom_speed := 0.5
@export var initial_distance := 4.0
@export var touch_zoom_sensitivity := 0.01
@export var transition_duration := 0.5
@export var bounce_overshoot := 0.1

var camera_distance: float
var rotation_quaternion: Quaternion
var up_vector := Vector3.UP
var is_orbiting := false
var touch_start_position := Vector2.ZERO
var focus_target: Node3D = null
var current_orbit_point := Vector3.ZERO
var target_orbit_point := Vector3.ZERO
var transition_tween: Tween

# Touch zoom variables
var touch_points := {}
var previous_touch_distance := 0.0
var is_zooming := false

func _ready() -> void:
	camera_distance = initial_distance
	rotation_quaternion = Quaternion.IDENTITY
	_update_camera_position()

func _unhandled_input(event: InputEvent) -> void:  # Changed from _input to _unhandled_input
	# Handle desktop zoom
	if event is InputEventMouseButton:
		match event.button_index:
			MOUSE_BUTTON_WHEEL_UP:
				_zoom_camera(-zoom_speed)
				get_viewport().set_input_as_handled()
			MOUSE_BUTTON_WHEEL_DOWN:
				_zoom_camera(zoom_speed)
				get_viewport().set_input_as_handled()
			MOUSE_BUTTON_RIGHT:  # Changed orbit to right mouse button
				is_orbiting = event.pressed
				if is_orbiting:
					touch_start_position = event.position
					get_viewport().set_input_as_handled()
	
	# Handle mouse motion only when orbiting
	elif event is InputEventMouseMotion and is_orbiting:
		_handle_rotation(event.position - touch_start_position)
		touch_start_position = event.position
		get_viewport().set_input_as_handled()
	
	# Handle touch input
	elif event is InputEventScreenTouch:
		if event.pressed:
			if touch_points.size() == 0:  # First touch
				touch_points[event.index] = event.position
			elif touch_points.size() == 1:  # Second touch - start zoom
				touch_points[event.index] = event.position
				is_zooming = true
				is_orbiting = false
				previous_touch_distance = _get_touch_distance()
				get_viewport().set_input_as_handled()
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
			elif touch_points.size() == 2:  # Two finger orbit
				is_orbiting = true
				_handle_rotation(event.position - touch_start_position)
				touch_start_position = event.position
				get_viewport().set_input_as_handled()

func set_focus_target(new_target: Node3D) -> void:
	print("Camera focusing on new target: ", new_target.name)  # Debug print
	
	if focus_target == new_target:
		print("Already focused on this target")
		return
		
	focus_target = new_target
	target_orbit_point = new_target.global_position
	print("Target orbit point set to: ", target_orbit_point)  # Debug print
	
	# Kill existing tween if any
	if transition_tween and transition_tween.is_valid():
		transition_tween.kill()
	
	# Create new tween for smooth transition
	transition_tween = create_tween()
	transition_tween.set_ease(Tween.EASE_OUT)
	transition_tween.set_trans(Tween.TRANS_CUBIC)
	
	# Add position tween with slight bounce
	var final_position = _calculate_camera_position(target_orbit_point)
	var bounce_position = final_position + (final_position - position) * bounce_overshoot
	
	print("Starting camera transition")  # Debug print
	transition_tween.tween_property(self, "current_orbit_point", target_orbit_point, transition_duration * 0.7)
	transition_tween.parallel().tween_property(self, "position", bounce_position, transition_duration * 0.7)
	transition_tween.chain().tween_property(self, "position", final_position, transition_duration * 0.3)

func _calculate_camera_position(orbit_point: Vector3) -> Vector3:
	# Calculate desired camera position based on orbit point
	var offset = rotation_quaternion * Vector3(0, 0, camera_distance)
	return orbit_point + offset

func _process(_delta: float) -> void:
	if focus_target:
		# Update target position in case the target moves
		target_orbit_point = focus_target.global_position
		
		if not transition_tween or not transition_tween.is_running():
			# No transition running, just update normally
			current_orbit_point = target_orbit_point
			position = _calculate_camera_position(current_orbit_point)
	
	# Always look at current orbit point
	look_at(current_orbit_point, up_vector)

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
