extends Camera3D

@export var orbit_speed := 1.0
@export var min_zoom := 2.0
@export var max_zoom := 10.0
@export var zoom_speed := 0.5
@export var initial_distance := 4.0
@export var touch_zoom_sensitivity := 0.01
@export var transition_duration := 0.5
@export var bounce_overshoot := 0.0

# Physics parameters
@export var base_damping := 0.995  # Base damping factor (higher = less friction)
@export var velocity_dependent_damping := 0.02  # Additional damping based on velocity
@export var input_sensitivity := 0.005  # How much input affects rotation
@export var min_rotation_speed := 0.001  # When to stop rotating
@export var input_acceleration_factor := 1.2  # How much sequential inputs compound

# State variables
var camera_distance: float
var rotation_quaternion: Quaternion
var up_vector := Vector3.UP
var is_orbiting := false
var touch_start_position := Vector2.ZERO
var focus_target: Node3D = null
var current_orbit_point := Vector3.ZERO
var target_orbit_point := Vector3.ZERO
var transition_tween: Tween

# Physics state
var rotation_velocity := Vector2.ZERO  # Current rotation speed
var last_input_time := 0.0  # Track timing of inputs
var input_chain_multiplier := 1.0  # Tracks sequential input momentum
var last_input_direction := Vector2.ZERO  # For comparing input directions

# Movement physics
var last_mouse_delta := Vector2.ZERO  # For calculating acceleration
var current_zoom_velocity := 0.0  # For smooth zooming
var zoom_damping := 0.85  # How quickly zoom slows down

# Touch zoom variables
var touch_points := {}
var previous_touch_distance := 0.0
var is_zooming := false

func _ready() -> void:
	camera_distance = initial_distance
	rotation_quaternion = Quaternion.IDENTITY
	_update_camera_position()

func _process(delta: float) -> void:
	# Apply rotation physics when not transitioning between targets
	if not transition_tween or not transition_tween.is_running():
		if rotation_velocity.length() > min_rotation_speed:
			# Calculate velocity-dependent damping
			var speed = rotation_velocity.length()
			var total_damping = base_damping - (velocity_dependent_damping * speed)
			total_damping = clamp(total_damping, 0.9, 0.999)  # Ensure reasonable bounds
			
			# Apply damping
			rotation_velocity *= total_damping
		
		# Apply rotation
		_apply_rotation(rotation_velocity * delta)
		
		# Apply zoom physics
		if abs(current_zoom_velocity) > 0.001:
			_zoom_camera(current_zoom_velocity)
			current_zoom_velocity *= zoom_damping
	
	if focus_target:
		target_orbit_point = focus_target.global_position
		if not transition_tween or not transition_tween.is_running():
			current_orbit_point = target_orbit_point
			position = _calculate_camera_position(current_orbit_point)
	
	# Set the camera's transform directly to avoid gimbal lock at poles
	transform = Transform3D(Basis(rotation_quaternion), position)


func _calculate_exponential_falloff(value: float, base: float, exponent: float) -> float:
	return 1.0 - pow(base, -value * exponent)

func _handle_rotation(delta: Vector2) -> void:
	var current_time = Time.get_ticks_msec() / 1000.0
	var time_since_last_input = current_time - last_input_time
	
	# Calculate input direction correlation with current velocity
	var input_direction = delta.normalized()
	var direction_alignment = input_direction.dot(last_input_direction) if last_input_direction.length() > 0 else 0.0
	
	# Reset or increase chain multiplier based on timing and direction
	if time_since_last_input > 0.5 or direction_alignment < 0.7:
		input_chain_multiplier = 1.0
	else:
		# Increase multiplier for sequential inputs in similar directions
		input_chain_multiplier = min(input_chain_multiplier * input_acceleration_factor, 5.0)
	
	# Calculate input force with chain multiplier
	var input_force = delta * input_sensitivity * input_chain_multiplier
	
	# Add to current velocity
	rotation_velocity += input_force
	
	# Update state tracking
	last_input_time = current_time
	last_input_direction = input_direction

func _apply_rotation(rotation_amount: Vector2) -> void:
	# Get our current orientation vectors
	var basis = Basis(rotation_quaternion)
	var camera_right = basis.x
	var camera_up = basis.y
	
	# Create a rotation for each axis using our stable camera vectors
	var vertical_rotation = Quaternion(camera_right, -rotation_amount.y)
	var horizontal_rotation = Quaternion(camera_up, -rotation_amount.x)
	
	# Apply in a fixed order (horizontal first)
	rotation_quaternion = horizontal_rotation * vertical_rotation * rotation_quaternion
	rotation_quaternion = rotation_quaternion.normalized()

func _update_camera_position() -> void:
	var target_pos = focus_target.global_position if focus_target else Vector3.ZERO
	
	# Get our current orientation as a basis
	var basis = Basis(rotation_quaternion)
	
	# Calculate camera position using basis
	var offset = basis * Vector3(0, 0, camera_distance)
	position = target_pos + offset
	
	# Look at target with current camera up
	look_at(target_pos, basis.y)

func set_focus_target(new_target: Node3D) -> void:
	if focus_target == new_target:
		return
		
	focus_target = new_target
	target_orbit_point = new_target.global_position
	
	# Kill existing tween
	if transition_tween and transition_tween.is_valid():
		transition_tween.kill()
	
	# Calculate transition parameters based on distance
	var distance = position.distance_to(target_orbit_point)
	var travel_time = clamp(distance * 0.1, 0.2, 1.0)  # Longer transitions for longer distances
	var bounce_factor = clamp(distance * 0.01, 0.0, bounce_overshoot)  # More bounce for longer distances
	
	# Create transition
	transition_tween = create_tween()
	transition_tween.set_ease(Tween.EASE_OUT)
	transition_tween.set_trans(Tween.TRANS_CUBIC)
	
	var final_position = _calculate_camera_position(target_orbit_point)
	var bounce_position = final_position + (final_position - position) * bounce_factor
	
	transition_tween.tween_property(self, "current_orbit_point", target_orbit_point, travel_time * 0.7)
	transition_tween.parallel().tween_property(self, "position", bounce_position, travel_time * 0.7)
	transition_tween.chain().tween_property(self, "position", final_position, travel_time * 0.3)

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

func _calculate_camera_position(orbit_point: Vector3) -> Vector3:
	# Calculate desired camera position based on orbit point
	var offset = rotation_quaternion * Vector3(0, 0, camera_distance)
	return orbit_point + offset

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

func _zoom_camera(zoom_delta: float) -> void:
	camera_distance = clamp(camera_distance + zoom_delta, min_zoom, max_zoom)
	_update_camera_position()

func _get_touch_distance() -> float:
	if touch_points.size() < 2:
		return 0.0
	var points = touch_points.values()
	return points[0].distance_to(points[1])
