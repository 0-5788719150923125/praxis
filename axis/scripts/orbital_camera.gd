extends Camera3D

@export var min_zoom = 2.0
@export var max_zoom = INF
@export var zoom_speed = 0.5
@export var initial_distance = 4.0
@export var touch_zoom_sensitivity = 0.01

# Physics parameters
@export var base_damping = 0.995
@export var velocity_dependent_damping = 0.02
@export var input_sensitivity = 0.005
@export var min_rotation_speed = 0.001
@export var input_acceleration_factor = 1.2

# Zoom physics parameters
@export var zoom_base_damping = 0.98
@export var zoom_velocity_damping = 0.01
@export var zoom_acceleration_factor =1.2
@export var zoom_input_sensitivity = 0.5
@export var min_zoom_speed = 0.001
@export var zoom_chain_timeout = 0.3

# State variables
var camera_distance: float
var rotation_quaternion: Quaternion
var is_orbiting = false
var touch_start_position = Vector2.ZERO
var focus_target: Node3D = null
var current_orbit_point = Vector3.ZERO
var target_orbit_point = Vector3.ZERO
var transition_tween: Tween

# Physics state
var rotation_velocity = Vector2.ZERO
var last_input_time = 0.0
var input_chain_multiplier = 1.0
var last_input_direction = Vector2.ZERO

# Zoom physics state
var zoom_velocity = 0.0
var last_zoom_time = 0.0
var zoom_chain_multiplier = 1.0
var last_zoom_direction = 0.0

# Touch zoom variables
var touch_points = {}
var previous_touch_distance = 0.0
var is_zooming = false

var is_in_interior_mode = false
const INTERIOR_APPROACH_FACTOR = 2.0  # Controls difficulty of approaching nucleus
const MIN_APPROACH_SPEED = 0.001      # Minimum movement speed
const NUCLEUS_DISTANCE = 0.999          # Distance at which nucleus effects start
const INTERIOR_SCALE_FACTOR = 1.0  # Match this with your atom system's scale factor

# Add transition smoothing
var interior_mode_blend = 0.0
var mode_transition_tween: Tween

func _ready() -> void:
	camera_distance = initial_distance
	rotation_quaternion = Quaternion.IDENTITY
	# Initialize current_orbit_point if focus_target is set
	if focus_target:
		current_orbit_point = focus_target.global_position

func _process(delta: float) -> void:
	# Always apply rotation physics
	if rotation_velocity.length() > min_rotation_speed:
		var speed = rotation_velocity.length()
		var total_damping = base_damping - (velocity_dependent_damping * speed)
		total_damping = clamp(total_damping, 0.9, 0.999)
		rotation_velocity *= total_damping
		_apply_rotation(rotation_velocity * delta)
	
	# Always apply zoom physics
	if abs(zoom_velocity) > min_zoom_speed:
		var speed = abs(zoom_velocity)
		var total_damping = zoom_base_damping - (zoom_velocity_damping * speed)
		total_damping = clamp(total_damping, 0.9, 0.999)
		zoom_velocity *= total_damping
		_apply_zoom(zoom_velocity * delta)
	
	# Update target orbit point
	if focus_target:
		target_orbit_point = focus_target.global_position
		
		# Update current orbit point when not transitioning
		if not transition_tween or not transition_tween.is_running():
			current_orbit_point = target_orbit_point
	
	# Always calculate position based on current orbit point
	position = _calculate_camera_position(current_orbit_point)
	
	# Set the camera's transform
	transform = Transform3D(Basis(rotation_quaternion), position)

# enter/exit interior mode
func set_interior_mode(enabled: bool) -> void:
	is_in_interior_mode = enabled
	
	# Kill existing tween
	if mode_transition_tween and mode_transition_tween.is_valid():
		mode_transition_tween.kill()
	
	# For entry, we don't need a tween since we're setting values directly
	if enabled:
		interior_mode_blend = 1.0
		const ENTRY_SAFE_DISTANCE = 2.0
		camera_distance = ENTRY_SAFE_DISTANCE
		zoom_velocity = 0.0
		zoom_chain_multiplier = 1.0
		return
	
	# Only create and set up tween for exit
	mode_transition_tween = create_tween()
	mode_transition_tween.set_trans(Tween.TRANS_CUBIC)
	mode_transition_tween.set_ease(Tween.EASE_IN_OUT)
	mode_transition_tween.tween_property(self, "interior_mode_blend", 0.0, 0.5)

# Modify _handle_zoom_input to handle interior mode
func _handle_zoom_input(delta: float) -> void:
	var current_time = Time.get_ticks_msec() / 1000.0
	var time_since_last_zoom = current_time - last_zoom_time
	
	var zoom_direction = sign(delta)
	var direction_match = zoom_direction == sign(last_zoom_direction)
	
	if time_since_last_zoom > zoom_chain_timeout or not direction_match:
		zoom_chain_multiplier = 1.0
	else:
		zoom_chain_multiplier = min(zoom_chain_multiplier * zoom_acceleration_factor, 10.0)
	
	var zoom_force = delta * zoom_input_sensitivity * zoom_chain_multiplier
	
	if interior_mode_blend > 0.0:
		# Scale the distance check by the interior scale factor
		var scaled_distance = position.length() / INTERIOR_SCALE_FACTOR
		if scaled_distance < NUCLEUS_DISTANCE:
			var approach_factor = log(scaled_distance / NUCLEUS_DISTANCE + 1.0)
			zoom_force *= max(approach_factor, MIN_APPROACH_SPEED)
		
		# Blend between normal and interior behavior
		zoom_force = lerp(zoom_force, zoom_force * 0.5, interior_mode_blend)
	
	zoom_velocity += zoom_force
	
	last_zoom_time = current_time
	last_zoom_direction = zoom_direction

func _apply_zoom(zoom_delta: float) -> void:
	var zoom_factor = 1.0
	
	if interior_mode_blend > 0.0:
		var scaled_distance = position.length() / INTERIOR_SCALE_FACTOR
		if scaled_distance < NUCLEUS_DISTANCE:
			zoom_factor = pow(scaled_distance / NUCLEUS_DISTANCE, INTERIOR_APPROACH_FACTOR)
			zoom_factor = max(zoom_factor, MIN_APPROACH_SPEED)
			
			# Apply scaling factor to the actual movement
			zoom_factor *= lerp(1.0, 0.5, interior_mode_blend)
	elif camera_distance > 10.0:
		zoom_factor = pow(camera_distance, 0.3)
	
	var move_amount = zoom_delta * zoom_factor
	var new_distance = camera_distance + move_amount
	
	camera_distance = max(min_zoom, new_distance)

func _calculate_camera_position(orbit_point: Vector3) -> Vector3:
	var offset = rotation_quaternion * Vector3(0, 0, camera_distance)
	
	# Scale the position check when in interior mode
	if interior_mode_blend > 0.0:
		var scaled_offset = offset / INTERIOR_SCALE_FACTOR
		if scaled_offset.length() < NUCLEUS_DISTANCE:
			# Apply logarithmic scaling to prevent clipping
			var scale_factor = log(scaled_offset.length() / NUCLEUS_DISTANCE + 1.0)
			offset *= max(scale_factor, MIN_APPROACH_SPEED)
	
	return orbit_point + offset

func _handle_touch_zoom(distance_delta: float) -> void:
	var normalized_delta = distance_delta * touch_zoom_sensitivity * 5.0
	_handle_zoom_input(normalized_delta)

func _handle_rotation(delta: Vector2) -> void:
	var current_time = Time.get_ticks_msec() / 1000.0
	var time_since_last_input = current_time - last_input_time
	
	var input_direction = delta.normalized()
	var direction_alignment = input_direction.dot(last_input_direction) if last_input_direction.length() > 0 else 0.0
	
	if time_since_last_input > 0.5 or direction_alignment < 0.7:
		input_chain_multiplier = 1.0
	else:
		input_chain_multiplier = min(input_chain_multiplier * input_acceleration_factor, 5.0)
	
	var input_force = delta * input_sensitivity * input_chain_multiplier
	rotation_velocity += input_force
	
	last_input_time = current_time
	last_input_direction = input_direction

func _apply_rotation(rotation_amount: Vector2) -> void:
	var b = Basis(rotation_quaternion)
	var camera_right = b.x
	var camera_up = b.y
	
	var vertical_rotation = Quaternion(camera_right, -rotation_amount.y)
	var horizontal_rotation = Quaternion(camera_up, -rotation_amount.x)
	
	rotation_quaternion = horizontal_rotation * vertical_rotation * rotation_quaternion
	rotation_quaternion = rotation_quaternion.normalized()

func set_focus_target(new_target: Node3D) -> void:
	if focus_target == new_target:
		return
	
	focus_target = new_target
	target_orbit_point = new_target.global_position
	
	if transition_tween and transition_tween.is_valid():
		transition_tween.kill()
	
	var distance = position.distance_to(target_orbit_point)
	var travel_time = clamp(distance * 0.1, 0.2, 1.0) * 0.7
	
	transition_tween = create_tween()
	transition_tween.set_ease(Tween.EASE_OUT)
	transition_tween.set_trans(Tween.TRANS_CUBIC)
	transition_tween.chain().tween_property(self, "current_orbit_point", target_orbit_point, travel_time)

func _input(event: InputEvent) -> void:
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
	
	# Handle touch input
	if event is InputEventScreenTouch:
		if event.pressed:
			touch_points[event.index] = event.position
			if touch_points.size() == 1:
				is_orbiting = true
				is_zooming = false
				touch_start_position = event.position
		else:
			touch_points.erase(event.index)
			if touch_points.size() == 0:
				is_orbiting = false
				is_zooming = false
	elif event is InputEventScreenDrag:
		if touch_points.has(event.index):
			touch_points[event.index] = event.position
			if is_orbiting and touch_points.size() == 1:
				_handle_rotation(event.position - touch_start_position)
				touch_start_position = event.position
				get_viewport().set_input_as_handled()
			elif is_zooming and touch_points.size() == 2:
				var new_touch_distance = _get_touch_distance()
				var zoom_delta = previous_touch_distance - new_touch_distance
				_handle_touch_zoom(zoom_delta)
				previous_touch_distance = new_touch_distance
				get_viewport().set_input_as_handled()

func _unhandled_input(event: InputEvent) -> void:
	# Handle desktop zoom
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			_handle_zoom_input(-1)
			get_viewport().set_input_as_handled()
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			_handle_zoom_input(1)
			get_viewport().set_input_as_handled()
	
	# Handle touch zoom
	elif event is InputEventScreenTouch:
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
				var new_touch_distance = _get_touch_distance()
				var zoom_delta = previous_touch_distance - new_touch_distance
				_handle_touch_zoom(zoom_delta)
				previous_touch_distance = new_touch_distance
				get_viewport().set_input_as_handled()

func _get_touch_distance() -> float:
	if touch_points.size() < 2:
		return 0.0
	var points = touch_points.values()
	return points[0].distance_to(points[1])
