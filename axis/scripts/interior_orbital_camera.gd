extends "res://scripts/orbital_camera.gd"

# Thresholds for different zoom behaviors
const NUCLEUS_VISIBLE_DISTANCE := 5.0   # When nucleus becomes clearly visible
const APPROACH_START_DISTANCE := 2.0    # When strong dampening starts
const MINIMUM_APPROACH_SPEED := 0.001   # Prevents complete stopping

# Dampening factors
const NORMAL_DAMPENING := 0.8          # Regular zoom slowdown
const CLOSE_DAMPENING := 3.0          # Extra slowdown when very close
const RETREAT_FACTOR := 2.0           # Makes zooming out easier

func _handle_zoom_input(delta: float) -> void:
	var current_time = Time.get_ticks_msec() / 1000.0
	var time_since_last_zoom = current_time - last_zoom_time
	
	var zoom_direction = sign(delta)
	var direction_match = zoom_direction == sign(last_zoom_direction)
	
	# Reset or increase chain multiplier
	if time_since_last_zoom > zoom_chain_timeout or not direction_match:
		zoom_chain_multiplier = 1.0
	else:
		zoom_chain_multiplier = min(zoom_chain_multiplier * zoom_acceleration_factor, 5.0)
	
	# Calculate distance to nucleus (center)
	var distance_to_center = position.length()
	
	# Calculate zoom force based on direction and distance
	var zoom_force = delta * zoom_input_sensitivity * zoom_chain_multiplier
	
	# Handle zooming in (approaching nucleus)
	if zoom_direction < 0:  # Zooming in
		if distance_to_center < APPROACH_START_DISTANCE:
			# Apply strong exponential dampening when close to nucleus
			var approach_factor = pow(distance_to_center / APPROACH_START_DISTANCE, CLOSE_DAMPENING)
			zoom_force *= max(approach_factor, MINIMUM_APPROACH_SPEED)
		else:
			# Apply mild dampening when further away
			zoom_force *= NORMAL_DAMPENING
	else:  # Zooming out
		# Make retreat easier by reducing dampening
		zoom_force *= RETREAT_FACTOR
	
	zoom_velocity += zoom_force
	
	last_zoom_time = current_time
	last_zoom_direction = zoom_direction

func _apply_zoom(zoom_delta: float) -> void:
	var distance_to_center = position.length()
	var scaled_delta = zoom_delta
	
	# Apply progressive scaling only when approaching nucleus
	if zoom_delta < 0 and distance_to_center < NUCLEUS_VISIBLE_DISTANCE:
		var approach_factor = 1.0
		
		# Progressive difficulty only when getting very close
		if distance_to_center < APPROACH_START_DISTANCE:
			approach_factor = pow(distance_to_center / APPROACH_START_DISTANCE, 2.0)
			approach_factor = max(approach_factor, MINIMUM_APPROACH_SPEED)
		
		scaled_delta *= approach_factor
	
	var new_distance = camera_distance + scaled_delta
	
	# Ensure we don't go below minimum zoom
	camera_distance = max(min_zoom, new_distance)
	_update_camera_position()
