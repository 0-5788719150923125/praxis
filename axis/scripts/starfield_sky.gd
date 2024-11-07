extends Node3D

const TEXTURE_WIDTH = 2048  # Increased resolution
const TEXTURE_HEIGHT = 1024  # Maintain 2:1 ratio for panoramic
const STAR_COUNT = 4000  # More stars
const STAR_SIZE_RANGES = [
	Vector2(0.5, 0.8),  # Tiny stars (most common)
	Vector2(0.8, 1.2),  # Small stars
	Vector2(1.2, 1.5)   # Medium stars (rare)
]
const STAR_SIZE_WEIGHTS = [  # Probability weights for each size range
	0.7,  # 70% tiny stars
	0.25, # 25% small stars
	0.05  # 5% medium stars
]
const STAR_COLORS = [
	Color(1.0, 1.0, 1.0, 0.8),    # Dimmer white
	Color(0.95, 0.95, 1.0, 0.7),  # Very slight blue
	Color(1.0, 0.98, 0.95, 0.7),  # Very slight yellow
	Color(0.98, 0.95, 0.95, 0.6)  # Very slight red
]

func _ready():
	call_deferred("create_starfield_sky")

func create_starfield_sky():
	var image = Image.create(TEXTURE_WIDTH, TEXTURE_HEIGHT, false, Image.FORMAT_RGBA8)
	image.fill(Color(0, 0, 0, 1))
	
	var rng = RandomNumberGenerator.new()
	rng.randomize()
	
	# Generate stars with position correction for panoramic projection
	for _i in range(STAR_COUNT):
		# Select star size range based on weights
		var size_range_idx = _weighted_random_index(STAR_SIZE_WEIGHTS, rng)
		var size_range = STAR_SIZE_RANGES[size_range_idx]
		
		# Get random position with panoramic correction
		var phi = rng.randf() * TAU  # Longitude (0 to 2π)
		var theta = acos(2 * rng.randf() - 1)  # Latitude (0 to π)
		
		# Convert spherical coordinates to panoramic UV
		var x = int((phi / TAU) * TEXTURE_WIDTH) % TEXTURE_WIDTH
		var y = int((theta / PI) * TEXTURE_HEIGHT)
		
		# Adjust size based on latitude to prevent stretching
		var size = rng.randf_range(size_range.x, size_range.y)
		size *= sin(theta)  # Reduce size near poles
		
		var color = STAR_COLORS[rng.randi() % STAR_COLORS.size()]
		
		# Add some variance to star brightness
		color.a *= rng.randf_range(0.5, 1.0)
		
		draw_star(image, x, y, size, color)
	
	var texture = ImageTexture.create_from_image(image)
	
	var sky_material = PanoramaSkyMaterial.new()
	sky_material.panorama = texture
	
	var sky = Sky.new()
	sky.sky_material = sky_material
	
	var world_env = get_node("../WorldEnvironment")
	if world_env and world_env.environment:
		world_env.environment.background_mode = Environment.BG_SKY
		world_env.environment.sky = sky
		world_env.environment.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
		world_env.environment.ambient_light_sky_contribution = 0.1  # Reduced ambient contribution
		world_env.environment.background_energy_multiplier = 1.0  # Normalized brightness

func draw_star(image: Image, center_x: int, center_y: int, size: float, color: Color):
	var radius = ceil(size)
	var start_x = max(0, center_x - radius)
	var end_x = min(TEXTURE_WIDTH - 1, center_x + radius)
	var start_y = max(0, center_y - radius)
	var end_y = min(TEXTURE_HEIGHT - 1, center_y + radius)
	
	for y in range(start_y, end_y + 1):
		for x in range(start_x, end_x + 1):
			var distance = Vector2(center_x, center_y).distance_to(Vector2(x, y))
			if distance <= size:
				var alpha = pow(1.0 - (distance / size), 3)  # Stronger falloff for sharper stars
				var pixel_color = Color(
					color.r,
					color.g,
					color.b,
					color.a * alpha * 0.8  # Overall dimmer stars
				)
				
				# Blend with existing pixel for better star overlap
				var existing = image.get_pixel(x, y)
				image.set_pixel(x, y, existing.blend(pixel_color))

func _weighted_random_index(weights: Array, rng: RandomNumberGenerator) -> int:
	var total_weight = 0.0
	for weight in weights:
		total_weight += weight
	
	var random_value = rng.randf() * total_weight
	var current_weight = 0.0
	
	for i in range(weights.size()):
		current_weight += weights[i]
		if random_value <= current_weight:
			return i
	
	return weights.size() - 1
