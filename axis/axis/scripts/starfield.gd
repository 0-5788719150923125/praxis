extends Node3D

const TEXTURE_WIDTH = 2048
const TEXTURE_HEIGHT = 1024
const STAR_COUNT = 4000

const NORMAL_COLORS = [
	Color(1.0, 1.0, 1.0, 0.8),    # Dimmer white
	Color(0.95, 0.95, 1.0, 0.7),  # Very slight blue
	Color(1.0, 0.98, 0.95, 0.7),  # Very slight yellow
	Color(0.98, 0.95, 0.95, 0.6)  # Very slight red
]

var current_mode: bool = false  # false = normal, true = inverse
var world_env: WorldEnvironment
var base_environment: Environment

# Cache for pre-computed skies
var normal_sky: Sky
var inverse_sky: Sky

func _ready():
	world_env = get_node("../WorldEnvironment")
	if world_env and world_env.environment:
		base_environment = world_env.environment
		
		# Pre-compute both sky variants during initialization
		print("Pre-computing normal starfield...")
		normal_sky = _create_sky(false)
		
		print("Pre-computing inverse starfield...")
		inverse_sky = _create_sky(true)
		
		# Set initial sky
		_apply_sky(normal_sky)
		print("Starfield initialization complete")

func toggle_inverse_mode(is_inverse: bool) -> void:
	if current_mode != is_inverse:
		current_mode = is_inverse
		# Simply switch between pre-computed skies
		_apply_sky(inverse_sky if is_inverse else normal_sky)

func _apply_sky(sky: Sky) -> void:
	if world_env and world_env.environment:
		world_env.environment.background_mode = Environment.BG_SKY
		world_env.environment.sky = sky
		world_env.environment.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
		world_env.environment.ambient_light_sky_contribution = 0.1
		world_env.environment.background_energy_multiplier = 1.0

func _create_sky(inverse: bool = false) -> Sky:
	var image = Image.create(TEXTURE_WIDTH, TEXTURE_HEIGHT, false, Image.FORMAT_RGBA8)
	image.fill(Color(0, 0, 0, 1) if not inverse else Color(1, 1, 1, 1))
	
	var rng = RandomNumberGenerator.new()
	rng.randomize()
	
	for _i in range(STAR_COUNT):
		if inverse:
			var x = rng.randi() % TEXTURE_WIDTH
			var y = rng.randi() % TEXTURE_HEIGHT
			var size = rng.randf_range(0.5, 1.5)
			var color = Color(0, 0, 0, rng.randf_range(0.5, 0.8))
			draw_star(image, x, y, size, color, inverse)
		else:
			var phi = rng.randf() * TAU
			var theta = acos(2 * rng.randf() - 1)
			var x = int((phi / TAU) * TEXTURE_WIDTH) % TEXTURE_WIDTH
			var y = int((theta / PI) * TEXTURE_HEIGHT)
			var size = rng.randf_range(0.5, 1.2)
			size *= sin(theta)
			var color = NORMAL_COLORS[rng.randi() % NORMAL_COLORS.size()]
			color.a *= rng.randf_range(0.5, 1.0)
			draw_star(image, x, y, size, color, inverse)
	
	var texture = ImageTexture.create_from_image(image)
	
	var sky_material = PanoramaSkyMaterial.new()
	sky_material.panorama = texture
	
	var sky = Sky.new()
	sky.sky_material = sky_material
	
	return sky

func draw_star(image: Image, center_x: int, center_y: int, size: float, color: Color, inverse: bool):
	var radius = ceil(size)
	var start_y = max(0, center_y - radius)
	var end_y = min(TEXTURE_HEIGHT - 1, center_y + radius + 1)
	var start_x = max(0, center_x - radius)
	var end_x = min(TEXTURE_WIDTH - 1, center_x + radius + 1)
	
	for y in range(start_y, end_y):
		for x in range(start_x, end_x):
			var dist = Vector2(center_x, center_y).distance_to(Vector2(x, y))
			if dist <= size:
				if inverse:
					var alpha = (1.0 - (dist / size)) * color.a
					var pixel_color = Color(color.r, color.g, color.b, alpha)
					image.set_pixel(x, y, pixel_color)
				else:
					var alpha = pow(1.0 - (dist / size), 3)
					var final_color = Color(
						color.r,
						color.g,
						color.b,
						color.a * alpha * 0.8
					)
					var existing = image.get_pixel(x, y)
					image.set_pixel(x, y, existing.blend(final_color))
