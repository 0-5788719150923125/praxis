extends GhostScene

## Rainfall - slanting rain over a brooding sky, fog rolling through it.
##
## Fast streaks fall at a wind-blown slant whose angle sways with the bass; a low,
## desaturated colour bed and drifting fog give it weather and depth. Density and slant
## ride the audio, so a loud passage is a downpour. `bed` + `fog` + `rain` composed.

func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	var hue: float = [0.6, 0.58, 0.55][rng.randi() % 3]
	# A sampled INTENSITY - a fine mist, a drizzle, steady rain, or a downpour - so the weather
	# varies run to run instead of being one rain.
	var count: int
	var fall: float
	var slant: float
	var width: float
	var fog_a: float
	var k := rng.randf()
	if k < 0.18:                                    # mist - very fine, slow, hazy
		count = rng.randi_range(55, 95); fall = rng.randf_range(0.5, 0.8)
		slant = rng.randf_range(0.05, 0.18); width = 1.0; fog_a = 0.10
	elif k < 0.45:                                  # drizzle - light
		count = rng.randi_range(70, 110); fall = rng.randf_range(0.8, 1.1)
		slant = rng.randf_range(0.08, 0.24); width = 1.1; fog_a = 0.06
	elif k < 0.78:                                  # steady rain
		count = rng.randi_range(120, 180); fall = rng.randf_range(1.05, 1.4)
		slant = rng.randf_range(0.12, 0.32); width = 1.3; fog_a = 0.05
	else:                                           # downpour - dense, fast, steep
		count = rng.randi_range(200, 300); fall = rng.randf_range(1.35, 1.75)
		slant = rng.randf_range(0.2, 0.45); width = 1.6; fog_a = 0.04
	add_layer("bed", rng, {"hue": hue, "sat": 0.3, "val": 0.20, "pools": 2})
	add_layer("fog", rng, {"hue": hue, "alpha": fog_a, "count": rng.randi_range(5, 9)})
	add_layer("rain", rng, {
		"hue": fposmod(hue + 0.02, 1.0),
		"count": count, "fall": fall, "slant": slant, "width": width,
	})
	# Harmonic obscuring veil: drifting grey sheets of rain that thicken on loud passages and soften
	# the view, then clear - moving patterns of visibility riding the music (the snow-squall effect,
	# generalized). Heavier rain pushes a denser veil.
	add_layer("veil", rng, {
		"hue": hue, "sat": 0.18, "val": 0.78,
		"floor": 0.10, "gain": 0.8, "max": clampf(0.30 + 0.5 * width, 0.3, 0.62),
	})
	return {"hue": hue}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.02, 0.03)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
