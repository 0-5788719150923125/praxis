extends GhostScene

## Motes - dust adrift in a shaft of light.
##
## The quietest scene: a soft beam of light cuts the frame and countless fine motes
## hang and turn in it on a slow curl-noise current, brightening as they cross the
## shaft. A warm, still, contemplative interlude. `bed` + a `dust` layer with its shaft
## enabled; a breath of `fog` for depth.

func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	var hue: float = [0.11, 0.09, 0.6][rng.randi() % 3]    # warm sun / amber / cool window
	add_layer("bed", rng, {"hue": fposmod(hue + 0.5, 1.0), "sat": 0.3, "val": 0.14, "pools": 2})
	add_layer("fog", rng, {"hue": hue, "sat": 0.1, "alpha": 0.03, "count": 4})
	add_layer("dust", rng, {
		"hue": hue,
		"count": rng.randi_range(120, 200),
		"shaft": true,
		"shaft_x": rng.randf_range(-0.35, 0.35),
		"speed": rng.randf_range(0.015, 0.03),
		"drift": rng.randf_range(0.004, 0.01),
	})
	return {"hue": hue}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.015, 0.02)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
