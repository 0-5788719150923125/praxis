extends GhostScene

## Bubbles - an underwater drift of rising bubbles in coloured depths.
##
## Bubbles wobble upward with rim highlights, fine suspended particles hang in the
## water, and a deep cool bed with slow colour pools gives the sense of light filtering
## down from above. Teal, deep blue, or green by seed. `bed` + `dust` + `bubbles`.

func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	var hue: float = [0.5, 0.55, 0.45][rng.randi() % 3]    # teal / blue / green water
	add_layer("bed", rng, {"hue": hue, "sat": 0.6, "val": 0.24, "pools": 3})
	add_layer("dust", rng, {"hue": hue, "count": 90, "shaft": true, "shaft_x": rng.randf_range(-0.3, 0.3),
		"drift": 0.004})
	add_layer("bubbles", rng, {
		"hue": hue,
		"count": rng.randi_range(36, 60),
		"rise": rng.randf_range(0.07, 0.12),
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
