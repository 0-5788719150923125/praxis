extends GhostScene

## Underwater - looking up through flowing water: shafts of light from the surface, bubbles
## rising, a deep blue-green wash. The submerged corner of the weather catalogue.
##
## A blue-green bed, swaying god-[Rays] from the surface above, and [Bubbles] drifting up
## through them. The light shafts brighten with the music; the whole frame drifts gently.

func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	var hue: float = rng.randf_range(0.50, 0.58)        # blue through blue-green
	add_layer("bed", rng, {"hue": hue, "sat": 0.55, "val": 0.15, "pools": 3})
	# The lit surface far above (fills the dead space at the top with a real light source + caustics),
	# then the god-rays through it, a kelp floor below, and bubbles rising through it all.
	add_layer("surface", rng, {"hue": fposmod(hue + 0.02, 1.0), "caustics": rng.randi_range(7, 11),
		"sun_x": rng.randf_range(-0.4, 0.4)})
	add_layer("rays", rng, {"hue": fposmod(hue + 0.02, 1.0), "count": rng.randi_range(4, 7)})
	add_layer("kelp", rng, {"hue": fposmod(hue - 0.04, 1.0), "fronds": rng.randi_range(12, 20),
		"glows": rng.randi_range(3, 5)})
	add_layer("bubbles", rng, {"count": rng.randi_range(40, 80), "hue": fposmod(hue + 0.05, 1.0)})
	return {"hue": hue}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.015, 0.025)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
