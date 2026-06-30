extends GhostScene

## Starfield - a deep night sky, twinkling, with the occasional shooting star.
##
## A parallax field of stars over a near-black nebula bed and a wisp of coloured fog;
## brighter stars glow, and every so often a meteor streaks across (more often on a
## beat). Calm and vast - the celestial corner of the catalogue. `bed` + `fog` + `stars`.

func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	var nebula: float = rng.randf()                      # the faint colour behind the stars
	add_layer("bed", rng, {"hue": nebula, "sat": 0.55, "val": 0.10, "pools": 3})
	add_layer("fog", rng, {"hue": nebula, "sat": 0.3, "alpha": 0.025, "count": 4})
	# Often, give the void DEPTH: a distant planet, nebula, or galaxy behind the stars, so the
	# field is not just dots. Drawn before the stars (add order) so they sit in front of it.
	if rng.randf() < 0.6:
		add_layer("cosmos", rng, {"z": "back", "hue": nebula, "count": rng.randi_range(1, 2)})
	add_layer("stars", rng, {
		"hue": fposmod(nebula + 0.5, 1.0),
		"count": rng.randi_range(140, 240),
	})
	return {"hue": nebula}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.015, 0.02)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
