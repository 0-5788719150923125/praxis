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
	# Near-black bed, only a faint gradient (no big soft "spotlight" pools - they looked like fake
	# lights stranded behind the planet; the bright glows belong out front as a lens flare instead).
	add_layer("bed", rng, {"hue": nebula, "sat": 0.55, "val": 0.08, "pools": 0})
	add_layer("fog", rng, {"hue": nebula, "sat": 0.3, "alpha": 0.025, "count": 4})
	# Diffuse depth BEHIND the stars: a distant nebula or galaxy, and drifting harmonic bands of
	# fog (volumetric, fog mode) - the celestial dust catching light and moving with the music.
	if rng.randf() < 0.55:
		add_layer("cosmos", rng, {"z": "back", "hue": nebula, "count": rng.randi_range(1, 2)})
	if rng.randf() < 0.5:
		add_layer("volumetric", rng, {"z": "back", "mode": "fog", "hue": fposmod(nebula + 0.5, 1.0)})
	add_layer("stars", rng, {
		"hue": fposmod(nebula + 0.5, 1.0),
		"count": rng.randi_range(140, 240),
	})
	# A real 3D planet IN FRONT of the stars (added last, so it OCCLUDES the stars behind it -
	# a solid body should block them, not show them through).
	if rng.randf() < 0.5:
		add_layer("planet", rng, {"hue": fposmod(nebula + rng.randf_range(-0.2, 0.2), 1.0)})
	# A lens flare OVER everything (incl. the planet): the bright glows the user wanted up front,
	# pulsing with the harmonics, gently fisheye-bowed. Often, not always.
	if rng.randf() < 0.62:
		add_layer("flare", rng, {"hue": fposmod(nebula + rng.randf_range(-0.15, 0.15), 1.0),
			"fisheye": rng.randf_range(0.12, 0.3)})
	return {"hue": nebula}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.015, 0.02)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
