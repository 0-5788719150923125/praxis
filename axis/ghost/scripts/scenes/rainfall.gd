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
	var heavy := rng.randf() < 0.5
	add_layer("bed", rng, {"hue": hue, "sat": 0.3, "val": 0.20, "pools": 2})
	add_layer("fog", rng, {"hue": hue, "alpha": 0.05, "count": rng.randi_range(5, 8)})
	add_layer("rain", rng, {
		"hue": fposmod(hue + 0.02, 1.0),
		"count": rng.randi_range(110, 200) if heavy else rng.randi_range(70, 120),
		"fall": rng.randf_range(1.0, 1.4),
		"slant": rng.randf_range(0.1, 0.3),
		"width": 1.5 if heavy else 1.2,
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
