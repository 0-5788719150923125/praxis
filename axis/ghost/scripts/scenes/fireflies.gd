extends GhostScene

## Fireflies - a dusk meadow sparkling with wandering lights.
##
## Warm motes drift along a curl-noise breeze and blink on their own phase; a beat
## lights the subset whose threshold it crosses (the embers trick), so the field
## twinkles in ripples rather than in unison. A deep dusk bed and a breath of low fog
## set the mood; faint dust hangs in the air. `bed` + `fog` + `dust` + `fireflies`.

func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	# Deep dusk: blue night, or a greener twilight.
	var bed_hue: float = [0.62, 0.66, 0.5][rng.randi() % 3]
	var bug_hue: float = [0.16, 0.13, 0.28][rng.randi() % 3]   # warm yellow / amber / green
	add_layer("bed", rng, {"hue": bed_hue, "sat": 0.5, "val": 0.16, "pools": 2})
	add_layer("fog", rng, {"hue": bed_hue, "alpha": 0.03, "count": 5})
	if rng.randf() < 0.6:
		add_layer("dust", rng, {"hue": bug_hue, "count": 70, "shaft": false})
	add_layer("fireflies", rng, {
		"hue": bug_hue,
		"count": rng.randi_range(30, 55),
		"speed": rng.randf_range(0.05, 0.09),
	})
	return {"hue": bug_hue}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.02, 0.025)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
