extends GhostScene

## Aurora - slow curtains of light over a starlit night.
##
## Wavy ribbons drift and undulate, each tied to a band of the spectrum so the curtains
## brighten and ripple with the music, hung over a faint starfield and a dark bed. The
## northern-lights corner. `bed` + `stars` + `aurora`, themed green / cyan / violet.

func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	# Classic aurora palettes.
	var hue: float = [0.38, 0.45, 0.78][rng.randi() % 3]    # green, cyan, violet
	add_layer("bed", rng, {"hue": fposmod(hue + 0.5, 1.0), "sat": 0.5, "val": 0.08, "pools": 2})
	add_layer("stars", rng, {"hue": 0.6, "count": rng.randi_range(90, 150)})
	add_layer("aurora", rng, {
		"hue": hue,
		"sat": rng.randf_range(0.5, 0.7),
		"count": rng.randi_range(3, 5),
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
