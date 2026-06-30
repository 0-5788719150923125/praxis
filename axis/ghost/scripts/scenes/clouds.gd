extends GhostScene

## Clouds - REAL 3D cloud masses drifting across the sky, lit by the sun.
##
## A coloured sky bed (cool day or warm dusk), sometimes a few stars behind, and a [Volumetric]
## cloudscape: soft gaussian puffs placed in 3D, self-shadowed (bright sunlit tops, darker
## undersides) and drifting/billowing over time. `bed` + `stars?` + `volumetric`.

func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	var dusk := rng.randf() < 0.4
	var hue: float = rng.randf_range(0.04, 0.10) if dusk else [0.58, 0.60, 0.62][rng.randi() % 3]
	add_layer("bed", rng, {"hue": hue, "sat": 0.45 if dusk else 0.25, "val": 0.12, "pools": 3})
	if rng.randf() < 0.5:
		add_layer("stars", rng, {"z": "back", "count": rng.randi_range(60, 110), "hue": 0.6})
	# Real 3D clouds. At dusk they take a warm palette tint; by day they stay sunlit-white (no tint).
	var p := {"mode": "cloud"}
	if dusk:
		p["hue"] = fposmod(hue + 0.02, 1.0)
	add_layer("volumetric", rng, p)
	return {"hue": hue}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.012, 0.02)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
