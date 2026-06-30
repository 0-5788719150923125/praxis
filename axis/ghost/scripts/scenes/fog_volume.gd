extends GhostScene

## Fog volume - REAL 3D fog: a low, wide bank of soft gaussian puffs receding into depth, lit
## volumetrically (a brighter sunward edge fading into a dim core) and slowly drifting. A genuine
## haze with simulated dynamics, not a flat 2D wash. `bed` + `volumetric` (fog mode).

func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	var hue: float = rng.randf_range(0.54, 0.66)
	add_layer("bed", rng, {"hue": hue, "sat": 0.30, "val": 0.10, "pools": 3})
	var p := {"mode": "fog"}
	if rng.randf() < 0.6:
		p["hue"] = hue                  # a cool tinted haze; otherwise neutral grey fog
	add_layer("volumetric", rng, p)
	return {"hue": hue}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.010, 0.015)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
