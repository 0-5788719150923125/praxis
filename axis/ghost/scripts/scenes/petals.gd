extends GhostScene

## Petals - blossom or leaves drifting down on a soft breeze.
##
## Flat petals tumble (in-plane spin + flutter, the flat-subject discipline) as they
## fall, riding a curl-noise breeze, with fine dust hanging in the warm light. Cherry
## pink, autumn amber, or green leaves by seed, over a gentle bed. `bed` + `dust` +
## `petals`.

func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	# season: cherry blossom / autumn / spring green
	var season := rng.randi() % 3
	var petal_hue: float = [0.95, 0.07, 0.28][season]
	var bed_hue: float = [0.5, 0.08, 0.35][season]
	add_layer("bed", rng, {"hue": bed_hue, "sat": 0.35, "val": 0.30, "pools": 3})
	add_layer("dust", rng, {"hue": petal_hue, "count": 80, "shaft": false, "drift": 0.01})
	add_layer("petals", rng, {
		"hue": petal_hue,
		"sat": rng.randf_range(0.45, 0.6),
		"count": rng.randi_range(36, 60),
		"fall": rng.randf_range(0.06, 0.10),
	})
	return {"hue": petal_hue}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.02, 0.025)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
