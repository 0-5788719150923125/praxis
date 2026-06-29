extends GhostScene

## Fog bank - rolling coloured fog, light glowing from within.
##
## The explicit "rolling fog with colours underneath / inside" idea: a rich colour bed
## with slow pools breathing on the spectrum, then several fog layers of different tint
## and speed rolling over and through it, so the colour bleeds up through the cloud. The
## bank lurches on the beat and coasts down (velocity + decay, not a uniform drift).
## `bed` + stacked `fog` layers - atmosphere from pure composition.

func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	var hue: float = rng.randf()
	var warm := rng.randf() < 0.5
	# A bright, saturated bed so the colour reads strongly *through* the fog above it.
	add_layer("bed", rng, {"hue": hue, "sat": rng.randf_range(0.6, 0.85),
		"val": rng.randf_range(0.42, 0.58), "pools": rng.randi_range(4, 6)})
	# One or two fog sheets, each a different tint and drift, for depth. Kept light so
	# the bank veils the colour rather than burying it in white.
	var sheets := rng.randi_range(1, 2)
	for i in sheets:
		var tint := fposmod(hue + (0.04 if warm else -0.4) + 0.06 * i, 1.0)
		add_layer("fog", rng, {
			"hue": tint,
			"sat": rng.randf_range(0.12, 0.3),
			"alpha": rng.randf_range(0.03, 0.05),
			"count": rng.randi_range(5, 7),
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
