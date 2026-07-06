extends GhostScene

## Fire - a living flame attuned to the harmonics.
##
## One GPU temperature field (see [Layer.Fire] / shaders/flame.gdshader): heat sources
## along the bed, each listening to its own harmonic band (bass at the centre, treble
## at the rim), raise columns that rising turbulence carves into licks. CPU sparks
## crackle out of whichever region is roaring, and smoke sometimes hazes the top.
## Quiet passages sit as embers; powerful ones send columns up the frame.

func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	add_layer("bed", rng, {"hue": 0.04, "sat": 0.6, "val": 0.06, "pools": 2})
	add_layer("fire", rng, {
		"count": rng.randi_range(70, 120),                 # the SPARK pool (born from heat)
		"spread": rng.randf_range(0.9, 1.1),               # fans across the whole width
		"hue": rng.randf_range(0.02, 0.06),
	})
	# A little smoke drifting off the top, sometimes.
	if rng.randf() < 0.55:
		add_layer("fog", rng, {"hue": 0.06, "sat": 0.2, "alpha": 0.03, "count": rng.randi_range(4, 7)})
	return {}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.015, 0.02)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
