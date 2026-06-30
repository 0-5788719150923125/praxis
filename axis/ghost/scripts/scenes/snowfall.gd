extends GhostScene

## Snowfall - a quiet field of falling snow over a soft colour bed.
##
## The atmospheric weather scene (distinct from `snowflakes`, the hero crystal field):
## dozens of out-of-focus flakes drift down and gust sideways with the treble, a few
## near ones crisp into six-fold dendrites, all over a cool gradient that breathes with
## the music. By seed it sometimes rolls low fog along the ground. Pure component
## composition - `bed` + `snow` (+ `fog`) from the [Layer] registry, no bespoke code.

func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	# Cool winter palettes: dusk blue, slate, or a faint aurora green.
	var hue: float = [0.58, 0.62, 0.55, 0.45][rng.randi() % 4]
	add_layer("bed", rng, {"hue": hue, "sat": 0.45, "val": 0.26, "pools": 3})
	# Real volumetric fog rolling low under the snow - a lit, drifting bank with harmonic motion,
	# behind the falling flakes (added before the snow layer).
	if rng.randf() < 0.6:
		add_layer("volumetric", rng, {"mode": "fog", "hue": hue})
	add_layer("snow", rng, {
		"hue": fposmod(hue + 0.02, 1.0),
		"count": rng.randi_range(160, 260),       # many fine flecks
		"fall": rng.randf_range(0.08, 0.14),
		"crystal_frac": rng.randf_range(0.03, 0.08),   # only a rare large dendrite
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
