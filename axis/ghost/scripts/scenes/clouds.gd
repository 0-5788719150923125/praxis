extends GhostScene

## Clouds - REAL 3D cloud masses drifting across the sky, lit by the sun.
##
## A coloured sky bed, sometimes stars behind, sometimes haze rolling in front, and a
## [Volumetric] cloudscape: soft gaussian puffs placed in 3D, self-shadowed (bright sunlit
## tops, darker undersides) and drifting/billowing over time. The seed picks a WEATHER
## first and the weather decides both which moods the sky may be drawn from and what else
## is in the frame. `bed` + `stars?` + `volumetric` + `haze?`.

# A sky is a weather before it is a colour - what accompanies the clouds (starlight behind,
# haze in front, how heavily the bed sits) follows from the same roll, so one choice moves
# the whole read of the scene rather than just its tint.
#   stars/haze/tint - probabilities; pools - the bed's colour-pool count range
const SKIES := {
	"clear": {
		"moods": ["glacier", "abyss", "teal", "violet", "bone"],
		"stars": 0.70, "haze": 0.15, "tint": 0.30, "pools": [2, 4],
	},
	"dusk": {
		"moods": ["dawn", "ember", "sodium", "brass", "rose", "magenta"],
		"stars": 0.45, "haze": 0.40, "tint": 1.00, "pools": [3, 5],
	},
	"overcast": {
		"moods": ["ash", "bone", "glacier", "abyss"],
		"stars": 0.05, "haze": 0.85, "tint": 0.90, "pools": [2, 6],
	},
}


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	var keys := SKIES.keys()
	var weather := String(keys[rng.randi() % keys.size()])
	var sky: Dictionary = SKIES[weather]
	var sch := Scheme.among(sky["moods"], rng)
	var pools: Array = sky["pools"]
	add_layer("bed", rng, {
		"hue": sch.hue,
		"sat": sch.sat * rng.randf_range(0.5, 0.95),
		"val": rng.randf_range(0.08, 0.17),
		"pools": rng.randi_range(int(pools[0]), int(pools[1])),
	})
	if rng.randf() < float(sky["stars"]):
		# Starlight on the accent, so a violet sky gets its own kind of stars.
		add_layer("stars", rng, {"z": "back", "count": rng.randi_range(40, 180), "hue": sch.accent})
	# Real 3D clouds. Most weathers tint them to the sky's mood; a clear one usually leaves
	# them sunlit-white, which is what makes a blue day read as a blue day.
	var p := {"mode": "cloud"}
	if rng.randf() < float(sky["tint"]):
		p["hue"] = sch.vary(rng)
	add_layer("volumetric", rng, p)
	if rng.randf() < float(sky["haze"]):
		# Haze drawn last - weather in FRONT of the masses, softening them into the sky.
		add_layer("fog", rng, {
			"hue": sch.vary(rng),
			"sat": sch.sat * rng.randf_range(0.2, 0.5),
			"alpha": rng.randf_range(0.02, 0.05),
			"count": rng.randi_range(3, 7),
		})
	return {"hue": sch.hue, "mood": sch.name, "weather": weather}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.012, 0.02)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
