extends GhostScene

## Rainfall - slanting rain over a brooding sky, fog rolling through it.
##
## Fast streaks fall at a wind-blown slant whose angle sways with the bass; a low colour
## bed and drifting fog give it weather and depth. Density and slant ride the audio, so a
## loud passage is a downpour. The seed picks the STORM first - hanging mist, drizzle,
## steady rain, downpour, or rain in a lit street at night - and the storm decides its own
## palette along with its density. `bed` + `fog` + `rain` + `veil` (+ `flare`).

# A storm is an intensity AND a light. Count, fall speed, slant and streak width belong
# together (a downpour is not a drizzle with more drops), and so does what the water
# catches: sunless mist is grey, a lit street is sodium and neon. One roll picks all of it.
#   streak - which end of the scheme the drops take: 0 the base water, 2 the accent light
const STORMS := {
	"mist": {
		"moods": ["glacier", "ash", "bone", "abyss", "teal", "violet", "dawn", "verdant"],
		"count": [55, 95], "fall": [0.5, 0.8], "slant": [0.05, 0.18], "width": [0.9, 1.1],
		"fog_a": [0.08, 0.12], "fog_n": [6, 10], "val": [0.16, 0.26], "sat": [0.4, 0.7],
		"veil": 0.50, "flare": 0.08, "streak": 0,
	},
	"drizzle": {
		"moods": ["ash", "glacier", "abyss", "bone", "teal", "violet", "verdant", "dawn"],
		"count": [70, 110], "fall": [0.8, 1.1], "slant": [0.08, 0.24], "width": [1.0, 1.2],
		"fog_a": [0.05, 0.075], "fog_n": [5, 9], "val": [0.14, 0.24], "sat": [0.35, 0.65],
		"veil": 0.42, "flare": 0.12, "streak": 0,
	},
	"steady": {
		"moods": ["abyss", "ash", "glacier", "teal", "violet", "bone", "verdant", "brass"],
		"count": [120, 180], "fall": [1.05, 1.4], "slant": [0.12, 0.32], "width": [1.2, 1.45],
		"fog_a": [0.04, 0.06], "fog_n": [5, 8], "val": [0.13, 0.22], "sat": [0.4, 0.75],
		"veil": 0.48, "flare": 0.15, "streak": 0,
	},
	"downpour": {
		"moods": ["abyss", "ash", "violet", "glacier", "teal", "brass", "bone", "ember"],
		"count": [200, 300], "fall": [1.35, 1.75], "slant": [0.2, 0.45], "width": [1.5, 1.8],
		"fog_a": [0.03, 0.05], "fog_n": [4, 7], "val": [0.10, 0.18], "sat": [0.45, 0.8],
		"veil": 0.62, "flare": 0.20, "streak": 0,
	},
	"lit": {
		# Rain in a lit street at night: the streaks take the LAMP's colour, not the sky's,
		# and there is a source in frame throwing a flare - the one warm, loud storm.
		"moods": ["sodium", "ember", "magenta", "brass", "violet", "toxic", "rose", "teal", "dawn"],
		"count": [150, 260], "fall": [1.1, 1.6], "slant": [0.15, 0.40], "width": [1.2, 1.8],
		"fog_a": [0.03, 0.06], "fog_n": [4, 8], "val": [0.10, 0.18], "sat": [0.7, 1.0],
		"veil": 0.40, "flare": 0.85, "streak": 2,
	},
}


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	var keys := STORMS.keys()
	var storm := String(keys[rng.randi() % keys.size()])
	var st: Dictionary = STORMS[storm]
	var sch := Scheme.among(st["moods"], rng)
	var count := rng.randi_range(int(st["count"][0]), int(st["count"][1]))
	var width := rng.randf_range(float(st["width"][0]), float(st["width"][1]))
	var val: Array = st["val"]
	var sat: Array = st["sat"]
	add_layer("bed", rng, {
		"hue": sch.hue,
		"sat": clampf(sch.sat * rng.randf_range(float(sat[0]), float(sat[1])), 0.03, 1.0),
		"val": rng.randf_range(float(val[0]), float(val[1])),
		"pools": rng.randi_range(1, 4),
	})
	add_layer("fog", rng, {
		"hue": sch.vary(rng),
		"sat": clampf(sch.sat * 0.35, 0.02, 0.5),
		"alpha": rng.randf_range(float(st["fog_a"][0]), float(st["fog_a"][1])),
		"count": rng.randi_range(int(st["fog_n"][0]), int(st["fog_n"][1])),
	})
	add_layer("rain", rng, {
		# Water has no colour of its own - it shows whatever lights it, which is why the
		# storm names the end of the scheme its streaks are drawn from.
		"hue": sch.hue_at(int(st["streak"]), 3),
		"sat": clampf(sch.sat * (0.7 if int(st["streak"]) > 0 else 0.3), 0.05, 0.8),
		"count": count, "fall": rng.randf_range(float(st["fall"][0]), float(st["fall"][1])),
		"slant": rng.randf_range(float(st["slant"][0]), float(st["slant"][1])), "width": width,
	})
	# Harmonic obscuring veil: drifting grey sheets of rain that thicken on loud passages and soften
	# the view, then clear - moving patterns of visibility riding the music (the snow-squall effect,
	# generalized). Heavier rain pushes a denser veil.
	add_layer("veil", rng, {
		"hue": sch.hue, "sat": clampf(sch.sat * 0.25, 0.02, 0.35), "val": 0.78,
		"floor": 0.10, "gain": 0.8,
		"max": clampf(float(st["veil"]) * (0.8 + 0.2 * width), 0.15, 0.7),
	})
	# The lamp itself, in front of the weather - what makes a lit storm read as lit.
	if rng.randf() < float(st["flare"]):
		add_layer("flare", rng, {"hue": sch.accent, "fisheye": rng.randf_range(0.1, 0.28)})
	return {"hue": sch.hue, "mood": sch.name, "storm": storm, "count": count, "width": width}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.02, 0.03)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
