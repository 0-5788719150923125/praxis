extends GhostScene

## Underwater - looking up through flowing water: shafts of light from the surface, bubbles
## rising, a deep blue-green wash. The submerged corner of the weather catalogue.
##
## A blue-green bed, swaying god-[Rays] from the surface above, and [Bubbles] drifting up
## through them. The light shafts brighten with the music; the whole frame drifts gently.

# WATERS. The hue was pinned to 0.50-0.58 and the layer mix never changed, so
# every dive was the same blue-green room with the same tall kelp. A water is a
# whole place: its colour, how much light reaches it, what grows there, and how
# much life is suspended in it.
#
# `depth` is the organising idea - a bright shallow reef gets many rays, small
# stiff plants and busy bubbles; an abyss gets almost no light, sparse tall
# whips and bioluminescence instead. `silt` adds suspended matter, which is what
# makes an estuary read as murky rather than merely darker.
const WATERS := [
	{"name": "reef",     "hue": [0.47, 0.52], "sat": 0.52, "val": 0.22,
	 "rays": [6, 9], "form": "fan",   "fronds": [16, 26], "glows": [1, 3],
	 "bubbles": [70, 120], "silt": 0.0},
	{"name": "kelp",     "hue": [0.30, 0.38], "sat": 0.50, "val": 0.15,
	 "rays": [4, 7], "form": "kelp",  "fronds": [14, 22], "glows": [2, 4],
	 "bubbles": [40, 80], "silt": 0.10},
	{"name": "meadow",   "hue": [0.38, 0.45], "sat": 0.46, "val": 0.18,
	 "rays": [5, 8], "form": "grass", "fronds": [30, 48], "glows": [1, 3],
	 "bubbles": [50, 90], "silt": 0.06},
	{"name": "abyss",    "hue": [0.60, 0.68], "sat": 0.62, "val": 0.07,
	 "rays": [1, 2], "form": "whip",  "fronds": [6, 12],  "glows": [5, 8],
	 "bubbles": [15, 35], "silt": 0.0},
	{"name": "estuary",  "hue": [0.13, 0.20], "sat": 0.38, "val": 0.13,
	 "rays": [2, 4], "form": "bulb",  "fronds": [10, 18], "glows": [1, 2],
	 "bubbles": [25, 55], "silt": 0.30},
	{"name": "twilight", "hue": [0.72, 0.80], "sat": 0.48, "val": 0.11,
	 "rays": [3, 5], "form": "whip",  "fronds": [8, 16],  "glows": [3, 6],
	 "bubbles": [30, 60], "silt": 0.08},
]


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	var w: Dictionary = WATERS[rng.randi() % WATERS.size()]
	var hr: Array = w["hue"]
	var hue: float = rng.randf_range(float(hr[0]), float(hr[1]))
	add_layer("bed", rng, {"hue": hue, "sat": float(w["sat"]), "val": float(w["val"]), "pools": 3})
	# The lit surface far above (fills the dead space at the top with a real light source + caustics),
	# then the god-rays through it, a floor of whatever grows in this water, and bubbles rising
	# through it all. Deep waters get few rays and many glows; shallow ones the reverse.
	add_layer("surface", rng, {"hue": fposmod(hue + 0.02, 1.0), "caustics": rng.randi_range(7, 11),
		"sun_x": rng.randf_range(-0.4, 0.4)})
	add_layer("rays", rng, {"hue": fposmod(hue + 0.02, 1.0),
		"count": rng.randi_range(int(w["rays"][0]), int(w["rays"][1]))})
	add_layer("kelp", rng, {"hue": fposmod(hue - 0.04, 1.0), "form": String(w["form"]),
		"fronds": rng.randi_range(int(w["fronds"][0]), int(w["fronds"][1])),
		"glows": rng.randi_range(int(w["glows"][0]), int(w["glows"][1]))})
	add_layer("bubbles", rng, {"hue": fposmod(hue + 0.05, 1.0),
		"count": rng.randi_range(int(w["bubbles"][0]), int(w["bubbles"][1]))})
	# Suspended matter: what separates murky water from merely dark water.
	if float(w["silt"]) > 0.01:
		add_layer("dust", rng, {"hue": fposmod(hue + 0.03, 1.0),
			"count": int(120.0 * float(w["silt"])), "drift": 0.02})
	return {"hue": hue, "water": String(w["name"])}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.015, 0.025)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
