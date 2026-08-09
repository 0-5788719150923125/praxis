extends GhostScene

## Fireflies - a dusk meadow sparkling with wandering lights.
##
## Motes drift along a curl-noise breeze and blink on their own phase; a beat lights
## the subset whose threshold it crosses (the embers trick), so the field twinkles in
## ripples rather than in unison. The night comes from a [Scheme] and the lights from
## that scheme's accent, so the pair is harmonious whichever night is drawn - a warm
## amber over teal dusk, a cold bioluminescent green over indigo. `bed` + `fog` +
## `dust` + `fireflies`.

# HOW MANY, AND HOW BUSY. The field was always 30-55 bugs at one speed, so every
# dusk had the same density. A swarm is a whole habit: a slow crowded cloud low in
# the grass, a scattered handful darting about, a meadow's worth in between. Fewer
# lights afford the expensive real-light render (it reads far better on a handful),
# so `light` - the chance of rendering them as actual sources - tracks the count.
const SWARMS := [
	{"name": "cloud", "count": [80, 130], "speed": [0.03, 0.05], "jitter": [0.08, 0.14], "light": 0.2},
	{"name": "meadow", "count": [40, 70], "speed": [0.05, 0.09], "jitter": [0.14, 0.22], "light": 0.5},
	{"name": "drift", "count": [24, 44], "speed": [0.04, 0.07], "jitter": [0.10, 0.16], "light": 0.6},
	{"name": "sparse", "count": [8, 18], "speed": [0.07, 0.13], "jitter": [0.18, 0.28], "light": 0.85},
]


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	# A dusk is dark and cool - blue night, green twilight, a colourless overcast one -
	# and the bugs glow in that night's accent, so the two never clash.
	var sch := Scheme.among(["abyss", "violet", "teal", "verdant", "ash", "glacier", "magenta"], rng)
	var sw: Dictionary = SWARMS[rng.randi() % SWARMS.size()]
	var bug_hue := sch.accent
	add_layer("bed", rng, {"hue": sch.hue, "sat": sch.sat * 0.9,
		"val": sch.val * rng.randf_range(0.16, 0.26), "pools": rng.randi_range(2, 4)})
	add_layer("fog", rng, {"hue": sch.hue, "sat": sch.sat * 0.5,
		"alpha": rng.randf_range(0.02, 0.05), "count": rng.randi_range(4, 8)})
	if rng.randf() < 0.6:
		add_layer("dust", rng, {"hue": bug_hue, "count": rng.randi_range(40, 110), "shaft": false})
	add_layer("fireflies", rng, {
		"hue": bug_hue,
		"count": rng.randi_range(int(sw["count"][0]), int(sw["count"][1])),
		"speed": rng.randf_range(float(sw["speed"][0]), float(sw["speed"][1])),
		"jitter": rng.randf_range(float(sw["jitter"][0]), float(sw["jitter"][1])),
		"real_light": rng.randf() < float(sw["light"]),   # render as actual light sources
	})
	return {"hue": bug_hue, "mood": sch.name, "swarm": String(sw["name"])}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.02, 0.025)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
