extends GhostScene

## Petals - blossom, leaves or ash drifting down on a soft breeze.
##
## Flat petals tumble (in-plane spin + flutter, the flat-subject discipline) as they fall,
## riding a curl-noise breeze, with fine dust hanging in the light. The seed picks the FALL
## first - cherry blossom, autumn leaves, spring green, or ash coming down over a burnt
## ground - and that decides the palette, how many are in the air, how heavily they drop,
## whether a sunbeam reaches in, and whether a second tree is shedding alongside the first.
## `bed` + `dust` + `petals` (+ a second `petals`, + `veil`).

# What is falling is one choice, and everything else follows it: a leaf is heavier than a
# petal, ash is finer and denser than either, and only a lit day gets a shaft of sun.
#   mix - chance of a SECOND petal layer on the accent (two trees shedding at once)
#   bed - which end of the scheme the ground/sky wash takes: "accent" (sky) or "base"
const FALLS := {
	"blossom": {
		"moods": ["rose", "magenta", "dawn", "bone", "violet", "glacier", "teal", "verdant"],
		"count": [46, 84], "fall": [0.05, 0.085], "sat": [0.45, 0.65], "dust": [60, 110],
		"shaft": 0.70, "bed": "accent", "val": [0.26, 0.38], "mix": 0.55, "veil": 0.0,
	},
	"autumn": {
		# A forest in autumn IS orange - and it is also still half green, so verdant stays in.
		"moods": ["dawn", "ember", "sodium", "brass", "rose", "ash", "bone", "verdant"],
		"count": [28, 52], "fall": [0.09, 0.15], "sat": [0.55, 0.80], "dust": [40, 90],
		"shaft": 0.60, "bed": "base", "val": [0.22, 0.34], "mix": 0.70, "veil": 0.0,
	},
	"spring": {
		"moods": ["verdant", "toxic", "teal", "sodium", "dawn", "bone", "glacier", "brass"],
		"count": [40, 72], "fall": [0.06, 0.11], "sat": [0.40, 0.60], "dust": [50, 100],
		"shaft": 0.75, "bed": "accent", "val": [0.24, 0.36], "mix": 0.40, "veil": 0.0,
	},
	"ashfall": {
		# Not blossom at all: burnt flakes coming down thick over a dark ground, no sunbeam,
		# smoke veiling the view. Same tumbling flat subject, an entirely different scene.
		"moods": ["ash", "bone", "abyss", "ember", "sodium", "violet", "glacier", "dawn"],
		"count": [90, 150], "fall": [0.035, 0.065], "sat": [0.05, 0.22], "dust": [90, 160],
		"shaft": 0.15, "bed": "base", "val": [0.08, 0.16], "mix": 0.25, "veil": 0.80,
	},
}


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	var keys := FALLS.keys()
	var season := String(keys[rng.randi() % keys.size()])
	var fl: Dictionary = FALLS[season]
	var sch := Scheme.among(fl["moods"], rng)
	var val: Array = fl["val"]
	var bed_hue: float = sch.accent if String(fl["bed"]) == "accent" else sch.hue
	add_layer("bed", rng, {
		"hue": bed_hue,
		"sat": clampf(sch.sat * rng.randf_range(0.45, 0.8), 0.04, 1.0),
		"val": rng.randf_range(float(val[0]), float(val[1])),
		"pools": rng.randi_range(2, 4),
	})
	add_layer("dust", rng, {
		"hue": sch.vary(rng),
		"count": rng.randi_range(int(fl["dust"][0]), int(fl["dust"][1])),
		"shaft": rng.randf() < float(fl["shaft"]),
		"shaft_x": rng.randf_range(-0.5, 0.5),
		"drift": rng.randf_range(0.004, 0.014),
	})
	var count := rng.randi_range(int(fl["count"][0]), int(fl["count"][1]))
	var sat: Array = fl["sat"]
	add_layer("petals", rng, {
		"hue": sch.vary(rng, 0.6),
		"sat": rng.randf_range(float(sat[0]), float(sat[1])),
		"count": count,
		"fall": rng.randf_range(float(fl["fall"][0]), float(fl["fall"][1])),
	})
	# A second shedding on the accent hue: two trees over one another, and the air visibly
	# fuller for it - the difference is in the count, not only the colour.
	if rng.randf() < float(fl["mix"]):
		var extra := rng.randi_range(int(count * 0.3), int(count * 0.7))
		add_layer("petals", rng, {
			"hue": fposmod(sch.accent + rng.randf_range(-0.03, 0.03), 1.0),
			"sat": rng.randf_range(float(sat[0]), float(sat[1])) * 0.85,
			"count": extra,
			# Lighter, later-falling second shedding, so the two layers do not move as one.
			"fall": rng.randf_range(float(fl["fall"][0]), float(fl["fall"][1])) * 0.75,
		})
		count += extra
	# Smoke rolling through the ashfall, thickening on loud passages.
	if rng.randf() < float(fl["veil"]):
		add_layer("veil", rng, {
			"hue": sch.vary(rng), "sat": clampf(sch.sat * 0.3, 0.02, 0.4), "val": 0.55,
			"floor": 0.10, "gain": 0.75, "max": rng.randf_range(0.25, 0.45),
		})
	return {"hue": sch.hue, "mood": sch.name, "season": season, "petals": count}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.02, 0.025)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
