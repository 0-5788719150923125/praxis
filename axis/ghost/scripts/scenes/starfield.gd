extends GhostScene

## Starfield - a deep night sky, twinkling, with the occasional shooting star.
##
## A parallax field of stars over a near-black nebula bed and a wisp of coloured fog;
## brighter stars glow, and every so often a meteor streaks across (more often on a
## beat). Calm and vast - the celestial corner of the catalogue. The seed picks WHERE
## in the galaxy the sky is looked at from first, and that decides the crowding, the
## dust and the company as one choice. `bed` + `fog` + `stars` (+ `cosmos` / `volumetric`
## / `planet` / `flare`).

# A sky is a PLACE before it is a colour. A crowded dusty core and an empty deep field
# differ in star count, in how much lit dust hangs between them and in what else shares
# the frame - so one roll moves all of it together, and the mood set follows from the
# same place rather than being sampled loose.
#   count - stars; val - the bed's brightness; fog - haze alpha; the rest are chances of company
const SKIES := {
	"core": {
		# Toward the galactic centre: thousands of stars reddened by the dust in front of them.
		"moods": ["sodium", "dawn", "brass", "bone", "ember", "magenta", "violet", "rose"],
		"count": [420, 700], "val": [0.10, 0.17], "fog": [0.035, 0.06], "fog_n": [5, 9],
		"cosmos": 0.85, "cosmos_n": [1, 3], "volumetric": 0.65, "planet": 0.15, "flare": 0.80,
	},
	"deep": {
		# A deep field between the arms: almost nothing, very far away, very cold.
		"moods": ["abyss", "violet", "ash", "glacier", "teal", "bone", "magenta", "verdant"],
		"count": [45, 110], "val": [0.02, 0.05], "fog": [0.008, 0.02], "fog_n": [2, 4],
		"cosmos": 0.55, "cosmos_n": [1, 2], "volumetric": 0.10, "planet": 0.30, "flare": 0.20,
	},
	"nebula": {
		# Inside the cloud, where the gas IS the subject - emission nebulae really are that
		# lurid green-teal, so "toxic" belongs here as much as violet does.
		"moods": ["magenta", "violet", "teal", "abyss", "rose", "toxic", "glacier", "ember", "dawn"],
		"count": [150, 280], "val": [0.06, 0.12], "fog": [0.03, 0.055], "fog_n": [4, 8],
		"cosmos": 1.00, "cosmos_n": [2, 3], "volumetric": 0.85, "planet": 0.25, "flare": 0.45,
	},
	"orbit": {
		# Close over a world with the sun just past its limb: the planet and the flare are
		# the scene, the stars only the backdrop.
		"moods": ["glacier", "abyss", "ash", "bone", "sodium", "teal", "verdant", "brass", "violet"],
		"count": [110, 220], "val": [0.03, 0.07], "fog": [0.01, 0.03], "fog_n": [2, 5],
		"cosmos": 0.30, "cosmos_n": [1, 1], "volumetric": 0.15, "planet": 1.00, "flare": 0.90,
	},
}


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	var keys := SKIES.keys()
	var sky_name := String(keys[rng.randi() % keys.size()])
	var sky: Dictionary = SKIES[sky_name]
	var sch := Scheme.among(sky["moods"], rng)
	var val: Array = sky["val"]
	# Near-black bed, only a faint gradient (no big soft "spotlight" pools - they looked like fake
	# lights stranded behind the planet; the bright glows belong out front as a lens flare instead).
	add_layer("bed", rng, {
		"hue": sch.hue,
		"sat": clampf(sch.sat * rng.randf_range(0.6, 1.0), 0.05, 1.0),
		"val": rng.randf_range(float(val[0]), float(val[1])),
		"pools": 0,
	})
	var fog_a: Array = sky["fog"]
	var fog_n: Array = sky["fog_n"]
	add_layer("fog", rng, {
		"hue": sch.vary(rng),
		"sat": clampf(sch.sat * 0.5, 0.02, 0.6),
		"alpha": rng.randf_range(float(fog_a[0]), float(fog_a[1])),
		"count": rng.randi_range(int(fog_n[0]), int(fog_n[1])),
	})
	# Diffuse depth BEHIND the stars: a distant nebula or galaxy, and drifting harmonic bands of
	# fog (volumetric, fog mode) - the celestial dust catching light and moving with the music.
	if rng.randf() < float(sky["cosmos"]):
		var cn: Array = sky["cosmos_n"]
		add_layer("cosmos", rng, {"z": "back", "hue": sch.hue_at(1, 3),
			"count": rng.randi_range(int(cn[0]), int(cn[1]))})
	if rng.randf() < float(sky["volumetric"]):
		add_layer("volumetric", rng, {"z": "back", "mode": "fog", "hue": sch.accent})
	# Starlight on the accent, so the stars stay legible against a nebula of the base hue.
	var count := rng.randi_range(int(sky["count"][0]), int(sky["count"][1]))
	add_layer("stars", rng, {"hue": sch.accent, "count": count})
	# A real 3D planet IN FRONT of the stars (added last, so it OCCLUDES the stars behind it -
	# a solid body should block them, not show them through).
	if rng.randf() < float(sky["planet"]):
		add_layer("planet", rng, {"hue": sch.vary(rng, 3.0)})
	# A lens flare OVER everything (incl. the planet): the bright glows the user wanted up front,
	# pulsing with the harmonics, gently fisheye-bowed. How likely is the sky's business.
	if rng.randf() < float(sky["flare"]):
		add_layer("flare", rng, {"hue": sch.hue_at(1, 3), "fisheye": rng.randf_range(0.12, 0.3)})
	return {"hue": sch.hue, "mood": sch.name, "sky": sky_name, "stars": count}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.015, 0.02)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
