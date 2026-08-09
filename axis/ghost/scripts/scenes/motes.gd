extends GhostScene

## Motes - dust adrift in a shaft of light.
##
## The quietest scene: light cuts the frame and countless fine motes hang and turn in
## it on a slow curl-noise current, brightening as they cross the beam. The colour of
## the light is a [Scheme] - late sun, a cold north window, green light under leaves -
## and the room's shadow is that scheme's accent, so the two agree. `bed` + `fog` +
## a `dust` layer.

# THE AIR IN THE ROOM. One shaft, 120-200 motes, always: the same still afternoon
# every time. An air is how much matter hangs in it and whether there is a beam at
# all - a thick haze with no visible shaft is a different room from a near-empty one
# with a hard column of light through it.
const AIRS := [
	{"name": "shaft", "count": [120, 200], "beam": true, "speed": [0.015, 0.030], "drift": [0.004, 0.010]},
	{"name": "haze", "count": [220, 340], "beam": false, "speed": [0.010, 0.022], "drift": [0.002, 0.006]},
	{"name": "still", "count": [45, 95], "beam": true, "speed": [0.006, 0.014], "drift": [0.001, 0.004]},
	{"name": "draught", "count": [140, 230], "beam": true, "speed": [0.035, 0.065], "drift": [0.012, 0.024]},
]


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	# Whatever light gets in: late sun, sodium street-glare, a cold north window,
	# green light under leaves, or the colourless grey of an overcast one.
	var sch := Scheme.among(["sodium", "dawn", "brass", "bone", "glacier", "ash", "verdant",
		"teal", "rose"], rng)
	var air: Dictionary = AIRS[rng.randi() % AIRS.size()]
	# The room behind the beam sits in the scheme's accent, kept dark - shadow that
	# belongs to this light rather than an unrelated second colour.
	add_layer("bed", rng, {"hue": sch.accent, "sat": sch.sat * 0.5,
		"val": sch.val * rng.randf_range(0.12, 0.22), "pools": rng.randi_range(2, 4)})
	add_layer("fog", rng, {"hue": sch.hue, "sat": sch.sat * 0.3,
		"alpha": rng.randf_range(0.02, 0.055), "count": rng.randi_range(3, 8)})
	add_layer("dust", rng, {
		"hue": sch.vary(rng, 0.6),
		"count": rng.randi_range(int(air["count"][0]), int(air["count"][1])),
		"shaft": bool(air["beam"]),
		"shaft_x": rng.randf_range(-0.35, 0.35),
		"speed": rng.randf_range(float(air["speed"][0]), float(air["speed"][1])),
		"drift": rng.randf_range(float(air["drift"][0]), float(air["drift"][1])),
	})
	return {"hue": sch.hue, "mood": sch.name, "air": String(air["name"])}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.015, 0.02)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
