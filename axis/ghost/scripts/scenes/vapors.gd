extends GhostScene

## Vapors - coloured ink-in-water: heavy masses twisting through the frame with fibrous
## strands drawn out of them, lit from inside.
##
## The gap this fills. ghost already had soft weather - `clouds`, `fog_bank`, `fog_volume`,
## `aurora` - and every one of them is built from stacked gaussian puffs, which can only
## ever be SOFT. The look asked for here is the opposite: diffuse but HARD, a mass with a
## visible front, bright threads streaming inside it, and its colour changing across the
## frame - ink in water, or heavy vapour turning over, or the stretched-cotton cobweb
## people hang at Halloween. None of that is a blob's to draw, so this scene is a FIELD:
## one quad, one density evaluated per pixel, and the structure is in the arithmetic. Read
## shaders/vapor_field.gdshader's header for how (a ridged multifractal for the filaments, a
## double domain warp for the twist, the plume envelope re-tapped toward the lamp for the
## volume) and [Layer.Vapor] for the plumes that place, colour and drive it.
##
## Its gate is tests/vapor_check.gd, and it is a gate about a LOOK: presence, a front, drawn-out
## filaments, more than one hue, and motion, each against a control that breaks that one
## property. Three earlier cuts of the field compiled, ran, reported plausible coverage and drew
## engraved satin; that is why the gate measures the properties rather than the plumbing.
##
## WHAT THE MUSIC IS ALLOWED TO TOUCH, and this is the scene's hardest-won rule: light and
## SPEED, never shape. The vapour travels steadily forward and turns over on its own clock; a
## swell brightens it, moves its rate of flow a little, and thickens each mass slightly over a
## second or two; a beat lights the plume it landed on. What it may NOT do is resize anything -
## the first version drove the plume amplitudes hard and fast and re-warped space with the
## loudness, and since amplitude IS the size of a mass, the whole frame inflated and collapsed
## with the harmonics ("EXTREMELY unstable... it should be continuous movement forward, not an
## expand and contract"). Every audio path here now ends in a brightness or a rate, and
## tests/vapor_check.gd holds it to that. The tonal centre pulls every hue toward the music's
## key, eased as a vector so a key change sweeps instead of jumping.
##
## What kind of vapour it is comes from one roll, because these are not independent
## choices: ink is dense and wet and dark, a nebula is wide and diffuse and starlit, and
## gossamer is pale and almost all fibre. Sampling them separately would mostly produce
## neither.

# The vapour's character. Each decides the moods it may be coloured from, the field's shape
# (how hard the front, how fibrous, how big the features, how violently it turns), where the
# masses sit, and what else is in the frame.
const CHARACTERS := {
	# INK - a drop of ink in still water: dense, wet, sharp-fronted, dark around it.
	"ink": {
		"moods": ["abyss", "violet", "magenta", "teal", "ember"],
		"layouts": ["scatter", "sweep"],
		"count": [5, 7], "hard": [0.72, 0.95], "thresh": [0.46, 0.60],
		"stretch": [2.2, 3.6], "crease": [1.9, 2.8], "scale": [0.85, 1.25],
		"swirl": [0.55, 0.85], "churn": [0.8, 1.3], "flow": [0.012, 0.026], "haze": [0.07, 0.14],
		"shadow": [1.9, 2.8], "sat": [0.80, 0.95], "size": [0.9, 1.15],
		"counter": 0.3, "bed": [0.008, 0.022], "stars": 0.15, "dust": 0.25,
	},
	# NEBULA - the same field seen across light years: broad, diffuse, colours far apart,
	# starlight behind it.
	"nebula": {
		"moods": ["violet", "magenta", "abyss", "glacier", "rose", "teal"],
		"layouts": ["edges", "scatter"],
		"count": [5, 7], "hard": [0.32, 0.58], "thresh": [0.40, 0.52],
		"stretch": [1.7, 2.7], "crease": [1.2, 1.9], "scale": [0.60, 0.90],
		"swirl": [0.40, 0.65], "churn": [0.55, 0.95], "flow": [0.016, 0.034], "haze": [0.16, 0.26],
		"shadow": [1.2, 1.9], "sat": [0.70, 0.90], "size": [1.05, 1.35],
		"counter": 0.5, "bed": [0.008, 0.022], "stars": 0.90, "dust": 0.20,
	},
	# GOSSAMER - the fake cobweb: pale, low mass, almost entirely strand, stretched between
	# whatever it is caught on.
	"gossamer": {
		"moods": ["glacier", "bone", "teal", "violet", "ash"],
		"layouts": ["sweep", "edges"],
		"count": [4, 6], "hard": [0.60, 0.88], "thresh": [0.44, 0.56],
		"stretch": [3.2, 4.8], "crease": [2.4, 3.2], "scale": [0.90, 1.30],
		"swirl": [0.35, 0.60], "churn": [0.5, 0.9], "flow": [0.022, 0.042], "haze": [0.06, 0.12],
		"shadow": [1.6, 2.4], "sat": [0.30, 0.55], "size": [0.95, 1.25],
		"counter": 0.2, "bed": [0.008, 0.022], "stars": 0.35, "dust": 0.60,
	},
	# PLASMA - vapour with power in it: hot, saturated, turning over fast.
	"plasma": {
		"moods": ["toxic", "magenta", "sodium", "teal", "ember", "verdant"],
		"layouts": ["scatter", "edges", "sweep"],
		"count": [6, 8], "hard": [0.68, 0.92], "thresh": [0.44, 0.58],
		"stretch": [2.0, 3.2], "crease": [1.8, 2.8], "scale": [0.80, 1.20],
		"swirl": [0.70, 1.00], "churn": [1.25, 1.90], "flow": [0.026, 0.050], "haze": [0.10, 0.18],
		"shadow": [1.5, 2.3], "sat": [0.85, 1.00], "size": [0.85, 1.10],
		"counter": 0.45, "bed": [0.008, 0.022], "stars": 0.10, "dust": 0.15,
	},
}

var _vapor: Layer.Vapor = null
# The tonal centre, held as a VECTOR and eased there - see the note in update().
var _ch_vec := Vector2.ZERO


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	var keys := CHARACTERS.keys()
	var kind := String(keys[rng.randi() % keys.size()])
	var ch: Dictionary = CHARACTERS[kind]
	var sch := Scheme.among(ch["moods"], rng)
	var layouts: Array = ch["layouts"]
	var layout := String(layouts[rng.randi() % layouts.size()])
	# The ground is nearly black on purpose: the field is ADDITIVE light, so every bit of
	# brightness under it is brightness the vapour cannot be darker than, and this look
	# lives or dies on having true dark between the masses.
	add_layer("bed", rng, {
		"hue": sch.hue,
		"sat": clampf(sch.sat * rng.randf_range(0.5, 0.9), 0.05, 1.0),
		"val": _pick(rng, ch["bed"]),
		"pools": rng.randi_range(2, 5),
	})
	if rng.randf() < float(ch["stars"]):
		add_layer("stars", rng, {
			"z": "back", "count": rng.randi_range(60, 220), "hue": sch.accent})
	_vapor = add_layer("vapor", rng, {
		"hue": sch.hue,
		"accent": sch.accent,              # the fringe / shadow chroma
		"spread": sch.spread * rng.randf_range(1.2, 2.6),
		"sat": _pick(rng, ch["sat"]),
		"val": rng.randf_range(0.95, 1.0),
		"count": rng.randi_range(int(ch["count"][0]), int(ch["count"][1])),
		"layout": layout,
		"counter": float(ch["counter"]),   # share of masses on the OPPOSING hue
		"size": _pick(rng, ch["size"]),
		"hard": _pick(rng, ch["hard"]),
		"thresh": _pick(rng, ch["thresh"]),
		"stretch": _pick(rng, ch["stretch"]),
		"crease": _pick(rng, ch["crease"]),
		"scale": _pick(rng, ch["scale"]),
		"swirl": _pick(rng, ch["swirl"]),
		"churn": _pick(rng, ch["churn"]),      # how fast the medium turns over...
		"flow": _pick(rng, ch["flow"]),        # ...and how fast it travels while doing it
		"haze": _pick(rng, ch["haze"]),
		"shadow": _pick(rng, ch["shadow"]),
		"gain": rng.randf_range(1.15, 1.50),
	}) as Layer.Vapor
	# Motes draw onto the scene's own canvas, so the field (a child quad - see
	# [Layer.FieldQuad]) composites OVER them whatever order they are added in. That is fine
	# and is why they are here at all: additive light does not occlude, so the motes read
	# THROUGH the vapour, brightened where it is thick.
	#
	# `flare` was here and is deliberately gone: a literal lens flare - rings, starburst,
	# anamorphic streak - drew hardware in front of the frame and read as a stock overlay
	# over a scene whose whole subject is light IN a medium. Rendered, looked at, removed.
	if rng.randf() < float(ch["dust"]):
		add_layer("dust", rng, {
			"hue": sch.accent, "count": rng.randi_range(40, 120),
			"shaft": false, "speed": rng.randf_range(0.01, 0.03)})
	return {"hue": sch.hue, "mood": sch.name, "character": kind, "layout": layout}


static func _pick(rng: RandomNumberGenerator, r: Array) -> float:
	return rng.randf_range(float(r[0]), float(r[1]))


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	# A whisper of camera: the field's own motion is the show, and panning a full-frame
	# field on top of it only ever reads as the camera slipping.
	drift_view(f, 0.010, 0.018)
	# THE TONAL CENTRE, EASED AS A VECTOR - tidepool's discipline, for its reason. An angle
	# is the wrong quantity to ease: when two tonal centres a tritone apart trade dominance
	# the resultant passes near the origin and its angle flips half a turn between frames,
	# which the vapour would show as every mass changing colour at once. Easing the vector
	# routes that through the middle: an ambiguous tonality reads as low STRENGTH and the
	# pull simply fades out instead of picking a side.
	var raw := chroma_hue()
	var want := Vector2(cos(raw.x * TAU), sin(raw.x * TAU)) * raw.y
	_ch_vec = _ch_vec.lerp(want, 1.0 - exp(-0.85 * delta))
	if _vapor != null:
		_vapor.tonal = Vector2(fposmod(_ch_vec.angle() / TAU, 1.0),
			clampf(_ch_vec.length(), 0.0, 1.0))
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
