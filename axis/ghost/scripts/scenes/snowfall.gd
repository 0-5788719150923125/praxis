extends GhostScene

## Snowfall - a quiet field of falling snow over a soft colour bed.
##
## The atmospheric weather scene (distinct from `snowflakes`, the hero crystal field):
## flakes drift down and gust sideways with the treble over a gradient that breathes with
## the music. The seed picks the KIND of snow first - fine powder, fat lazy flurry, a
## driven blizzard, a still hard frost - and that decides how many flakes there are, how
## big, how fast they fall and how many of them crisp into six-fold dendrites, so the
## silhouette changes with the weather and not just the tint. Pure component composition -
## `bed` + `snow` (+ `volumetric` fog, + `veil`) from the [Layer] registry.

# Snow is a WEATHER before it is a colour. Flake size, count and fall speed are one choice,
# not three: powder is many tiny fast-settling specks, a flurry is a handful of fat slow
# ones, a blizzard is a wall, and a hoar frost is sparse but genuinely crystalline.
#   count/size/fall/crystal go straight to the snow layer; fog/veil are chances; val is the bed
#
# `habits` is the crystal morphology this weather produces, as relative weights over
# [constant Crystal.HABITS], and it is not decoration: habit is set by TEMPERATURE and
# complexity by SUPERSATURATION (the Nakaya diagram), so the conditions that decide
# whether snow is powder or a flurry are the same ones that decide whether its crystals
# are columns or ferns. Tying them together is what makes the four weathers read as four
# different substances rather than four densities of the same one.
#
# `density` is the floor and gain of the audio-driven flake count - the drizzle-to-
# downpour axis. A blizzard is loud at its quietest, so its floor is high and its gain
# small; a hoar frost barely moves at all, which is the point of a still frost.
const SNOWS := {
	"powder": {
		# Dry fine snow. Cool by default, but snow takes the colour of whatever lights it -
		# dawn on a field, or a lamp - so the warm moods are in here too.
		# Cold and dry: columns and needles, almost no branching. The old table gave
		# powder full dendrites, which is meteorologically backwards.
		"moods": ["glacier", "bone", "ash", "abyss", "teal", "violet", "dawn", "rose"],
		"count": [260, 380], "size": [0.0035, 0.005], "fall": [0.06, 0.10],
		"crystal": [0.10, 0.25], "fog": 0.70, "veil": 0.0, "val": [0.20, 0.30],
		"habits": {"column": 0.45, "needle": 0.30, "plate": 0.25},
		"complexity": [0.0, 0.2], "rime": [0.0, 0.1], "density": [0.30, 0.75],
	},
	"flurry": {
		# Fat wet flakes, few and slow - the ones you can follow with your eye down.
		# Warm and wet, near freezing: big stellar dendrites and sectored plates.
		"moods": ["glacier", "sodium", "dawn", "rose", "violet", "bone", "teal", "abyss"],
		"count": [55, 110], "size": [0.012, 0.020], "fall": [0.045, 0.08],
		"crystal": [0.55, 0.85], "fog": 0.45, "veil": 0.0, "val": [0.22, 0.34],
		"habits": {"stellar": 0.45, "sectored": 0.30, "fern": 0.15, "capped": 0.10},
		"complexity": [0.5, 0.8], "rime": [0.1, 0.35], "density": [0.35, 0.70],
	},
	"blizzard": {
		# Driven and dense enough to hide the ground; brass/sodium are headlights in it.
		# Driven snow is battered snow: heavily rimed, broken, barely branched.
		"moods": ["ash", "glacier", "abyss", "bone", "teal", "violet", "sodium", "brass"],
		"count": [340, 480], "size": [0.005, 0.008], "fall": [0.20, 0.34],
		"crystal": [0.15, 0.35], "fog": 0.90, "veil": 0.95, "val": [0.12, 0.22],
		"habits": {"plate": 0.35, "column": 0.30, "bullet": 0.20, "stellar": 0.15},
		"complexity": [0.0, 0.25], "rime": [0.5, 1.0], "density": [0.62, 0.95],
	},
	"hoar": {
		# A still, hard frost: fewer flakes but most of the near ones are drawn dendrites,
		# so this is the variant whose SHAPES read differently, not just its density.
		# Still air and high supersaturation - the showcase, and the only unrimed one.
		"moods": ["glacier", "teal", "bone", "violet", "abyss", "magenta", "ash", "dawn"],
		"count": [150, 240], "size": [0.009, 0.014], "fall": [0.03, 0.06],
		"crystal": [0.80, 1.0], "fog": 0.35, "veil": 0.0, "val": [0.24, 0.36],
		"habits": {"fern": 0.40, "stellar": 0.35, "sectored": 0.25},
		"complexity": [0.75, 1.0], "rime": [0.0, 0.05], "density": [0.55, 0.45],
	},
}


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	var keys := SNOWS.keys()
	var kind := String(keys[rng.randi() % keys.size()])
	var s: Dictionary = SNOWS[kind]
	var sch := Scheme.among(s["moods"], rng)
	var val: Array = s["val"]
	add_layer("bed", rng, {
		"hue": sch.hue,
		"sat": clampf(sch.sat * rng.randf_range(0.5, 0.9), 0.05, 1.0),
		"val": rng.randf_range(float(val[0]), float(val[1])),
		"pools": rng.randi_range(2, 4),
	})
	# Real volumetric fog rolling low under the snow - a lit, drifting bank with harmonic motion,
	# behind the falling flakes (added before the snow layer).
	if rng.randf() < float(s["fog"]):
		add_layer("volumetric", rng, {"mode": "fog", "hue": sch.vary(rng)})
	var count := rng.randi_range(int(s["count"][0]), int(s["count"][1]))
	var crystal := rng.randf_range(float(s["crystal"][0]), float(s["crystal"][1]))
	var dens: Array = s["density"]
	add_layer("snow", rng, {
		"hue": sch.hue_at(1, 3),
		# Snow is near-white whatever lights it, so the mood only tints it faintly.
		"sat": clampf(sch.sat * 0.18, 0.02, 0.35),
		"count": count,
		"size": rng.randf_range(float(s["size"][0]), float(s["size"][1])),
		"fall": rng.randf_range(float(s["fall"][0]), float(s["fall"][1])),
		"crystal_frac": crystal,
		# The crystal bank: enough distinct geometries that a viewer counting flakes runs
		# out of patience before they run out of shapes, without paying to generate one
		# per flake. Scaled to the field, since a blizzard shows far more at once.
		"bank": clampi(int(count / 12) + 8, 10, 32),
		"habits": s["habits"],
		"complexity": rng.randf_range(float(s["complexity"][0]), float(s["complexity"][1])),
		"rime": rng.randf_range(float(s["rime"][0]), float(s["rime"][1])),
		"density_floor": float(dens[0]),
		"density_gain": float(dens[1]),
	})
	# A blizzard needs more obscuring than the snow layer's own squalls give: a second veil in
	# front, swelling on loud passages, is what closes the view down to a whiteout.
	if rng.randf() < float(s["veil"]):
		add_layer("veil", rng, {
			"hue": sch.hue, "sat": clampf(sch.sat * 0.12, 0.01, 0.2), "val": 1.0,
			"floor": 0.08, "gain": 0.9, "max": rng.randf_range(0.35, 0.55),
		})
	return {"hue": sch.hue, "mood": sch.name, "snow": kind, "count": count, "crystal": crystal}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.02, 0.03)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
