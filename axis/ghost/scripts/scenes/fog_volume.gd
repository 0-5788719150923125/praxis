extends GhostScene

## Fog volume - REAL 3D fog: a low, wide bank of soft gaussian puffs receding into depth, lit
## volumetrically (a brighter sunward edge fading into a dim core) and slowly drifting. A genuine
## haze with simulated dynamics, not a flat 2D wash. `bed` + `volumetric` (fog mode), and
## depending on the AIR the seed draws: stars above it, or 2D sheets rolling through it.

# The air the haze hangs in. A thin one is nearly clear and shows the sky through it; a soup
# is tinted, layered and starless. What accompanies the volume is the scene's shape here,
# since the puff field itself is built inside the layer.
#   tint - chance the volume takes the mood's hue rather than neutral grey
const AIRS := {
	"thin":   {"tint": 0.45, "sheets": 0, "stars": 0.65, "val": [0.08, 0.15]},
	"tinted": {"tint": 1.00, "sheets": 1, "stars": 0.30, "val": [0.09, 0.17]},
	"soup":   {"tint": 0.90, "sheets": 2, "stars": 0.00, "val": [0.05, 0.11]},
}


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	# Haze is the colour of whatever lights it - sodium street lamps, a green swamp, a cold
	# dawn - so no mood is off the table. It was locked to one narrow blue before.
	var sch := Scheme.pick(rng)
	var keys := AIRS.keys()
	var air := String(keys[rng.randi() % keys.size()])
	var a: Dictionary = AIRS[air]
	var val_r: Array = a["val"]
	add_layer("bed", rng, {
		"hue": sch.hue,
		"sat": clampf(sch.sat * rng.randf_range(0.4, 0.8), 0.03, 1.0),
		"val": rng.randf_range(float(val_r[0]), float(val_r[1])),
		"pools": rng.randi_range(2, 5),
	})
	if rng.randf() < float(a["stars"]):
		add_layer("stars", rng, {"z": "back", "count": rng.randi_range(50, 140), "hue": sch.accent})
	var p := {"mode": "fog"}
	if rng.randf() < float(a["tint"]):
		p["hue"] = sch.vary(rng)        # a tinted haze; otherwise neutral grey fog
	add_layer("volumetric", rng, p)
	# 2D sheets rolling THROUGH the volume: the flat drift reads as nearer air moving past
	# the depth behind it, which is what makes a soup feel like it has layers.
	for i in int(a["sheets"]):
		add_layer("fog", rng, {
			"hue": sch.hue_at(i + 1, int(a["sheets"]) + 1),
			"sat": clampf(sch.sat * rng.randf_range(0.15, 0.45), 0.02, 0.6),
			"alpha": rng.randf_range(0.025, 0.055),
			"count": rng.randi_range(4, 8),
		})
	return {"hue": sch.hue, "mood": sch.name, "air": air}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.010, 0.015)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
