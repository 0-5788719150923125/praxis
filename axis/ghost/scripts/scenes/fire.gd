extends GhostScene

## Fire - a living flame attuned to the harmonics.
##
## One GPU temperature field (see [Layer.Fire] / shaders/flame.gdshader): heat sources
## along the bed, each listening to its own harmonic band (bass at the centre, treble
## at the rim), raise columns that rising turbulence carves into licks. CPU sparks
## crackle out of whichever region is roaring, and smoke sometimes hazes the top.
## Quiet passages sit as embers; powerful ones send columns up the frame.
##
## The flame's own colours are the shader's (a real fire is the temperature of its
## fuel), so what the seed varies here is the HEARTH around it: the ambient the fire
## burns in, how wide it is laid, and how much smoke it throws.

# The hearth. Each is a place a fire is found, and it decides the ambient wash behind the
# flame, how far the sources are laid across the frame, and how heavily it smokes - so a
# forge, a bonfire and a fire on a cold night are three different scenes, not one tint.
const HEARTHS := {
	"forge":   {"moods": ["ember", "sodium", "brass"], "spread": [0.45, 0.75],
		"val": [0.07, 0.11], "smoke": 0.35, "sheets": [1, 1]},
	"bonfire": {"moods": ["ember", "dawn", "rose", "sodium"], "spread": [0.85, 1.15],
		"val": [0.05, 0.09], "smoke": 0.60, "sheets": [1, 2]},
	"night":   {"moods": ["abyss", "violet", "ash", "teal"], "spread": [0.70, 1.05],
		"val": [0.03, 0.07], "smoke": 0.70, "sheets": [1, 2]},
}


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	var keys := HEARTHS.keys()
	var hearth := String(keys[rng.randi() % keys.size()])
	var h: Dictionary = HEARTHS[hearth]
	var sch := Scheme.among(h["moods"], rng)
	var val_r: Array = h["val"]
	var spread_r: Array = h["spread"]
	add_layer("bed", rng, {
		"hue": sch.hue,
		"sat": clampf(sch.sat * rng.randf_range(0.6, 1.0), 0.05, 1.0),
		"val": rng.randf_range(float(val_r[0]), float(val_r[1])),
		"pools": rng.randi_range(1, 4),
	})
	# No hue passed to the flame itself: its palette is the temperature ramp in
	# flame.gdshader, and a fire that took an arbitrary hue would stop reading as one.
	add_layer("fire", rng, {
		"count": rng.randi_range(50, 160),                       # the SPARK pool (born from heat)
		"spread": rng.randf_range(float(spread_r[0]), float(spread_r[1])),
	})
	# Smoke drifting off the top, sometimes - on the mood's accent, so a cold-night fire
	# smokes cold and a forge smokes warm.
	if rng.randf() < float(h["smoke"]):
		var sheets: Array = h["sheets"]
		for i in rng.randi_range(int(sheets[0]), int(sheets[1])):
			add_layer("fog", rng, {
				"hue": sch.hue_at(i + 1, 3),
				"sat": clampf(sch.sat * rng.randf_range(0.15, 0.4), 0.02, 0.5),
				"alpha": rng.randf_range(0.02, 0.045),
				"count": rng.randi_range(3, 8),
			})
	return {"hue": sch.hue, "mood": sch.name, "hearth": hearth}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.015, 0.02)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
