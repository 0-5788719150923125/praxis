extends GhostScene

## Fog bank - rolling coloured fog, light glowing from within.
##
## The explicit "rolling fog with colours underneath / inside" idea: a rich colour bed
## with slow pools breathing on the spectrum, then several fog layers of different tint
## and speed rolling over and through it, so the colour bleeds up through the cloud. The
## bank lurches on the beat and coasts down (velocity + decay, not a uniform drift).
## `bed` + stacked `fog` layers - atmosphere from pure composition.

# How heavy the bank is. A thin one barely veils the colour and a deep one buries it in
# sheets, which is the difference between a glow with mist over it and weather - so the
# sheet count, their opacity and the bed's brightness all move together.
const BANKS := {
	"thin":    {"sheets": [1, 1], "alpha": [0.020, 0.035], "count": [4, 6],  "val": [0.46, 0.60]},
	"rolling": {"sheets": [2, 3], "alpha": [0.030, 0.050], "count": [5, 8],  "val": [0.38, 0.54]},
	"deep":    {"sheets": [3, 4], "alpha": [0.040, 0.070], "count": [6, 10], "val": [0.28, 0.44]},
}


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	# Any mood is fair here: the bank is the constant, the colour under it is the variable.
	var sch := Scheme.pick(rng)
	var keys := BANKS.keys()
	var bank := String(keys[rng.randi() % keys.size()])
	var b: Dictionary = BANKS[bank]
	var val_r: Array = b["val"]
	var alpha_r: Array = b["alpha"]
	var cnt_r: Array = b["count"]
	var sheet_r: Array = b["sheets"]
	# A bright, saturated bed so the colour reads strongly *through* the fog above it.
	add_layer("bed", rng, {
		"hue": sch.hue,
		"sat": clampf(sch.sat * rng.randf_range(0.85, 1.2), 0.05, 1.0),
		"val": rng.randf_range(float(val_r[0]), float(val_r[1])),
		"pools": rng.randi_range(3, 7),
	})
	var sheets := rng.randi_range(int(sheet_r[0]), int(sheet_r[1]))
	for i in sheets:
		# The stack walks from the base hue toward the accent, so a pile of sheets is one
		# family of tints rather than an arbitrary offset invented per sheet.
		var tint: float = sch.hue_at(i + 1, sheets + 1)
		add_layer("fog", rng, {
			"hue": tint,
			"sat": clampf(sch.sat * rng.randf_range(0.2, 0.5), 0.03, 0.6),
			"alpha": rng.randf_range(float(alpha_r[0]), float(alpha_r[1])),
			"count": rng.randi_range(int(cnt_r[0]), int(cnt_r[1])),
		})
	return {"hue": sch.hue, "mood": sch.name, "bank": bank, "sheets": sheets}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.02, 0.03)
	update_layers(f, delta)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
