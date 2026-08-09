extends Node
## Do repeated seeds now give genuinely different waters?

func _ready() -> void:
	var seen := {}
	for sv in range(1, 13):
		var rng := RandomNumberGenerator.new()
		rng.seed = sv
		var sc = load("res://scripts/scenes/underwater.gd").new()
		add_child(sc)
		var info: Dictionary = sc.build_params(rng)
		var name := String(info.get("water", "?"))
		seen[name] = int(seen.get(name, 0)) + 1
		if sv <= 6:
			print("PROBE: seed %2d  water=%-9s hue=%.2f  layers=%d" % [sv, name, float(info["hue"]), sc._layers.size()])
		sc.queue_free()
	print("PROBE: distinct waters over 12 seeds = %d  %s" % [seen.size(), str(seen)])
	# and the plant forms must actually differ in shape, not just count
	for form in ["kelp", "grass", "fan", "whip", "bulb"]:
		var r := RandomNumberGenerator.new()
		r.seed = 99
		var k = Layer.make("kelp", r, {"form": form, "fronds": 40, "glows": 0})
		var hmin := 9.0; var hmax := 0.0; var wmax := 0.0
		for fr in k._fronds:
			hmin = minf(hmin, float(fr["h"])); hmax = maxf(hmax, float(fr["h"]))
			wmax = maxf(wmax, float(fr["w"]))
		print("PROBE: form %-6s height %.2f-%.2f  max width %.3f" % [form, hmin, hmax, wmax])
	get_tree().quit()
