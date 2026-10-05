extends Node

## WHERE THE CANDLES STAND ON THE TAROT TABLE, HOW BRIGHT THEY MAY BE, AND HOW THEY FLICKER.
## Reported 2026-10-04: "candles pushed all the way to the edge of the table"; then a candle that
## ran out of room stood in FRONT of the cards, where it became the key light and blew out the card
## held up to the lens; a key candle on pale boards flooded half the frame through the bloom (the
## same light on felt reads as a candle); and "the candles flicker at exactly the same rate". Over
## many seeds, on a fixture of its own:
##
##   - every candle a look asks for stands, up to the most a look may ask for;
##   - none nearer the reader than the middle of the table, all wholly in the frame, none in front
##     of another;
##   - every candle throws a shadow from every flame but its own (it floated, casting none);
##   - no candle is brighter than the cloth by it allows ([constant TarotMedium.HEAT]), the key
##     is a candle only where its cloth can take [constant TarotMedium.KEY_MIN], and exactly one
##     light throws shadows. Two-sided: on a pale cloth the lamp must be the key, on a dark one a
##     candle at [constant TarotMedium.KEY_ENERGY] - and the retired rule (nearest the middle, at
##     full light) must break the cap on this fixture, or the half-pale cloth tests nothing;
##   - no two flames keep time: each its own tempo, its drafts its own - against the retired
##     flicker (one tempo for all) as the control; and the room's out-of-shot candles are out of
##     shot, throw no shadows, and flicker too.
##
##   tests/run_boot_probe.sh tests/tarot_place_check.gd 180
##
## A BOOT probe (the medium reaches the Director); no GPU needed - placement is projection
## arithmetic and the cloth's lightness is read from its file.

const DIR := "user://tarot_place_check"
const SEEDS := 40

var _fails := 0
var medium: TarotMedium
var subs: Subtitles
var _script := ""


func _ready() -> void:
	_run.call_deferred()


func _ok(cond: bool, what: String) -> void:
	print(("  ok    " if cond else "  FAIL  ") + what)
	if not cond:
		_fails += 1


func _run() -> void:
	_fixture()
	var stage := SubViewport.new()
	stage.size = Vector2i(640, 360)
	stage.own_world_3d = true
	add_child(stage)
	Director.detach()
	medium = Medium.make("tarot") as TarotMedium
	medium.mount(stage)
	Director.attach(stage, medium)
	Director.hold(true)
	subs = preload("res://scripts/subtitles.gd").new()
	add_child(subs)
	medium.bind_captions(subs)
	_script = TarotScript.compose([{"kind": "shuffle", "card": 0, "text": "Shuffle the deck."},
		{"kind": "draw", "card": 1, "text": "One."}, {"kind": "draw", "card": 2, "text": "Two."},
		{"kind": "draw", "card": 3, "text": "Three."}, {"kind": "spread", "card": 0, "text": "Done."}])

	print("where the candles stand (2, a look's usual)")
	var a := _sweep("half", 2)
	_ok(a["candles"] == 2 * SEEDS, "every candle stands (%d/%d)" % [a["candles"], 2 * SEEDS])
	_ok(a["front"] == 0, "none in front of the middle (%d)" % a["front"])
	_ok(a["out"] == 0, "all wholly in the frame (%d)" % a["out"])
	_ok(a["overlap"] == 0, "none in front of another (%d)" % a["overlap"])
	print("where the candles stand (the most a look may ask for)")
	var b := _sweep("half", TarotTable.MAX_CANDLES)
	_ok(b["candles"] == TarotTable.MAX_CANDLES * SEEDS, "every candle stands (%d/%d)" % [b["candles"], TarotTable.MAX_CANDLES * SEEDS])
	_ok(b["front"] == 0 and b["out"] == 0 and b["overlap"] == 0,
		"none in front, out of frame or overlapping (%d, %d, %d)" % [b["front"], b["out"], b["overlap"]])

	print("the instrument")
	var spot := Vector3(0.25, 0.0, -0.15)
	_ok(not _inside(medium._screen_box(Vector3(0.0, 0.0, -0.2), 0.8, 0.1)), "a candle wider than the frame is out of it")
	_ok(medium._screen_box(spot, 0.02, 0.1).intersects(medium._screen_box(spot + Vector3(0.01, 0, 0), 0.02, 0.1)),
		"two candles on one spot overlap")

	print("how bright a candle may be")
	_ok(a["over_cap"] == 0 and b["over_cap"] == 0, "no candle brighter than its cloth allows (%d, %d)" % [a["over_cap"], b["over_cap"]])
	_ok(a["bad_key"] == 0 and b["bad_key"] == 0, "the key stands where its cloth can take it (%d, %d)" % [a["bad_key"], b["bad_key"]])
	_ok(a["casters"] == 0 and b["casters"] == 0, "exactly one light throws shadows (%d, %d off)" % [a["casters"], b["casters"]])
	_ok(b["own_shadow"] == 0, "no candle shadows its own base (%d)" % b["own_shadow"])
	_ok(b["no_shadow"] == 0, "every candle stands in the other flames' light, shadow and all (%d left out)" % b["no_shadow"])
	_ok(a["old_over"] > 0, "control: the retired key (nearest the middle, full light) breaks the cap here (%d seeds)" % a["old_over"])
	var pale := _sweep("pale", 2)
	_ok(pale["lamp_keys"] == SEEDS, "a pale cloth: the lamp is the key every time (%d/%d)" % [pale["lamp_keys"], SEEDS])
	var dark := _sweep("dark", 2)
	_ok(dark["full_keys"] == SEEDS, "a dark cloth: a candle is the key at full light every time (%d/%d)" % [dark["full_keys"], SEEDS])

	print("a pale stripe behind the candles (the blanket that blew out)")
	var st := _sweep("stripe", 2)
	_ok(st["over_cap"] == 0, "no candle brighter than the stripe allows (%d)" % st["over_cap"])
	_ok(st["heat_at"] < st["heat_aim"] * 0.8 and SEEDS - st["lamp_keys"] >= int(SEEDS * 0.75), "candles keep off the stripe: the cloth by them %.2f, at their aims %.2f (a candle the key %d/%d)"
		% [st["heat_at"], st["heat_aim"], SEEDS - st["lamp_keys"], SEEDS])
	# the measure itself, at the stripe's near edge: the cloth's AVERAGE round a candle there (the
	# retired measure) let a key burn; its hottest spot does not
	var edge := Vector3(0.27, 0.0, (0.26 - 0.5) * TarotMedium.CLOTH.y + medium._cloth.position.z)
	var old_cap := 0.27 / maxf(_mean_lum(edge, 0.1), 0.02)
	var new_cap := TarotMedium.HEAT / maxf(medium._heat(edge, 0.13), 0.05)
	_ok(old_cap >= TarotMedium.KEY_ENERGY and new_cap < TarotMedium.KEY_MIN,
		"control: by the stripe the average allowed a key (%.2f), the hottest spot does not (%.2f)" % [old_cap, new_cap])

	print("how they flicker")
	_flicker()

	Director.hold(false)
	Director.detach()
	_clear()
	print("tarot_place_check: %s" % ("ALL OK" if _fails == 0 else "%d FAILED" % _fails))
	get_tree().quit(0 if _fails == 0 else 1)


## Build the table for [constant SEEDS] seeds on the [param cloth] fixture and count.
func _sweep(cloth: String, candles: int) -> Dictionary:
	var dir := DIR.path_join(cloth)
	var cards: Array = []
	for i in 3:
		cards.append({"key": "c%d" % i, "name": "Card %d" % i, "numeral": str(i), "reversed": false,
			"jumper": false, "position": {}, "booklet": {}, "art": ""})
	var n := {"candles": 0, "front": 0, "out": 0, "overlap": 0, "over_cap": 0, "bad_key": 0,
		"casters": 0, "old_over": 0, "lamp_keys": 0, "full_keys": 0, "heat_at": 0.0, "heat_aim": 0.0,
		"own_shadow": 0, "no_shadow": 0}
	for s in range(1, SEEDS + 1):
		var doc := {"show": "place-check", "seed": s, "dir": dir, "images": {"surface": dir.path_join("surface.png")},
			"plan": {"look": {"candles": candles}}, "cards": cards}
		subs.document = {"source": _script, "title": "Place Check", "tarot": doc}
		medium._ensure_doc()
		var rects: Array = []
		for c in medium._props.get_children():
			if c is MeshInstance3D and (c as MeshInstance3D).mesh is CylinderMesh and not (c as Node).is_queued_for_deletion():
				var cm := (c as MeshInstance3D).mesh as CylinderMesh
				var p := (c as Node3D).position
				rects.append(medium._screen_box(Vector3(p.x, 0.0, p.z), cm.top_radius, cm.height + 0.035))
				n["front"] += 1 if p.z > medium._mid.z + 0.001 else 0
		for r in rects:
			n["out"] += 0 if _inside(r) else 1
		for i in rects.size():
			for j in range(i + 1, rects.size()):
				n["overlap"] += 1 if (rects[i] as Rect2).intersects(rects[j] as Rect2) else 0
		n["candles"] += medium._flames.size()
		# the light
		# each candle's body is left out of its own flame's shadows, and only its own
		var bodies: Array = medium._props.get_children().filter(func(c: Node) -> bool:
			return c is MeshInstance3D and (c as MeshInstance3D).mesh is CylinderMesh and not c.is_queued_for_deletion())
		for i in medium._flames.size():
			var mask: int = ((medium._flames[i] as Dictionary)["light"] as OmniLight3D).shadow_caster_mask
			for j in bodies.size():
				var cast := (mask & (bodies[j] as MeshInstance3D).layers) != 0
				if i == j and cast:
					n["own_shadow"] += 1
				elif i != j and not cast:
					n["no_shadow"] += 1
		# how pale the cloth is by each candle, against at the spot it wants (its aim, its side)
		for f in medium._flames:
			var fb: Vector3 = (f as Dictionary)["base"]
			n["heat_at"] += medium._heat(Vector3(fb.x, 0.0, fb.z), TarotMedium.HEAT_H) / float(SEEDS * candles)
			n["heat_aim"] += medium._heat(Vector3(signf(fb.x) * TarotMedium.CANDLE_AIM.x, 0.0, TarotMedium.CANDLE_AIM.y),
				TarotMedium.HEAT_H) / float(SEEDS * candles)
		var casters := 1 if medium._lamp.shadow_enabled else 0
		var nearest := -1
		var best := INF
		for i in medium._flames.size():
			var f: Dictionary = medium._flames[i]
			var base: Vector3 = f["base"]
			var lb: Vector3 = f["light_base"]
			var cap := TarotMedium.HEAT / maxf(medium._heat(Vector3(base.x, 0.0, base.z), lb.y), 0.05)
			if float(f["energy"]) > cap + 0.0001:
				n["over_cap"] += 1
			var light: OmniLight3D = f["light"]
			if light.shadow_enabled:
				casters += 1
				if cap < TarotMedium.KEY_MIN:
					n["bad_key"] += 1
				if absf(float(f["energy"]) - TarotMedium.KEY_ENERGY) < 0.0001:
					n["full_keys"] += 1
			var d := (base * Vector3(1, 0, 1)).length()
			if d < best:
				best = d
				nearest = i
		if casters != 1:
			n["casters"] += 1
		if medium._lamp.shadow_enabled:
			n["lamp_keys"] += 1
		if nearest >= 0:
			var nb: Vector3 = (medium._flames[nearest] as Dictionary)["base"]
			var nl: Vector3 = (medium._flames[nearest] as Dictionary)["light_base"]
			if TarotMedium.KEY_ENERGY > TarotMedium.HEAT / maxf(medium._heat(Vector3(nb.x, 0.0, nb.z), nl.y), 0.05):
				n["old_over"] += 1
	return n


## NO TWO FLAMES KEEP TIME. Over a minute at 30 frames a second, on [constant SEEDS] tables of
## four candles: each flame's TEMPO (how often its brightness crosses its own mean) must differ
## from the others', and their drafts must not come together more than chance would bring them.
## The control is the retired flicker on the same flames - one tempo for all - which must not.
func _flicker() -> void:
	var dir := DIR.path_join("dark")
	var ratios := PackedFloat32Array()
	var old_ratios := PackedFloat32Array()
	var drafted := 0.0
	var both := 0.0
	var expected := 0.0
	var glows := 0
	var glows_seen := 0
	var glow_shadows := 0
	var glow_moves := 0
	for s in range(1, 13):
		var doc := {"show": "place-check", "seed": s, "dir": dir, "plan": {"look": {"candles": 4}}, "cards": []}
		subs.document = {"source": _script, "title": "Place Check", "tarot": doc}
		medium._ensure_doc()
		var tempos := PackedFloat32Array()
		var old_tempos := PackedFloat32Array()
		var dips: Array = []
		for f in medium._flames:
			var fk: Dictionary = (f as Dictionary)["flicker"]
			var series := PackedFloat32Array()
			var old := PackedFloat32Array()
			var dip := PackedByteArray()
			for i in 1800:
				var t := float(i) / 30.0
				var b := medium._flame_at(fk, t).x
				series.append(b)
				dip.append(1 if b < 0.9 else 0)
				var sd := float(fk["seed"])
				old.append(0.7 + 0.45 * (0.5 + 0.5 * medium._noise.get_noise_2d(t * 2.3, sd))
					+ 0.18 * (0.5 + 0.5 * medium._noise.get_noise_2d(t * 9.0, sd + 31.0)))
			tempos.append(_crossings(series))
			old_tempos.append(_crossings(old))
			dips.append(dip)
		ratios.append(_spread(tempos))
		old_ratios.append(_spread(old_tempos))
		for i in dips.size():
			for j in range(i + 1, dips.size()):
				var pi_ := 0.0
				var pj := 0.0
				var pij := 0.0
				for k in 1800:
					pi_ += (dips[i] as PackedByteArray)[k]
					pj += (dips[j] as PackedByteArray)[k]
					pij += 1.0 if (dips[i] as PackedByteArray)[k] == 1 and (dips[j] as PackedByteArray)[k] == 1 else 0.0
				drafted += (pi_ + pj) / 3600.0
				both += pij / 1800.0
				expected += (pi_ / 1800.0) * (pj / 1800.0)
		for g in medium._glows:
			glows += 1
			glows_seen += 1 if medium._in_shot((g as Dictionary)["base"]) else 0
			glow_shadows += 1 if ((g as Dictionary)["light"] as OmniLight3D).shadow_enabled else 0
			var e0 := medium._flame_at((g as Dictionary)["flicker"], 3.0).x
			var moved := false
			for i in 60:
				if absf(medium._flame_at((g as Dictionary)["flicker"], 3.0 + float(i) * 0.5).x - e0) > 0.02:
					moved = true
			glow_moves += 1 if moved else 0
	ratios.sort()
	old_ratios.sort()
	var med := ratios[ratios.size() / 2]
	var old_med := old_ratios[old_ratios.size() / 2]
	_ok(med >= 1.3, "each flame has its own tempo: the fastest of a table's four against the slowest, median %.2fx" % med)
	_ok(old_med < 1.15, "control: the retired flicker keeps one tempo (%.2fx)" % old_med)
	var pairs := 12.0 * 6.0
	_ok(drafted / pairs > 0.01 and drafted / pairs < 0.2, "a flame gutters now and then, not always (%.1f%% of the time)" % (drafted / pairs * 100.0))
	_ok(both <= expected * 2.0 + 0.002, "drafts come apart, as chance has them (together %.2f%% of the time, chance %.2f%%)"
		% [both / pairs * 100.0, expected / pairs * 100.0])
	_ok(glows >= 24 and glows_seen == 0, "two or three room candles a table, none in the shot (%d, %d seen)" % [glows, glows_seen])
	_ok(glow_shadows == 0, "the room's candles throw no shadows (%d do)" % glow_shadows)
	_ok(glow_moves == glows, "the room's candles flicker (%d of %d)" % [glow_moves, glows])


## The retired measure: the cloth's mean lightness within [param r] meters of [param at].
func _mean_lum(at: Vector3, r: float) -> float:
	var g := TarotMedium.LUM_GRID
	var cell := Vector2(TarotMedium.CLOTH.x / g.x, TarotMedium.CLOTH.y / g.y)
	var sum := 0.0
	var n := 0
	for gy in g.y:
		for gx in g.x:
			var c := Vector2(medium._cloth.position.x + (gx + 0.5 - g.x * 0.5) * cell.x,
				medium._cloth.position.z + (gy + 0.5 - g.y * 0.5) * cell.y)
			if c.distance_to(Vector2(at.x, at.z)) <= r:
				sum += medium._lum[gy * g.x + gx]
				n += 1
	return sum / maxf(1.0, float(n))


## How often a series crosses its own mean, a second.
func _crossings(v: PackedFloat32Array) -> float:
	var mean := 0.0
	for x in v:
		mean += x
	mean /= float(v.size())
	var n := 0
	for i in range(1, v.size()):
		if (v[i] - mean) * (v[i - 1] - mean) < 0.0:
			n += 1
	return float(n) / (float(v.size()) / 30.0)


## The fastest of [param v] over the slowest.
func _spread(v: PackedFloat32Array) -> float:
	var lo := INF
	var hi := 0.0
	for x in v:
		lo = minf(lo, x)
		hi = maxf(hi, x)
	return hi / maxf(lo, 0.001)


func _inside(r: Rect2) -> bool:
	return r.position.x >= 0.0 and r.position.y >= 0.0 and r.end.x <= 1.0 and r.end.y <= 1.0


## Four cloths - half felt and half pale boards (as the episode that found the flood), all pale,
## all dark, and a near-black blanket with a cream stripe toward its far edge (as the one that blew
## out behind its candles).
func _fixture() -> void:
	_clear()
	var felt := Color(0.36, 0.5, 0.14)     # the episode's felt: linear luminance 0.18
	var pine := Color(0.92, 0.86, 0.74)    # ...and its boards: 0.7
	var deep := Color(0.3, 0.1, 0.12)      # an oxblood cloth: 0.02
	var spruce := Color(0.22, 0.26, 0.2)   # the blanket: 0.045, with a cream stripe (0.46) near its far edge
	var cream := Color(0.74, 0.7, 0.6)
	for cloth in ["half", "pale", "dark", "stripe"]:
		var dir := DIR.path_join(cloth)
		DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(dir))
		var surf := Image.create(256, 256, false, Image.FORMAT_RGB8)
		for y in 256:
			for x in 256:
				var c := deep if cloth == "dark" else (felt if cloth == "half" and x < 128 else pine)
				if cloth == "stripe":
					c = cream if y >= 16 and y < 51 else spruce
				surf.set_pixel(x, y, c.darkened(0.1 * float((x * 7 + y * 13) % 5) / 4.0))
		surf.save_png(dir.path_join("surface.png"))


func _clear() -> void:
	for cloth in ["half", "pale", "dark", "stripe"]:
		var da := DirAccess.open(DIR.path_join(cloth))
		if da == null:
			continue
		for fn in da.get_files():
			da.remove(fn)
		DirAccess.remove_absolute(ProjectSettings.globalize_path(DIR.path_join(cloth)))
	DirAccess.remove_absolute(ProjectSettings.globalize_path(DIR))
