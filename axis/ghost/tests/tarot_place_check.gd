extends Node

## WHERE THINGS STAND ON THE TAROT TABLE, HOW BRIGHT ITS CANDLES MAY BE, HOW THEY FLICKER, AND THAT
## THE CARDS GO ROUND WHAT STANDS. Reported 2026-10-04: "candles pushed all the way to the edge of
## the table"; then a candle that ran out of room stood in FRONT of the cards, where it became the
## key light and blew out the card held up to the lens; a key candle on pale boards flooded half the
## frame through the bloom (the same light on felt reads as a candle); "the candles flicker at
## exactly the same rate"; and, 2026-10-05, cards clipping through candles in the spread shuffle.
## Over many seeds, on a fixture of its own:
##
##   - every candle a look asks for stands, up to the most a look may ask for; a set dresser's
##     table stands nearly whole - on the cloth, off everywhere the cards go, nothing standing in
##     another, no group in front of another in the picture;
##   - no lit thing nearer the reader than the middle of the table, all wholly in the frame;
##   - every thing throws a shadow from every light but its own (it floated, casting none), and a
##     lit thing is ONE light however many wicks it has (every wick still burns);
##   - candles standing together light the cloth no hotter than it takes, all at once - against
##     each at its own limit as the control, which overheats it;
##   - in a wash no card passes through anything standing, and every card still ends in the deck -
##     against the same washes with the cards let through, which must cross;
##   - no candle is brighter than the cloth by it allows ([constant TarotMedium.HEAT]), the key
##     is a candle only where its cloth can take [constant TarotMedium.KEY_MIN], and every candle
##     and the lamp throw shadows. Two-sided: on a pale cloth the lamp must be the key, on a dark one a
##     candle at [constant TarotMedium.KEY_ENERGY] - and the retired rule (nearest the middle, at
##     full light) must break the cap on this fixture, or the half-pale cloth tests nothing;
##   - no two flames keep time: each its own tempo, its drafts its own - against the retired
##     flicker (one tempo for all) as the control; and the room's out-of-shot candles are out of
##     shot, throw no shadows, and flicker too;
##   - the outro: lit while the voice speaks, fading to black after its last word and black as the
##     outro runs out; the channel's name over the intro and never again (it came back at the end).
##
##   tests/run_boot_probe.sh tests/tarot_place_check.gd 180
##
## A BOOT probe (the medium reaches the Director); no GPU needed - placement is projection
## arithmetic and the cloth's lightness is read from its file.

const DIR := "user://tarot_place_check"
const SEEDS := 40

## A SET TABLE as a set dresser writes one: groups, lone things, two lit candles and a third in a
## holder, things wide and low beside the cards (where a wash reaches them).
const SET_TABLE := {"materials": {"brass": {"kind": "metal", "color": "#b48a43"}, "wax": {"kind": "wax", "color": "#e8dcc0"},
	"quartz": {"kind": "crystal", "color": "#d8c8f0", "clarity": 0.5}, "clay": {"kind": "ceramic", "color": "#8a5a3a"},
	"leather": {"kind": "leather", "color": "#5a3a2a"}, "glass": {"kind": "glass", "color": "#cfe0e0"}},
	"things": [
		{"name": "a taper in a brass stick", "place": "back left", "group": "light", "parts": [
			{"shape": "lathe", "profile": [[0, 0], [3.5, 0], [3.5, 0.6], [1, 1], [0.9, 4], [1.4, 4.4], [0, 4.4]], "material": "brass"},
			{"shape": "lathe", "profile": [[0, 0], [1, 0], [1, 7], [0, 7]], "at": [0, 4.4, 0], "material": "wax", "wick": true}]},
		{"name": "a quartz cluster", "place": "back left", "group": "light", "parts": [
			{"shape": "cluster", "count": 7, "radius": 3, "length": [1.5, 4], "material": "quartz"}]},
		{"name": "two pillars", "place": "back right", "group": "pair", "parts": [
			{"shape": "lathe", "profile": [[0, 0], [2.2, 0], [2.2, 6], [0, 6]], "material": "wax", "wick": true,
				"copies": {"line": {"count": 2, "step": [5, 0, 1]}}}]},
		{"name": "a clay bowl", "place": "right", "parts": [
			{"shape": "lathe", "profile": [[0, 0], [3, 0], [6, 3.5], [6.3, 3.7], [5.8, 3.5], [2.8, 0.5], [0, 0.5]], "material": "clay"}]},
		{"name": "a leather book", "place": "left", "turn": 20, "parts": [
			{"shape": "box", "size": [14, 3, 10], "round": 0.4, "material": "leather"}]},
		{"name": "a crystal ball on a stand", "place": "back", "parts": [
			{"shape": "lathe", "profile": [[0, 0], [3, 0], [3, 1], [2, 1.6], [0, 1.6]], "material": "brass"},
			{"shape": "ball", "size": [6, 6, 6], "at": [0, 1.2, 0], "material": "glass"}]},
		{"name": "scattered coins", "place": "by the deck", "parts": [
			{"shape": "lathe", "profile": [[0, 0], [1.2, 0], [1.2, 0.2], [0, 0.2]], "material": "brass",
				"copies": {"scatter": {"count": 5, "radius": 4}}}]},
	]}
## THREE CANDLES IN ONE GROUP, where their pools overlap.
const TOGETHER := {"materials": {"wax": {"kind": "wax", "color": "#e8dcc0"}}, "things": [
	{"name": "a tall pillar", "place": "back left", "group": "c", "parts": [{"shape": "lathe", "profile": [[0, 0], [2.4, 0], [2.4, 9], [0, 9]], "material": "wax", "wick": true}]},
	{"name": "a pillar", "place": "back left", "group": "c", "parts": [{"shape": "lathe", "profile": [[0, 0], [2.4, 0], [2.4, 7], [0, 7]], "material": "wax", "wick": true}]},
	{"name": "a short pillar", "place": "back left", "group": "c", "parts": [{"shape": "lathe", "profile": [[0, 0], [2.4, 0], [2.4, 5], [0, 5]], "material": "wax", "wick": true}]},
	]}

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

	print("a set table: things in groups, candles among them")
	var set := _sweep("dark", 2, SET_TABLE)
	var wanted := (SET_TABLE["things"] as Array).size() * SEEDS
	_ok(set["things"] >= wanted * 0.9, "nearly every thing finds room (%d of %d)" % [set["things"], wanted])
	_ok(set["flames"] == 3 * SEEDS, "every wick burns (%d of %d)" % [set["flames"], 3 * SEEDS])
	_ok(set["candles"] == 2 * SEEDS, "each lit thing is one light, the pair of pillars too (%d of %d)" % [set["candles"], 2 * SEEDS])
	_ok(set["off_cloth"] == 0 and set["on_cards"] == 0, "all on the cloth, none where the cards go (%d off, %d on)" % [set["off_cloth"], set["on_cards"]])
	_ok(set["touching"] == 0, "none standing in another (%d)" % set["touching"])
	_ok(set["out"] == 0 and set["overlap"] == 0, "all wholly in the frame, no group in front of another (%d, %d)" % [set["out"], set["overlap"]])
	_ok(set["front"] == 0, "no lit thing in front of the middle (%d)" % set["front"])
	_ok(set["own_shadow"] == 0 and set["no_shadow"] == 0, "every thing in every flame's shadows but its own (%d own, %d left out)"
		% [set["own_shadow"], set["no_shadow"]])

	print("candles standing together")
	var tg := _sweep("half", 0, TOGETHER)
	_ok(tg["candles"] == 3 * SEEDS, "the three stand (%d of %d)" % [tg["candles"], 3 * SEEDS])
	_ok(tg["over_joint"] == 0, "the cloth under all of them at once is never hotter than it takes (%d tables over)" % tg["over_joint"])
	_ok(tg["alone_over"] > SEEDS / 2, "control: each at its own limit, candles together overheat it (%d of %d tables)" % [tg["alone_over"], SEEDS])

	print("cards go round what stands")
	var w := _cards_go_round(SET_TABLE)
	_ok(w["near"] > 0, "the washes bring cards up to the things (%d samples within 3 cm)" % w["near"])
	_ok(w["through"] == 0, "no card passes through a thing (%d of %d samples)" % [w["through"], w["samples"]])
	_ok(w["through_off"] > 0, "control: let through, cards cross the things (%d samples)" % w["through_off"])
	_ok(w["home"] == w["cards"], "every card still ends in the deck (%d of %d)" % [w["home"], w["cards"]])

	print("the instrument")
	var spot := Vector3(0.25, 0.0, -0.15)
	_ok(not _inside(_box_rect(Vector3(0.0, 0.0, -0.2), 0.8, 0.1)), "a thing wider than the frame is out of it")
	_ok(_box_rect(spot, 0.02, 0.1).intersects(_box_rect(spot + Vector3(0.01, 0, 0), 0.02, 0.1)),
		"two things on one spot overlap in the picture")

	print("how bright a candle may be")
	_ok(a["over_cap"] == 0 and b["over_cap"] == 0, "no candle brighter than its cloth allows (%d, %d)" % [a["over_cap"], b["over_cap"]])
	_ok(a["bad_key"] == 0 and b["bad_key"] == 0, "the key stands where its cloth can take it (%d, %d)" % [a["bad_key"], b["bad_key"]])
	_ok(a["casters"] == 0 and b["casters"] == 0, "every candle and the lamp throw shadows (%d, %d tables not)" % [a["casters"], b["casters"]])
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
	_ok(st["heat_at"] < st["heat_aim"] * 0.8 and SEEDS - st["lamp_keys"] >= int(SEEDS * 0.75), "candles keep off the stripe: the cloth by them %.2f, at their zones' middles %.2f (a candle the key %d/%d)"
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

	print("the room past the table")
	_backdrop()
	print("the outro")
	_outro()

	Director.hold(false)
	Director.detach()
	_clear()
	print("tarot_place_check: %s" % ("ALL OK" if _fails == 0 else "%d FAILED" % _fails))
	get_tree().quit(0 if _fails == 0 else 1)


## Build the table for [constant SEEDS] seeds on the [param cloth] fixture and count. Its things
## are [param table] (a set dresser's reply), or the look's [param candles] alone when empty.
func _sweep(cloth: String, candles: int, table := {}) -> Dictionary:
	var dir := DIR.path_join(cloth)
	var tf := dir.path_join("table.json")
	if not table.is_empty():
		var f := FileAccess.open(tf, FileAccess.WRITE)
		f.store_string(JSON.stringify(table))
		f.close()
	elif FileAccess.file_exists(tf):
		DirAccess.remove_absolute(ProjectSettings.globalize_path(tf))
	var cards: Array = []
	for i in 3:
		cards.append({"key": "c%d" % i, "name": "Card %d" % i, "numeral": str(i), "reversed": false,
			"jumper": false, "position": {}, "booklet": {}, "art": ""})
	var n := {"candles": 0, "flames": 0, "front": 0, "out": 0, "overlap": 0, "over_cap": 0, "bad_key": 0,
		"casters": 0, "old_over": 0, "lamp_keys": 0, "full_keys": 0, "heat_at": 0.0, "heat_aim": 0.0,
		"own_shadow": 0, "no_shadow": 0, "things": 0, "off_cloth": 0, "on_cards": 0, "touching": 0,
		"over_joint": 0, "alone_over": 0, "behind": 0, "lit_things": 0}
	for s in range(1, SEEDS + 1):
		var doc := {"show": "place-check", "seed": s, "dir": dir, "images": {"surface": dir.path_join("surface.png")},
			"plan": {"look": {"candles": candles}}, "cards": cards}
		subs.document = {"source": _script, "title": "Place Check", "tarot": doc}
		medium._key = ""
		medium._ensure_doc()
		var things: Array = medium._things
		n["things"] += things.size()
		for th in things:
			var t: Dictionary = th
			var node: Node3D = t["node"]
			var r: Rect2 = t["rect"]
			n["out"] += 0 if _inside(r) else 1
			if int(t["lit"]) > 0:
				n["lit_things"] += 1
				n["front"] += 1 if node.position.z > medium._mid.z + 0.001 else 0
			for q in (t["outline"] as PackedVector2Array):
				if absf(q.x) > TarotMedium.CLOTH.x * 0.5 or q.y < -0.02 - TarotMedium.CLOTH.y * 0.5 or q.y > -0.02 + TarotMedium.CLOTH.y * 0.5:
					n["off_cloth"] += 1
					break
			for k in medium._keep_out():
				if TarotMedium._convex_overlap(t["outline"], TarotMedium._rect_poly(k as Rect2), 0.0):
					n["on_cards"] += 1
		for i in things.size():
			for j in range(i + 1, things.size()):
				var a: Dictionary = things[i]
				var b: Dictionary = things[j]
				if TarotMedium._convex_overlap(a["outline"], b["outline"], 0.0):
					n["touching"] += 1
				if String(a["group"]) != String(b["group"]) and (a["rect"] as Rect2).intersects(b["rect"] as Rect2):
					n["overlap"] += 1
		n["candles"] += medium._lights.size()
		for l in medium._lights:
			n["flames"] += ((l as Dictionary)["flames"] as Array).size()
		# the light: each candle's thing is left out of its own light's shadows, and only its own
		for th in things:
			var t: Dictionary = th
			for i in medium._lights.size():
				var mask: int = ((medium._lights[i] as Dictionary)["light"] as OmniLight3D).shadow_caster_mask
				for m in t["meshes"]:
					var cast := (mask & (m as MeshInstance3D).layers) != 0
					if (t["lights"] as Array).has(i) and cast:
						n["own_shadow"] += 1
					elif not (t["lights"] as Array).has(i) and not cast:
						n["no_shadow"] += 1
		# how pale the cloth is by each candle, against at the spot its zone aims for
		for th in things:
			var t: Dictionary = th
			if int(t["lit"]) == 0:
				continue
			var p: Vector3 = (t["node"] as Node3D).position
			var aim := TarotTable.zone_aim(String(t["place"]), medium._deck_base)
			n["heat_at"] += medium._heat(Vector3(p.x, 0.0, p.z), TarotMedium.HEAT_H)
			n["heat_aim"] += medium._heat(Vector3(aim.x, 0.0, aim.y), TarotMedium.HEAT_H)
		# THE CLOTH UNDER ALL THE FLAMES AT ONCE, cell by cell - and, as the control, under each alone at
		# its own limit, the rule before candles stood together: the key (nearest the middle of those
		# whose cloth can take it) at full light, the rest at the fill, each capped by its own cloth
		var total := {}
		var alone := {}
		var fields: Array = []
		var old_key := -1
		var old_best := INF
		for i in medium._lights.size():
			var fd: Dictionary = medium._lights[i]
			var base: Vector3 = fd["base"]
			var lb: Vector3 = fd["light_base"]
			var field := medium._heat_field(Vector3(base.x, 0.0, base.z), lb.y)
			fields.append(field)
			var d := (base * Vector3(1, 0, 1)).length()
			if TarotMedium.HEAT / maxf(TarotMedium._field_max(field), 0.05) >= TarotMedium.KEY_MIN and d < old_best:
				old_best = d
				old_key = i
		for i in medium._lights.size():
			var field: Dictionary = fields[i]
			var own_cap := minf(TarotMedium.KEY_ENERGY if i == old_key else TarotMedium.FILL_ENERGY,
				TarotMedium.HEAT / maxf(TarotMedium._field_max(field), 0.05))
			for c in field:
				total[c] = float(total.get(c, 0.0)) + float((medium._lights[i] as Dictionary)["energy"]) * float(field[c])
				alone[c] = float(alone.get(c, 0.0)) + own_cap * float(field[c])
		for c in total:
			if float(total[c]) > TarotMedium.HEAT + 0.001:
				n["over_joint"] += 1
				break
		for c in alone:
			if float(alone[c]) > TarotMedium.HEAT + 0.001:
				n["alone_over"] += 1
				break
		var casters := 1 if medium._lamp.shadow_enabled else 0
		var nearest := -1
		var best := INF
		for i in medium._lights.size():
			var f: Dictionary = medium._lights[i]
			var base: Vector3 = f["base"]
			var lb: Vector3 = f["light_base"]
			var cap := TarotMedium.HEAT / maxf(medium._heat(Vector3(base.x, 0.0, base.z), lb.y), 0.05)
			if float(f["energy"]) > cap + 0.0001:
				n["over_cap"] += 1
			var light: OmniLight3D = f["light"]
			if light.shadow_enabled:
				casters += 1
			if i == medium._key_flame:
				if float(f["energy"]) < TarotMedium.KEY_MIN - 0.0001:
					n["bad_key"] += 1
				if absf(float(f["energy"]) - TarotMedium.KEY_ENERGY * float((f["flames"] as Array).size())) < 0.0001:
					n["full_keys"] += 1
			var d := (base * Vector3(1, 0, 1)).length()
			if d < best:
				best = d
				nearest = i
		if casters != medium._lights.size() + 1:
			n["casters"] += 1
		if medium._key_flame < 0:
			n["lamp_keys"] += 1
		if nearest >= 0:
			var nb: Vector3 = (medium._lights[nearest] as Dictionary)["base"]
			var nl: Vector3 = (medium._lights[nearest] as Dictionary)["light_base"]
			if TarotMedium.KEY_ENERGY > TarotMedium.HEAT / maxf(medium._heat(Vector3(nb.x, 0.0, nb.z), nl.y), 0.05):
				n["old_over"] += 1
	if FileAccess.file_exists(tf):
		DirAccess.remove_absolute(ProjectSettings.globalize_path(tf))
	n["heat_at"] = float(n["heat_at"]) / maxf(float(n["lit_things"]), 1.0)
	n["heat_aim"] = float(n["heat_aim"]) / maxf(float(n["lit_things"]), 1.0)
	return n


## NOTHING PASSES THROUGH WHAT STANDS ON THE TABLE. Washes planned over [constant SEEDS] tables of
## [param table]'s things: every card at every sample, its real turned rectangle against every
## thing's foot - none may cross one (a hair of tolerance), and every card still ends in the deck.
## The control is the same washes with the cards let through: they must cross, or this fixture never
## put a card near a thing and the check holds nothing.
func _cards_go_round(table: Dictionary) -> Dictionary:
	var dir := DIR.path_join("dark")
	var tf := dir.path_join("table.json")
	var f := FileAccess.open(tf, FileAccess.WRITE)
	f.store_string(JSON.stringify(table))
	f.close()
	var out := {"samples": 0, "through": 0, "through_off": 0, "home": 0, "cards": 0, "near": 0}
	for s in range(1, SEEDS + 1):
		var doc := {"show": "place-check", "seed": s, "dir": dir, "plan": {"look": {"candles": 2}}, "cards": []}
		subs.document = {"source": _script, "title": "Place Check", "tarot": doc}
		medium._key = ""
		medium._ensure_doc()
		for collide in [true, false]:
			medium._collide = collide
			var plan := medium._wash_plan(s * 7919, 14.0)
			medium._collide = true
			var tracks: Array = plan["tracks"]
			for i in tracks.size():
				var tr: PackedVector4Array = PackedVector4Array(tracks[i] as Array) if tracks[i] is Array else tracks[i]
				for q in tr:
					var at := Vector2(q.x + medium._mid.x, q.z + medium._mid.z)
					var card := TarotMedium._card_poly(at, q.w)
					var hit := false
					for st in medium._standing:
						# the foot as it stands, without the margin a card keeps from it, less a hair
						if _sat_strict(card, TarotMedium._grown(st as PackedVector2Array, -TarotMedium.FOOT_MARGIN - 0.001)):
							hit = true
							break
					if collide:
						out["samples"] += 1
						out["through"] += 1 if hit else 0
						for st in medium._standing:
							if TarotMedium._convex_overlap(card, st as PackedVector2Array, 0.03):
								out["near"] += 1
								break
					else:
						out["through_off"] += 1 if hit else 0
				if collide:
					out["cards"] += 1
					var last: Vector4 = tr[tr.size() - 1]
					var slot: Vector3 = medium._slot_jit[(plan["order"] as Array).find(i)] if (plan["order"] as Array).find(i) >= 0 else medium._slot_jit[i]
					out["home"] += 1 if Vector2(last.x, last.z).distance_to(Vector2(slot.x, slot.y)) < 0.002 else 0
	DirAccess.remove_absolute(ProjectSettings.globalize_path(tf))
	return out


## Two convex shapes truly overlapping (no gap at all on any axis).
func _sat_strict(a: PackedVector2Array, b: PackedVector2Array) -> bool:
	for poly in [a, b]:
		var p: PackedVector2Array = poly
		for i in p.size():
			var e := p[(i + 1) % p.size()] - p[i]
			if e.length_squared() < 1e-12:
				continue
			var ax := Vector2(-e.y, e.x).normalized()
			var ra := TarotMedium._span(a, ax)
			var rb := TarotMedium._span(b, ax)
			if ra.x >= rb.y or rb.x >= ra.y:
				return false
	return true


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
		var flames: Array = []
		for l in medium._lights:
			flames.append_array((l as Dictionary)["flames"])
		for f in flames:
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


## THE OUTRO, on a reading whose word times are known: the table's brightness from the end fade,
## and the title's from its own curve, sampled across the whole reading.
## THE ROOM IS SEEN FROM THE READER'S EYE: a picture asked for level (`backdrop.json` beside it)
## stands on an upright plane straight ahead of the camera, its middle at the eye's height and as
## wide as its lens saw - two-sided: with no view beside it, the picture made before keeps its old
## place, square to the tilted camera.
func _backdrop() -> void:
	var dir := DIR.path_join("dark")
	var vf := dir.path_join("backdrop.json")
	var level := 0
	var old := 0
	for s in range(1, 9):
		for v in [true, false]:
			if v:
				var f := FileAccess.open(vf, FileAccess.WRITE)
				f.store_string(JSON.stringify({"view": "level", "lens_mm": 20}))
				f.close()
			elif FileAccess.file_exists(vf):
				DirAccess.remove_absolute(ProjectSettings.globalize_path(vf))
			var doc := {"show": "place-check", "seed": s, "dir": dir, "plan": {"look": {"candles": 1}}, "cards": []}
			subs.document = {"source": _script, "title": "Place Check", "tarot": doc}
			medium._key = ""
			medium._ensure_doc()
			var xf := medium._backdrop.global_transform
			var eye := medium._cam_base.origin
			var size := (medium._backdrop.mesh as QuadMesh).size
			var ahead := -medium._cam_base.basis.z
			if v:
				var upright := absf(xf.basis.y.normalized().dot(Vector3.UP) - 1.0) < 0.001
				var at_eye := absf(xf.origin.y - eye.y) < 0.001
				var facing := xf.basis.z.normalized().dot(-Vector3(ahead.x, 0.0, ahead.z).normalized()) > 0.999
				var lens := size.distance_to(Vector2(36.0, 24.0) * 3.2 / 20.0) < 0.001
				level += 1 if upright and at_eye and facing and lens else 0
			else:
				old += 1 if xf.basis.y.normalized().dot(medium._cam_base.basis.y.normalized()) > 0.999 else 0
	if FileAccess.file_exists(vf):
		DirAccess.remove_absolute(ProjectSettings.globalize_path(vf))
	_ok(level == 8, "a level picture stands upright at the eye, straight ahead, as wide as its lens (%d of 8)" % level)
	_ok(old == 8, "a picture made before keeps its place, square to the camera (%d of 8)" % old)


func _outro() -> void:
	var outro_was := Director.outro_hold
	Director.outro_hold = 6.0
	var cards: Array = []
	for i in 3:
		cards.append({"key": "c%d" % i, "name": "Card %d" % i, "numeral": str(i), "reversed": false,
			"jumper": false, "position": {}, "booklet": {}, "art": ""})
	var doc := {"show": "place-check", "seed": 7, "dir": DIR.path_join("dark"), "plan": {"look": {"candles": 2}}, "cards": cards}
	subs.words = load("res://tests/tarot_look_probe.gd").timeline(TarotScript.parse(_script), 0.36, Director.intro_hold)
	subs.document = {"source": _script, "title": "Place Check", "tarot": doc}
	medium._ensure_doc()
	medium._follow.extend(subs.words)
	medium._sched = medium._follow.place(medium._parse["actions"], maxf(Director.intro_hold, 0.6), TarotMedium.LEAD, TarotMedium.TAIL)
	var spoken: PackedStringArray = medium._parse["spoken"]
	var last := medium._follow.known_last()
	_ok(last == spoken.size() - 1, "the last word's time is known (%d of %d)" % [last + 1, spoken.size()])
	var end_t: float = medium._follow.st1[last]
	_ok(medium._end_fade(end_t - 1.0) == 1.0 and medium._end_fade(end_t + 0.1) == 1.0,
		"lit while the voice speaks, and for a beat after its last word")
	var mid := medium._end_fade(end_t + 3.0)
	_ok(mid > 0.05 and mid < 0.95, "fading through the outro (%.2f halfway)" % mid)
	_ok(medium._end_fade(end_t + 6.01) < 0.001, "black as the outro runs out (%.3f)" % medium._end_fade(end_t + 6.01))
	var ts := float(medium._times()["shuffle"])
	_ok(ts < INF and medium._title_alpha(ts - 1.0) > 0.5, "the channel's name is up over the intro (%.2f)" % medium._title_alpha(ts - 1.0))
	var shown := 0
	var t := ts + 1.0
	while t < end_t + 9.0:
		shown += 1 if medium._title_alpha(t) > 0.001 else 0
		t += 0.25
	_ok(shown == 0, "and never again, the end included (%d moments)" % shown)
	Director.outro_hold = outro_was


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


## The picture's rectangle for a box [param r] wide each way of its middle and [param h] tall at
## [param at], as the table's search measures it.
func _box_rect(at: Vector3, r: float, h: float) -> Rect2:
	var corners := PackedVector3Array()
	for cx in [-r, r]:
		for cy in [0.0, h]:
			for cz in [-r, r]:
				corners.append(Vector3(cx, cy, cz))
	var k := tan(deg_to_rad(medium._cam.fov * 0.5))
	return TarotMedium._screen_rect_fast(corners, at, medium._cam_base.affine_inverse(), Vector2(k * 16.0 / 9.0, k))


func _inside(r: Rect2) -> bool:
	return r.position.x >= 0.0 and r.position.y >= 0.0 and r.end.x <= 1.0 and r.end.y <= 1.0


## Four cloths - half felt and half pale boards (as the episode that found the flood), all pale,
## all dark, and a near-black blanket with a cream stripe toward its far edge (as the one that blew
## out behind its candles). Painted landscape, as the painters make a cloth now, and laid out in
## the CLOTH's own fractions through [method TarotTable.cloth_crop] - the stripe was once drawn in
## the picture's top rows, which the cloth no longer shows since its pixels are kept square.
func _fixture() -> void:
	_clear()
	var felt := Color(0.36, 0.5, 0.14)     # the episode's felt: linear luminance 0.18
	var pine := Color(0.92, 0.86, 0.74)    # ...and its boards: 0.7
	var deep := Color(0.3, 0.1, 0.12)      # an oxblood cloth: 0.02
	var spruce := Color(0.22, 0.26, 0.2)   # the blanket: 0.045, with a cream stripe (0.46) near its far edge
	var cream := Color(0.74, 0.7, 0.6)
	var size := Vector2i(384, 256)
	var crop := TarotTable.cloth_crop(Vector2(size), TarotMedium.CLOTH)
	for cloth in ["half", "pale", "dark", "stripe"]:
		var dir := DIR.path_join(cloth)
		DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(dir))
		var surf := Image.create(size.x, size.y, false, Image.FORMAT_RGB8)
		for y in size.y:
			for x in size.x:
				# where this pixel lies on the cloth, 0..1 across and from the far edge
				var on := (Vector2(x + 0.5, y + 0.5) / Vector2(size) - crop.position) / crop.size
				var c := deep if cloth == "dark" else (felt if cloth == "half" and on.x < 0.5 else pine)
				if cloth == "stripe":
					c = cream if on.y >= 0.0625 and on.y < 0.2 else spruce
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
