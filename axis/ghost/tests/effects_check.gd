extends SceneTree

## The gate of [Effects] - the air: fog, motes and bursts - with no renderer.
##
##   godot --headless --path . --script res://tests/effects_check.gd
##
## - WHATEVER IS WRITTEN BUILDS: unknown kinds, looks, places and moments are dropped, and so is air
##   past its caps (effects, fogs, motes in all); sizes stay in their look's range, colors are
##   colors, and a look's own color stands in for none.
## - THE VOCABULARY names every kind, look, place and moment, with each look's sizes.
## - A FUNCTION OF SHOW TIME: a mote's place and brightness are the same whenever asked; a mote that
##   is away is where it went (out of the picture); a firefly blinks; an ember rises through its
##   region and round again; nothing sinks under its region's floor.
## - BURSTS are born the same way from the same moments - within their moment, most at its start, at
##   the emitter - and a moved moment is born again; one not moved is left alone.
## - A MOTE'S HOME IS SEEN: in the picture and in front of the stage's occluders. Two-sided: homes drawn
##   from the region's volume alone are hidden behind the table often.
## - NOTHING FLIES THROUGH WHAT STANDS: a mote over the table or a thing on it stays above its top - a lit
##   one by LIT_CLEAR, where its light at full glow burns no spot that blooms - and the floor lifting it
##   over a thing never makes it jump. Two-sided: with nothing under them (the old floor) motes go into
##   the table.
## - BUILT: a fog volume per fog, a sprite per mote, no more lights than the cap.

var _fails := 0


func _init() -> void:
	_run.call_deferred()


func _ok(cond: bool, what: String) -> void:
	if not cond:
		_fails += 1
		print("  FAIL: " + what)


## A stage like the tarot table's: a camera looking down across a table, the air's regions round it.
func _stage() -> Dictionary:
	var lay := TarotTable.layout_of(1234)
	return {"regions": TarotTable.AIR, "camera": lay["camera"], "fov": float(lay["fov"]), "aspect": 16.0 / 9.0,
		"occluders": [AABB(Vector3(-0.95, -0.05, -0.405), Vector3(1.9, 0.05, 0.85))]}


func _run() -> void:
	for check in [_sanitize, _vocabulary, _motes, _bursts, _homes, _floors, _built]:
		var done: Variant = await (check as Callable).call()
		_ok(done == true, "%s stopped part way (a script error - see above)" % (check as Callable).get_method())
	print("effects_check: %s (%d failure%s)" % ["ALL OK" if _fails == 0 else "FAILED", _fails, "" if _fails == 1 else "s"])
	quit(1 if _fails > 0 else 0)


func _clean(raw: Array) -> Array:
	return Effects.sanitize(raw, ["#102030", "#aabbcc"], TarotTable.AIR.keys(), TarotTable.MOMENTS.keys())


func _sanitize() -> bool:
	var junk := _clean([{"kind": "weather"}, "not an effect", {"kind": "fog", "where": "the ceiling"},
		{"kind": "motes", "look": "dragons", "where": "over the cloth"}, {"kind": "burst", "look": "sparks", "on": "the end"},
		{"kind": "FOG", "where": "Over The Cloth", "density": 9, "color": "blue"}])
	_ok(junk.size() == 1 and String(junk[0]["where"]) == "over the cloth" and float(junk[0]["density"]) == 1.0
		and String(junk[0]["color"]) == "#aabbcc", "junk was not dropped, or a fog not made safe: %s" % str(junk))
	var many: Array = []
	for i in 6:
		many.append({"kind": "fog", "where": "over the cloth", "name": "fog %d" % i})
	var fogs := _clean(many)
	_ok(fogs.size() == Effects.MAX_FOG, "%d fogs were kept, past the cap of %d" % [fogs.size(), Effects.MAX_FOG])
	var crowd: Array = []
	for i in 12:
		crowd.append({"kind": "motes", "look": "dust", "where": "over the cloth", "count": 40})
	var motes := 0
	var kept := _clean(crowd)
	for m in kept:
		motes += int(m["count"])
	_ok(kept.size() <= Effects.MAX_EFFECTS and motes <= Effects.MAX_MOTES_ALL, "%d effects with %d motes were kept" % [kept.size(), motes])
	var sized := _clean([{"kind": "motes", "look": "pixie", "where": "over the cloth", "size": 300, "glow": -2, "colors": ["red", "#ff00ff"]},
		{"kind": "burst", "look": "smoke", "on": "jumper", "size": 0.01}])
	var pix: Dictionary = sized[0]
	_ok(float(pix["size"]) == (Effects.MOTES["pixie"]["sizes"] as Vector2).y and float(pix["glow"]) == 0.0 and pix["colors"] == ["#ff00ff"],
		"a pixie's size, glow or colors were not kept within its range: %s" % str(pix))
	_ok(float(sized[1]["size"]) == (Effects.BURSTS["smoke"]["sizes"] as Vector2).x and sized[1]["colors"] == [Effects.BURSTS["smoke"]["color"]],
		"a burst's size was not kept within its look's range, or its own color did not stand in: %s" % str(sized[1]))
	return true


func _vocabulary() -> bool:
	var text := Effects.describe({"here": "a place"}, {"now": "a moment"})
	for k in Effects.KINDS.keys() + Effects.MOTES.keys() + Effects.BURSTS.keys():
		_ok(text.contains("- %s:" % k), "the vocabulary does not name %s" % k)
	_ok(text.contains("- \"here\": a place") and text.contains("- \"now\": a moment"), "the vocabulary does not name the host's places and moments")
	_ok(text.contains("pixie: ") and text.contains("(3 mm, 1.5 to 8)"), "the vocabulary does not give a pixie's sizes")
	return true


func _motes() -> bool:
	var stage := _stage()
	var root := Node3D.new()
	var air := Effects.build(_clean([{"kind": "motes", "look": "pixie", "where": "beyond the table", "count": 12, "away": 0.5},
		{"kind": "motes", "look": "firefly", "where": "over the cloth", "count": 6},
		{"kind": "motes", "look": "ember", "where": "over the cloth", "count": 6}]), stage, 99)
	root.add_child(air.root)
	var pix: Dictionary = air.motes[0]
	var same := true
	var away_out := 0
	var aways := 0
	for m in pix["each"]:
		for t in [0.0, 13.7, 61.2]:
			var a := Effects.mote_at(pix, m, t)
			var b := Effects.mote_at(pix, m, t)
			same = same and a["pos"] == b["pos"] and a["bright"] == b["bright"]
			if float(a["presence"]) == 0.0:
				aways += 1
				var s: Variant = Effects._screen(stage, a["pos"])
				away_out += 1 if s == null or (s as Vector2).x < 0.0 or (s as Vector2).x > 1.0 or (s as Vector2).y < 0.0 or (s as Vector2).y > 1.0 else 0
	_ok(same, "a mote is not the same whenever it is asked for at one time")
	_ok(aways > 0 and away_out == aways, "a mote that is away is in the picture (%d of %d)" % [aways - away_out, aways])
	var ff: Dictionary = air.motes[1]
	var lo := INF
	var hi := -INF
	for i in 400:
		var b := float(Effects.mote_at(ff, ff["each"][0], float(i) * 0.05)["bright"])
		lo = minf(lo, b)
		hi = maxf(hi, b)
	_ok(hi > 0.8 and lo < 0.15, "a firefly does not blink (%.2f to %.2f)" % [lo, hi])
	var em: Dictionary = air.motes[2]
	var box: AABB = em["box"]
	var ys := PackedFloat32Array()
	for i in 600:
		ys.append((Effects.mote_at(em, em["each"][0], float(i) * 0.1)["pos"] as Vector3).y)
	var rises := 0
	for i in range(1, ys.size()):
		rises += 1 if ys[i] > ys[i - 1] else 0
	_ok(rises > ys.size() * 0.7 and ys[ys.size() - 1] <= box.end.y + 1e-3, "an ember does not rise through its region and round (%d of %d steps up)" % [rises, ys.size()])
	var under := 0
	for pop in air.motes:
		var b2: AABB = (pop as Dictionary)["box"]
		for m in (pop as Dictionary)["each"]:
			for i in 50:
				under += 1 if (Effects.mote_at(pop, m, float(i) * 1.7)["pos"] as Vector3).y < b2.position.y - 1e-4 else 0
	_ok(under == 0, "%d mote places were under their region's floor" % under)
	root.free()
	return true


func _bursts() -> bool:
	var fx: Dictionary = _clean([{"kind": "burst", "look": "sparks", "on": "jumper", "count": 80}])[0]
	var path := [[10.0, Transform3D(Basis.IDENTITY, Vector3(0.0, 0.05, 0.0))], [11.0, Transform3D(Basis.IDENTITY, Vector3(0.2, 0.05, 0.0))]]
	var moments := [{"t": 10.0, "dur": 1.0, "from": "card", "path": path}]
	var a := Effects.births(fx, moments, 7)
	var b := Effects.births(fx, moments, 7)
	_ok(a.size() == 80 and str(a) == str(b), "a burst is not born the same way twice (%d particles)" % a.size())
	var inside := 0
	var early := 0
	var near := 0
	for p in a:
		var born := float((p as Dictionary)["born"])
		inside += 1 if born >= 10.0 and born <= 11.0 else 0
		early += 1 if born < 10.3 else 0
		var at := Effects._along(path, born)
		near += 1 if ((p as Dictionary)["pos"] as Vector3).distance_to(at.origin) < 0.08 else 0
	_ok(inside == a.size(), "%d of %d particles were born outside their moment" % [a.size() - inside, a.size()])
	_ok(early > a.size() / 2, "a burst does not throw most of itself at its start (%d of %d in the first 30%%)" % [early, a.size()])
	_ok(near == a.size(), "%d of %d particles were born away from the emitter" % [a.size() - near, a.size()])
	var moved := Effects.births(fx, [{"t": 12.0, "dur": 1.0, "from": "card", "path": path}], 7)
	_ok(str(moved) != str(a), "a moved moment was born the same")
	# PLANNED ONCE PER MOMENT: a burst whose moments did not move is not born again
	var root := Node3D.new()
	var air := Effects.build([fx], _stage(), 5)
	root.add_child(air.root)
	air.plan({"jumper": moments})
	var mm: MultiMesh = ((air.bursts[0] as Dictionary)["layer"] as Dictionary)["mm"]
	air.plan({"jumper": moments})
	_ok(int((air.bursts[0] as Dictionary)["plans"]) == 1 and mm.instance_count == 80, "an unmoved moment was planned again")
	air.plan({"jumper": [{"t": 20.0, "dur": 1.0, "from": "card", "path": path}]})
	_ok(int((air.bursts[0] as Dictionary)["plans"]) == 2, "a moved moment was not planned again")
	root.free()
	return true


func _homes() -> bool:
	var stage := _stage()
	var rng := RandomNumberGenerator.new()
	rng.seed = 11
	var box: AABB = (TarotTable.AIR["beyond the table"] as Dictionary)["motes"]
	var seen := 0
	var hidden := 0
	for i in 200:
		var p := Effects._in_view(box, stage, rng)
		var s: Variant = Effects._screen(stage, p)
		seen += 1 if s != null and (s as Vector2).x >= 0.0 and (s as Vector2).x <= 1.0 and (s as Vector2).y >= 0.0 and (s as Vector2).y <= 1.0 else 0
		hidden += 1 if Effects._hidden(stage, p) else 0
	_ok(seen == 200 and hidden == 0, "mote homes were out of the picture (%d) or behind the table (%d)" % [200 - seen, hidden])
	var naive := 0
	for i in 200:
		var p := box.position + Vector3(rng.randf(), rng.randf(), rng.randf()) * box.size
		naive += 1 if Effects._hidden(stage, p) else 0
	_ok(naive > 20, "control: homes drawn from the volume alone are hidden as seldom as %d of 200" % naive)
	var span := Effects._ray_box(Vector3(0, 0, 5), Vector3(0, 0, -1), AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2)))
	_ok(is_equal_approx(span.x, 4.0) and is_equal_approx(span.y, 6.0), "a ray through a box was measured as %s" % str(span))
	return true


func _floors() -> bool:
	var stage := _stage()
	var table: AABB = stage["occluders"][0]
	(stage["occluders"] as Array).append(AABB(Vector3(-0.25, 0.0, -0.39), Vector3(0.12, 0.22, 0.12)))
	var root := Node3D.new()
	var air := Effects.build(_clean([{"kind": "motes", "look": "pixie", "where": "beyond the table", "count": 24, "glow": 1.0},
		{"kind": "motes", "look": "firefly", "where": "over the cloth", "count": 12},
		{"kind": "motes", "look": "dust", "where": "over the cloth", "count": 12}]), stage, 41)
	root.add_child(air.root)
	var into := 0
	var low := 0
	var bare_into := 0
	var step := 0.0
	var bare_step := 0.0
	for pop in air.motes:
		var lit := bool(((pop as Dictionary)["fx"] as Dictionary)["light"])
		var bare := (pop as Dictionary).duplicate()
		bare["under"] = []
		for m in (pop as Dictionary)["each"]:
			var last := Vector3.INF
			var bare_last := Vector3.INF
			for i in 2400:
				var t := float(i) / 30.0
				var p: Vector3 = Effects.mote_at(pop, m, t)["pos"]
				var q: Vector3 = Effects.mote_at(bare, m, t)["pos"]
				for o in stage["occluders"]:
					var b: AABB = o
					if p.x > b.position.x and p.x < b.end.x and p.z > b.position.z and p.z < b.end.z:
						into += 1 if p.y < b.end.y else 0
						low += 1 if lit and p.y < b.end.y + Effects.LIT_CLEAR - 1e-4 else 0
				bare_into += 1 if table.has_point(q) else 0
				if i > 0:
					step = maxf(step, p.distance_to(last))
					bare_step = maxf(bare_step, q.distance_to(bare_last))
				last = p
				bare_last = q
	_ok(into == 0 and low == 0, "motes went into what stands (%d) or a lit one flew low over it (%d)" % [into, low])
	_ok(bare_into > 0, "control: with nothing under them motes never went into the table")
	print("  a frame's longest step: %.1f cm with the floor, %.1f without" % [step * 100.0, bare_step * 100.0])
	_ok(step < bare_step * 1.5, "the floor makes a mote jump: %.1f cm in a frame against %.1f without it" % [step * 100.0, bare_step * 100.0])
	# THE LIGHT AT ITS CLOSEST: at full glow, LIT_CLEAR off a pale surface, it burns no spot that blooms
	var glare := 0.0
	var fog := INF
	for i in 40:
		air.tick(float(i) * 1.3)
		for l in (air.motes[0] as Dictionary)["lights"]:
			var light: OmniLight3D = (l as Dictionary)["light"]
			var d := Effects.LIT_CLEAR
			var fall := pow(maxf(1.0 - pow(d / light.omni_range, 4.0), 0.0), 2.0) * pow(d, -light.omni_attenuation)
			glare = maxf(glare, light.light_energy * fall * 0.84 / PI)
			fog = minf(fog, light.light_volumetric_fog_energy)
	_ok(glare > 0.0 and glare < 1.0, "a mote's light %.0f mm off a pale surface lights it to %.2f (blooms past 1.25)" % [Effects.LIT_CLEAR * 1000.0, glare])
	_ok(fog >= 10.0, "a mote's light is not mostly the fog's (%.1f times a surface's)" % fog)
	root.free()
	return true


func _built() -> bool:
	var root := Node3D.new()
	var air := Effects.build(_clean([{"kind": "fog", "where": "beyond the table"}, {"kind": "fog", "where": "low on the cloth"},
		{"kind": "motes", "look": "pixie", "where": "over the cloth", "count": 30, "light": true},
		{"kind": "motes", "look": "wisp", "where": "beyond the table", "count": 20, "light": true}]), _stage(), 3)
	root.add_child(air.root)
	var vols := 0
	var lights := 0
	for c in air.root.get_children():
		vols += 1 if c is FogVolume else 0
		lights += 1 if c is OmniLight3D else 0
	_ok(vols == 2 and air.has_fog(), "%d fog volumes were built for 2 fogs" % vols)
	_ok(lights == Effects.MAX_LIGHTS, "%d mote lights were made, the cap being %d" % [lights, Effects.MAX_LIGHTS])
	_ok(((air.motes[0] as Dictionary)["layer"]["mm"] as MultiMesh).instance_count == 30, "a mote population was not one sprite each")
	air.tick(12.5)
	root.free()
	return true
