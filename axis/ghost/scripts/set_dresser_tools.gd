extends RefCounted
class_name SetDresserTools

## SetDresserTools - what the set dresser can do while it sets a tarot reader's table: MAKE a thing,
## LOOK at it, FIX it, and see the whole table as the camera will before handing it in.
##
## Written as one reply, a table was ~5,000 tokens of geometry its author never saw: a squid of balls
## and rods that read as a giant bug, stones perched on stones, things left off the table for want of
## room with nobody told. Served as tools ([AgentTools]) instead, the set dresser keeps a DRAFT here
## and works on it:
##
##   put     things (and their materials) onto the draft - a thing of the same name is replaced -
##           answered with what was built: each thing's size, anything the builder had to leave out or
##           guess, whether it is taller than its place can show, and a picture of what was put
##   remove  things off it, by name
##   look    one thing close up from four sides ([method TablePreview.thing])
##   set     the whole draft standing on the episode's own table, photographed from the camera's
##           place, with where each thing stood, what was made smaller or left off, which light leads
##           ([method TablePreview.table]) - and its AIR as it is some way into the reading
##   watch   one effect of the air ([Effects]) in motion: a burst at the moment it marks, fog or
##           motes a few seconds apart ([method TablePreview.watch])
##   submit  hand the draft in: written beside the job ([constant SUBMITTED]) for the producer to land
##           exactly as an answer in words would have landed ([method TarotProducer._land_table])
##
## Every picture counts against [constant LOOKS]; words cost nothing. A picture comes with the
## question to ask of it ([constant JUDGE_THING], [constant JUDGE_TABLE]): told only to fix what was
## wrong, a model looked at a 1 cm disc it had named an oil lamp and called it perfect. Before it is handed in the
## table is SET at least once since its last change, wherever a table can be stood - the one look the
## old single reply never had.
##
## IT KNOWS NO CARD, as the set dresser never did: the table it sets is photographed with the cards
## face down, and nothing here reads the draw.

## The file a handed-in table is written to, in the job's folder.
const SUBMITTED := "submitted.json"
## What to ask of a picture before going on: say what it shows, then judge it against what was meant.
const JUDGE_THING := "Before you go on, say what each picture actually shows - its shape and proportions against the grid (1 cm squares, a brighter line every 5), its material - and whether a stranger would name it as you did. Put again what does not read: a part floating or sunk, a thing too flat or too small to read, a vessel that reads as a plate, a bundle that reads as a stick."
const JUDGE_TABLE := "Before you submit, say what the frame actually shows: can each thing be told for what it is at this size, is any hidden, crowded or left off, and does the light come from where you meant? Fix what does not, and set the table again."
const JUDGE_AIR := "Say what the pictures show of it: is it where you meant, as thick or as bright as you meant, in colors that belong to this table - and can every card still be seen through it? Put it again to change it."
## How many pictures one set dresser may take, and how long its whole run may last (seconds).
const LOOKS := 30
const TIMEOUT := 1500

var episode: TarotEpisode
var plan: Dictionary
var dir := ""                    # the job's folder: its prompt, its reply, its tool log
var submitted := false

var _look := {}
var _things: Array = []          # the draft, in the order put
var _effects: Array = []         # ...its air, by name
var _materials := {}
var _idea := ""
var _looks := 0
var _set_since_change := false
var _warned := false
var _preview: TablePreview


func _init(ep: TarotEpisode, episode_plan: Dictionary, job_dir: String) -> void:
	episode = ep
	plan = episode_plan
	dir = job_dir
	_look = TarotTable.sanitize_look(plan.get("look", {}) if plan.get("look") is Dictionary else {})
	_preview = TablePreview.new(ep, plan, dir.path_join("preview"))
	# A RUN STARTS CLEAN: a rerun works in the folder the last run left, and its handed-in table or
	# its pictures must not pass for this run's
	DirAccess.make_dir_recursive_absolute(dir)
	for f in DirAccess.get_files_at(dir):
		var fs := String(f)
		if fs == SUBMITTED or fs == "tools.jsonl" or (fs.begins_with("look_") and fs.ends_with(".jpg")):
			DirAccess.remove_absolute(dir.path_join(fs))


## Give back the pictures' stages.
func release() -> void:
	_preview.release()


func instructions() -> String:
	return "Tools for setting a tarot reader's table: put things on it, look at them, set the whole table as the camera will film it, and submit it."


func list_tools() -> Array:
	return [
		{"name": "put", "description": "Put things on the table you are setting: each new thing is added, and a thing with the same name as one already there is replaced. Give the materials they use (added to the table's, or replacing one of the same name), the table's air (`effects`, each replacing one of the same name) and, if you like, the table's idea. Things, materials and effects are written exactly as in the format you were shown. Answers with what was built - each thing's size, anything the builder had to leave out or guess, whether it is taller than its place can show - and a picture of the things you put, each on its own tile on a centimeter grid.",
			"inputSchema": {"type": "object", "properties": {
				"things": {"type": "array", "items": {"type": "object"}, "description": "things, each {name, why, place, group, turn, parts}"},
				"materials": {"type": "object", "description": "materials by name, each {kind, color, ...}"},
				"effects": {"type": "array", "items": {"type": "object"}, "description": "the air: fog, motes and bursts, each {name, kind, ...}"},
				"idea": {"type": "string", "description": "two or three sentences: this table, and the reader who set it"}}}},
		{"name": "remove", "description": "Take things - or effects of its air - off the table you are setting, by name.",
			"inputSchema": {"type": "object", "properties": {"names": {"type": "array", "items": {"type": "string"}}}, "required": ["names"]}},
		{"name": "look", "description": "Look at one thing close up, from four sides - its front, its right side, from above, and as the camera at the reader's chair sees it - on a centimeter grid (a brighter line every 5 cm).",
			"inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
		{"name": "set", "description": "Stand everything on this episode's own table as the show will - its cloth, its room, its lights, the deck at its side and the cards laid face down where the reading lays them - and photograph it from the camera's place: what the viewer will see. Answers with the picture, where each thing stood, what was made smaller or left off for want of room, and which light leads.",
			"inputSchema": {"type": "object", "properties": {}}},
		{"name": "watch", "description": "Watch one effect of the table's air in motion, on this episode's own table: a burst photographed four times in the second after the moment it marks (a card leaping from the deck, one held up and twirled...), fog or motes four times a few seconds apart. A sheet of four pictures.",
			"inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
		{"name": "submit", "description": "Hand in the table as it now stands. Answers with what keeps it from being set, if anything; otherwise the table is done, and so is your work.",
			"inputSchema": {"type": "object", "properties": {}}},
	]


func call_tool(name: String, args: Dictionary) -> Dictionary:
	match name:
		"put":
			return await _put(args)
		"remove":
			return _remove(args)
		"look":
			return await _look_at(args)
		"set":
			return await _set_table()
		"watch":
			return await _watch(args)
		"submit":
			return _submit()
	return {"text": "There is no tool called \"%s\": put, remove, look, set, watch and submit are the tools." % name, "error": true}


## The draft as the set dresser has written it - the shape a one-reply table had.
func draft() -> Dictionary:
	var out := {"idea": _idea, "things": _things.duplicate(true), "materials": _materials.duplicate(true)}
	if not _effects.is_empty():
		out["effects"] = _effects.duplicate(true)
	return out


# --- the tools ----------------------------------------------------------------------------------------

func _put(args: Dictionary) -> Dictionary:
	var raw: Variant = args.get("things", [])
	if raw is Dictionary:
		raw = [raw]
	var mats: Dictionary = args.get("materials", {}) if args.get("materials") is Dictionary else {}
	var air: Variant = args.get("effects", [])
	if air is Dictionary:
		air = [air]
	var any_air: bool = air is Array and not (air as Array).is_empty()
	if not (raw is Array) or ((raw as Array).is_empty() and mats.is_empty() and not any_air and not (args.get("idea") is String)):
		return {"text": "Nothing was put: give `things` (a list of things), `materials`, `effects` or an `idea`.", "error": true}
	var notes := PackedStringArray()
	for k in mats:
		_materials[String(k).strip_edges()] = mats[k]
		notes.append_array(_material_troubles(String(k).strip_edges(), mats[k]))
	if args.get("idea") is String:
		_idea = String(args["idea"]).strip_edges()
	var put: Array = []
	var junk := 0
	for t in raw:
		if not (t is Dictionary):
			junk += 1
			continue
		var thing := (t as Dictionary).duplicate(true)
		var nm := String(thing.get("name", "")).strip_edges()
		if nm.is_empty():
			nm = "thing %d" % (_things.size() + 1)
		thing["name"] = nm
		var at := _index_of(nm)
		if at >= 0:
			_things[at] = thing
		else:
			_things.append(thing)
		if not put.has(nm):
			put.append(nm)
	var aired: Array = []
	if any_air:
		for e in air:
			if not (e is Dictionary):
				continue
			var fx := (e as Dictionary).duplicate(true)
			var en := String(fx.get("name", "")).strip_edges()
			if en.is_empty():
				en = "%s %d" % [String(fx.get("kind", "effect")), _effects.size() + 1]
			fx["name"] = en
			var at := _effect_of(en)
			if at >= 0:
				_effects[at] = fx
			else:
				_effects.append(fx)
			aired.append(en)
	_set_since_change = false
	var lines := PackedStringArray()
	if not put.is_empty():
		lines.append("Put %d thing%s (the table now holds %d):" % [put.size(), "" if put.size() == 1 else "s", _things.size()])
		lines.append_array(_describe(put))
	if not aired.is_empty():
		lines.append("The air (%d effect%s in all):" % [_effects.size(), "" if _effects.size() == 1 else "s"])
		lines.append_array(_describe_air(aired))
	if junk > 0:
		lines.append("%d of the things given %s not an object {name, parts, ...} - left out." % [junk, "was" if junk == 1 else "were"])
	if not notes.is_empty():
		lines.append("Materials:")
		for n in notes:
			lines.append("  - " + n)
	lines.append(_tally())
	var images: Array = []
	if not put.is_empty():
		var numbers := {}
		for i in _things.size():
			numbers[String((_things[i] as Dictionary).get("name", ""))] = i + 1
		var img: Image = await _picture(func() -> Image: return await _preview.things(_safe(), put, numbers))
		if img != null:
			images.append(img)
			lines.append(JUDGE_THING)
	lines.append(_looks_line(not images.is_empty()))
	return {"text": "\n".join(lines), "images": images}


func _remove(args: Dictionary) -> Dictionary:
	var names: Array = args.get("names", []) if args.get("names") is Array else ([args["names"]] if args.get("names") is String else [])
	var gone := PackedStringArray()
	var missing := PackedStringArray()
	for n in names:
		var at := _index_of(String(n).strip_edges())
		var fx := _effect_of(String(n).strip_edges())
		if at >= 0:
			_things.remove_at(at)
			gone.append(String(n))
		elif fx >= 0:
			_effects.remove_at(fx)
			gone.append(String(n))
		else:
			missing.append(String(n))
	if not gone.is_empty():
		_set_since_change = false
	var lines := PackedStringArray()
	if not gone.is_empty():
		lines.append("Taken off: %s." % ", ".join(gone))
	if not missing.is_empty():
		lines.append("Not on the table: %s. On it: %s." % [", ".join(missing), _names()])
	lines.append(_tally())
	return {"text": "\n".join(lines), "error": gone.is_empty()}


func _look_at(args: Dictionary) -> Dictionary:
	var nm := String(args.get("name", "")).strip_edges()
	if _index_of(nm) < 0:
		return {"text": "No thing called \"%s\" is on the table. On it: %s." % [nm, _names()], "error": true}
	var safe := _safe()
	var built := false
	for t in safe["things"]:
		built = built or String((t as Dictionary).get("name", "")) == nm
	if not built:
		return {"text": "\"%s\" has nothing the builder can make: %s" % [nm, "; ".join(_troubles(_things[_index_of(nm)]))], "error": true}
	var img: Image = await _picture(func() -> Image: return await _preview.thing(safe, nm))
	var lines := PackedStringArray(_describe([nm]))
	if img != null:
		lines.append(JUDGE_THING)
	lines.append(_looks_line(img != null))
	return {"text": "\n".join(lines), "images": [img] if img != null else []}


func _set_table() -> Dictionary:
	var safe := _safe()
	if (safe["things"] as Array).is_empty():
		return {"text": "Nothing on the table can be built yet - put things first.", "error": true}
	if _looks >= LOOKS:
		return {"text": _looks_line(false), "error": true}
	if not _can_stand():
		_set_since_change = true
		return {"text": "The table cannot be stood in this run (%s), so there is no picture of it: judge it by what put told you." %
			("no renderer" if not _can_see() else "no ghost session")}
	var got: Dictionary = await _preview.table(draft())
	if got.has("error"):
		return {"text": "The table could not be stood: %s" % String(got["error"]), "error": true}
	_looks += 1
	_set_since_change = true
	var lines := PackedStringArray(["The table, from the camera's place - the deck at the %s, %d cards face down in their spread:" %
		[String(got["deck"]), int(got["cards"])]])
	for s in got["stood"]:
		var d: Dictionary = s
		var at: Vector2 = d["at"]
		var where := "%d cm %s of the middle, %d cm %s" % [roundi(absf(at.x)), "right" if at.x >= 0.0 else "left",
			roundi(absf(at.y)), "toward the camera" if at.y >= 0.0 else "back"]
		var small := (" - made smaller, to %d%%, to fit" % roundi(float(d["k"]) * 100.0)) if float(d["k"]) < 0.995 else ""
		lines.append("- %s (%s): %s; %d%% of the picture's height%s" % [String(d["name"]), String(d["place"]), where,
			roundi(float(d["tall"]) * 100.0), small])
	for n in got["left_off"]:
		lines.append("- %s: LEFT OFF - no room for it in its place, clear of the cards, in the shot and not in front of another group" % String(n))
	lines.append("The light is led by %s." % String(got["key"]) + (" Held down by pale cloth near it: %s." % ", ".join(PackedStringArray(got["held_down"])) if not (got["held_down"] as Array).is_empty() else ""))
	var air := _air_line(safe)
	if not air.is_empty():
		lines.append(air)
	if got.get("image") != null:
		lines.append(JUDGE_TABLE)
	lines.append(_looks_line(got.get("image") != null))
	return {"text": "\n".join(lines), "images": [got["image"]] if got.get("image") != null else []}


func _watch(args: Dictionary) -> Dictionary:
	var nm := String(args.get("name", "")).strip_edges()
	if _effect_of(nm) < 0:
		return {"text": "No effect called \"%s\" is in the table's air. In it: %s." % [nm, _effect_names()], "error": true}
	var safe := _safe()
	var built := {}
	for e in safe.get("effects", []):
		if String((e as Dictionary).get("name", "")) == nm:
			built = e
	if built.is_empty():
		return {"text": "\"%s\" cannot be built: %s" % [nm, "; ".join(_air_troubles(_effects[_effect_of(nm)]))], "error": true}
	if _looks >= LOOKS:
		return {"text": _looks_line(false), "error": true}
	if not _can_stand():
		return {"text": "The table cannot be stood in this run (%s), so its air cannot be watched: judge it by what put told you." %
			("no renderer" if not _can_see() else "no ghost session")}
	var got: Dictionary = await _preview.watch(draft(), nm)
	if got.has("error"):
		return {"text": "\"%s\" could not be watched: %s" % [nm, String(got["error"])], "error": true}
	_looks += 1
	var frames := PackedStringArray()
	for f in got.get("frames", []):
		frames.append(("%.2f s after" % float(f)) if bool(got.get("burst", false)) else ("%d s on" % roundi(float(f))))
	var lines := PackedStringArray(_describe_air([nm]))
	lines.append("Four pictures, left to right and down: %s%s." % [", ".join(frames),
		(" the moment it marks, as the table stages it") if bool(got.get("burst", false)) else ""])
	lines.append(JUDGE_AIR)
	lines.append(_looks_line(true))
	return {"text": "\n".join(lines), "images": [got["image"]]}


func _submit() -> Dictionary:
	var safe := _safe()
	if (safe["things"] as Array).is_empty():
		return {"text": "Nothing on the table can be built yet - put things first.", "error": true}
	# SEEN BEFORE IT IS HANDED IN: once, where a table can be stood at all
	if not _set_since_change and not _warned and _can_stand() and _looks < LOOKS:
		_warned = true
		return {"text": "You have not seen the table as it now stands: call set, look at the frame, fix what it shows, then submit. (Submit again unchanged to hand it in as it is.)", "error": true}
	var raw := draft()
	var path := dir.path_join(SUBMITTED)
	var err := TextGen.put(path + ".part", JSON.stringify(raw, "\t"))
	if err.is_empty() and DirAccess.rename_absolute(path + ".part", path) != OK:
		err = "could not move the table into place"
	if not err.is_empty():
		return {"text": "The table could not be handed in: %s" % err, "error": true}
	submitted = true
	return {"text": "Handed in: %d thing%s. The table is set - your work is done; end with one short line." %
		[(safe["things"] as Array).size(), "" if (safe["things"] as Array).size() == 1 else "s"]}


# --- what the builder makes of the draft --------------------------------------------------------------

## The draft made safe, as the table will build it.
func _safe() -> Dictionary:
	return TarotTable.sanitize_table(draft(), _look)


## One block per thing named: its place, its size, its flames - and everything the builder could
## not make as written.
func _describe(names: Array) -> PackedStringArray:
	var safe := _safe()
	var head := TarotTable.headroom(episode.seed)
	var out := PackedStringArray()
	for nm in names:
		var at := _index_of(String(nm))
		if at < 0:
			continue
		var raw: Dictionary = _things[at]
		var troubles := _troubles(raw)
		var built := {}
		for t in safe["things"]:
			if String((t as Dictionary).get("name", "")) == String(nm):
				built = t
				break
		if built.is_empty():
			out.append("%d. %s: NOTHING BUILT - %s" % [at + 1, String(nm), "; ".join(troubles) if not troubles.is_empty() else "it has no part the builder can make"])
			continue
		var b := Props.build(built, safe["materials"], hash([episode.seed, at, "describe"]))
		var box: AABB = b["size"]
		var node: Node3D = b["node"]
		var shrunk := node.get_child_count() > 0 and (node.get_child(0) as Node3D).scale.x < 0.999
		var flames := (b["wicks"] as Array).size()
		node.free()
		var place := String(built.get("place", "back"))
		var group := String(built.get("group", ""))
		out.append("%d. %s (%s%s): %s, %d part%s%s" % [at + 1, String(nm), place, (", group " + group) if not group.is_empty() else "",
			TablePreview.size_text(box), _count_parts(built.get("parts", [])), "" if _count_parts(built.get("parts", [])) == 1 else "s",
			(", %d flame%s" % [flames, "" if flames == 1 else "s"]) if flames > 0 else ""])
		if shrunk:
			troubles.append("larger than %d cm across: built smaller, whole" % roundi(Props.MAX_SIZE))
		var wrote := 0
		for p in raw.get("parts", []) if raw.get("parts") is Array else []:
			if p is Dictionary:
				wrote += _flames_written(p as Dictionary)
		if wrote > flames:
			troubles.append("%d of its flames stay unlit: a table lights at most %d things, %d flames on one" % [wrote - flames, TarotTable.MAX_CANDLES, TarotTable.MAX_FLAMES])
		var room := int(head.get(place, 99))
		if box.size.y * 100.0 > float(room) + 0.5:
			troubles.append("%s cm tall, and \"%s\" shows only %d cm: it will be made smaller or left off" % [TablePreview._cm(box.size.y), place, room])
		for tr in troubles:
			out.append("   - " + tr)
	return out


## What the builder cannot make of thing [param t] as written: one line each.
func _troubles(t: Dictionary) -> PackedStringArray:
	var out := PackedStringArray()
	if not (t.get("parts") is Array) or (t["parts"] as Array).is_empty():
		out.append("it has no parts")
		return out
	_part_troubles(t["parts"], 0, [Props.MAX_PARTS], "", out)
	var place := String(t.get("place", "")).strip_edges().to_lower()
	if not place.is_empty() and not TarotTable.ZONES.has(place):
		out.append("\"%s\" is not a place: it will stand at the back" % place)
	return out


func _part_troubles(list: Array, depth: int, budget: Array, path: String, out: PackedStringArray) -> void:
	for i in list.size():
		var where := "%s%d" % [path, i + 1]
		if int(budget[0]) <= 0:
			out.append("part %s and the rest: past the %d parts a thing may have - left out" % [where, Props.MAX_PARTS])
			return
		if not (list[i] is Dictionary):
			out.append("part %s is not a part - left out" % where)
			continue
		var d: Dictionary = list[i]
		if d.get("parts") is Array:
			if depth >= Props.MAX_DEPTH:
				out.append("group %s is more than %d groups deep - left out, with everything in it" % [where, Props.MAX_DEPTH])
			else:
				_part_troubles(d["parts"], depth + 1, budget, where + ".", out)
			continue
		var shape := String(d.get("shape", "")).strip_edges().to_lower()
		if not Props.SHAPES.has(shape):
			out.append("part %s: \"%s\" is not a shape - left out" % [where, str(d.get("shape", ""))])
			continue
		if shape == "lathe" and Props._points2(d.get("profile"), Vector2(0.0, -1.0), Vector2(Props.MAX_SIZE, Props.MAX_SIZE)).size() < 2:
			out.append("part %s: a lathe's profile needs two [radius, height] points or more - left out" % where)
			continue
		if shape == "tube" and Props._points3(d.get("path"), Props.MAX_SIZE).size() < 2:
			out.append("part %s: a tube's path needs two [x, y, z] points or more - left out" % where)
			continue
		budget[0] = int(budget[0]) - 1
		var names: Array = d["material"] if d.get("material") is Array else [d.get("material", "")]
		for m in names:
			if m is Dictionary:
				continue
			var nm := String(m).strip_edges() if m is String else ""
			if nm.is_empty():
				out.append("part %s names no material - painted in the deck's colors" % where)
			elif not _materials.has(nm) and not Props.MATERIALS.has(nm.to_lower()):
				out.append("part %s: \"%s\" is not one of the table's materials - painted in the deck's colors" % [where, nm])


## What the builder cannot make of material [param m] as written.
static func _material_troubles(name: String, m: Variant) -> PackedStringArray:
	var out := PackedStringArray()
	if not (m is Dictionary):
		out.append("\"%s\" is not a material {kind, color, ...} - painted" % name)
		return out
	var d: Dictionary = m
	var kind := String(d.get("kind", "")).strip_edges().to_lower()
	if not Props.MATERIALS.has(kind):
		out.append("\"%s\": \"%s\" is not a kind of material - painted" % [name, kind])
	if not Props._is_color(d.get("color", "")):
		out.append("\"%s\": its color is not #rrggbb - one of the deck's colors" % name)
	if d.has("play") and not Props.PLAYS.has(String(d["play"]).strip_edges().to_lower()):
		out.append("\"%s\": \"%s\" is not a play of light - none" % [name, str(d["play"])])
	return out


## The flames a part asks for as written, every copy of it.
static func _flames_written(p: Dictionary) -> int:
	var n := 0
	if p.get("parts") is Array:
		for q in p["parts"]:
			if q is Dictionary:
				n += _flames_written(q as Dictionary)
	elif p.get("wicks") is Array:
		n = (p["wicks"] as Array).size()
	elif p.get("wicks") is float or p.get("wicks") is int:
		n = int(p["wicks"])
	elif p.get("wick") == true:
		n = 1
	var c: Variant = p.get("copies", {})
	var count := 1
	if c is Dictionary:
		for kind in c:
			if (c as Dictionary)[kind] is Dictionary:
				count = maxi(1, int(((c as Dictionary)[kind] as Dictionary).get("count", 1)))
	return n * count


static func _count_parts(parts: Variant) -> int:
	var n := 0
	for p in parts if parts is Array else []:
		n += _count_parts((p as Dictionary)["parts"]) if (p as Dictionary).get("parts") is Array else 1
	return n


## The table as a whole against what the look asked for: lit things, and the others.
func _tally() -> String:
	var safe := _safe()
	var lit := 0
	for t in safe["things"]:
		var f := 0
		for p in (t as Dictionary).get("parts", []):
			f += Props.flames_of(p as Dictionary)
		lit += 1 if f > 0 else 0
	var others := (safe["things"] as Array).size() - lit
	var asked := clampi(int(_look.get("candles", 1)), 0, TarotTable.MAX_CANDLES)
	var size := TarotPrompts.table_size(episode.seed)
	return "The table: %d lit thing%s (%d asked for), %d other thing%s (%d to %d asked for)." % [lit, "" if lit == 1 else "s",
		asked, others, "" if others == 1 else "s", size.x, size.y]


## One block per effect named: what it is, where or when - and anything the builder could not use.
func _describe_air(names: Array) -> PackedStringArray:
	var safe := _safe()
	var out := PackedStringArray()
	for nm in names:
		var at := _effect_of(String(nm))
		if at < 0:
			continue
		var built := {}
		for e in safe.get("effects", []):
			if String((e as Dictionary).get("name", "")) == String(nm):
				built = e
		var troubles := _air_troubles(_effects[at])
		if built.is_empty():
			out.append("- %s: NOT BUILT - %s" % [String(nm), "; ".join(troubles) if not troubles.is_empty() else "there is more air than a table holds"])
			continue
		var what := ""
		match String(built["kind"]):
			"fog":
				what = "fog %s, density %.2f, reaching %d cm" % [String(built["where"]), float(built["density"]), roundi(float(built["height"]))]
			"motes":
				what = "%s %s, %s mm%s" % [Effects.count_of(String(built["look"]), int(built["count"])), String(built["where"]),
					Effects._mm(float(built["size"])), ", lit" if bool(built["light"]) else ""]
			"burst":
				what = "%s on %s, %d of them, %s mm" % [String(built["look"]), String(built["on"]), int(built["count"]), Effects._mm(float(built["size"]))]
		out.append("- %s: %s" % [String(nm), what])
		for tr in troubles:
			out.append("   - " + tr)
	return out


## What the air builder cannot use of effect [param e] as written: one line each.
func _air_troubles(e: Dictionary) -> PackedStringArray:
	var out := PackedStringArray()
	var kind := String(e.get("kind", "")).strip_edges().to_lower()
	if not Effects.KINDS.has(kind):
		out.append("\"%s\" is not a kind of effect (fog, motes, burst)" % kind)
		return out
	if kind == "fog" or kind == "motes":
		var where := String(e.get("where", "")).strip_edges().to_lower()
		if not TarotTable.AIR.has(where):
			out.append("\"%s\" is not a place in the air: %s" % [where, ", ".join(PackedStringArray(TarotTable.AIR.keys()))])
	if kind == "motes" or kind == "burst":
		var look := String(e.get("look", "")).strip_edges().to_lower()
		var looks: Dictionary = Effects.MOTES if kind == "motes" else Effects.BURSTS
		if not looks.has(look):
			out.append("\"%s\" is not a look of %s: %s" % [look, kind, ", ".join(PackedStringArray(looks.keys()))])
		elif e.has("size") and (e["size"] is float or e["size"] is int):
			var r: Vector2 = (looks[look] as Dictionary)["sizes"]
			if float(e["size"]) < r.x or float(e["size"]) > r.y:
				out.append("a %s is %s to %s mm: its size was kept within that" % [look, Effects._mm(r.x), Effects._mm(r.y)])
		if kind == "motes" and looks.has(look) and String((looks[look] as Dictionary)["shape"]) == "speck" and Props._flag(e.get("light"), false):
			out.append("a %s is a dark speck and carries no light: it was left unlit" % look)
	if kind == "burst":
		var on := String(e.get("on", "")).strip_edges().to_lower()
		if not TarotTable.MOMENTS.has(on):
			out.append("\"%s\" is not a moment: %s" % [on, ", ".join(PackedStringArray(TarotTable.MOMENTS.keys()))])
	return out


## The air in a line, for the table's own picture: what is in it, and that a burst is watched.
func _air_line(safe: Dictionary) -> String:
	var parts := PackedStringArray()
	var bursts := 0
	for e in safe.get("effects", []):
		var d: Dictionary = e
		match String(d["kind"]):
			"fog":
				parts.append("%s (fog, %s)" % [String(d["name"]), String(d["where"])])
			"motes":
				parts.append("%s (%s, %s)" % [String(d["name"]), Effects.count_of(String(d["look"]), int(d["count"])), String(d["where"])])
			"burst":
				bursts += 1
	if parts.is_empty() and bursts == 0:
		return ""
	return "The air in this picture: %s.%s" % [", ".join(parts) if not parts.is_empty() else "none still", (" Its %d burst%s only show at %s moment - watch %s." % [bursts,
		"" if bursts == 1 else "s", "its" if bursts == 1 else "their", "it" if bursts == 1 else "each"]) if bursts > 0 else ""]


func _effect_of(name: String) -> int:
	for i in _effects.size():
		if String((_effects[i] as Dictionary).get("name", "")) == name:
			return i
	return -1


func _effect_names() -> String:
	var out := PackedStringArray()
	for e in _effects:
		out.append(String((e as Dictionary).get("name", "")))
	return ", ".join(out) if not out.is_empty() else "nothing yet"


func _index_of(name: String) -> int:
	for i in _things.size():
		if String((_things[i] as Dictionary).get("name", "")) == name:
			return i
	return -1


func _names() -> String:
	var out := PackedStringArray()
	for t in _things:
		out.append(String((t as Dictionary).get("name", "")))
	return ", ".join(out) if not out.is_empty() else "nothing yet"


## A picture from [param take], counted - or null when none can be taken or none are left.
func _picture(take: Callable) -> Image:
	if _looks >= LOOKS or not _can_see():
		return null
	var img: Image = await take.call()
	if img != null:
		_looks += 1
	return img


func _looks_line(took: bool) -> String:
	if not _can_see():
		return "(No pictures in this run - it has no renderer.)"
	if _looks >= LOOKS:
		return "No looks left: submit the table." if not took else "That was your last look: submit the table."
	return "Looks left: %d." % (LOOKS - _looks)


## Whether this run can take pictures, and stand the episode's table ([TablePreview]) - a gate's seam.
func _can_see() -> bool:
	return TablePreview.can_see()


func _can_stand() -> bool:
	return TablePreview.can_set()
