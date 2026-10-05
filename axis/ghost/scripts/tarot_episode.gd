extends RefCounted
class_name TarotEpisode

## TarotEpisode - one episode of a tarot show, as it lies on disk.
##
## An episode is everything one seed of a show produced: the plan (title, spread, look), the
## shuffle, each drawn card's design and booklet entry, the pictures, the reader's words and the
## script they make. EVERY STEP IS A FILE under `user://tarot/<show>/<seed>/`, written the
## moment it is made, so:
##
##   - re-running a seed re-uses what it produced - the same reading, the same pictures, every
##     time - and nothing is made twice by accident (it costs the author's quota);
##   - a step that failed, or a ghost that quit half way, picks up where it stopped;
##   - REDOING A STEP IS DELETING IT: [method invalidate] removes the step and everything made
##     from it, and the producer makes whatever is missing. There is no other kind of reroll.
##
## The job directories under `jobs/` keep every prompt exactly as sent, beside the reply - the
## record that the reader was never shown a card before it was drawn.

const ROOT := "user://tarot"

## Where shows live. Moved aside under test, so a gate never touches the author's episodes.
static var root := ROOT
## How a deleted episode's folder is thrown away, when set: the gate's seam, so a check never
## fills the author's trash. Unset, it goes to the system's trash ([method trash]).
static var discard: Callable = Callable()

var show := ""
var seed := 0
## Absolute.
var dir := ""
var _script_cache := {}


static func open(show_key: String, seed_value: int) -> TarotEpisode:
	var e := TarotEpisode.new()
	e.show = show_key
	e.seed = seed_value
	e.dir = ProjectSettings.globalize_path(root.path_join(show_key).path_join(str(seed_value)))
	return e


## A show's key from its title: lowercase words joined by dashes, nothing a path could trip on.
static func slug(title: String) -> String:
	var s := Manuscript._rx("[^a-z0-9]+").sub(title.to_lower(), "-", true).strip_edges()
	while s.begins_with("-"):
		s = s.substr(1)
	while s.ends_with("-"):
		s = s.left(-1)
	return s if not s.is_empty() else "untitled"


## Every seed of [param show_key] with at least a plan, newest first: `[{seed, plan, at}]`.
static func history(show_key: String) -> Array:
	var base := ProjectSettings.globalize_path(root.path_join(show_key))
	var out: Array = []
	if not DirAccess.dir_exists_absolute(base):
		return out
	for d in DirAccess.get_directories_at(base):
		if not String(d).is_valid_int():
			continue
		var p := base.path_join(d).path_join("plan.json")
		if not FileAccess.file_exists(p):
			continue
		var j := JSON.new()
		if j.parse(FileAccess.get_file_as_string(p)) != OK or not (j.data is Dictionary):
			continue
		out.append({"seed": int(d), "plan": j.data, "at": FileAccess.get_modified_time(p)})
	out.sort_custom(func(a: Dictionary, b: Dictionary) -> bool: return int(a["at"]) > int(b["at"]))
	return out


# --- files -----------------------------------------------------------------------------

## The file a step is kept in (see [method steps] for the keys).
func file_of(step: String) -> String:
	var parts := step.split(":")
	match String(parts[0]):
		"plan":
			return dir.path_join("plan.json")
		"draw":
			return dir.path_join("draw.json")
		"design":
			return dir.path_join("design_%s.json" % parts[1])
		"image":
			if parts.size() == 3:
				return dir.path_join("%s_%s.png" % [parts[1], parts[2]])     # card_K
			return dir.path_join("%s.png" % parts[1])
		"say":
			return dir.path_join("say_%s.txt" % parts[1])
		"script":
			return dir.path_join("script.md")
		"meta":
			return dir.path_join("meta.json")
		"upload":
			return dir.path_join("upload.md")
	return ""


## A step exists when its file does: every write lands by rename (here, and the painter's
## pictures in [AgentJobs]), so a file that is there is whole - and asking costs no read.
func has(step: String) -> bool:
	var p := file_of(step)
	return not p.is_empty() and FileAccess.file_exists(p)


## The directory a step's agent job runs in - kept, with its prompt.
func job_dir(step: String) -> String:
	return dir.path_join("jobs").path_join(step.replace(":", "_"))


func read_json(step: String) -> Variant:
	var raw := FileAccess.get_file_as_string(file_of(step))
	if raw.is_empty():
		return null
	var j := JSON.new()
	return j.data if j.parse(raw) == OK else null


func read_text(step: String) -> String:
	return FileAccess.get_file_as_string(file_of(step))


## Written ATOMICALLY - a file that exists is a step that is done, so a half-written one must
## never be seen: temp beside it, then renamed over.
func write_text(step: String, text: String) -> String:
	var p := file_of(step)
	DirAccess.make_dir_recursive_absolute(p.get_base_dir())
	var tmp := p + ".part"
	var f := FileAccess.open(tmp, FileAccess.WRITE)
	if f == null:
		return "could not write " + p
	f.store_string(text)
	f.close()
	if DirAccess.rename_absolute(tmp, p) != OK:
		return "could not move %s into place" % p
	return ""


func write_json(step: String, value: Variant) -> String:
	return write_text(step, JSON.stringify(value, "\t"))


# --- the steps -------------------------------------------------------------------------

## How many cards the episode draws: the shuffle's, once made; else the plan's spread.
func card_count() -> int:
	var d: Variant = read_json("draw")
	if d is Dictionary:
		return ((d as Dictionary).get("cards", []) as Array).size()
	var p: Variant = read_json("plan")
	if p is Dictionary:
		return ((((p as Dictionary).get("spread", {}) as Dictionary).get("positions", [])) as Array).size()
	return 0


## EVERY STEP, in the order it is made: `plan`, `draw`, `design:K`, `image:back`,
## `image:surface`, `image:backdrop`, `image:card:K`, `say:intro`, `say:K`,
## `say:close`, `script`. Card steps exist only once the plan says how many cards there are.
func steps() -> Array:
	var out := ["plan", "draw"]
	var n := card_count()
	for k in range(1, n + 1):
		out.append("design:%d" % k)
	out.append_array(["image:back", "image:surface", "image:backdrop"])
	for k in range(1, n + 1):
		out.append("image:card:%d" % k)
	out.append("say:intro")
	for k in range(1, n + 1):
		out.append("say:%d" % k)
	out.append_array(["say:close", "script"])
	return out


## What [param step] is made FROM - the steps that must exist before it can be.
func needs(step: String) -> Array:
	var parts := step.split(":")
	var n := card_count()
	match String(parts[0]):
		"plan":
			return []
		"draw":
			return ["plan"]
		"design":
			return ["draw"]
		"image":
			if parts.size() == 3:
				# a card's picture is painted in the deck's hand: after the back, and after the
				# card before it (they are sent to it as references - see TarotProducer)
				var k := int(parts[2])
				var out := ["design:%d" % k, "image:back"]
				if k > 1:
					out.append("image:card:%d" % (k - 1))
				return out
			return ["plan"]
		"say":
			var who := String(parts[1])
			if who == "intro":
				return ["draw"]
			if who == "close":
				return ["say:%d" % n] if n > 0 else ["say:intro"]
			# a card's passage is written LOOKING AT the card: it waits for its picture, and a
			# card painted again is a passage written again (see TarotProducer.say_prompt)
			var k := int(who)
			return ["design:%d" % k, "image:card:%d" % k, "say:intro" if k == 1 else "say:%d" % (k - 1)]
		"script":
			return ["say:close"]
	return []


## Everything made from [param step], directly or through another step. A card's picture is NOT
## made from the card before it in this sense - it is only shown it - so redoing one picture
## leaves the rest of the deck alone (the [Illustrations] rule: the chain is not the signature).
func dependents(step: String) -> Array:
	var out: Array = []
	var frontier := [step]
	while not frontier.is_empty():
		var s := String(frontier.pop_front())
		for t in steps():
			var ts := String(t)
			if out.has(ts) or ts == step:
				continue
			if _soft(s, ts):
				continue
			if needs(ts).has(s):
				out.append(ts)
				frontier.append(ts)
	return out


## The edges [method needs] lists only for ORDER: a card's picture waits for the back and for
## the card before it, because it is sent them as references, but it is not made from them.
static func _soft(from: String, to: String) -> bool:
	return to.begins_with("image:card:") and (from.begins_with("image:card:") or from == "image:back")


## REDO [param step]: delete it and everything made from it. Returns what went.
func invalidate(step: String) -> Array:
	var gone := [step] + dependents(step)
	# the plan decides how many cards there are: going back to it takes every card step with it,
	# including ones a smaller spread no longer lists
	if (step == "plan" or step == "draw") and DirAccess.dir_exists_absolute(dir):
		for f in DirAccess.get_files_at(dir):
			var fs := String(f)
			if fs.begins_with("design_") or fs.begins_with("card_") or fs.begins_with("say_") \
					or fs == "script.md" or (step == "plan" and fs in ["draw.json", "back.png", "surface.png",
					"backdrop.png"]):
				DirAccess.remove_absolute(dir.path_join(fs))
	for s in gone:
		var p := file_of(String(s))
		if not p.is_empty() and FileAccess.file_exists(p):
			DirAccess.remove_absolute(p)
	return gone


## DELETE THE EPISODE: its whole folder - plan, draw, designs, pictures, passages, script, upload
## notes, and every job's prompt beside its reply - goes to the SYSTEM'S TRASH, not to nothing: an
## episode is minutes of an author's quota, and a mistaken delete should be undoable from there.
## Exported videos live elsewhere and are not touched. "" on success, else why not.
func trash() -> String:
	if not DirAccess.dir_exists_absolute(dir):
		return "there is nothing on disk for episode #%d" % seed
	var err: int = discard.call(dir) if discard.is_valid() else OS.move_to_trash(dir)
	if err != OK or DirAccess.dir_exists_absolute(dir):
		return "could not move %s to the trash (error %d)" % [dir, err]
	_script_cache = {}
	return ""


## The whole reading, ready to speak - "" until every passage is written. Asked every frame
## (the exporter's button asks whether there is a reading), so it is read again only when the
## file changes; a redo deletes it, which drops the copy.
func script() -> String:
	var p := file_of("script")
	if not FileAccess.file_exists(p):
		_script_cache = {}
		return ""
	var mt := FileAccess.get_modified_time(p)
	if int(_script_cache.get("at", -1)) != mt:
		_script_cache = {"at": mt, "text": FileAccess.get_file_as_string(p)}
	return String(_script_cache["text"])


## Every step is made.
func complete() -> bool:
	for s in steps():
		if not has(String(s)):
			return false
	return true


## UPLOAD NOTES: `upload.md` - the title, the description and tags the producer wrote, and a
## chapter per card timed from the take at [param take] (its sidecar's word timings). "" on
## success, else why not.
func write_upload_notes(take: String) -> String:
	var side := FileAccess.get_file_as_string(take.get_basename() + ".json")
	var j := JSON.new()
	if j.parse(side) != OK or not (j.data is Dictionary):
		return "the take's sidecar is unreadable"
	var words: Array = (j.data as Dictionary).get("words", [])
	var plan: Variant = read_json("plan")
	var p: Dictionary = plan if plan is Dictionary else {}
	var doc := document()
	var cards: Array = doc.get("cards", [])
	var lines := PackedStringArray()
	lines.append("# " + String(p.get("episode_title", "")))
	lines.append("")
	var desc := String(p.get("description", "")).strip_edges()
	lines.append(desc if not desc.is_empty() else String(p.get("premise", "")))
	lines.append("")
	lines.append("Chapters")
	for c in TarotScript.chapters(script(), words):
		var d: Dictionary = c
		var label := "Intro"
		match String(d["kind"]):
			"draw", "jumper":
				var k := int(d["card"])
				var card: Dictionary = cards[k - 1] if k >= 1 and k <= cards.size() else {}
				var pos := String((card.get("position", {}) as Dictionary).get("name", ""))
				label = "%s%s%s%s" % [(pos + " - ") if not pos.is_empty() else "", String(card.get("name", "Card %d" % k)),
					" (reversed)" if bool(card.get("reversed", false)) else "",
					" - a jumper" if String(d["kind"]) == "jumper" else ""]
			"spread":
				label = "The spread"
		var t := int(d["t"])
		lines.append("%d:%02d %s" % [floori(float(t) / 60.0), t % 60, label])
	var tags := PackedStringArray(p.get("tags", []) if p.get("tags") is Array else [])
	if not tags.is_empty():
		lines.append("")
		lines.append("Tags: " + ", ".join(tags))
	return write_text("upload", "\n".join(lines) + "\n")


# --- what the table is given ------------------------------------------------------------

## THE EPISODE AS THE TABLE DRAWS IT, handed to the medium beside the script (live, through the
## Generative panel's book document; in a render, through the take's sidecar). Plain data and
## ABSOLUTE paths, because a render is a second process with no producer in it.
func document() -> Dictionary:
	var plan: Variant = read_json("plan")
	var draw: Variant = read_json("draw")
	var out := {"show": show, "seed": seed, "dir": dir, "plan": plan if plan is Dictionary else {},
		"images": {}, "cards": []}
	for k in ["back", "surface", "backdrop"]:
		if has("image:" + k):
			out["images"][k] = file_of("image:" + k)
	if draw is Dictionary:
		var spread: Array = (((out["plan"] as Dictionary).get("spread", {}) as Dictionary)
			.get("positions", [])) as Array
		var i := 0
		for c in (draw as Dictionary).get("cards", []):
			i += 1
			# the draw keeps the cards themselves (see TarotProducer._make_draw)
			var card := (c as Dictionary).duplicate()
			card["reversed"] = bool(card.get("reversed", false))
			card["jumper"] = bool(card.get("jumper", false))
			card["position"] = spread[i - 1] if i - 1 < spread.size() else {}
			var design: Variant = read_json("design:%d" % i)
			card["booklet"] = (design as Dictionary).get("booklet", {}) if design is Dictionary else {}
			card["art"] = file_of("image:card:%d" % i) if has("image:card:%d" % i) else ""
			(out["cards"] as Array).append(card)
	return out
