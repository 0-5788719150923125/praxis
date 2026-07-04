extends RefCounted
class_name Storyboard

## Storyboard - the loader for user-authored scene scores (Manual mode).
##
## Extracted from the Director so the data spec has one owner. A storyboard file is
## YAML (the on-disk format - it holds comments) or JSON (what the future manual
## editor will emit); both parse to the SAME canonical Dictionary, so everything
## downstream is format-blind. Bare names resolve inside `res://storyboards/`
## (`.yaml` first, then `.json`); explicit paths are taken as-is.
##
## Beyond parsing, the loader owns the spec's three data conveniences:
##   defs/use  - `defs:` is a top-level map of named fragments; `use: <name>` inside
##               any map deep-merges the fragment UNDER the map's own keys (explicit
##               keys win). A map that is ONLY `{use: name}` becomes the fragment
##               verbatim (so a whole cast / track block can be a reference). This is
##               how a storyboard composes reusable pieces - "a scene calling a scene".
##   sample    - "cattle, not pets": anywhere a number is expected, `[lo, hi]` is a
##               RANGE, sampled once per instance from the scene's seeded rng via
##               [method sample]. (A two-number array is therefore always a range -
##               positions are 3-component `[x, y, z]`.)
##   validate  - structural errors come back as one clear per-path message instead of
##               a silent misload.
##
## [method load_file] returns `{ok, error, path, name, loop, transition, sensitivity,
## sequence}`; on failure only `ok` / `error` matter.

const DIR := "res://storyboards/"
const MAX_USE_DEPTH := 8


static func load_file(name_or_path: String) -> Dictionary:
	var path := resolve_path(name_or_path)
	if path.is_empty():
		return _fail("storyboard not found: %s" % name_or_path)
	var text := FileAccess.get_file_as_string(path)
	var data: Variant
	if path.ends_with(".json"):
		data = JSON.parse_string(text)
		if data == null:
			return _fail("storyboard %s: invalid JSON" % path)
	else:
		var y := MiniYaml.parse(text)
		if not y.ok:
			return _fail("storyboard %s: %s" % [path, y.error])
		data = y.data
	if typeof(data) != TYPE_DICTIONARY:
		return _fail("storyboard %s: top level must be a map" % path)

	# Expand defs/use into a plain tree, then validate the result.
	var defs: Variant = (data as Dictionary).get("defs", {})
	if typeof(defs) != TYPE_DICTIONARY:
		return _fail("storyboard %s: 'defs' must be a map" % path)
	var ctx := {"error": ""}
	var expanded: Dictionary = _expand(data, defs, 0, ctx)
	if ctx.error != "":
		return _fail("storyboard %s: %s" % [path, String(ctx.error)])
	var verr := _validate(expanded, path)
	if verr != "":
		return _fail(verr)

	# A top-level `elastic` (the music-breathing timeline clock, see scenes/stage.gd)
	# stamps into every entry that doesn't set its own, so one number tunes the show.
	if expanded.has("elastic"):
		for e in expanded["sequence"]:
			if not (e as Dictionary).has("elastic"):
				e["elastic"] = expanded["elastic"]
		for e in expanded.get("tail", []):
			if not (e as Dictionary).has("elastic"):
				e["elastic"] = expanded["elastic"]

	return {
		"ok": true, "error": "", "path": path,
		"name": String(expanded.get("name", name_or_path)),
		"loop": bool(expanded.get("loop", true)),
		"transition": String(expanded.get("transition", "")),
		"sensitivity": float(expanded.get("sensitivity", -1.0)),
		"sequence": expanded["sequence"],
		"tail": expanded.get("tail", []),
	}


## Resolve a bare name or explicit path to an existing storyboard file, or "".
static func resolve_path(name_or_path: String) -> String:
	var cands: Array = [name_or_path] \
		if name_or_path.ends_with(".json") or name_or_path.ends_with(".yaml") or name_or_path.ends_with(".yml") \
		else [DIR + name_or_path + ".yaml", DIR + name_or_path + ".yml", DIR + name_or_path + ".json"]
	for p in cands:
		if FileAccess.file_exists(p):
			return p
	return ""


## Sample a spec value with a seeded rng: a two-number array is a `[lo, hi]` range
## (one draw per call site = once per instance); everything else recurses. Call this
## on any user-authored numeric config before baking it into an instance.
static func sample(v: Variant, rng: RandomNumberGenerator) -> Variant:
	match typeof(v):
		TYPE_ARRAY:
			var a: Array = v
			if a.size() == 2 and _is_num(a[0]) and _is_num(a[1]):
				return rng.randf_range(float(a[0]), float(a[1]))
			var out := []
			for it in a:
				out.append(sample(it, rng))
			return out
		TYPE_DICTIONARY:
			var d := {}
			for k in v:
				d[k] = sample(v[k], rng)
			return d
		_:
			return v


## A spec vector: `[x, y, z]` where each component may itself be a `[lo, hi]` range.
static func sample_vec3(v: Variant, rng: RandomNumberGenerator, fallback := Vector3.ZERO) -> Vector3:
	if typeof(v) != TYPE_ARRAY or (v as Array).size() != 3:
		return fallback
	var a: Array = v
	return Vector3(float(sample(a[0], rng)) if typeof(a[0]) == TYPE_ARRAY else float(a[0]),
		float(sample(a[1], rng)) if typeof(a[1]) == TYPE_ARRAY else float(a[1]),
		float(sample(a[2], rng)) if typeof(a[2]) == TYPE_ARRAY else float(a[2]))


static func _fail(msg: String) -> Dictionary:
	return {"ok": false, "error": msg, "path": "", "name": "", "loop": true,
		"transition": "", "sensitivity": -1.0, "sequence": [], "tail": []}


static func _is_num(v: Variant) -> bool:
	return typeof(v) == TYPE_INT or typeof(v) == TYPE_FLOAT


# Recursive defs/use expansion. `depth` counts USE hops only (cycle guard), not tree
# depth. Node keys always win over fragment keys; dict-vs-dict conflicts merge deep.
static func _expand(node: Variant, defs: Dictionary, depth: int, ctx: Dictionary) -> Variant:
	if ctx.error != "":
		return node
	match typeof(node):
		TYPE_DICTIONARY:
			var d: Dictionary = node
			if d.has("use"):
				if depth >= MAX_USE_DEPTH:
					ctx.error = "'use' nesting deeper than %d (a def cycle?)" % MAX_USE_DEPTH
					return node
				var key := String(d["use"])
				if not defs.has(key):
					ctx.error = "unknown def '%s' (available: %s)" % [key, ", ".join(defs.keys())]
					return node
				var frag: Variant = _expand(defs[key], defs, depth + 1, ctx)
				if ctx.error != "":
					return node
				if d.size() == 1:
					return frag                       # a pure reference becomes the fragment itself
				if typeof(frag) != TYPE_DICTIONARY:
					ctx.error = "def '%s' must be a map to merge extra keys over it" % key
					return node
				var out: Dictionary = (frag as Dictionary).duplicate(true)
				for k in d:
					if k == "use":
						continue
					var val: Variant = _expand(d[k], defs, depth, ctx)
					if out.has(k) and typeof(out[k]) == TYPE_DICTIONARY and typeof(val) == TYPE_DICTIONARY:
						out[k] = _merge(out[k], val)
					else:
						out[k] = val
				return out
			var plain := {}
			for k in d:
				plain[k] = _expand(d[k], defs, depth, ctx)
			return plain
		TYPE_ARRAY:
			var arr := []
			for it in node:
				arr.append(_expand(it, defs, depth, ctx))
			return arr
		_:
			return node


# Deep merge: `over`'s keys win; nested maps merge recursively; arrays replace whole.
static func _merge(base: Dictionary, over: Dictionary) -> Dictionary:
	var out := base.duplicate(true)
	for k in over:
		if out.has(k) and typeof(out[k]) == TYPE_DICTIONARY and typeof(over[k]) == TYPE_DICTIONARY:
			out[k] = _merge(out[k], over[k])
		else:
			out[k] = over[k]
	return out


# Structural validation with per-path messages. Kept to what would otherwise fail
# silently or crash deep inside a scene; scenes still own their semantic defaults.
static func _validate(data: Dictionary, path: String) -> String:
	if not data.has("sequence"):
		return "storyboard %s has no 'sequence' array" % path
	if typeof(data["sequence"]) != TYPE_ARRAY or (data["sequence"] as Array).is_empty():
		return "storyboard %s sequence is empty" % path
	var seq: Array = data["sequence"]
	for idx in seq.size():
		var err := _validate_entry(seq[idx], "%s: sequence[%d]" % [path, idx])
		if err != "":
			return err
	if data.has("tail"):
		if typeof(data["tail"]) != TYPE_ARRAY:
			return "storyboard %s 'tail' must be a list of entries" % path
		var tail: Array = data["tail"]
		for idx in tail.size():
			var err := _validate_entry(tail[idx], "%s: tail[%d]" % [path, idx])
			if err != "":
				return err
	return ""


static func _validate_entry(entry: Variant, where: String) -> String:
	if typeof(entry) != TYPE_DICTIONARY:
		return where + " must be a map"
	var e: Dictionary = entry
	if String(e.get("scene", "")).is_empty():
		return where + " is missing 'scene'"
	for k in ["hold", "min_hold", "max_hold", "sensitivity", "elastic"]:
		if e.has(k) and not _is_num(e[k]):
			return "%s.%s must be a number" % [where, k]
	if String(e["scene"]) == "stage":
		return _validate_stage(e, where)
	return ""


# Shape checks for a data-driven `stage` entry: a cast of uniquely-id'd actors and a
# track of timed spans (see scenes/stage.gd for the semantics).
static func _validate_stage(e: Dictionary, where: String) -> String:
	if e.has("cast"):
		if typeof(e["cast"]) != TYPE_ARRAY:
			return where + ".cast must be a list of actors"
		var ids := {}
		for j in (e["cast"] as Array).size():
			var a: Variant = e["cast"][j]
			var aw := "%s.cast[%d]" % [where, j]
			if typeof(a) != TYPE_DICTIONARY:
				return aw + " must be a map"
			var id := String((a as Dictionary).get("id", ""))
			if id.is_empty():
				return aw + " is missing 'id'"
			if String((a as Dictionary).get("kind", "")).is_empty():
				return aw + " is missing 'kind'"
			if ids.has(id):
				return "%s duplicates actor id '%s'" % [aw, id]
			ids[id] = true
	if e.has("track"):
		if typeof(e["track"]) != TYPE_DICTIONARY:
			return where + ".track must be a map with 'spans'"
		var tr: Dictionary = e["track"]
		if typeof(tr.get("spans", [])) != TYPE_ARRAY:
			return where + ".track.spans must be a list"
		for j in (tr.get("spans", []) as Array).size():
			var s: Variant = tr["spans"][j]
			var sw := "%s.track.spans[%d]" % [where, j]
			if typeof(s) != TYPE_DICTIONARY:
				return sw + " must be a map"
			var sd: Dictionary = s
			if String(sd.get("action", "")).is_empty():
				return sw + " is missing 'action'"
			if not (sd.has("at") or sd.has("from")):
				return sw + " needs 'at' or 'from'"
			for k in ["at", "from", "to", "by"]:
				if sd.has(k) and not _is_num(sd[k]):
					return "%s.%s must be a number (seconds of the track's nominal)" % [sw, k]
	return ""
