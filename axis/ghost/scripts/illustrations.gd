extends RefCounted
class_name Illustrations

## Illustrations - the library of generated pictures a book prints.
##
## A chapter describes its pictures in `<!-- image: ... -->` markers ([Manuscript.images]);
## this turns each description into a PNG through an [ImageGen] backend and remembers it.
## The book vehicle only ever asks [method path_for], so it neither knows nor cares who
## painted what.
##
## THE CACHE IS THE POINT. A picture costs about a minute and a share of the author's quota,
## so nothing is ever made twice by accident: the key is [method Manuscript.image_key] of the
## description, the files live under `user://illustrations/<key>/vNNN.png`, and the index of
## which versions exist and which one is chosen is persisted through [Settings] like every
## other setting. Relaunching, re-reading the chapter or re-wrapping a comment changes nothing.
##
## GENERATION IS EXPLICIT. Nothing here starts a job on its own - not on launch, not on a
## chapter being opened - because the cost is the author's. The panel's buttons are the only
## callers of [method generate].
##
## A REROLL IS A NEW VERSION, never a replacement. The previous ones stay on disk and the
## panel can step back to any of them, since "the second try was worse" is a normal outcome.
##
## STALE, NOT INVALID. Changing the style or the reference set does not throw pictures away:
## each version records the signature of the look it was made under ([method look_signature]),
## and one made under a different look is flagged stale and still used until the author
## rerolls it. Invalidating the whole book because a comma changed in the style would be the
## expensive kind of correct.
##
## Settings is reached by NAME at runtime rather than as the autoload identifier, so this file
## also loads in a bare `--script` gate where no autoload exists (see [method use_for_test]).

## Pictures under `<key>/vNNN.png`, reference copies under `refs/`, job dirs under `jobs/`.
const DIR := "user://illustrations"
const SECTION := "illustrations"
## How many pictures may be painting at once. Each is a network round trip that mostly waits,
## so more than one is cheap - and more than a couple is a burst against the author's quota.
const MAX_JOBS := 2
## A job that runs this long is abandoned rather than watched forever. Measured ~60 s each.
const JOB_TIMEOUT_S := 600
const IMAGE_EXTS := ["png", "jpg", "jpeg", "webp"]

## Bumped whenever a picture lands or the chosen version changes, so a vehicle holding page
## textures knows to re-typeset without polling the disk.
static var revision := 0

## key -> job Dictionary, for the pictures being painted right now.
static var _jobs := {}
## Pictures waiting for a free slot: [{key, prompt, placement}], in request order.
static var _queue: Array = []
## key -> the last failure, for the panel. Session-only: a failure is news, not state.
static var _errors := {}

# THE TEST SEAM, the same shape as Films.use_for_test: an in-memory store instead of the
# author's settings, and a flag, because an EMPTY library is a state worth testing.
static var _test_active := false
static var _test_store := {}
static var _test_read_only := false
## Where pictures, references and jobs live. Moved aside under test so a gate never writes
## into the author's own library.
static var _root := DIR


static func use_for_test(store := {}, read_only := false) -> void:
	_test_active = true
	_test_store = store
	_test_read_only = read_only
	_root = "user://illustrations_test"
	_jobs = {}
	_queue = []
	_errors = {}


# --- storage -----------------------------------------------------------------

static func _settings() -> Node:
	var tree := Engine.get_main_loop() as SceneTree
	return tree.root.get_node_or_null("Settings") if tree != null else null


static func _read(key: String, dflt: Variant) -> Variant:
	if _test_active:
		var v: Variant = _test_store.get(key, dflt)
		return v.duplicate(true) if (v is Dictionary or v is Array) else v
	var st := _settings()
	return st.read(SECTION, key, dflt) if st != null else dflt


static func _write(key: String, value: Variant) -> void:
	if read_only():
		return
	if _test_active:
		_test_store[key] = value.duplicate(true) if (value is Dictionary or value is Array) else value
		return
	var st := _settings()
	if st != null:
		st.write(SECTION, key, value)


## A render, the offline analyzer and a test probe boot this app against the author's own
## settings. None of them may spend quota or rewrite the index.
static func read_only() -> bool:
	if _test_active:
		return _test_read_only
	var st := _settings()
	return st == null or bool(st.is_read_only())


# --- the look ----------------------------------------------------------------

static func style() -> String:
	return String(_read("style", ""))


static func set_style(text: String) -> void:
	_write("style", text)


## The reference images, as `user://` paths of the COPIES this library holds.
static func references() -> Array:
	var out: Array = []
	for p in (_read("refs", []) as Array):
		if FileAccess.file_exists(String(p)):
			out.append(String(p))
	return out


## Copy [param paths] into the library. A copy, because a reference that vanishes when the
## author tidies their Downloads folder would silently change every later picture. Named by
## content, so importing the same file twice is one reference. Returns one line per file that
## could not be taken.
static func add_references(paths: Array) -> PackedStringArray:
	var errs := PackedStringArray()
	if read_only():
		errs.append("read-only session")
		return errs
	var list := references()
	for src in paths:
		var dest := _take(String(src), errs)
		if not dest.is_empty() and not list.has(dest):
			list.append(dest)
	_write("refs", list)
	return errs


## One file into the library, by content: the copy's `user://` path, or "" with the reason
## appended to [param errs]. A file that IS already a library copy comes back as itself.
static func _take(s: String, errs: PackedStringArray) -> String:
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(_root.path_join("refs")))
	var ext := s.get_extension().to_lower()
	if not IMAGE_EXTS.has(ext):
		errs.append("%s: not an image" % s.get_file())
		return ""
	var bytes := FileAccess.get_file_as_bytes(s)
	if bytes.is_empty():
		errs.append("%s: unreadable" % s.get_file())
		return ""
	var dest := _root.path_join("refs").path_join(_sha(bytes).substr(0, 16) + "." + ext)
	if not FileAccess.file_exists(dest):
		var f := FileAccess.open(dest, FileAccess.WRITE)
		if f == null:
			errs.append("%s: could not copy" % s.get_file())
			return ""
		f.store_buffer(bytes)
		f.close()
	return dest


static func remove_reference(i: int) -> void:
	var list := references()
	if i < 0 or i >= list.size():
		return
	list.remove_at(i)
	_write("refs", list)
	# The COPY stays: references are per document now (see [method look]), so another
	# chapter may still name this file, and a copy named by its content costs nothing to keep.


static func _sha(bytes: PackedByteArray) -> String:
	var h := HashingContext.new()
	h.start(HashingContext.HASH_SHA256)
	h.update(bytes)
	return h.finish().hex_encode()


## The signature of the look a picture is made under: the style text and the reference set.
## Reference files are named by content, so their names ARE their content.
static func look_signature(style_text: String, refs: Array) -> String:
	var names := PackedStringArray()
	for r in refs:
		names.append(String(r).get_file())
	names.sort()
	return (style_text.strip_edges() + "|" + ",".join(names)).sha256_text().substr(0, 12)


static func current_signature() -> String:
	return look_signature(style(), references())


## THE LOOK AS A DOCUMENT CARRIES IT - the painter, the style and the reference images -
## so a chapter keeps the look its pictures were made under, the way it keeps its cast. The
## references are ABSOLUTE paths to the library's own copies: links a person can open, and
## files that stay put whatever happens to the originals.
static func look() -> Dictionary:
	var refs: Array = []
	for r in references():
		refs.append(ProjectSettings.globalize_path(String(r)))
	return {"painter": backend(), "style": style(), "references": refs}


## ...and back. Any image path is accepted (it is copied in by content, so a library copy is
## itself and a new file is imported); the reference list becomes EXACTLY this one. A key
## the block does not carry is left as it is. Returns one line per file that could not be taken.
static func set_look(d: Dictionary) -> PackedStringArray:
	var errs := PackedStringArray()
	if read_only():
		return errs
	if d.has("painter"):
		set_backend(String(d["painter"]))
	if d.has("style"):
		set_style(String(d["style"]))
	if d.get("references") is Array:
		var list: Array = []
		for p in d["references"] as Array:
			var dest := _take(String(p), errs)
			if not dest.is_empty() and not list.has(dest):
				list.append(dest)
		_write("refs", list)
	return errs


static func backend() -> String:
	var k := String(_read("backend", ImageGen.REGISTRY.keys()[0]))
	return k if ImageGen.REGISTRY.has(k) else String(ImageGen.REGISTRY.keys()[0])


static func set_backend(key: String) -> void:
	if ImageGen.REGISTRY.has(key):
		_write("backend", key)


# --- the prompt --------------------------------------------------------------

## THE PROMPT, assembled. Pure, so the gate can hold it to its rules.
##
## The description goes in VERBATIM: it is the author's own sentence and every clause in it
## was chosen. The rest is what every picture in the book shares - the style, what the
## references are FOR, the shape of the space it will be printed in - and the prohibitions,
## because an image model left to itself captions a courtroom and frames a portrait.
##
## THE REFERENCES ARE STYLE, NOT SUBJECT, and the prompt has to say so outright. Handed a
## picture of a wolf and asked for a courtroom, a model will happily put the wolf in the
## courtroom.
static func build_prompt(description: String, placement: String, style_text: String,
		ref_count: int, target: String) -> String:
	var lines := PackedStringArray()
	lines.append("Use your built-in image generation tool to create exactly ONE image, then "
		+ "save it as a PNG at this exact path: %s" % target)
	lines.append("Do not create or modify any other file. Reply with the saved path only.")
	lines.append("")
	lines.append("THE PICTURE (the author's description - follow it exactly):")
	lines.append(description.strip_edges())
	lines.append("")
	lines.append("FORMAT: " + placement_guide(placement))
	if not style_text.strip_edges().is_empty():
		lines.append("STYLE (applies to every illustration in this book): " + style_text.strip_edges())
	if ref_count > 0:
		lines.append("REFERENCES: the %d attached image%s %s STYLE REFERENCES ONLY. Match "
			% [ref_count, "" if ref_count == 1 else "s", "is a" if ref_count == 1 else "are"]
			+ "their rendering style: medium, palette, linework, texture, lighting and level of "
			+ "detail. Do NOT copy their subjects, characters, objects or composition - the "
			+ "content comes only from the description above.")
	lines.append("ALWAYS: no text, no captions, no lettering, no borders, no frames, "
		+ "no watermark, no signature.")
	return "\n".join(lines)


## What each placement asks for. A full page is a plate; an inline picture has to read at a
## few inches wide inside a column of prose, so it wants one clear subject.
static func placement_guide(placement: String) -> String:
	if placement == "full":
		return ("a full-page book illustration plate, PORTRAIT orientation, 2:3 aspect "
			+ "(1024x1536), composed to fill the whole page edge to edge.")
	return ("a smaller vignette printed inside a column of text in a novel, LANDSCAPE "
		+ "orientation, about 3:2 (1536x1024), one clear subject that reads at a small size.")


# --- the index ---------------------------------------------------------------

## The stored record for [param key]: {prompt, placement, versions: [{file, sig, at}], current}.
static func entry(key: String) -> Dictionary:
	var idx: Dictionary = _read("index", {})
	var e: Variant = idx.get(key, {})
	return e if e is Dictionary else {}


static func _put_entry(key: String, e: Dictionary) -> void:
	var idx: Dictionary = _read("index", {})
	idx[key] = e
	_write("index", idx)


## The versions that still exist on disk, oldest first.
static func versions(key: String) -> Array:
	var out: Array = []
	for v in (entry(key).get("versions", []) as Array):
		if v is Dictionary and FileAccess.file_exists(String((v as Dictionary).get("file", ""))):
			out.append(v)
	return out


static func current_index(key: String) -> int:
	var n := versions(key).size()
	if n == 0:
		return -1
	return clampi(int(entry(key).get("current", n - 1)), 0, n - 1)


## The picture currently chosen for [param key], absolute, or "" when there is none yet.
static func path_for(key: String) -> String:
	var i := current_index(key)
	if i < 0:
		return ""
	return ProjectSettings.globalize_path(String((versions(key)[i] as Dictionary)["file"]))


static func select_version(key: String, i: int) -> void:
	var n := versions(key).size()
	if n == 0 or read_only():
		return
	var e := entry(key)
	e["current"] = clampi(i, 0, n - 1)
	_put_entry(key, e)
	revision += 1


## Made under a different style or reference set than the one now in force.
static func is_stale(key: String) -> bool:
	var i := current_index(key)
	if i < 0:
		return false
	return String((versions(key)[i] as Dictionary).get("sig", "")) != current_signature()


## "missing", "queued", "running", "ready" or "error".
static func status(key: String) -> String:
	if _jobs.has(key):
		return "running"
	for q in _queue:
		if String((q as Dictionary)["key"]) == key:
			return "queued"
	if _errors.has(key) and current_index(key) < 0:
		return "error"
	return "ready" if current_index(key) >= 0 else "missing"


static func error_of(key: String) -> String:
	return String(_errors.get(key, ""))


static func busy() -> int:
	return _jobs.size() + _queue.size()


# --- generation --------------------------------------------------------------

## Ask for a picture of [param image] (a [Manuscript.images] row: key, prompt, placement). A
## reroll when one already exists. Returns "" when queued, else why not.
static func generate(image: Dictionary) -> String:
	if read_only():
		return "this session is read-only (a render or a probe) - nothing is generated here"
	var key := String(image.get("key", ""))
	if key.is_empty() or String(image.get("prompt", "")).strip_edges().is_empty():
		return "no description"
	if status(key) in ["queued", "running"]:
		return ""
	if not ImageGen.make(backend()).available():
		return "%s is not installed" % String(ImageGen.LABELS.get(backend(), backend()))
	_errors.erase(key)
	_queue.append({"key": key, "prompt": String(image["prompt"]),
		"placement": String(image.get("placement", "inline"))})
	pump()
	return ""


## Start what fits, notice what finished. Called from main._process and from the panel, the
## Films.pump rule: a job is a subprocess, and something with a frame has to see it end.
static func pump() -> void:
	if read_only():
		return
	for key in _jobs.keys():
		var job: Dictionary = _jobs[key]
		var pid := int(job["pid"])
		if Subprocess.alive(pid):
			if int(Time.get_unix_time_from_system()) - int(job["started"]) > JOB_TIMEOUT_S:
				Subprocess.stop(pid)
				_errors[key] = "timed out after %d s" % JOB_TIMEOUT_S
				_jobs.erase(key)
			continue
		Subprocess.forget(pid)
		_jobs.erase(key)
		_land(String(key), job)
	while _jobs.size() < MAX_JOBS and not _queue.is_empty():
		_start(_queue.pop_front())


static func _start(req: Dictionary) -> void:
	var key := String(req["key"])
	var gen := ImageGen.make(backend())
	var stamp := Time.get_ticks_msec()
	var dir := ProjectSettings.globalize_path(_root.path_join("jobs").path_join("%s_%d" % [key, stamp]))
	var target := dir.path_join("image.png")
	var refs: Array = []
	for r in references():
		refs.append(ProjectSettings.globalize_path(String(r)))
	var job := {
		"key": key, "dir": dir, "target": target, "refs": refs,
		"placement": String(req["placement"]), "description": String(req["prompt"]),
		"sig": current_signature(), "backend": backend(),
		# A second of slack: file mtimes are whole seconds and the clock read is not.
		"started": int(Time.get_unix_time_from_system()) - 1,
		"prompt": build_prompt(String(req["prompt"]), String(req["placement"]), style(),
			refs.size(), target),
	}
	job["gen"] = gen
	var pid := gen.start(job)
	if pid <= 0:
		_errors[key] = gen.failure(job)
		return
	job["pid"] = pid
	_jobs[key] = job
	print("ghost: painting %s (%s, %d reference%s)" % [key, job["placement"], refs.size(),
		"" if refs.size() == 1 else "s"])


## A job ended. Take its picture into the library as a new version, or record why not.
static func _land(key: String, job: Dictionary) -> void:
	var gen: ImageGen.Backend = job["gen"]
	var made := gen.resolve(job)
	var img := Image.new()
	if made.is_empty() or img.load(made) != OK:
		_errors[key] = gen.failure(job)
		push_warning("ghost: illustration %s failed - %s" % [key, _errors[key]])
		return
	var e := entry(key)
	var vs: Array = e.get("versions", [])
	var file := _root.path_join(key).path_join("v%03d.png" % (vs.size() + 1))
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(file.get_base_dir()))
	# Re-encoded rather than copied: whatever the painter wrote (a JPEG named .png has been
	# seen from image tools), the library holds a real PNG.
	if img.save_png(ProjectSettings.globalize_path(file)) != OK:
		_errors[key] = "could not write %s" % file
		return
	vs.append({"file": file, "sig": String(job["sig"]), "backend": String(job["backend"]),
		"at": int(Time.get_unix_time_from_system())})
	e["versions"] = vs
	e["current"] = vs.size() - 1
	e["prompt"] = String(job["description"])
	e["placement"] = String(job["placement"])
	_put_entry(key, e)
	_errors.erase(key)
	revision += 1
	print("ghost: illustration %s ready (v%d)" % [key, vs.size()])
	# The job's scratch is done with once the picture is in the library. A failed job keeps
	# its directory: the event log in it is the only record of why.
	_remove_tree(String(job.get("dir", "")))


static func _remove_tree(dir: String) -> void:
	if dir.is_empty() or not DirAccess.dir_exists_absolute(dir):
		return
	for f in DirAccess.get_files_at(dir):
		DirAccess.remove_absolute(dir.path_join(f))
	DirAccess.remove_absolute(dir)
