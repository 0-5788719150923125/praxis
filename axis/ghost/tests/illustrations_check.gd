extends SceneTree

## The illustration library's contract, headless: no Settings autoload, no codex, no network.
##
##   godot --headless --path . --script tests/illustrations_check.gd
##
## What fails SILENTLY here, and so is asserted: a prompt that drops the author's words or the
## "references are style only" instruction (the model then paints the reference's subject); a
## cache key that changes when a comment is re-wrapped (every picture repaints on the next
## edit - quota spent for nothing); a reroll that REPLACES instead of adding a version; an
## output resolver that picks up a png from a different run; and a read-only process (a
## render, a probe) that starts jobs or writes the index.

var _fails := 0


func _check(ok: bool, what: String) -> void:
	print(("  ok    " if ok else "  FAIL  ") + what)
	if not ok:
		_fails += 1


func _init() -> void:
	var I = load("res://scripts/illustrations.gd")
	var G = load("res://scripts/image_gen.gd")
	var M = load("res://scripts/manuscript.gd")

	print("-- prompt")
	var desc := "A gaunt gray wolf detective, one paw raised to his brow as if peering through fog"
	var p: String = I.build_prompt(desc, "full", "ink and wash", 2, "/x/image.png")
	_check(p.contains(desc), "the description goes in verbatim")
	_check(p.contains("/x/image.png"), "the exact save path is named")
	_check(p.contains("ink and wash"), "the style is included")
	_check(p.contains("STYLE REFERENCES ONLY") and p.contains("Do NOT copy their subjects"),
		"references are declared style-only")
	_check(p.contains("PORTRAIT") and p.contains("2:3"), "a full page asks for a portrait plate")
	_check(p.contains("no text") and p.contains("no watermark"), "the prohibitions are always there")
	var q: String = I.build_prompt(desc, "inline", "", 0, "/x/image.png")
	_check(not q.contains("REFERENCES"), "no reference instruction without references")
	_check(not q.contains("STYLE (") , "no style line without a style")
	_check(q.contains("LANDSCAPE"), "an inline picture asks for a landscape vignette")

	print("-- cache key")
	var k1: String = M.image_key(desc)
	_check(k1 == M.image_key("  A gaunt gray wolf detective, one paw raised\n to his  brow as if peering through fog "),
		"re-wrapping the description keeps the key")
	_check(k1 != M.image_key(desc + " at night"), "an edited description is a new key")
	_check(I.look_signature("ink", ["a/1.png", "b/2.png"]) == I.look_signature(" ink ", ["b/2.png", "a/1.png"]),
		"the look signature ignores reference order and padding")
	_check(I.look_signature("ink", []) != I.look_signature("wash", []), "a style change changes the look")

	print("-- versions")
	var store := {}
	I.use_for_test(store)
	var dir := ProjectSettings.globalize_path("user://illustrations_check")
	DirAccess.make_dir_recursive_absolute(dir)
	var img := Image.create(8, 8, false, Image.FORMAT_RGB8)
	var files: Array = []
	for n in 2:
		var f := dir.path_join("v%d.png" % n)
		img.fill(Color(n, 0, 0))
		img.save_png(f)
		files.append(f)
	I._put_entry(k1, {"prompt": desc, "placement": "full", "current": 1, "versions": [
		{"file": files[0], "sig": I.current_signature()},
		{"file": files[1], "sig": "other"}]})
	_check(I.path_for(k1) == files[1], "the chosen version is served")
	_check(I.is_stale(k1), "a version made under another look is stale")
	I.select_version(k1, 0)
	_check(I.path_for(k1) == files[0] and not I.is_stale(k1), "stepping back selects the earlier version")
	_check(I.versions(k1).size() == 2, "selecting keeps every version")
	_check(I.path_for("nope") == "" and I.status("nope") == "missing", "an unknown key is missing, not an error")
	_check(I.status(k1) == "ready", "a key with a version is ready")

	print("-- landing a job")
	var fake := {"key": k1, "gen": G.make("codex"), "target": files[1], "sig": "s",
		"backend": "codex", "description": desc, "placement": "full"}
	var rev: int = I.revision
	I._land(k1, fake)
	_check(I.versions(k1).size() == 3, "a reroll ADDS a version")
	_check(I.current_index(k1) == 2, "...and selects it")
	_check(I.revision > rev, "...and bumps the revision")

	print("-- output resolution")
	var home := dir.path_join("codex_home")
	var tid := "01test-thread"
	var gdir := home.path_join("generated_images").path_join(tid)
	DirAccess.make_dir_recursive_absolute(gdir)
	img.save_png(gdir.path_join("exec-a.png"))
	var ev := dir.path_join("events.jsonl")
	var fa := FileAccess.open(ev, FileAccess.WRITE)
	fa.store_string('{"type":"thread.started","thread_id":"%s"}\nnot json\n' % tid
		+ '{"type":"item.completed","item":{"type":"agent_message","text":"saved it"}}\n')
	fa.close()
	OS.set_environment("CODEX_HOME", home)
	var cx = G.make("codex")
	_check(G.Codex.thread_id(ev) == tid, "the thread id is read from the event log")
	_check(G.Codex.last_message(ev) == "saved it", "the agent's last words are kept for errors")
	var job := {"events": ev, "target": dir.path_join("absent.png"), "started": 0}
	_check(cx.resolve(job) == gdir.path_join("exec-a.png"), "no copy at the target: fall back to the thread's own png")
	job["target"] = files[0]
	_check(cx.resolve(job) == files[0], "the requested target wins when it exists")
	job["target"] = dir.path_join("absent.png")
	job["started"] = int(Time.get_unix_time_from_system()) + 100
	_check(cx.resolve(job) == "", "a png older than the job is never taken")

	print("-- read-only")
	var ro := {}
	I.use_for_test(ro, true)
	_check(I.generate({"key": "k", "prompt": "a cat", "placement": "inline"}) != "",
		"a read-only session refuses to generate")
	_check(I.busy() == 0, "...and queues nothing")
	I.set_style("changed")
	I.add_references([files[0]])
	_check(ro.is_empty(), "...and writes nothing")

	print("illustrations_check: %s" % ("PASS" if _fails == 0 else "%d FAILED" % _fails))
	quit(1 if _fails > 0 else 0)
