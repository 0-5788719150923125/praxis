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

	print("-- styles per kind")
	I.use_for_test({"style": "old single style"})
	_check(I.style("image") == "old single style" and I.style("sketch") == "",
		"a library from before kinds keeps its one style as the pictures' and none for sketches")
	I.set_style("ballpoint, schematic", "sketch")
	_check(I.style("image") == "old single style" and I.style("sketch") == "ballpoint, schematic",
		"setting the sketch style touched the pictures'")
	_check(I.current_signature("image") != I.current_signature("sketch"), "the two kinds share a signature")
	I.set_look({"style": {"image": "finger-painted", "sketch": "pen"}})
	_check(I.style("image") == "finger-painted" and I.style("sketch") == "pen", "a style map did not set both kinds")
	_check(I.look()["style"] is Dictionary, "the look is not written back as a map")
	I.set_look({"style": "a bare string"})
	_check(I.style("image") == "a bare string" and I.style("sketch") == "",
		"a bare-string style is not the pictures' alone - the block is the WHOLE look")
	var sk: String = I.build_prompt("a flask", "sketch", "pen", 0, "/t.png")
	_check(sk.contains("every sketch") and sk.contains("pen") and not sk.contains("no lettering"),
		"a sketch prompt does not carry its own style, or forbids the labels its description asks for")
	var skf := ProjectSettings.globalize_path("user://illustrations_check_sketch.png")
	Image.create(4, 4, false, Image.FORMAT_RGB8).save_png(skf)
	I._put_entry("skk", {"placement": "sketch", "current": 0, "versions": [{"file": skf, "sig": I.current_signature("sketch")}]})
	I.set_style("finger-painted, different", "image")
	_check(not I.is_stale("skk"), "changing the pictures' style made a sketch stale")
	I.set_style("pen, different", "sketch")
	_check(I.is_stale("skk"), "changing the sketches' style did not make a sketch stale")

	print("-- references per kind, and the chain")
	var rdir := ProjectSettings.globalize_path("user://illustrations_check_refs")
	DirAccess.make_dir_recursive_absolute(rdir)
	var pics: Array = []
	for n in 7:
		var f := rdir.path_join("p%d.png" % n)
		var pimg := Image.create(4, 4, false, Image.FORMAT_RGB8)
		pimg.fill(Color(float(n) / 7.0, 0.5, 0.2))
		pimg.save_png(f)
		pics.append(f)
	var rs := {"refs": [pics[0]]}
	I.use_for_test(rs)
	_check(I.references("image").size() == 1 and I.references("sketch").is_empty(),
		"a library from before kinds lost its references, or gave them to sketches")
	I.add_references([pics[1]], "sketch")
	_check(I.references("sketch").size() == 1 and I.references("image").size() == 1,
		"adding a sketch reference touched the pictures'")
	_check(I.current_signature("sketch") != I.look_signature(I.style("sketch"), []),
		"a sketch reference does not change the sketches' look")
	var lk: Dictionary = I.look()
	_check(lk["references"] is Dictionary and (lk["references"] as Dictionary).has("sketch"),
		"the references are not written back by kind")
	I.set_look({"painter": "codex"})
	_check(I.references("image").is_empty() and I.references("sketch").is_empty(),
		"a chapter naming no references kept the last chapter's")
	_check(not I.look().has("references") and not I.look().has("style"),
		"an empty look writes empty keys into the document")
	I.set_look({"references": [pics[2]]})
	_check(I.references("image").size() == 1, "a bare reference list is not the pictures'")
	I.set_look({})
	# THE CHAIN: seven pictures and a sketch, the first six made.
	var chap: Array = []
	for n in 7:
		chap.append({"key": "c%d" % n, "prompt": "p", "placement": "inline"})
		if n < 6:
			I._put_entry("c%d" % n, {"placement": "inline", "current": 0, "versions": [{"file": pics[n], "sig": ""}]})
	chap.insert(3, {"key": "s0", "prompt": "s", "placement": "sketch"})
	I.set_chapter(chap)
	_check(I.chain_refs("c0").is_empty(), "the first picture has something to be referenced to")
	_check(I.chain_refs("c2") == [pics[0], pics[1]], "picture 3 is not referenced to 1 and 2")
	var last: Array = I.chain_refs("c6")
	_check(last.size() == I.CHAIN_MAX and last[0] == pics[0] and last[last.size() - 1] == pics[5],
		"the chain is not the first plus the most recent, %d at most: %s" % [I.CHAIN_MAX, last])
	_check(I.chain_refs("s0").is_empty(), "a sketch was chained to the pictures")
	var cp: String = I.build_prompt("x", "inline", "", 0, "/t.png", 2)
	_check(cp.contains("EARLIER PICTURES") and not cp.contains("STYLE REFERENCES ONLY"),
		"a chained request is not told its attachments are this book's earlier pictures")
	var bp: String = I.build_prompt("x", "inline", "", 3, "/t.png", 2)
	_check(bp.contains("the first 3 attached images") and bp.contains("the last 2 attached images"),
		"references and the chain together are not told apart by their place in the attachments")
	# ONE AT A TIME for a self-referenced kind; a kind with its own references runs in parallel.
	I._queue = [{"key": "c1", "placement": "inline"}, {"key": "c2", "placement": "inline"},
		{"key": "s0", "placement": "sketch"}]
	I._jobs = {"c0": {"placement": "inline"}}
	_check(I._next_startable() == 2, "a chained picture started beside one of its own kind")
	I.add_references([pics[3]], "image")
	_check(I._next_startable() == 2, "static references switched the chain off - they ride together")
	I.set_self_reference(false, "image")
	_check(I._next_startable() == 0, "a kind with self-reference off is held back")
	_check(I.references("image") == [I.references("image")[0]] and I.references("image").size() == 1,
		"the chain leaked into the static references")
	var lk2: Dictionary = I.look()
	_check(lk2["self_reference"] == {"image": false, "sketch": true},
		"the switch is not written for every kind: %s" % [lk2.get("self_reference")])
	I.set_look({})
	_check(I.self_reference("image") and I.self_reference("sketch"),
		"a document naming no switch did not default it on")
	I.set_look({"self_reference": {"sketch": false}})
	_check(I.self_reference("image") and not I.self_reference("sketch"), "the switch is not per kind")
	I._queue = []
	I._jobs = {}

	I.set_look({})
	print("-- reference instructions")
	I.use_for_test({})
	var lk3: Dictionary = I.look()
	_check((lk3["reference_prompt"] as Dictionary).has("image") and (lk3["reference_prompt"] as Dictionary).has("sketch")
		and (lk3["self_reference_prompt"] as Dictionary).has("sketch"),
		"the reference instructions are not written out, by kind, for the author to edit")
	var sig0: String = I.current_signature("image")
	I.set_look(lk3)
	_check(I.current_signature("image") == sig0, "writing the default instructions back made the pictures stale")
	I.set_look({"reference_prompt": {"image": "Copy the palette only."}})
	_check(I.ref_prompt("references", "image") == "Copy the palette only."
		and I.ref_prompt("references", "sketch") == String(I.REF_PROMPT_DEFAULTS["references"]["sketch"]),
		"an edited instruction did not land on its own kind alone")
	_check(I.current_signature("image") != sig0, "an edited instruction did not change the look")
	_check(I.current_signature("sketch") == I.look_signature("", []), "editing the pictures' instruction touched the sketches' look")
	var rp: String = I.build_prompt("x", "inline", "", 2, "/t.png", 1, I.ref_prompt("references", "image"),
		I.ref_prompt("self_reference", "image"))
	_check(rp.contains("REFERENCES (the first 2 attached images): Copy the palette only.")
		and rp.contains("EARLIER PICTURES (the last 1 attached image):"),
		"the author's instruction does not reach the request with its attachments named")
	I.set_look({})
	_check(I.ref_prompt("references", "image") == String(I.REF_PROMPT_DEFAULTS["references"]["image"]),
		"a chapter naming no instruction kept the last chapter's")

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

	print("-- deleting")
	var dk := "delk"
	var dfiles: Array = []
	for n in 3:
		var f := ProjectSettings.globalize_path("user://illustrations_test").path_join(dk).path_join("v%03d.png" % (n + 1))
		DirAccess.make_dir_recursive_absolute(f.get_base_dir())
		img.save_png(f)
		dfiles.append(f)
	I._put_entry(dk, {"placement": "inline", "current": 2, "versions": [
		{"file": dfiles[0], "sig": ""}, {"file": dfiles[1], "sig": ""}, {"file": dfiles[2], "sig": ""}]})
	var rv: int = I.revision
	_check(I.delete_version(dk, 2), "deleting the current version refused")
	_check(not FileAccess.file_exists(dfiles[2]), "the deleted version's file is still on disk")
	_check(I.path_for(dk) == dfiles[1] and I.versions(dk).size() == 2,
		"deleting the current version did not fall back to the one before it")
	_check(I.revision > rv, "a delete did not tell the pages to redraw")
	var land := {"key": dk, "gen": G.make("codex"), "target": files[0], "sig": "s",
		"backend": "codex", "description": "d", "placement": "inline"}
	# v001 goes too, leaving only v002 - which is exactly the name "the count plus one" gives the
	# next picture. A job that lands now must not overwrite it.
	I.delete_version(dk, 0)
	var before := FileAccess.get_file_as_bytes(dfiles[1])
	I._land(dk, land)
	_check(FileAccess.get_file_as_bytes(dfiles[1]) == before and I.versions(dk).size() == 2,
		"a picture landing after a delete overwrote a version that still exists")
	I.delete_version(dk, 0)
	I.delete_version(dk, 0)
	_check(I.path_for(dk) == "" and I.status(dk) == "missing", "deleting every version did not leave it missing")
	_check(not I.delete_version(dk, 0), "deleting from nothing claimed to delete something")

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
