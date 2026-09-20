extends SceneTree

## doc_source_check - the gate for [FrontMatter] and the YAML emitter it writes through.
##
## WHY THIS FILE IS SHARP. Every other gate in this directory protects a picture or a sound;
## this one protects THE AUTHOR'S MANUSCRIPT. Sync mode points ghost at a real file and the
## ↑ button writes into it, so the failure mode is not "the reading sounds wrong", it is a
## chapter with a paragraph missing and no way to tell which one. Nothing about that is
## visible from inside ghost, and a document is usually only read again days later.
##
## So every claim here is two-sided, and the second side is not a formality: a check that
## only shows the good case passing cannot distinguish a careful write from one that
## rewrites the whole file and happens to produce something parseable. Each destructive
## edit is CONSTRUCTED and handed to the verifier, which must refuse it.
##
## Four groups:
##   1. THE ROUND TRIP. Everything [MiniYaml.emit] writes, [MiniYaml.parse] reads back
##      equal - including the cases that are only equal by care: a float on a whole number
##      (which would come back an int without the forced decimal point), a string that
##      looks like a number or a boolean, an empty collection, a key with a colon in it.
##   2. THE SPLIT. What is frontmatter and what is the document, including the cases that
##      look like frontmatter and are not - a horizontal rule, an unterminated fence.
##   3. THE SURGERY. One key replaced, inserted or appended, with the body byte-identical
##      and every foreign line - comments, blank lines, quoting styles, constructs MiniYaml
##      cannot even parse - exactly where it was.
##   4. THE DISK. A real file, written and read back, and a refusal that leaves it alone.
##
## Run: godot --headless --path axis/ghost --script tests/doc_source_check.gd

const MiniYaml_ := preload("res://scripts/yaml.gd")
const FrontMatter_ := preload("res://scripts/front_matter.gd")

const DIR := "user://doc_source_check"

var fails := 0
var checks := 0


func _init() -> void:
	_round_trip()
	_split()
	_surgery()
	_disk()
	if fails == 0:
		print("doc_source_check: ALL OK (%d checks)" % checks)
	else:
		print("doc_source_check: %d FAILURE(S) of %d checks" % [fails, checks])
	quit(1 if fails > 0 else 0)


func _fail(msg: String) -> void:
	fails += 1
	print("doc_source_check: FAIL  " + msg)


func _ok(cond: bool, msg: String) -> void:
	checks += 1
	if not cond:
		_fail(msg)


# --- 1. the round trip --------------------------------------------------------


func _round_trip() -> void:
	# A generative block as the panel really writes one, plus the scalar shapes that are
	# only preserved by care.
	var cases: Array = [
		{"turn": 1.0, "tab": 0, "voices": [
			{"voice": "en_US-libritts-high", "speaker": 0, "tone": 2, "pace": 1.0,
				"pause": 1.0, "dynamics": 0.5, "arc": 0.4, "effort": 0.35,
				"echo": 0.0, "room": 0.0, "resonance": 0.0, "presence": 1.0,
				"ambience": 0.0},
			{"voice": "en_GB-northern_english_male-medium", "speaker": 3, "tone": 0,
				"pace": 0.92, "pause": 2.5, "dynamics": 0.5, "arc": 0.4,
				"effort": 0.35, "echo": 0.1, "room": 0.2, "resonance": 0.0,
				"presence": 0.8, "ambience": 0.0},
		]},
		{"lineage": [1, 7, 12], "traits": {"grit": 0.0, "air": 1.0, "breath": 0.125},
			"genome": {}, "reception": {"prox": 0.0, "freq": 400.0}},
		# THE SCALARS THAT BITE. A bare `1.0` parses back as an int without the forced
		# decimal; `"true"`, `"7"` and `"~"` come back as a bool, an int and a null
		# without quotes; an empty string vanishes; `a: b` inside a value ends the scalar.
		{"whole": 1.0, "zero": 0.0, "neg": -2.5, "tiny": 1e-07, "int": 7,
			"yes": true, "no": false, "nothing": null, "empty_s": "",
			"looks_bool": "true", "looks_int": "7", "looks_null": "~",
			"has_colon": "a: b", "has_hash": "x #y", "dash": "-lead",
			"empty_map": {}, "empty_list": [], "unicode": "café — ok"},
	]
	for case in cases:
		var text: String = MiniYaml_.emit(case)
		var res: Dictionary = MiniYaml_.parse(text)
		if not res.ok:
			_fail("emit produced YAML that will not parse: %s\n%s" % [res.error, text])
			checks += 1
			continue
		_ok(_same(res.data, case), "round trip differs\nemitted:\n%s\nback: %s"
			% [text, str(res.data)])

	# TYPE, not just value: a float that lands on a whole number must come back a float.
	var one: Dictionary = MiniYaml_.parse(MiniYaml_.emit({"x": 1.0})).data
	_ok(typeof(one["x"]) == TYPE_FLOAT, "1.0 came back as a %s, not a float"
		% type_string(typeof(one["x"])))
	# ...and the control: this is exactly what an unquoted emitter would have done.
	var naive: Dictionary = MiniYaml_.parse("x: 1").data
	_ok(typeof(naive["x"]) == TYPE_INT,
		"the control is wrong - a bare `1` should parse as an int")


## Deep equality that does not care whether two dictionaries were built in the same order.
func _same(a: Variant, b: Variant) -> bool:
	if a is Dictionary and b is Dictionary:
		var da := a as Dictionary
		var db := b as Dictionary
		if da.size() != db.size():
			return false
		for k in da:
			if not db.has(k) or not _same(da[k], db[k]):
				return false
		return true
	if a is Array and b is Array:
		var aa := a as Array
		var ab := b as Array
		if aa.size() != ab.size():
			return false
		for i in aa.size():
			if not _same(aa[i], ab[i]):
				return false
		return true
	if a is float or b is float:
		if a == null or b == null:
			return a == b
		return is_equal_approx(float(a), float(b))
	return a == b


# --- 2. the split -------------------------------------------------------------


func _split() -> void:
	var plain := "# Chapter One\n\nThe rain had not stopped.\n"
	var p := FrontMatter_.split(plain)
	_ok(not p.has, "a document with no frontmatter was said to have some")
	_ok(String(p.body) == plain, "a document with no frontmatter lost its body")

	var doc := "---\ntitle: Chapter One\n---\n\n# Chapter One\n\nThe rain.\n"
	p = FrontMatter_.split(doc)
	_ok(p.has, "frontmatter was not recognised")
	_ok(String(p.head) == "title: Chapter One", "frontmatter head is wrong: %s" % p.head)
	_ok(String(p.body) == "\n# Chapter One\n\nThe rain.\n",
		"the body after frontmatter is wrong: %s" % p.body)

	# A HORIZONTAL RULE IS NOT A FENCE. The loose "find a --- anywhere" rule eats the
	# prose above one, which on a Markdown document is most of a section.
	var rule := "Some prose.\n\n---\n\nMore prose.\n"
	p = FrontMatter_.split(rule)
	_ok(not p.has, "a horizontal rule mid-document was read as frontmatter")
	_ok(String(p.body) == rule, "a horizontal rule cost the document its opening")

	# An opening fence with no closing one is a document, not a 4000-line frontmatter.
	p = FrontMatter_.split("---\ntitle: x\n\nno close here\n")
	_ok(not p.has, "an unterminated fence was read as frontmatter")

	# `...` closes a YAML document too, and it is not ours to rewrite.
	p = FrontMatter_.split("---\ntitle: x\n...\nbody\n")
	_ok(p.has and String(p.fence) == "...", "the `...` closing fence was not honoured")

	# CRLF, and a BOM: both survive a round trip through put_block.
	var crlf := "---\r\ntitle: x\r\n---\r\n\r\nBody line.\r\n"
	p = FrontMatter_.split(crlf)
	_ok(p.has and String(p.nl) == "\r\n", "a CRLF document was not detected as one")
	var out := FrontMatter_.put_block(crlf, {"generative": {"tab": 0}})
	_ok(out.contains("\r\n") and not out.contains("\n\n\n"),
		"a CRLF document came back with the wrong line endings")
	_ok(String(FrontMatter_.split(out).body) == String(p.body),
		"a CRLF document's body changed")

	var bom := "﻿---\ntitle: x\n---\nBody.\n"
	out = FrontMatter_.put_block(bom, {"generative": {"tab": 0}})
	_ok(out.begins_with("﻿"), "the byte-order mark was dropped")
	_ok(String(FrontMatter_.split(out).body) == "Body.\n", "a BOM document's body changed")


# --- 3. the surgery -----------------------------------------------------------


## The document every "nothing else moved" check is made against. Deliberately awkward:
## a comment, a blank line between keys, a single-quoted value, a nested list and A BLOCK
## SCALAR, which MiniYaml refuses outright - so any implementation that re-serializes the
## frontmatter rather than editing its lines destroys this file.
const AWKWARD := """---
# the author's own note, which must survive
title: 'Chapter One'
date: 2019-04-01

tags:
  - rain
  - doors
summary: |
  A block scalar. MiniYaml cannot read this,
  and it must not have to.
---

# Chapter One

The rain had not stopped, and the door was still open.

---

A horizontal rule, below the frontmatter.
"""


func _surgery() -> void:
	var block := {"generative": {"turn": 1.0, "tab": 0,
		"voices": [{"voice": "en_US-libritts-high", "pace": 1.0}]}}

	# INSERT into a document that has frontmatter but no ghost key.
	var once := FrontMatter_.put_block(AWKWARD, block)
	_ok(String(FrontMatter_.split(once).body) == String(FrontMatter_.split(AWKWARD).body),
		"inserting a block changed the document body")
	_ok(once.contains("# the author's own note"), "the author's comment was lost")
	_ok(once.contains("summary: |"), "an unparseable key was destroyed by the insert")
	_ok(once.contains("title: 'Chapter One'"), "a quoting style was rewritten")
	_ok(once.contains("ghost:"), "the block was not written at all")
	_ok(FrontMatter_._verify(AWKWARD, once, "ghost") == "",
		"a correct insert was refused by the verifier")

	# REPLACE, twice over: the second write must be idempotent for the same value, and
	# must not accumulate.
	var twice := FrontMatter_.put_block(once, block)
	_ok(twice == once, "writing the same block twice changed the document")
	var changed := FrontMatter_.put_block(once, {"generative": {"turn": 2.0, "tab": 1}})
	_ok(String(FrontMatter_.split(changed).body) == String(FrontMatter_.split(AWKWARD).body),
		"replacing a block changed the document body")
	_ok(changed.contains("# the author's own note"), "replacing a block lost the comment")
	_ok(changed.contains("summary: |"), "replacing a block destroyed an unparseable key")
	_ok(not changed.contains("turn: 1.0"), "the old block survived the replacement")
	_ok(changed.contains("turn: 2.0"), "the new block was not written")
	# ...and the count: one `ghost:` in the file, not two.
	_ok(_count_lines(changed, "ghost:") == 1,
		"there are %d `ghost:` keys after a replace" % _count_lines(changed, "ghost:"))

	# THE BLANK LINE BETWEEN KEYS. A span that swallowed the blank before the next key
	# would eat one line of the author's layout at every save, silently, forever.
	_ok(_count_lines(changed, "") == _count_lines(once, ""),
		"a save changed how many blank lines the frontmatter has")

	# A SECOND PANEL'S VOICE IS NOT THE FIRST'S BUSINESS. Both blocks live under the one
	# `ghost:` key, so writing one must carry the other through.
	var both := FrontMatter_.put_block(once, {"generative": {"turn": 3.0},
		"synthesis": {"lineage": [1, 4]}})
	var read := FrontMatter_.read_block(both)
	_ok(read.ok, "a document carrying two voices could not be read: %s" % read.error)
	_ok((read.data as Dictionary).has("generative") and (read.data as Dictionary).has("synthesis"),
		"a document lost one of its two voices")

	# A DOCUMENT WITH NO FRONTMATTER gets one, and keeps every byte of what it had.
	var bare := "# Chapter One\n\nThe rain.\n"
	var fresh := FrontMatter_.put_block(bare, block)
	_ok(fresh.begins_with("---\n"), "a fresh frontmatter did not open the document")
	_ok(fresh.ends_with(bare), "a fresh frontmatter did not leave the document beneath it")
	_ok(String(FrontMatter_.split(fresh).body).strip_edges() == bare.strip_edges(),
		"a fresh frontmatter changed the body")

	# READ IT BACK, which is the only check that the two halves agree.
	var back := FrontMatter_.read_block(once)
	_ok(back.ok, "a written block could not be read back: %s" % back.error)
	_ok(_same((back.data as Dictionary).get("generative", {}), block["generative"]),
		"a written block read back different: %s" % str(back.data))

	# A KEY MINIYAML CANNOT PARSE IS NOT OUR PROBLEM. AWKWARD carries `summary: |`, which is
	# ordinary frontmatter and outside MiniYaml's subset - and reading the whole head would
	# mean a voice could never be restored from a perfectly normal Markdown document. The
	# control is the same document with the block scalar taken out: both must read.
	_ok(back.ok, "a block scalar elsewhere in the frontmatter blocked the voice")
	_ok(not String(FrontMatter_.split(AWKWARD).body).is_empty(),
		"an unparseable frontmatter cost the document its body")
	# ...and OUR OWN key malformed IS reported, rather than raised or silently ignored.
	var broken := once.replace("ghost:", "ghost:\n  *alias")
	var hard := FrontMatter_.read_block(broken)
	_ok(not hard.ok and not String(hard.error).is_empty(),
		"a malformed `ghost:` block was reported as readable")

	_refusals(once)


## THE SECOND SIDE. Each of these is a destructive edit the verifier MUST refuse - and the
## control is `once` itself, which it must accept (asserted above). A gate with only the
## good cases would pass on a `_verify` that returns "" unconditionally.
func _refusals(good: String) -> void:
	var lines := good.split("\n")
	var cases := {}

	# A body with a paragraph deleted - the failure this whole file exists for.
	var cut: Array = []
	var dropped := false
	for line in lines:
		if not dropped and String(line).begins_with("The rain had not stopped"):
			dropped = true
			continue
		cut.append(line)
	cases["a deleted body paragraph"] = "\n".join(cut)

	# A body with a word changed. Byte-identity, not "looks similar".
	cases["a single changed word in the body"] = good.replace("the door", "a door")

	# The author's own comment deleted.
	var nocomment: Array = []
	for line in lines:
		if String(line).begins_with("# the author's own note"):
			continue
		nocomment.append(line)
	cases["a deleted frontmatter comment"] = "\n".join(nocomment)

	# Another key's quoting rewritten - the signature of a re-serializing writer.
	cases["a requoted foreign key"] = good.replace("title: 'Chapter One'", "title: Chapter One")

	# The frontmatter removed entirely.
	cases["frontmatter removed altogether"] = String(FrontMatter_.split(good).body)

	for why in cases:
		checks += 1
		var reason: String = FrontMatter_._verify(good, String(cases[why]), "ghost")
		if reason.is_empty():
			_fail("the verifier ACCEPTED %s - it must refuse it" % why)


func _count_lines(text: String, prefix: String) -> int:
	var n := 0
	for line in text.split("\n"):
		if prefix.is_empty():
			if String(line).strip_edges().is_empty():
				n += 1
		elif String(line).begins_with(prefix):
			n += 1
	return n


# --- 4. the disk --------------------------------------------------------------


func _disk() -> void:
	DirAccess.make_dir_recursive_absolute(DIR)
	var path := DIR + "/chapter.md"
	_write(path, AWKWARD)

	var err := FrontMatter_.write_block(path, {"generative": {"turn": 1.5, "tab": 0}})
	_ok(err.is_empty(), "writing to a real file failed: %s" % err)
	var after := _read(path)
	_ok(String(FrontMatter_.split(after).body) == String(FrontMatter_.split(AWKWARD).body),
		"writing to a real file changed its body")
	var read := FrontMatter_.read_block(after)
	# AWKWARD's block scalar is still in there and must not matter: the write went through
	# it, and so does the read.
	_ok(read.ok and is_equal_approx(
			float(((read.data as Dictionary).get("generative", {}) as Dictionary).get("turn", 0.0)),
			1.5),
		"the voice did not survive a document with a block scalar in it: %s" % str(read.data))
	_ok(after.contains("turn: 1.5"), "the block did not reach the file")
	_ok(after.contains("summary: |"), "writing to a real file destroyed a block scalar")

	# NO TEMP FILE LEFT BEHIND. An interrupted write is the reason it exists; a successful
	# one that leaves it is litter in the author's own directory.
	_ok(not FileAccess.file_exists(path + FrontMatter_.TEMP_SUFFIX),
		"the temporary file was left beside the document")

	# A CLEAN DOCUMENT, round-tripped through the disk with the voice read back out.
	var clean := DIR + "/clean.md"
	_write(clean, "---\ntitle: x\n---\n\nBody.\n")
	var voice := {"generative": {"turn": 2.0, "voices": [{"voice": "a", "pace": 0.9}]}}
	err = FrontMatter_.write_block(clean, voice)
	_ok(err.is_empty(), "writing a clean document failed: %s" % err)
	read = FrontMatter_.read_block(_read(clean))
	_ok(read.ok and _same(read.data, voice),
		"a voice did not survive the disk: %s" % str(read.data))

	# A MISSING FILE IS REFUSED, not created. Sync mode remembers a path across sessions
	# and the document may have been moved since.
	err = FrontMatter_.write_block(DIR + "/gone.md", {"generative": {}})
	_ok(not err.is_empty(), "writing to a missing file was allowed")
	_ok(not FileAccess.file_exists(DIR + "/gone.md"),
		"a missing document was CREATED by a save")

	for f in [path, clean]:
		DirAccess.remove_absolute(f)
	DirAccess.remove_absolute(DIR)


func _write(path: String, text: String) -> void:
	var fh := FileAccess.open(path, FileAccess.WRITE)
	fh.store_string(text)
	fh.close()


func _read(path: String) -> String:
	var fh := FileAccess.open(path, FileAccess.READ)
	if fh == null:
		return ""
	var text := fh.get_as_text()
	fh.close()
	return text
