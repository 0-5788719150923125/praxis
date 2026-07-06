extends SceneTree

## One-shot loader self-check (run headless):
##   godot --path axis/ghost --headless --script res://tests/loader_check.gd
## Exercises MiniYaml (subset accepts, rejects, typing) and Storyboard (resolution,
## defs/use, sampling, validation). Prints PASS/FAIL lines; exits 1 on any FAIL.

var fails := 0


func check(name: String, ok: bool, detail := "") -> void:
	if ok:
		print("PASS  " + name)
	else:
		fails += 1
		print("FAIL  " + name + ("  -- " + detail if detail != "" else ""))


func deep_eq(a: Variant, b: Variant) -> bool:
	if (typeof(a) == TYPE_INT or typeof(a) == TYPE_FLOAT) \
			and (typeof(b) == TYPE_INT or typeof(b) == TYPE_FLOAT):
		return is_equal_approx(float(a), float(b))
	if typeof(a) != typeof(b):
		return false
	if typeof(a) == TYPE_ARRAY:
		if (a as Array).size() != (b as Array).size():
			return false
		for i in (a as Array).size():
			if not deep_eq(a[i], b[i]):
				return false
		return true
	if typeof(a) == TYPE_DICTIONARY:
		if (a as Dictionary).size() != (b as Dictionary).size():
			return false
		for k in a:
			if not (b as Dictionary).has(k) or not deep_eq(a[k], b[k]):
				return false
		return true
	return a == b


func expect_err(name: String, text: String, needle := "line") -> void:
	var r := MiniYaml.parse(text)
	check(name, not r.ok and String(r.error).contains(needle),
		"expected an error containing '%s', got ok=%s error='%s'" % [needle, r.ok, r.error])


func _initialize() -> void:
	# --- MiniYaml: accepts ---------------------------------------------------
	var r := MiniYaml.parse("""
# comment line
name: the-point          # trailing comment
loop: false
sensitivity: 1.4
count: 7
ratio: .5
neg: -3
sci: 1e3
nul: ~
str: hello world
colon: a: b
quoted: "a # not a comment \\n line2"
single: 'it''s fine? no - plain'
flow: {a: 1, b: [1, 2.5, true, x y], c: {d: [0.62, 0, 0]}}
list:
  - 1
  - {id: left, kind: eye}
  - id: right
    kind: eye
    at: [0.62, 0, 0]
  - nested:
      deep: yes_string
seq:
- top_level_list_item
- 2
""")
	check("yaml parses", r.ok, String(r.error))
	if r.ok:
		var d: Dictionary = r.data
		check("string typed", d.name == "the-point")
		check("bool typed", d.loop == false and typeof(d.loop) == TYPE_BOOL)
		check("float typed", is_equal_approx(float(d.sensitivity), 1.4) and typeof(d.sensitivity) == TYPE_FLOAT)
		check("int typed", d.count == 7 and typeof(d.count) == TYPE_INT)
		check("leading-dot float", is_equal_approx(float(d.ratio), 0.5))
		check("negative int", d.neg == -3)
		check("scientific float", is_equal_approx(float(d.sci), 1000.0))
		check("null tilde", d.nul == null)
		check("bare string", d.str == "hello world")
		check("colon in value", d.colon == "a: b")
		check("double-quote escapes", String(d.quoted).contains("#") and String(d.quoted).contains("\n"))
		check("single-quote literal", String(d.single).begins_with("it"))
		var fl: Dictionary = d.flow
		check("flow map", fl.a == 1 and typeof(fl.b) == TYPE_ARRAY)
		check("flow list mixed", fl.b[2] == true and fl.b[3] == "x y")
		check("flow nesting", fl.c.d[0] is float and is_equal_approx(fl.c.d[0], 0.62))
		var li: Array = d.list
		check("list scalar", li[0] == 1)
		check("list inline map", li[1].id == "left")
		check("list block map", li[2].kind == "eye" and li[2].at[0] is float)
		check("list nested block", li[3].nested.deep == "yes_string")
		check("same-indent list under key", d.seq.size() == 2 and d.seq[0] == "top_level_list_item")
	check("empty doc is a map", MiniYaml.parse("\n# only comments\n").ok
		and typeof(MiniYaml.parse("").data) == TYPE_DICTIONARY)

	# --- MiniYaml: rejects (loud, line-numbered) -----------------------------
	expect_err("tab indent rejected", "a:\n\tb: 1")
	expect_err("anchor rejected", "a: &x 1")
	expect_err("alias rejected", "a: *x")
	expect_err("tag rejected", "a: !!int 3")
	expect_err("block scalar rejected", "a: |\n  text")
	expect_err("multi-doc rejected", "---\na: 1")
	expect_err("merge key rejected", "<<: {a: 1}")
	expect_err("duplicate key rejected", "a: 1\na: 2", "duplicate")
	expect_err("multiline flow rejected", "a: [1,\n  2]")
	expect_err("ragged indent rejected", "m:\n  a: 1\n   b: 2")
	expect_err("unterminated quote rejected", "a: \"oops")
	expect_err("unterminated flow rejected", "a: {k: 1")

	# --- Storyboard: resolution + canonical load -----------------------------
	var sb := Storyboard.load_file("default")
	check("default resolves to yaml", sb.ok and String(sb.path).ends_with("default.yaml"), sb.error)
	check("default fields", sb.ok and sb.name == "the-point" and sb.loop == false
		and sb.sequence.size() == 7)
	if sb.ok:
		var e0: Dictionary = sb.sequence[0]
		check("default is stage entries", e0.scene == "stage" and e0.behavior == "static"
			and float(e0.hold) == 6.0)
		check("stage cast/track shape", typeof(e0.cast) == TYPE_ARRAY and e0.cast[0].id == "left"
			and typeof(e0.track) == TYPE_DICTIONARY and (e0.track.spans as Array).size() >= 3)
		check("defs expanded into entries", typeof(e0.camera) == TYPE_DICTIONARY
			and int(e0.camera.fov) == 48)   # from the front-camera def via use:

	# The same board in both formats must normalize identically (JSON = editor format).
	var ytext := """
name: both-formats
loop: false
sequence:
  - scene: stage
    behavior: static
    hold: 4
    cast:
      - {id: left, kind: eye, at: [0, 0, 0]}
    track:
      nominal: 4
      spans:
        - {at: 0.5, action: blink, target: left}
"""
	var jtext := JSON.stringify({"name": "both-formats", "loop": false, "sequence": [
		{"scene": "stage", "behavior": "static", "hold": 4,
			"cast": [{"id": "left", "kind": "eye", "at": [0, 0, 0]}],
			"track": {"nominal": 4, "spans": [{"at": 0.5, "action": "blink", "target": "left"}]}}]})
	var ypath2 := "/tmp/ghost_check_pair.yaml"
	var jpath := "/tmp/ghost_check_pair.json"
	var yf2 := FileAccess.open(ypath2, FileAccess.WRITE)
	yf2.store_string(ytext)
	yf2.close()
	var jf := FileAccess.open(jpath, FileAccess.WRITE)
	jf.store_string(jtext)
	jf.close()
	var sy := Storyboard.load_file(ypath2)
	var sj := Storyboard.load_file(jpath)
	check("yaml board loads", sy.ok, sy.error)
	check("json board loads", sj.ok, sj.error)
	if sy.ok and sj.ok:
		# Numbers compare as values: YAML types integrals as int, JSON parses all
		# numbers as float; every consumer casts, so 4 == 4.0 here.
		check("yaml == json normalized", deep_eq(sy.sequence, sj.sequence),
			JSON.stringify(sy.sequence) + " vs " + JSON.stringify(sj.sequence))

	# --- Storyboard: defs/use + validation + sampling ------------------------
	var ypath := "/tmp/ghost_check_board.yaml"
	var yf := FileAccess.open(ypath, FileAccess.WRITE)
	yf.store_string("""
name: use-check
defs:
  cam: {eye: [0, 0, 4.0], fov: 48}
  base-cast:
    - {id: left, kind: eye, radius: [0.30, 0.40]}
  common: {behavior: static, camera: {use: cam}}
sequence:
  - scene: stage
    use: common
    hold: 4
    camera: {fov: 60}
    cast: {use: base-cast}
    track:
      nominal: 4
      spans:
        - {at: 0.5, action: blink, target: left}
""")
	yf.close()
	var su := Storyboard.load_file(ypath)
	check("defs/use board loads", su.ok, su.error)
	if su.ok:
		var e: Dictionary = su.sequence[0]
		check("use merges fragment under entry", e.behavior == "static" and float(e.hold) == 4.0)
		check("explicit keys win deep", int(e.camera.fov) == 60 and float(e.camera.eye[2]) == 4.0)
		check("whole-block use becomes fragment", typeof(e.cast) == TYPE_ARRAY and e.cast[0].id == "left")
		var rng := RandomNumberGenerator.new()
		rng.seed = 7
		var rad: float = Storyboard.sample(e.cast[0].radius, rng)
		rng.seed = 7
		var rad2: float = Storyboard.sample(e.cast[0].radius, rng)
		check("range sampling in [lo,hi] and seeded", rad >= 0.30 and rad <= 0.40 and rad == rad2)

	# --- tail + elastic stamping ---------------------------------------------
	var d0 := Storyboard.load_file("default")
	check("default has a tail", d0.ok and (d0.tail as Array).size() >= 1, d0.error)
	if d0.ok and not (d0.tail as Array).is_empty():
		check("elastic stamped into entries",
			is_equal_approx(float((d0.sequence[0] as Dictionary).get("elastic", 0.0)), 0.25)
			and is_equal_approx(float((d0.tail[0] as Dictionary).get("elastic", 0.0)), 0.25))
	var badt := FileAccess.open("/tmp/ghost_check_badtail.yaml", FileAccess.WRITE)
	badt.store_string("name: x\nsequence:\n  - {scene: eye, hold: 2}\ntail:\n  - {behavior: static}\n")
	badt.close()
	var sbt := Storyboard.load_file("/tmp/ghost_check_badtail.yaml")
	check("tail entries validated", not sbt.ok and String(sbt.error).contains("tail[0]"), sbt.error)

	var bad := FileAccess.open("/tmp/ghost_check_bad.yaml", FileAccess.WRITE)
	bad.store_string("name: broken\nsequence:\n  - {behavior: static}\n")
	bad.close()
	var sbad := Storyboard.load_file("/tmp/ghost_check_bad.yaml")
	check("missing scene rejected", not sbad.ok and String(sbad.error).contains("missing 'scene'"), sbad.error)
	var bad2 := FileAccess.open("/tmp/ghost_check_bad2.yaml", FileAccess.WRITE)
	bad2.store_string("a: [1,\n  2]\nsequence: []\n")
	bad2.close()
	var sbad2 := Storyboard.load_file("/tmp/ghost_check_bad2.yaml")
	check("yaml error carries line number", not sbad2.ok and String(sbad2.error).contains("line 1"), sbad2.error)
	check("unknown board fails safe", not Storyboard.load_file("no_such_board").ok)

	print("----")
	print("loader_check: %s (%d failures)" % ["OK" if fails == 0 else "FAILED", fails])
	quit(1 if fails > 0 else 0)
