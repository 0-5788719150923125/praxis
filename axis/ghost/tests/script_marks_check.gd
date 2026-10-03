extends SceneTree

## script_marks_check - that every mark the script editor offers DOES what it says.
##
## [ScriptMarks] is a list of promises: "click this and the reading will rest here", "this
## hands the text to Emily". It implements none of them - [Manuscript], [TextNorm] and
## [Phonemes] do - so the failure this exists for is DRIFT: a registry entry whose example no
## longer means anything to the parser that is supposed to read it, which looks exactly like
## a working palette button. So every entry's example is inserted and the REAL parser is asked
## what it makes of it, and an entry with no check here fails too - a new mark has to come
## with its proof.
##
## Also held: each entry's highlight pattern recognises its own example (the colour on screen
## is the same claim), the insertion rules (a selection becomes the fill, the fill is left
## selected, an own-line mark gets its own line, one undo step), and the highlighter colouring
## a multi-line picture description and leaving prose plain.
##
## Run: godot --headless --path axis/ghost --script tests/script_marks_check.gd

var fails := 0
var checks := 0


func _init() -> void:
	_registry_shape()
	_patterns_match_examples()
	_parsers_agree()
	_insertion()
	_highlighter()
	if fails == 0:
		print("script_marks_check: ALL OK (%d checks)" % checks)
	else:
		print("script_marks_check: %d FAILURE(S) of %d checks" % [fails, checks])
	quit(1 if fails > 0 else 0)


func _ok(cond: bool, msg: String) -> void:
	checks += 1
	if not cond:
		fails += 1
		print("script_marks_check: FAIL  " + msg)


func _registry_shape() -> void:
	for k in ScriptMarks.REGISTRY:
		var e: Dictionary = ScriptMarks.REGISTRY[k]
		for f in ["label", "group", "blurb", "before", "fill", "after", "line", "modes", "pattern"]:
			_ok(e.has(f), "%s has no `%s`" % [k, f])
		_ok(ScriptMarks.GROUPS.has(e.get("group", "")), "%s names an unknown group" % k)
		_ok(String(e.get("line", "")) in ["", "own", "start"], "%s has an unknown line rule" % k)
		for m in e.get("modes", []):
			_ok(m in ["generative", "synthesis"], "%s names an unknown mode %s" % [k, m])
		var re := RegEx.new()
		_ok(re.compile(String(e.get("pattern", ""))) == OK, "%s's pattern does not compile" % k)
	# Both panels have something to offer.
	_ok(ScriptMarks.for_mode("generative").size() > ScriptMarks.for_mode("synthesis").size()
			and ScriptMarks.for_mode("synthesis").size() > 0,
		"the per-mode palettes are not what they should be")


## THE COLOUR IS A CLAIM TOO: the highlight pattern must find the whole example, on a line
## of its own inside prose (the own-line marks are anchored to a line).
func _patterns_match_examples() -> void:
	for k in ScriptMarks.REGISTRY:
		var e: Dictionary = ScriptMarks.REGISTRY[k]
		var ex := ScriptMarks.example(k)
		var doc := "Some prose before.\n\n%s%s\n\nSome prose after." % [
			"" if String(e["line"]) != "start" else "", ex + ("The line goes on." if k == "timestamp" else "")]
		var re := RegEx.new()
		re.compile(String(e["pattern"]))
		var hit := false
		for m in re.search_all(doc):
			var s := m.get_string()
			if s.strip_edges() == ex.strip_edges() or (k == "heading" and s.begins_with(ex)) \
					or (k == "timestamp" and ex.begins_with(s.strip_edges())):
				hit = true
		_ok(hit, "%s's pattern does not find its own example %s" % [k, ex.c_escape()])


func _kinds(d: Dictionary) -> Array:
	var out := []
	for a in d["actions"]:
		out.append(String(a["kind"]))
	return out


## THE PROOF: each example, read by the parser of record. A key missing from this match is a
## failure, so a new entry cannot ship unverified.
func _parsers_agree() -> void:
	for k in ScriptMarks.REGISTRY:
		var e: Dictionary = ScriptMarks.REGISTRY[k]
		var ex := ScriptMarks.example(k)
		var fill := String(e["fill"])
		match k:
			"speaker":
				var body := "Before.\n%s\nAfter.\n" % ex
				_ok(Manuscript.speakers(body).has(fill), "a speaker cue is not a speaker")
				var ps := Manuscript.passages(body)
				_ok(ps.size() == 2 and String(ps[1]["speaker"]) == fill
						and String(ps[1]["text"]).strip_edges() == "After.",
					"a speaker cue did not hand the text after it over")
			"hesitation", "hesitation_timed":
				var h := Manuscript.hesitations("Wait %s for it." % ex)
				_ok(h.size() == 1, "%s is not a hesitation" % k)
				if h.size() == 1:
					var want := -1.0 if fill.is_empty() else float(fill)
					_ok(is_equal_approx(float(h[0]["seconds"]), want),
						"%s rests %s, not %s" % [k, h[0]["seconds"], want])
			"outro":
				# Read here by the parsers that see it; the fade itself is the Generative panel's,
				# gated in multi_voice_check (_check_outro_mark).
				_ok(RegEx.create_from_string(Manuscript.OUTRO).search("A %s B." % ex) != null,
					"the outro example is not the outro mark")
				_ok(not _words(Manuscript.unspoken("Before. %s After." % ex)).has("outro"),
					"the outro mark would be spoken")
				_ok(TabletScript.speakable("<!-- url: a.test -->\n\nOne. %s Two.\n" % ex).contains(ex),
					"a tablet chapter loses the outro mark before the voice sees it")
			"url":
				var d := TabletScript.parse("%s\n\nText.\n" % ex)
				_ok((d["pages"] as Array).size() == 1 and String(d["pages"][0]["url"]) == fill,
					"a url mark does not open a page at %s" % fill)
			"search":
				var d := TabletScript.parse("<!-- url: engine.test -->\n\n# Engine\n\n%s\n\nA result.\n" % ex)
				_ok(_kinds(d).has("search") and String((d["pages"] as Array).back()["query"]) == fill,
					"a search mark does not search for '%s'" % fill)
			"new_tab":
				_ok(_kinds(TabletScript.parse("<!-- url: a.test -->\n\nText.\n\n%s\n<!-- url: b.test -->\n\nMore.\n" % ex)).has("tab"),
					"a new tab mark opens no tab")
			"back":
				var d := TabletScript.parse("<!-- url: a.test -->\n\nOne.\n\n<!-- url: b.test -->\n\nTwo.\n\n%s\n" % ex)
				_ok(_kinds(d).has("back"), "a back mark goes nowhere")
			"landscape", "portrait":
				var pre := "<!-- landscape -->\n" if k == "portrait" else ""
				var d := TabletScript.parse("<!-- url: a.test -->\n%s%s\n\nText.\n" % [pre, ex])
				var to := -1
				for a in d["actions"]:
					if a["kind"] == "rotate":
						to = int(a["to"])
				_ok(to == (1 if k == "landscape" else 0), "%s does not turn to %s" % [k, k])
			"skip":
				var sp := TabletScript.speakable("<!-- url: a.test -->\n\nRead this. %s Not this.\n" % ex)
				_ok(sp.contains("Read this") and not sp.contains("Not this"), "a skip mark does not stop the reading")
			"filler":
				var d := TabletScript.parse("<!-- url: a.test -->\n\nText.\n\n%s\n" % ex)
				var n := 0
				for b in (d["pages"][0]["blocks"] as Array):
					if b["kind"] == "filler":
						n += int(b["n"])
				_ok(n == int(fill), "a filler mark does not put %s stories in" % fill)
			"timestamp":
				var t := Manuscript.mark_timestamp_pauses(ex + "The subject was moved.")
				_ok(Manuscript.hesitations(t).size() == 1, "a log-entry time gets no rest")
			"hum":
				# Read off the source: the editor itself cannot load without the autoloads.
				var word := fill.to_lower().rstrip(".!?…")
				var src := FileAccess.get_file_as_string("res://scripts/generative_editor.gd")
				var m := RegEx.create_from_string("const HUM_SECONDS := \\{([^}]*)\\}").search(src)
				_ok(m != null and m.get_string(1).contains("\"%s\":" % word),
					"the hum example is not a held hum")
			"image", "image_full", "image_left", "sketch":
				var imgs := Manuscript.images("Prose.\n\n%s\n\nMore prose.\n" % ex)
				_ok(imgs.size() == 1 and String(imgs[0]["prompt"]) == fill,
					"%s is not a picture of '%s': %s" % [k, fill, imgs])
				if imgs.size() == 1:
					var want: String = {"image": "inline", "image_full": "full", "image_left": "inline",
						"sketch": "sketch"}[k]
					_ok(String(imgs[0]["placement"]) == want,
						"%s is placed %s, not %s" % [k, imgs[0]["placement"], want])
					if k == "image_left":
						_ok(String(imgs[0].get("side", "")) == "left", "a left pin is not left")
			"phonetic":
				var ws: Array = Phonemes.parse("The %s sat." % ex)[0]
				_ok(ws.size() == 3 and bool(ws[1].get("literal", false))
						and (ws[1]["phones"] as Array) == ["K", "AE", "T"],
					"an inline pronunciation is not read as its phonemes")
			"macro":
				var dflt := fill.get_slice(":", 1)
				_ok(TextNorm.normalize("Chapter %s." % ex) == TextNorm.normalize("Chapter %s." % dflt),
					"a macro does not read as its default")
			"heading":
				var b := Manuscript.blocks(ex + "\n\nText.\n")
				_ok(b.size() > 0 and String(b[0]["kind"]) == "heading"
						and String(b[0]["text"]) == fill, "a heading is not a heading")
			"scene_line":
				_ok(Manuscript.is_scene_line(ex), "a scene line is not a scene line")
			"italic", "bold":
				var want := TextNorm.EMPH_I if k == "italic" else TextNorm.EMPH_B
				var ws: Array = Phonemes.parse("A %s word." % ex)[0]
				var lit := false
				var spoken := PackedStringArray()
				for w in ws:
					spoken.append(String(w["text"]))
					if int(w.get("emph", 0)) & want:
						lit = true
				_ok(lit, "%s does not reach the subtitle as emphasis" % k)
				_ok(not "".join(spoken).contains("*"), "%s's marks are spoken" % k)
			"rule":
				var b := Manuscript.blocks("One.\n\n%s\n\nTwo.\n" % ex)
				_ok(b.size() == 3 and String(b[1]["kind"]) == "rule", "a scene break is not a rule")
			"note":
				var b := Manuscript.blocks("Before %s after.\n" % ex)
				_ok(b.size() == 1 and not String(b[0]["text"]).contains(fill),
					"a note reached the page")
				_ok(not Manuscript.unspoken("Before %s after." % ex).contains(fill),
					"a note survives into the synthesis reading")
			_:
				_ok(false, "%s has no check in script_marks_check - prove it before shipping it" % k)
	# THE SYNTHESIS FIX, two-sided: the raw text DOES read a cue aloud, the stripped text not.
	var raw := "Hello. <!-- speaker: Emily --> Then. <!-- hesitation --> End."
	_ok(_words(raw).has("emily") and not _words(Manuscript.unspoken(raw)).has("emily")
			and not _words(Manuscript.unspoken(raw)).has("hesitation"),
		"Manuscript.unspoken does not keep comments out of a synthesis reading")


func _words(text: String) -> PackedStringArray:
	var out := PackedStringArray()
	for s in Phonemes.parse(text):
		for w in s:
			out.append(String(w["text"]))
	return out


func _insertion() -> void:
	var te := TextEdit.new()
	# A SELECTION BECOMES THE FILL, and an own-line mark in mid-line gets its own line.
	te.text = "Then Emily said hello."
	te.select(0, 5, 0, 10)
	ScriptMarks.insert(te, "speaker")
	_ok(te.text == "Then \n<!-- speaker: Emily -->\n said hello.",
		"a selected name did not become its own-line cue: %s" % te.text.c_escape())
	_ok(Manuscript.speakers(te.text).has("Emily"), "the inserted cue is not read as a cue")
	# THE FILL IS SELECTED so typing replaces the example.
	te.text = "Prose."
	te.deselect()
	te.set_caret_line(0)
	te.set_caret_column(6)
	ScriptMarks.insert(te, "image")
	_ok(te.get_selected_text() == "what the picture shows",
		"the example was not left selected: '%s'" % te.get_selected_text())
	_ok(te.text == "Prose.\n<!-- image: what the picture shows -->",
		"a picture after prose did not get its own line: %s" % te.text.c_escape())
	# ONE UNDO STEP takes the whole insertion back.
	te.undo()
	_ok(te.text == "Prose.", "one undo did not remove the whole mark: %s" % te.text.c_escape())
	# A MARK WITH NO FILL goes AFTER a selection, never over it.
	te.text = "Wait for it."
	te.select(0, 0, 0, 4)
	ScriptMarks.insert(te, "hesitation")
	_ok(te.text == "Wait<!-- hesitation --> for it.",
		"a hesitation replaced the selected word: %s" % te.text.c_escape())
	# A KNOWN NAME (the speaker chips) fills the mark and is not left selected.
	te.text = ""
	ScriptMarks.insert(te, "speaker", "Ryan")
	_ok(te.text == "<!-- speaker: Ryan -->" and not te.has_selection(),
		"a chip did not insert its speaker: %s" % te.text.c_escape())
	# A LINE-START MARK goes to the head of the line whatever the caret.
	te.text = "Chapter text"
	te.set_caret_column(8)
	ScriptMarks.insert(te, "heading")
	_ok(te.text.begins_with("# Heading") and te.text.ends_with("Chapter text"),
		"a heading was not put at the start of the line: %s" % te.text.c_escape())
	te.free()


func _highlighter() -> void:
	var te := TextEdit.new()
	var hl := ScriptHighlighter.new("generative")
	te.syntax_highlighter = hl
	te.text = "Plain prose.\n<!-- image: a wolf\nin a coat -->\n<!-- speaker: Emily -->\nMore."
	var plain: Color = te.get_theme_color("font_color")
	var pic := ScriptMarks.color_of("image")
	var voice := ScriptMarks.color_of("speaker")
	var l0 := hl.get_line_syntax_highlighting(0)
	_ok(l0.has(0) and (l0[0]["color"] as Color).is_equal_approx(plain), "prose is coloured")
	var l2 := hl.get_line_syntax_highlighting(2)
	_ok(l2.has(0) and (l2[0]["color"] as Color).is_equal_approx(pic),
		"the second line of a picture description lost its colour: %s" % l2)
	var l3 := hl.get_line_syntax_highlighting(3)
	_ok(l3.has(0) and (l3[0]["color"] as Color).is_equal_approx(voice),
		"a speaker cue is not coloured as one: %s" % l3)
	# A CUE THAT DOES NOT OWN ITS LINE IS ONLY A NOTE, and is shown as one.
	te.text = "She said <!-- speaker: Emily --> hello."
	var l := hl.get_line_syntax_highlighting(0)
	var note := ScriptMarks.color_of("note")
	var at := te.text.find("<!--")
	_ok(l.has(at) and (l[at]["color"] as Color).is_equal_approx(note),
		"an inline speaker cue is coloured as a working cue: %s" % l)
	# THE SYNTHESIS PANEL honours no cues: the same line there is a note.
	var te2 := TextEdit.new()
	var hl2 := ScriptHighlighter.new("synthesis")
	te2.syntax_highlighter = hl2
	te2.text = "<!-- speaker: Emily -->"
	var s0 := hl2.get_line_syntax_highlighting(0)
	_ok(s0.has(0) and (s0[0]["color"] as Color).is_equal_approx(note),
		"the synthesis highlighter colours a cue it does not honour")
	te.free()
	te2.free()
