extends Node

## Gate for the NOTEBOOK medium's page: that the writing is a notebook's and the pictures lie
## on it the way the medium says. Each of these fails as a page that merely looks a little off.
##
##   tests/run_boot_probe.sh tests/notebook_check.gd 90
##
## Held: every baseline lies on a rule; a line is ragged (never justified); a timestamp that
## opens a paragraph is set in the margin, left of the red line, and a time mid-sentence is
## not; a photo does NOT push the writing aside (the text runs on under it, as it would under
## a print clipped to a real page) while a sketch DOES take rows of its own; photos that follow
## one another share one clip; `<!-- sketch: -->` is parsed as a sketch with its own key; and
## keying a sketch leaves paper clear and ink solid.

var _fails := 0

const DOC := """---
title: Station Notes
---

## Tuesday

09:40 Arrived before the others and the door was unlocked, which it should not have been, and nothing obviously missing from the bench.

<!-- image: a door -->

10:15 am Channel three reads *consistently high* again, four percent over the reference, and I replaced the cable a second time this morning to be sure of it. We met at 12:30 to talk it over.

<!-- image: bottles -->
<!-- image: the pool -->

Nothing between these two pictures, so they share a clip. And now a drawing, which the writing must go around rather than under.

<!-- sketch: a flask -->

After the drawing the writing picks up again below it, on the next free rule of the page, as a hand would.
"""


func _ready() -> void:
	_check_layout()
	_check_markers()
	_check_keying()
	_check_headings_are_read()
	_check_inks()
	_check_single_spaced()
	if _fails == 0:
		print("notebook_check: ALL OK")
	else:
		print("notebook_check: %d FAILED" % _fails)
	get_tree().quit(1 if _fails > 0 else 0)


func _ok(cond: bool, msg: String) -> void:
	if not cond:
		_fails += 1
		print("notebook_check: FAIL - " + msg)


func _layout(with_images: bool) -> NotebookLayout:
	var l := NotebookLayout.new()
	l.hand = "kalam"
	l.hand_seed = 7
	l.body_fs = int(NotebookLayout.HANDS["kalam"]["size"])
	var doc := DOC if with_images else Manuscript.strip_frontmatter(DOC)
	if not with_images:
		var re := RegEx.new()
		re.compile("<!--\\s*image[^>]*-->")
		doc = re.sub(doc, "", true)
	l.build(doc, func(_k: String) -> Vector2: return Vector2(1536, 1024), "Station Notes")
	return l


func _check_layout() -> void:
	var l := _layout(true)
	_ok(l.words.size() > 60, "the chapter typeset to only %d words" % l.words.size())
	# ON THE RULES: the baseline sits a fixed step above a rule, give or take the hand's wobble.
	var off := 0
	for w in l.words:
		var y: float = (w["base"] as Vector2).y
		if y < NotebookLayout.HEADER:
			continue          # the title, in the header strip above the first rule
		var k := (y + 8.0 - NotebookLayout.HEADER) / NotebookLayout.RULE
		if absf(k - round(k)) * NotebookLayout.RULE > 4.0:
			off += 1
	_ok(off == 0, "%d words are off the rules" % off)
	# RAGGED: the right ends of full lines disagree by more than a justified column's would.
	var ends := {}
	for w in l.words:
		var key := "%d|%d" % [int(w["page"]), int(round((w["base"] as Vector2).y / NotebookLayout.RULE))]
		var r: Rect2 = w["rect"]
		ends[key] = maxf(float(ends.get(key, 0.0)), r.end.x)
	var xs: Array = ends.values()
	xs.sort()
	_ok(xs.size() > 4 and float(xs[xs.size() - 2]) - float(xs[xs.size() / 2]) > 20.0,
		"the lines end together - the writing is justified")
	# THE MARGIN: "09:40" and "10:15 am" left of the red line; "12:30" mid-sentence is not.
	var margin: Array = []
	for w in l.words:
		if (w["rect"] as Rect2).end.x < NotebookLayout.MARGIN_X:
			margin.append(String(w["text"]))
	_ok(margin == ["09:40", "10:15", "am"], "the margin holds %s, wanted 09:40, 10:15 am" % [margin])
	# PHOTOS LIE OVER THE WRITING: the same words on the same lines with and without them.
	var bare := _layout(false)
	var moved := 0
	var n := mini(bare.words.size(), l.words.size())
	for i in n:
		if String(l.words[i]["text"]) == "drawing,":
			break
		if (l.words[i]["base"] as Vector2).distance_to(bare.words[i]["base"] as Vector2) > 0.5:
			moved += 1
	_ok(moved == 0, "%d words moved for a photo - the writing should run on under it" % moved)
	var photos: Array = []
	var sketches: Array = []
	for pg in l.pages:
		for im in pg["images"]:
			if bool(im.get("photo", false)):
				photos.append(im)
			if bool(im.get("sketch", false)):
				sketches.append(im)
	_ok(photos.size() == 3, "%d photos, wanted 3" % photos.size())
	var clips := 0
	for ph in photos:
		if not (ph["clip"] as Dictionary).is_empty():
			clips += 1
	_ok(clips == 2, "%d clips for 3 photos - two in a row should share one" % clips)
	# A SKETCH TAKES ROWS: no word's box crosses it.
	_ok(sketches.size() == 1, "%d sketches, wanted 1" % sketches.size())
	if sketches.size() == 1:
		var sr: Rect2 = sketches[0]["rect"]
		var hit := 0
		for w in l.words:
			if (w["rect"] as Rect2).intersects(sr):
				hit += 1
		_ok(hit == 0, "%d words were written through the sketch" % hit)


func _check_markers() -> void:
	var ims := Manuscript.images("Text.\n\n<!-- sketch: a flask -->\n\n<!-- image: a flask -->\n")
	_ok(ims.size() == 2, "sketch + image parsed to %d blocks" % ims.size())
	if ims.size() == 2:
		_ok(bool(ims[0].get("sketch", false)) and String(ims[0]["placement"]) == "sketch",
			"the sketch marker was not read as a sketch")
		_ok(not bool(ims[1].get("sketch", false)), "the image marker was read as a sketch")
		_ok(String(ims[0]["key"]) != String(ims[1]["key"]),
			"a sketch and a picture with the same words share a key - one file for both")
	# The sketch's own style and references (per kind) are held by illustrations_check.
	var p := Illustrations.build_prompt("a flask", "sketch", "", 0, "/x.png")
	_ok(p.contains("BLACK INK") and p.contains("WHITE"), "the sketch prompt does not ask for ink on white")


func _check_keying() -> void:
	var img := Image.create(64, 32, false, Image.FORMAT_RGB8)
	img.fill(Color(0.95, 0.95, 0.95))               # paper, a touch grey
	img.fill_rect(Rect2i(8, 8, 16, 16), Color.BLACK)  # a stroke
	img.fill_rect(Rect2i(40, 8, 16, 16), Color(0.55, 0.55, 0.55))  # hatching grey
	var ink := Color(0.1, 0.16, 0.46)
	var out := BookMedium.ink_texture(img, ink).get_image()
	_ok(out.get_pixel(2, 2).a < 0.02, "paper survived keying (alpha %.2f)" % out.get_pixel(2, 2).a)
	_ok(out.get_pixel(14, 14).a > 0.98, "a black stroke keyed away (alpha %.2f)" % out.get_pixel(14, 14).a)
	var mid := out.get_pixel(48, 14).a
	_ok(mid > 0.3 and mid < 0.9, "a grey stroke is not a lighter stroke (alpha %.2f)" % mid)
	_ok(out.get_pixel(14, 14).b > 0.4, "the stroke is not in the pen's ink")


## A HEADING IS SPOKEN, so it must be WORDS the reading can follow. Set as a label, a dated
## entry ("## September 19, 2026") had nothing on the page to match: the highlight fell back to
## the previous paragraph and re-lit its last word for every word of the date, and the camera
## sat on it. Held for both layouts, on the chapter the report came from when it is present.
func _check_headings_are_read() -> void:
	var path := "/home/crow/repos/rift/books/north-star/chapters/41-the-gift-of-guilt.md"
	var doc := FileAccess.get_file_as_string(path) if FileAccess.file_exists(path) else DOC
	var heads: Array = []
	for b in Manuscript.blocks(doc):
		if String(b["kind"]) == "heading":
			heads.append(String(b["text"]))
	_ok(not heads.is_empty(), "the control is wrong - the chapter has no headings")
	for lay in [BookLayout.new(), _layout(true)]:
		if lay is NotebookLayout:
			lay = NotebookLayout.new()
			lay.hand = "kalam"
			lay.body_fs = int(NotebookLayout.HANDS["kalam"]["size"])
		lay.build(doc, func(_k: String) -> Vector2: return Vector2.ZERO, "Title Words")
		# the title is read aloud first, so it is the first thing the reading can follow
		_ok(lay.words.size() > 1 and String(lay.words[0]["norm"]) == "title"
			and String(lay.words[1]["norm"]) == "words",
			"%s: the title is not the first words on the page" % lay.get_script().get_global_name())
		var printed := ""
		for w in lay.words:
			printed += String(w["norm"]) + " "
		var missing := 0
		for h in heads:
			var want := ""
			for t in lay.tokens(h):
				want += lay.norm(String(t["text"])) + " "
			if not printed.contains(want):
				missing += 1
		_ok(missing == 0, "%s: %d of %d headings are not words the reading can follow"
			% [lay.get_script().get_global_name(), missing, heads.size()])


## EACH VOICE WRITES IN ITS OWN INK, black by default: words carry their speaker (headings and
## margin times included), a named ink is an ink colour, and the medium resolves a speaker to
## the ink the document names for it.
func _check_inks() -> void:
	var doc := "Opening.\n\n<!-- speaker: Angel -->\n## Monday\n\n09:40 Blue words here.\n\n<!-- speaker: Ryan -->\nRed words."
	var l := NotebookLayout.new()
	l.build(doc, func(_k: String) -> Vector2: return Vector2.ZERO, "T")
	var who := {}
	for w in l.words:
		who[String(w["text"])] = String(w.get("speaker", "?"))
	_ok(who.get("Opening.") == Manuscript.NARRATOR and who.get("Monday") == "Angel"
		and who.get("09:40") == "Angel" and who.get("Blue") == "Angel" and who.get("Red") == "Ryan",
		"words do not carry their speaker: %s" % [who])
	_ok(NotebookLayout.ink_color("blue") == NotebookLayout.INKS["blue"]
		and NotebookLayout.ink_color("#335577") == Color.html("#335577")
		and NotebookLayout.ink_color("nonsense") == NotebookLayout.INKS["black"],
		"ink names, hex and the black fallback do not resolve")
	var m := NotebookMedium.new()
	var subs: Subtitles = preload("res://scripts/subtitles.gd").new()
	subs.document = {"source": doc, "inks": {"Angel": "blue", "Ryan": "red"}}
	m._subs = subs
	m._ink = NotebookLayout.INKS["black"]
	_ok(m._ink_for({"speaker": "Angel"}) == NotebookLayout.INKS["blue"]
		and m._ink_for({"speaker": "Ryan"}) == NotebookLayout.INKS["red"]
		and m._ink_for({"speaker": Manuscript.NARRATOR}) == NotebookLayout.INKS["black"],
		"a speaker is not written in the ink the document names for it")
	subs.free()
	m.free()


## EVERY HAND WRITES ON EVERY RULE: consecutive lines of one paragraph are one rule apart, in
## each hand. Caveat's body size once double-spaced every line.
func _check_single_spaced() -> void:
	var para := "word ".repeat(120)
	for h in NotebookLayout.HANDS:
		var l := NotebookLayout.new()
		l.hand = h
		l.body_fs = int(NotebookLayout.HANDS[h]["size"])
		l.build(para, func(_k: String) -> Vector2: return Vector2.ZERO, "")
		var ys: Array = []
		for w in l.words:
			var y := int(round(((w["base"] as Vector2).y + 8.0 - NotebookLayout.HEADER) / NotebookLayout.RULE))
			if not ys.has(y):
				ys.append(y)
		_ok(ys.size() > 3 and int(ys[1]) - int(ys[0]) == 1 and int(ys[2]) - int(ys[1]) == 1,
			"%s is not written on consecutive rules: %s" % [h, ys.slice(0, 4)])
