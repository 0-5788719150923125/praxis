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
	_check_heading_kept_with_text()
	_check_photo_covers()
	_check_peel_holds()
	_check_clip_sides()
	_check_hand_drift()
	_check_one_print_size()
	_check_curl_shadow_fills()
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


## A CURLED PHOTO'S SHADOW IS ALWAYS FILLABLE. An overnight export logged "Invalid polygon data,
## triangulation failed" from the curl: with the lamp shining back along the curl, the shadow's
## rising edge folds over itself and the engine drops the whole shadow for that frame. Swept over
## every hinge, the angles a print is turned to, and the whole peel - and the RAW outline is
## counted too, because a sweep that never produces the fault proves nothing.
func _check_curl_shadow_fills() -> void:
	var raw_bad := 0
	var bad := 0
	var n := 0
	for hinge in ["top", "right", "left"]:
		for portrait in [false, true]:
			var sz := Vector2(420, 290) if not portrait else Vector2(290, 420)
			for deg in range(-10, 11, 2):
				var ang := deg_to_rad(float(deg))
				for pk in 21:
					var peel := float(pk) / 20.0
					var a := Vector2(0, 1)
					var o := Vector2(0, -sz.y * 0.5)
					var length := sz.y
					var width := sz.x
					if hinge == "right":
						a = Vector2(-1, 0)
						o = Vector2(sz.x * 0.5, 0)
						length = sz.x
						width = sz.y
					elif hinge == "left":
						a = Vector2(1, 0)
						o = Vector2(-sz.x * 0.5, 0)
						length = sz.x
						width = sz.y
					var side := Vector2(-a.y, a.x) * width * 0.5
					var t0 := 0.3
					var theta := peel * PI
					var rem := (1.0 - t0) * length
					var r := minf(NotebookMedium.CURL_RADIUS, rem / PI)
					var fold := o + a * t0 * length
					var us := PackedFloat32Array()
					var zs := PackedFloat32Array()
					var ph := PackedFloat32Array()
					for i in 41:
						var sv := rem * float(i) / 40.0
						var bent := minf(sv / r, theta) if theta > 0.0 else 0.0
						var u := r * sin(bent)
						var z := r * (1.0 - cos(bent))
						var rest := sv - bent * r
						if rest > 0.0:
							u += rest * cos(theta)
							z += rest * sin(theta)
						us.append(u)
						zs.append(z)
						ph.append(bent)
					for g in [4.0, 0.0]:
						n += 1
						var poly := NotebookMedium.curl_shadow(o, fold, side, a, us, zs, ph, ang, g)
						if Geometry2D.triangulate_polygon(poly).is_empty():
							bad += 1
					# the raw outline, as it was drawn before
					var light := NotebookMedium.LIGHT_DIR.rotated(-ang)
					var base_off := NotebookMedium.SHADOW_BASE.rotated(-ang)
					var upper := PackedVector2Array([o + side + base_off, fold + side + base_off])
					var lower := PackedVector2Array([o - side + base_off, fold - side + base_off])
					for i in range(1, 41):
						if ph[i] > PI * 0.5 + 0.01:
							break
						upper.append(fold + a * us[i] + side + base_off + light * zs[i])
						lower.append(fold + a * us[i] - side + base_off + light * zs[i])
					lower.reverse()
					if Geometry2D.triangulate_polygon(upper + lower).is_empty():
						raw_bad += 1
	print("notebook_check: curl shadow - %d raw outlines unfillable, %d of %d after" % [raw_bad, bad, n])
	_ok(raw_bad > 0, "the sweep never produces the unfillable outline - it tests nothing")
	_ok(bad == 0, "%d curl shadows the engine cannot fill" % bad)


## ONE CAMERA, ONE PRINT: every inline photo is the same size, and a portrait picture is that
## print turned on its side. Sizes used to be drawn per photo (44-56% of the page, a fanned one
## 90-105% of the one on top), and a portrait was set to a landscape's WIDTH - 2.25x the area.
func _check_one_print_size() -> void:
	var longs: Array = []
	for portrait in [false, true]:
		var l := NotebookLayout.new()
		l.hand = "kalam"
		l.hand_seed = 7
		l.body_fs = int(NotebookLayout.HANDS["kalam"]["size"])
		var px := Vector2(1024, 1536) if portrait else Vector2(1536, 1024)
		l.build(DOC, func(_k: String) -> Vector2: return px, "Station Notes")
		for pg in l.pages:
			for im in pg["images"]:
				if not bool(im.get("photo", false)) or bool(im.get("full", false)):
					continue
				var sz: Vector2 = (im["rect"] as Rect2).size
				longs.append(maxf(sz.x, sz.y))
				_ok((sz.y > sz.x) == portrait, "a %s picture was printed %s" % [
					"portrait" if portrait else "landscape", "tall" if sz.y > sz.x else "wide"])
	_ok(longs.size() >= 4, "the control is wrong - %d inline photos" % longs.size())
	var lo: float = longs.min()
	var hi: float = longs.max()
	_ok(hi - lo < 0.5, "inline prints differ in size: long edges %.1f to %.1f" % [lo, hi])


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


## A PAGE NEVER ENDS ON A HEADING: every heading has two lines of its own text under it on the
## same page (unless the chapter ends there). Reported as a dated entry at the foot of a page
## with its whole text over the leaf. Held on chapter 41 in both layouts.
func _check_heading_kept_with_text() -> void:
	var path := "/home/crow/repos/rift/books/north-star/chapters/41-the-gift-of-guilt.md"
	var doc := FileAccess.get_file_as_string(path) if FileAccess.file_exists(path) else DOC
	for lay in [BookLayout.new(), NotebookLayout.new()]:
		if lay is NotebookLayout:
			lay.hand = "caveat"
			lay.body_fs = int(NotebookLayout.HANDS["caveat"]["size"])
		lay.build(doc, func(_k: String) -> Vector2: return Vector2(1536, 1024), "T")
		var stranded := 0
		for i in lay.words.size():
			if not bool(lay.words[i].get("heading", false)):
				continue
			var j: int = i + 1
			while j < lay.words.size() and bool(lay.words[j].get("heading", false)) \
					and int(lay.words[j]["page"]) == int(lay.words[i]["page"]):
				j += 1
			if j >= lay.words.size():
				continue                   # the chapter's last words
			if int(lay.words[j]["page"]) != int(lay.words[i]["page"]):
				stranded += 1
		_ok(stranded == 0, "%s: %d headings end a page with their text over the leaf"
			% [lay.get_script().get_global_name(), stranded])


## EVERY CLIPPED PHOTO KNOWS WHICH WORDS IT HIDES, so it can lift while they are read: a photo
## over writing names a real range of words on its own page, and a stack shares one.
func _check_photo_covers() -> void:
	var l := _layout(true)
	var n := 0
	for pg in l.pages:
		for im in pg["images"]:
			if not bool(im.get("photo", false)):
				continue
			var cv: Array = im.get("cover", [-1, -1])
			_ok(cv.size() == 2 and int(cv[0]) >= 0 and int(cv[1]) >= int(cv[0]),
				"a photo over the writing hides no words: %s" % [cv])
			if int(cv[0]) >= 0:
				_ok(int(l.words[int(cv[0])]["page"]) == int(l.pages.find(pg)),
					"a photo claims words on another page")
			n += 1
	_ok(n == 3, "the control is wrong - %d photos" % n)


## A LIFTED PHOTO DOES NOT FLICKER. Reported: curled, dropped, curled again - the early test
## flipped as the take's timings arrived. Held: once up it stays up while "early" goes false,
## it drops only after the reading passes its last word, never within PEEL_HOLD of lifting, and
## it does not lift for a reading that is nowhere near it.
func _check_peel_holds() -> void:
	var L := NotebookMedium.peel_latch
	var at: float = L.call(NAN, 10.0, 50, 60, 90, true)
	_ok(not is_nan(at), "an early lift did not lift")
	at = L.call(at, 10.5, 51, 60, 90, false)
	_ok(not is_nan(at), "the photo dropped when the early test changed its mind")
	at = L.call(at, 11.0, 91, 60, 90, false)
	_ok(not is_nan(at), "the photo dropped within PEEL_HOLD of lifting")
	at = L.call(at, 14.5, 92, 60, 90, false)
	_ok(is_nan(at), "the photo stayed up after its words were read and the hold was over")
	_ok(is_nan(L.call(NAN, 1.0, 5, 60, 90, false)), "a photo lifted for a reading far above it")


## A CLIP GRIPS A SHEET, so both its pages show it: a right-hand page is backed by the next
## left-hand one (page 0 is the cover's inside, with no sheet), and the back is a different
## drawing from the front - the inner tongue and the end past the edge, not the whole clip.
func _check_clip_sides() -> void:
	_ok(NotebookMedium._sheet_partner(1) == 2 and NotebookMedium._sheet_partner(2) == 1
		and NotebookMedium._sheet_partner(5) == 6 and NotebookMedium._sheet_partner(0) == -1,
		"pages are not paired into sheets")
	var front: Array = NotebookMedium._clip_path(false)
	var back: Array = NotebookMedium._clip_path(true)
	var fmax := 0.0
	for pth in front:
		for pt in pth:
			fmax = maxf(fmax, (pt as Vector2).y)
	var bmax := 0.0
	for pth in back:
		for pt in pth:
			bmax = maxf(bmax, (pt as Vector2).y)
	# the front is the small loop, the back the big one - one loop a side
	_ok(fmax < bmax * 0.85, "the front of a clip is not its small loop (%.0f vs %.0f)" % [fmax, bmax])
	_ok(front.size() == 1 and back.size() == 1, "a side of the clip is drawn as more than one wire")


## THE HAND DRIFTS, IT DOES NOT JITTER: words sit exactly on the line (no per-word offset), and
## the drift is SMOOTH - a letter a little further along sits almost where its neighbour does,
## and lines next to each other lean alike. A per-word random offset was reported as unnatural.
func _check_hand_drift() -> void:
	var l := NotebookLayout.new()
	l.hand = "kalam"
	l.body_fs = int(NotebookLayout.HANDS["kalam"]["size"])
	l.build("word ".repeat(40), func(_k: String) -> Vector2: return Vector2.ZERO, "")
	var ys := {}
	var rules := {}
	for w in l.words:
		ys[snappedf((w["base"] as Vector2).y, 0.01)] = true
		rules[int(round((w["base"] as Vector2).y / NotebookLayout.RULE))] = true
	_ok(ys.size() == rules.size(), "words are offset one by one: %d baselines on %d lines" % [ys.size(), rules.size()])
	var m := NotebookMedium.new()
	m._seed = 1234
	var worst := 0.0
	var y := NotebookLayout.HEADER + NotebookLayout.RULE * 4.0
	for x in range(200, 1000, 4):
		worst = maxf(worst, absf(m._drift(3, float(x) + 4.0, y) - m._drift(3, float(x), y)))
	_ok(worst < 0.2, "the drift jumps %.2f px between letters 4 px apart" % worst)
	var a := m._drift(3, 900.0, y) - m._drift(3, 250.0, y)
	var b := m._drift(3, 900.0, y + NotebookLayout.RULE) - m._drift(3, 250.0, y + NotebookLayout.RULE)
	_ok(absf(a - b) < 2.0, "neighbouring lines do not lean alike (%.1f vs %.1f px)" % [a, b])
	# THE HAND LEANS RIGHT (positive skew in Godot leans a glyph's top right - measured): every
	# line's slant, per-letter wobble included, stays right of upright or at it, never left.
	var lefts := 0
	for li in 60:
		for k in 5:
			var xf: Transform2D = m._glyph_xform_uncached(li * 7 + k, k,
				Vector2(400.0, NotebookLayout.HEADER + NotebookLayout.RULE * float(li)), 3)
			# the direction a glyph's upright stroke is drawn in: its top, from its baseline
			var up := xf.basis_xform(Vector2(0.0, -1.0))
			if up.x < -0.02:
				lefts += 1
	_ok(lefts == 0, "%d of 300 letters lean left - the hand should lean right or stand upright" % lefts)
	m.free()
