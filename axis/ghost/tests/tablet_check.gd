extends SceneTree

## THE TABLET'S SCRIPT: what the hand does is inferred right, and the voice and the screen agree
## about which words are read.
##
##   godot --headless --path . --script tests/tablet_check.gd
##
## The sharp check is the AGREEMENT. The medium finds the word being said by walking the
## chapter's spoken words alongside the voice's; a word the voice reads that the screen thinks
## is skipped (or the reverse) desynchronises every highlight, scroll and tap after it, silently.
## So the voice's text ([method TabletScript.speakable]) is tokenized the way the voice would see
## it and compared, word for word, with the words the parse marks spoken. And a chapter that is
## NOT a tablet chapter must come back byte-identical, because this runs on every reading.

var _fail := 0

const DOC := """---
title: Test
---

Somebody picks it up.

<!-- url: news.test -->

<!-- skip -->
# The News

<!-- filler: 2 -->

## [A Story](news.test/a-story)

<!-- skip -->
The stub of it.

## Nobody wrote this one

<!-- url: news.test/a-story -->

# A Story

Read this. <!-- skip --> Not this, or the rest of the paragraph.

Read to the end <!-- skip --> of this sentence. But not this one.

<!-- skip -->
None of this paragraph.

Some *emphasis* and a [link here](other.test/page) to follow.

<!-- new tab -->
<!-- url: www.engine.test -->

<!-- skip -->
# Engine

<!-- search: what is it? -->

### [First result](far.test/one)

A snippet.

<!-- url: far.test/one -->
<!-- landscape -->

# Far away

The end.
"""


func _initialize() -> void:
	_check_inference()
	_check_agreement()
	_check_untouched()
	_check_back()
	_check_real()
	_check_arrivals()
	_check_omnibox()
	print("tablet_check: %s" % ("PASS" if _fail == 0 else "FAIL (%d)" % _fail))
	quit(1 if _fail > 0 else 0)


func _ok(cond: bool, what: String) -> void:
	if not cond:
		_fail += 1
		print("  FAIL: ", what)
	else:
		print("  ok: ", what)


func _check_inference() -> void:
	var d := TabletScript.parse(DOC)
	var kinds := []
	for a in d["actions"]:
		kinds.append(String(a["kind"]))
	_ok(kinds == ["wake", "open", "skim", "link", "tab", "type", "search", "link", "rotate"],
		"actions inferred: %s" % str(kinds))
	var pages: Array = d["pages"]
	_ok(pages.size() == 5, "five pages (got %d)" % pages.size())
	var acts: Array = d["actions"]
	var link: Dictionary = acts[3]
	_ok(String(d["words"][int(link["word"])]["text"]) == "A", "the link tapped is the headline's first word")
	_ok(String(acts[5]["text"]) == "www.engine.test", "a url with no link to it is typed")
	# below a skipped masthead and two placeholder stories, the reader skims to the headline;
	# on the article, a short skipped tail is no skim
	_ok(int(acts[2]["word"]) == int(link["word"]) and int(acts[2]["from"]) == 0,
		"the hand skims past what is not read to what is")
	_ok(bool(pages[2]["search_box"]) and not bool(pages[2]["results"]), "a page searched from has a box")
	_ok(bool(pages[3]["results"]) and String(pages[3]["query"]) == "what is it?", "the results page knows its query")
	var stubs := []
	for b in pages[0]["blocks"]:
		if b["kind"] == "heading" and bool(b.get("stub", false)):
			stubs.append(b["raw"])
	_ok(stubs == ["Nobody wrote this one"], "a heading with nothing under it is a stub, a skipped body is not: %s" % str(stubs))
	# NOTHING IS READ OFF THE SCREEN: the narration before the first url is not a word at all
	_ok(int(acts[0]["after"]) == 0 and int(acts[1]["after"]) == 0, "nothing is read before the browser opens")
	var first: Dictionary = d["words"][(d["spoken"] as PackedInt32Array)[0]]
	_ok(int(first["page"]) == 0 and String(first["text"]) == "A", "the first word read is on the first page")
	# a skip mark mid-sentence holds off to the sentence's end
	var sp := String(d["speakable"])
	_ok(sp.contains("Read to the end of this sentence.") and not sp.contains("But not"),
		"a skip mid-sentence takes effect at the sentence's end")


func _check_agreement() -> void:
	var d := TabletScript.parse(DOC)
	var said := []
	for wi in (d["spoken"] as PackedInt32Array):
		var n := String(d["words"][wi]["norm"])
		if not n.is_empty():
			said.append(n)
	var voice := []
	var text := String(d["speakable"])
	text = TabletScript._rx("<!--[\\s\\S]*?-->").sub(text, " ", true)
	for tok in text.split("\n"):
		for w in String(tok).lstrip("#").split(" ", false):
			var n := TabletScript.norm(String(w))
			if not n.is_empty():
				voice.append(n)
	_ok(said == voice, "the voice reads exactly the words marked spoken\n    voice %s\n    page  %s" % [str(voice), str(said)])
	for gone in ["Somebody", "Test", "News", "stub", "Not", "None", "Engine", "news.test"]:
		_ok(not String(d["speakable"]).contains(gone), "'%s' is not read" % gone)
	for kept in ["Read this.", "link here", "First result"]:
		_ok(String(d["speakable"]).contains(kept), "'%s' is read" % kept)
	# one rest per run of actions, each as long as the run
	var holds := []
	for m in TabletScript._rx(TabletScript.HOLD).search_all(String(d["speakable"])):
		holds.append(float(m.get_string(1)))
	var want := []
	var run := 0.0
	var after := -1
	for a in d["actions"]:
		if int(a["after"]) != after and after >= 0:
			want.append(snappedf(run, 0.01))
			run = 0.0
		after = int(a["after"])
		run += float(a["dur"])
	want.append(snappedf(run, 0.01))
	var got := []
	for h in holds:
		got.append(snappedf(h, 0.01))
	_ok(got == want, "a rest per run of actions, sized to it: %s vs %s" % [str(got), str(want)])
	_ok(TabletScript.hold_of("<!-- action-hold: 3.5 -->") == 3.5 and TabletScript.hold_of("<!-- hesitation -->") < 0.0,
		"hold_of reads only the action rest")


const BACK := """<!-- url: r.test -->

# Results

[One](one.test)

[Two](two.test)

<!-- url: two.test -->

<!-- skip -->
Nothing read here at all.

<!-- back -->

Dropped words.

<!-- url: one.test -->

# One
"""


func _check_back() -> void:
	var d := TabletScript.parse(BACK)
	var kinds := []
	for a in d["actions"]:
		kinds.append(String(a["kind"]))
	_ok(kinds == ["wake", "open", "link", "skim", "back", "link"], "back inferred, with a glance first: %s" % str(kinds))
	var acts: Array = d["actions"]
	_ok(int(acts[4]["page"]) == 0 and int(acts[4]["from"]) == 1, "back returns to the page before")
	_ok(int(acts[3]["word"]) == -1, "an unread page is glanced at before leaving it")
	_ok(int(acts[5]["from"]) == 0 and String(d["words"][int(acts[5]["word"])]["text"]) == "One",
		"after back, the next link is found on the page returned to")
	var all := []
	for w in d["words"]:
		all.append(String(w["text"]))
	_ok(not all.has("Dropped") and not String(d["speakable"]).contains("Dropped"),
		"text after a back is neither shown nor read")
	# typing: the screen and the voice's rest come from the same clock
	var text := "www.duckduckduck.mom"
	var ct := TabletScript.char_times(text)
	var ph := TabletScript.phases("type", text)
	var mono := true
	for i in text.length():
		mono = mono and ct[i + 1] > ct[i]
	_ok(mono and is_equal_approx(float(ph["chars1"]) - float(ph["chars0"]), ct[text.length()]),
		"each letter typed after the last, ending when the rest says")
	_ok(ct[text.length()] / text.length() >= 0.18, "typing is unhurried (%.2f s a letter)" % (ct[text.length()] / text.length()))


const REAL := """<!-- url: news.test/a -->

# An article

Read this, then [the listing](https://www.shop.test/item/42).

<!-- url: https://www.shop.test/item/42 -->

<!-- new tab -->
<!-- url: engine.test -->

# Engine
"""


func _check_real() -> void:
	var d := TabletScript.parse(REAL)
	var pages: Array = d["pages"]
	_ok(bool(pages[1].get("real", false)) and not bool(pages[0].get("real", false)) and not bool(pages[2].get("real", false)),
		"a url with nothing written under it is the real page; written ones are not")
	_ok(String(pages[1]["snap"]) == TabletScript.snap_key("shop.test/item/42"), "its capture is keyed by its address")
	var kinds := []
	for a in d["actions"]:
		kinds.append(String(a["kind"]))
	_ok(kinds == ["wake", "open", "link", "skim", "tab", "type"], "the real page is reached by its link and lingered on: %s" % str(kinds))
	var linger: Dictionary = d["actions"][3]
	_ok(int(linger["from"]) == 1 and int(linger["word"]) == -1 and float(linger["dur"]) >= 6.0,
		"the linger is long (%.1f s): nothing on it is read" % float(linger["dur"]))
	_ok(not String(d["speakable"]).contains("An article") and String(d["speakable"]).contains("Read this"),
		"a page's opening title is shown, not read; what follows it is read")
	var snaps := TabletScript.snapshots(REAL)
	_ok(snaps.size() == 1 and String(snaps[0]["url"]) == "https://www.shop.test/item/42" and String(snaps[0]["placement"]) == "page",
		"the panel is offered the real page to capture")


const OMNI := """<!-- url: engine.test -->

# Engine

<!-- search: first -->

### [A result](site.test/article)

<!-- url: site.test/article -->

# An article

One. Two. Three.

Four. Five.

Six. Seven.

Eight.

<!-- search: second -->

### Another result
"""


func _check_omnibox() -> void:
	var d := TabletScript.parse(OMNI)
	var acts: Array = []
	for a in d["actions"]:
		if a["kind"] == "search":
			acts.append(a)
	var pages: Array = d["pages"]
	_ok(acts.size() == 2 and not bool(acts[0].get("bar", false)), "a search on an engine's page goes in its box")
	_ok(bool(acts[1].get("bar", false)) and not bool(pages[2]["search_box"]),
		"a search from an article goes in the address bar - no box grows on the article")
	_ok(String(pages[3]["host"]) == "engine.test", "...and its results come from the engine last used")


func _check_arrivals() -> void:
	# a page that comes up is looked at before anything else happens on it
	for kind in ["open", "link", "type", "search"]:
		var ph := TabletScript.phases(kind, "what is it?")
		_ok(float(ph["end"]) - float(ph["show"]) >= TabletScript.ARRIVE_LOOK - 0.001,
			"%s ends on a look at the new page (%.1f s)" % [kind, float(ph["end"]) - float(ph["show"])])
	var bk := TabletScript.phases("back")
	_ok(float(bk["end"]) - float(bk["show"]) >= TabletScript.RETURN_LOOK - 0.001, "back ends on a shorter look")


func _check_untouched() -> void:
	var plain := "---\ntitle: X\n---\n\nA [link](x.y) and <!-- skip --> words.\n"
	_ok(TabletScript.speakable(plain) == plain, "a chapter without url/search marks is returned untouched")
	# a picture between two read passages is stopped on
	var pics := TabletScript.parse("<!-- url: a.test -->\n\nOne.\n\n<!-- image: a cat -->\n\nTwo.\n")
	var sk: Array = []
	for a in pics["actions"]:
		if a["kind"] == "skim":
			sk.append(a)
	_ok(sk.size() == 1 and int(sk[0]["m"]) == 1 and float(sk[0]["dur"]) > TabletScript.PICTURE_DWELL,
		"a skim stops on the picture it passes")
	_ok(TabletScript.url_key("https://www.Duck.mom/") == "duck.mom", "addresses compare without scheme, www or slash")
	var om := TabletScript.speakable("<!-- url: a.test -->\n\nOne. <!-- outro --> Two.\n\n<!-- outro -->\n\nThree.\n")
	_ok(om.count("<!-- outro -->") == 2, "the outro mark reaches the voice, inline or on its own line")
