extends RefCounted
class_name TabletScript

## TabletScript - a chapter written as somebody browsing on a tablet.
##
## The `tablet` medium shows the reading as web pages on a tablet lying on a desk. The chapter
## is ordinary markdown - what is on each page - plus a handful of own-line marks for what the
## hand does between them:
##
##     <!-- url: www.duckduckduck.mom -->    the page that follows lives at this address
##     <!-- search: What is the 7th Realm? -->   type this into the page's search box; the
##                                             results page follows
##     <!-- new tab -->                        open a blank tab (the next url is typed into it)
##     <!-- back -->                           back to the page before, scrolled where it was left
##     <!-- landscape -->  <!-- portrait -->   turn the picture (the camera turns, not the tablet)
##     <!-- skip -->                           shown, not read: the rest of this paragraph from
##                                             the end of its sentence, or on its own line, the
##                                             whole paragraph below it
##     <!-- filler: 3 -->                      three placeholder stories (squiggles) here
##
## HOW A PAGE IS REACHED IS INFERRED, never written. The first url opens the browser from the
## home screen; a url linked from the page on screen (`[word](url)`) is scrolled to and tapped;
## any other url is typed into the address bar. A page grows a search box when a search is made
## on it. A heading with no text under it gets squiggles for a body, and every page is padded
## with placeholder stories so there is something to scroll past.
##
## NOTHING IS READ OFF THE SCREEN. No title is announced and text before the first url is not
## spoken: the voice is the reader's, and the reader only reads what is on a page.
##
## A READER SKIMS. Wherever the reading passes over something it does not read - a skipped
## tail, a skipped paragraph, a picture, placeholder stories - the hand gets a SKIM: a beat of
## stillness, then a slow drag down the page, stopping on each picture to look at it, to the
## next words read. The voice rests for it like any other action. A skip mark set mid-sentence
## takes effect at the end of that sentence: a reader drops a paragraph, not half a clause.
##
## ONE WALK, TWO READERS. [method parse] builds the pages and the words the medium draws, and in
## the same pass the text the VOICE reads ([member speakable]): skipped text removed, link
## targets dropped, and every run of actions replaced by one rest long enough to perform them
## (`<!-- action-hold: S -->`, which the Generative panel splices like a hesitation). The rest
## and the medium's schedule come from the same [method phases], so the voice waits exactly as
## long as the hand is busy.

## Own-line actions. Group 1 is the verb, 2 its argument.
const MARKER := "^\\s*<!--\\s*(url|search|new tab|back|landscape|portrait|rotate|filler)\\s*(?::\\s*(.*?))?\\s*-->\\s*$"
## Inline or own-line: stop reading here.
const SKIP := "<!--\\s*skip\\s*-->"
## The rest the voice takes for a run of actions. Written by [method speakable] only.
const HOLD := "<!--\\s*action-hold\\s*:\\s*([0-9]*\\.?[0-9]+)\\s*-->"
const LINK := "\\[([^\\]]+)\\]\\(([^)\\s]+)\\)"
## What makes a chapter a tablet chapter: nothing is rewritten for any other.
const TABLET := "<!--\\s*(?:url|search)\\s*:"
const SKIP_MARK := ""

## Seconds per typed character, on average - a lazy typist on glass, one finger, unhurried
## (a first cut at 0.13 read as "blazing fast"). Each letter is a little early or late
## ([method char_times]), and a word break costs WORD_PAUSE more.
const CHAR := 0.21
const WORD_PAUSE := 0.3
## What a skim costs a picture it passes: the drag to it and a look at it.
const PICTURE_DWELL := 3.4
## A skim is owed for this many unread words, or for any picture or placeholder story.
const SKIM_WORDS := 12

static var _re := {}


static func _rx(pattern: String) -> RegEx:
	if not _re.has(pattern):
		var r := RegEx.new()
		r.compile(pattern)
		_re[pattern] = r
	return _re[pattern]


## Does [param body] use the tablet's marks at all?
static func is_tablet(body: String) -> bool:
	return _rx(TABLET).search(body) != null


## [param body] as the voice should read it - see the class note. Any other chapter is
## returned untouched.
static func speakable(body: String) -> String:
	if not is_tablet(body):
		return body
	return String(parse(body)["speakable"])


## The length of an `action-hold` comment, or -1 for any other comment.
static func hold_of(comment: String) -> float:
	var m := _rx(HOLD).search(comment)
	return float(m.get_string(1)) if m != null else -1.0


## THE TIMING OF EACH ACTION, in seconds from its start: every key is a moment the medium draws
## something at, `end` is how long the voice rests for it. Scaled as a whole when a reading
## leaves less room than this.
static func phases(kind: String, text := "", n := 0, m := 0) -> Dictionary:
	match kind:
		"skim":
			# past [param n] words nobody reads and [param m] pictures somebody looks at: still a
			# moment, then a slow drag, a pause on each picture, and on to the next words read
			var e := clampf(1.8 + 0.025 * float(n), 2.0, 4.5) + PICTURE_DWELL * float(m)
			return {"rest": 0.8, "scroll0": 0.8, "scroll1": e - 0.35, "end": e}
		"wake":
			# dark, then the screen comes up, then a beat on the home screen
			return {"on0": 2.0, "on1": 2.8, "end": 4.4}
		"open":
			return {"tap": 0.3, "open0": 0.55, "open1": 1.05, "load0": 1.05, "show": 1.6,
				"load1": 1.9, "end": 2.4}
		"link":
			return {"scroll0": 0.0, "scroll1": 1.5, "tap": 1.75, "load0": 1.9, "show": 2.5,
				"load1": 2.8, "end": 3.3}
		"type", "search":
			# reach for the field, the keyboard rises, type, look at it, go, wait for the page
			var c0 := 1.6
			var c1 := c0 + float(char_times(text)[text.length()])
			var go := c1 + 0.8
			return {"tap": 0.5, "edit": 0.85, "kb1": 1.3, "chars0": c0, "chars1": c1,
				"go": go, "kb0": go + 0.45, "load0": go + 0.1, "show": go + 1.0,
				"load1": go + 1.4, "end": go + 2.0}
		"tab":
			return {"tap": 0.6, "add": 1.05, "end": 2.2}
		"back":
			# the page was seen already: it comes straight back, no load to wait for
			return {"tap": 0.6, "show": 1.0, "end": 1.9}
		"rotate":
			# the camera turns with the screen still showing the old layout ON it, then the new
			# layout dissolves in over it - the content never leaves the glass
			return {"cam1": 1.5, "fade0": 1.3, "fade1": 1.9, "end": 2.2}
	return {"end": 0.0}


## When each letter of [param text] is typed, from the first: `n + 1` offsets, the last being
## when typing ends. Uneven on purpose, and the same every time for the same text, because the
## voice's rest and the screen are both sized from it.
static func char_times(text: String) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	var t := 0.0
	for i in text.length():
		out.append(t)
		var jitter := float(hash([text, i]) & 0xFF) / 255.0
		t += CHAR * lerpf(0.7, 1.35, jitter)
		if text[i] == " ":
			t += WORD_PAUSE
	out.append(t)
	return out


## Lower case, letters and digits only: what decides that a spoken word is a printed one.
static func norm(s: String) -> String:
	return _rx("[^\\p{L}\\p{N}]").sub(s.to_lower(), "", true)


## An address as a key: no scheme, no `www.`, no trailing slash, lower case.
static func url_key(url: String) -> String:
	var u := url.strip_edges().to_lower()
	for p in ["https://", "http://"]:
		if u.begins_with(p):
			u = u.substr(p.length())
	if u.begins_with("www."):
		u = u.substr(4)
	return u.rstrip("/")


static func host_of(url: String) -> String:
	return url_key(url).get_slice("/", 0).get_slice("?", 0)


## THE CHAPTER, walked once:
##
##     pages:   [{url, host, blocks, links: {url_key: word}, search_box, query, results}]
##     words:   [{text, norm, link, emph, spoken, page, block}]   page -1 = not on screen
##     spoken:  word indices the voice reads, in order
##     actions: [{kind, after, dur, text, page, from, word, to}]  `after` = spoken words before it
##     speakable: the text for the voice
##
## Text before the first url is on no page and is neither shown nor read.
static func parse(source: String) -> Dictionary:
	var w := _Walk.new()
	w.run(_collapse_comments(Manuscript.strip_frontmatter(source)))
	return {"pages": w.pages, "words": w.words, "spoken": w.spoken, "actions": w.actions,
		"speakable": "\n".join(w.say)}


## Newlines inside a comment become spaces, so every mark sits on one line - a picture's
## description may run over several.
static func _collapse_comments(text: String) -> String:
	var out := ""
	var at := 0
	for m in _rx("<!--[\\s\\S]*?-->").search_all(text):
		out += text.substr(at, m.get_start() - at) + m.get_string().replace("\n", " ")
		at = m.get_end()
	return out + text.substr(at)


class _Walk:
	var pages: Array = []
	var words: Array = []
	var spoken := PackedInt32Array()
	var actions: Array = []
	var say := PackedStringArray()

	var _cur := -1               # the page being written, -1 before the first url
	var _para := PackedStringArray()
	var _skip_next := false
	var _browser := false
	var _fresh_tab := false
	var _landscape := false
	var _hold := 0.0             # seconds of action waiting for the next spoken words
	# what the reading has passed over since the last words read on this page
	var _skim_words := 0
	var _skim_fill := 0
	var _skim_pics: Array = []
	var _hist: Array = [[]]      # per tab: the pages behind the one shown
	var _tab := 0
	var _returned := false       # back on a page already written: nothing more may be added to it

	func run(body: String) -> void:
		for raw in body.split("\n"):
			_line(String(raw))
		_flush()
		if _hold > 0.0:
			say.append("<!-- action-hold: %.2f -->" % _hold)
		for p in pages:
			_finish_page(p)

	func _line(line: String) -> void:
		var s := line.strip_edges()
		if not Manuscript.speaker_of_line(line).is_empty():
			_flush()
			say.append(s)
			return
		if s.is_empty():
			_flush()
			return
		var m := TabletScript._rx(TabletScript.MARKER).search(line)
		if m != null:
			_flush()
			_marker(m.get_string(1), m.get_string(2).strip_edges())
			return
		var im := TabletScript._rx(Manuscript.IMAGE).search(s)
		if im != null and im.get_start() == 0 and im.get_end() == s.length():
			_flush()
			var prompt := im.get_string(3).strip_edges()
			var key := Manuscript.image_key(prompt)
			if im.get_string(1) == "sketch":
				key = Manuscript.image_key("sketch: " + prompt)
			_block({"kind": "image", "prompt": prompt, "key": key})
			return
		if TabletScript._rx("^" + TabletScript.SKIP + "$").search(s) != null:
			_flush()
			_skip_next = true
			return
		if s == "---" or s == "***" or s == "* * *":
			_flush()
			_block({"kind": "rule"})
			return
		if s.begins_with("#"):
			_flush()
			var level := 0
			while level < s.length() and s[level] == "#":
				level += 1
			_block({"kind": "heading", "level": level, "raw": s.substr(level).strip_edges()})
			return
		_para.append(s)

	func _flush() -> void:
		if _para.is_empty():
			return
		var raw := " ".join(_para)
		_para = PackedStringArray()
		# a line of nothing but notes neither prints nor reads
		if TabletScript._rx(Manuscript.COMMENT).sub(raw, "", true).strip_edges().is_empty() \
				and TabletScript._rx(TabletScript.SKIP).search(raw) == null:
			return
		_block({"kind": "para", "raw": raw})

	## Close a block: tokenize it onto the page, and hand what of it is spoken to the voice,
	## after any rest the hand is owed. Nothing before the first page exists.
	func _block(b: Dictionary) -> void:
		var lead_skip := _skip_next
		_skip_next = false
		if _cur < 0:
			return
		if _returned:
			# the page came back as it was left; writing more onto it would change a page the
			# reader has already seen, from its first visit on
			push_warning("tablet: text after <!-- back --> belongs to no page - follow it with a url, a search or another back")
			return
		var page: Dictionary = pages[_cur]
		var bi := (page["blocks"] as Array).size()
		match String(b["kind"]):
			"image":
				_skim_pics.append(bi)
			"filler":
				_skim_fill += int(b["n"])
		if b.has("raw"):
			var raw := _snap_skip(String(b["raw"]))
			b["raw"] = raw
			var text := _spoken_text(raw, lead_skip)
			var skim := {}
			if not text.strip_edges().is_empty() and (_skim_words >= TabletScript.SKIM_WORDS
					or _skim_fill > 0 or not _skim_pics.is_empty()):
				skim = {"kind": "skim", "from": _cur, "n": _skim_words + 20 * _skim_fill,
					"m": _skim_pics.size(), "pics": _skim_pics.duplicate()}
				_act(skim)
			b["words"] = _tokenize(raw, lead_skip, bi)
			if not text.strip_edges().is_empty():
				_skim_words = 0
				_skim_fill = 0
				_skim_pics = []
				if _hold > 0.0:
					say.append("<!-- action-hold: %.2f -->" % _hold)
					say.append("")
					_hold = 0.0
				var hashes := "#".repeat(int(b["level"])) + " " if b["kind"] == "heading" else ""
				say.append(hashes + text)
				say.append("")
				page["read"] = true
			# words after the skip count toward the next skim
			for wi in (b["words"] as PackedInt32Array):
				if not bool((words[wi] as Dictionary)["spoken"]):
					_skim_words += 1
			if not skim.is_empty():
				for wi in (b["words"] as PackedInt32Array):
					if bool((words[wi] as Dictionary)["spoken"]):
						skim["word"] = wi
						break
		(page["blocks"] as Array).append(b)

	## A skip mark mid-sentence moves to the end of that sentence; one with no sentence end after
	## it reads the paragraph to its end.
	static func _snap_skip(raw: String) -> String:
		var m := TabletScript._rx(TabletScript.SKIP).search(raw)
		if m == null:
			return raw
		var before := TabletScript._rx("<!--[\\s\\S]*?-->").sub(raw.substr(0, m.get_start()), "", true).strip_edges()
		if before.is_empty() or TabletScript._rx("[.!?:;][\"'\\)\\]”’*_]*$").search(before) != null:
			return raw
		var rest := raw.substr(m.get_end())
		var end := TabletScript._rx("[.!?][\"'\\)\\]”’*_]*(?=\\s|$)").search(rest)
		var head := raw.substr(0, m.get_start()).rstrip(" ")
		if end == null:
			return head + " " + rest.strip_edges()
		return head + " " + rest.substr(0, end.get_end()).strip_edges() + " " \
			+ m.get_string() + " " + rest.substr(end.get_end()).strip_edges()

	## The words of [param raw], onto the global list: links kept as targets, emphasis as a
	## level, everything after a skip mark shown but not spoken.
	func _tokenize(raw: String, skipping: bool, block: int) -> PackedInt32Array:
		var out := PackedInt32Array()
		var text := TabletScript._rx(Manuscript.HESITATION).sub(raw, "", true)
		text = TabletScript._rx(TabletScript.SKIP).sub(text, " " + TabletScript.SKIP_MARK + " ", true)
		text = _strip_other_notes(text)
		var segs: Array = []
		var at := 0
		for m in TabletScript._rx(TabletScript.LINK).search_all(text):
			segs.append([text.substr(at, m.get_start() - at), ""])
			segs.append([m.get_string(1), m.get_string(2)])
			at = m.get_end()
		segs.append([text.substr(at), ""])
		var emph := 0
		for sg in segs:
			var link := String(sg[1])
			var first := true
			for tok in String(sg[0]).split(" ", false):
				var t := String(tok).strip_edges()
				if t == TabletScript.SKIP_MARK:
					skipping = true
					continue
				if t.is_empty():
					continue
				# emphasis: a run of `*`/`_` opening a word turns a level on, closing one off
				var lvl := emph
				var open := _run_len(t, true)
				if open > 0:
					lvl = 2 if open >= 2 else 1
					emph = lvl
					t = t.substr(open)
				var close := _run_len(t, false)
				if close > 0:
					t = t.substr(0, t.length() - close)
					emph = 0
				if t.is_empty():
					continue
				var wi := words.size()
				words.append({"text": t, "norm": TabletScript.norm(t), "link": link,
					"emph": lvl, "spoken": not skipping, "page": _cur, "block": block})
				out.append(wi)
				if not skipping:
					spoken.append(wi)
				if not link.is_empty() and first and _cur >= 0:
					var links: Dictionary = (pages[_cur] as Dictionary)["links"]
					var k := TabletScript.url_key(link)
					if not links.has(k):
						links[k] = wi
				first = false
		return out

	static func _run_len(t: String, lead: bool) -> int:
		var n := 0
		var L := t.length()
		while n < L - 1:
			var c := t[n] if lead else t[L - 1 - n]
			if c != "*" and c != "_":
				break
			n += 1
		return n

	static func _strip_other_notes(text: String) -> String:
		return TabletScript._rx("<!--[\\s\\S]*?-->").sub(text, "", true)

	## What the voice reads of [param raw]: up to a skip mark, links as their words, every note
	## but a hesitation gone (the panel turns those into rests).
	func _spoken_text(raw: String, lead_skip: bool) -> String:
		if lead_skip:
			return ""
		var m := TabletScript._rx(TabletScript.SKIP).search(raw)
		var t := raw.substr(0, m.get_start()) if m != null else raw
		t = TabletScript._rx(TabletScript.LINK).sub(t, "$1", true)
		var res := ""
		var at := 0
		for c in TabletScript._rx(Manuscript.COMMENT).search_all(t):
			res += t.substr(at, c.get_start() - at)
			if TabletScript._rx(Manuscript.HESITATION).search(c.get_string()) != null:
				res += c.get_string()
			at = c.get_end()
		return (res + t.substr(at)).strip_edges()

	func _act(a: Dictionary) -> void:
		if actions.is_empty() and String(a["kind"]) != "wake":
			_act({"kind": "wake"})
		a["after"] = spoken.size()
		a["dur"] = float(TabletScript.phases(String(a["kind"]), String(a.get("text", "")),
			int(a.get("n", 0)), int(a.get("m", 0)))["end"])
		actions.append(a)
		_hold += float(a["dur"])

	func _new_page(url: String) -> int:
		pages.append({"url": url, "host": TabletScript.host_of(url), "blocks": [], "links": {},
			"search_box": false, "query": "", "results": false})
		return pages.size() - 1

	func _marker(verb: String, arg: String) -> void:
		match verb:
			"url":
				if arg.is_empty():
					return
				var from := _cur
				var p := _new_page(arg)
				if from >= 0 and not _fresh_tab:
					(_hist[_tab] as Array).append(from)
				_returned = false
				if not _browser:
					_act({"kind": "open", "page": p, "from": -1})
					_browser = true
				elif _fresh_tab or from < 0:
					_act({"kind": "type", "text": arg, "page": p, "from": from})
				else:
					var links: Dictionary = (pages[from] as Dictionary)["links"]
					var k := TabletScript.url_key(arg)
					if links.has(k):
						_act({"kind": "link", "page": p, "from": from, "word": int(links[k])})
					else:
						_act({"kind": "type", "text": arg, "page": p, "from": from})
				_fresh_tab = false
				_cur = p
				_skim_words = 0
				_skim_fill = 0
				_skim_pics = []
			"search":
				if arg.is_empty() or _cur < 0 or _fresh_tab:
					push_warning("tablet: a search needs a page to search from - '%s' ignored" % arg)
					return
				var from := _cur
				(_hist[_tab] as Array).append(from)
				_returned = false
				(pages[from] as Dictionary)["search_box"] = true
				var host := String((pages[from] as Dictionary)["host"])
				var p := _new_page("%s/?q=%s" % [host, arg.uri_encode()])
				(pages[p] as Dictionary)["query"] = arg
				(pages[p] as Dictionary)["results"] = true
				(pages[p] as Dictionary)["search_box"] = true
				_act({"kind": "search", "text": arg, "page": p, "from": from})
				_cur = p
				_skim_words = 0
				_skim_fill = 0
				_skim_pics = []
			"new tab":
				if not _browser:
					_act({"kind": "open", "page": -1, "from": -1})
					_browser = true
				else:
					_act({"kind": "tab"})
					_hist.append([])
					_tab = _hist.size() - 1
				_fresh_tab = true
				_returned = false
			"back":
				var behind: Array = _hist[_tab]
				if behind.is_empty() or _cur < 0:
					push_warning("tablet: <!-- back --> with nothing to go back to - ignored")
					return
				# A PAGE LEFT UNREAD IS GLANCED AT FIRST: a reader who opens a page and goes
				# straight back has still looked down it
				var page: Dictionary = pages[_cur]
				if not bool(page.get("read", false)) or _skim_words >= TabletScript.SKIM_WORDS \
						or _skim_fill > 0 or not _skim_pics.is_empty():
					_act({"kind": "skim", "from": _cur, "n": _skim_words + 20 * _skim_fill,
						"m": _skim_pics.size(), "pics": _skim_pics.duplicate(), "word": -1})
				var p := int(behind.pop_back())
				_act({"kind": "back", "page": p, "from": _cur})
				_cur = p
				_skim_words = 0
				_skim_fill = 0
				_skim_pics = []
				_returned = true
			"landscape", "portrait", "rotate":
				var want := not _landscape if verb == "rotate" else verb == "landscape"
				if want != _landscape:
					_landscape = want
					_act({"kind": "rotate", "to": 1 if want else 0})
			"filler":
				if _cur >= 0:
					_block({"kind": "filler", "n": maxi(1, int(arg)) if not arg.is_empty() else 1})

	## A heading with nothing written under it before the next heading of its rank or above is a
	## STUB: the page draws squiggles for its body.
	func _finish_page(p: Dictionary) -> void:
		var blocks: Array = p["blocks"]
		for i in blocks.size():
			var b: Dictionary = blocks[i]
			if b["kind"] != "heading":
				continue
			var stub := true
			for j in range(i + 1, blocks.size()):
				var n: Dictionary = blocks[j]
				if n["kind"] == "heading" and int(n["level"]) <= int(b["level"]):
					break
				if n["kind"] != "filler" and n["kind"] != "rule":
					stub = false
					break
			b["stub"] = stub and int(b["level"]) >= 2
