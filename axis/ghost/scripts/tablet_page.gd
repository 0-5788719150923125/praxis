extends RefCounted
class_name TabletPage

## TabletPage - one web page of a [TabletScript], set into a column of a given width.
##
## Pages are typeset, not rendered from HTML: the blocks the chapter wrote (headings, paragraphs,
## pictures, links) in a site's own face and colour, and PLACEHOLDERS for everything it did not
## write. A placeholder is drawn as squiggles - lines of wavy strokes in the shape of text - so
## it reads at a glance as "there is something here" and never as words someone forgot to
## write. A heading with nothing under it gets a squiggled body; `<!-- filler: N -->` puts N
## placeholder stories in; and every page is padded with them to [member min_height], because a
## page with nothing past the fold has nothing to scroll.
##
## THE SITE'S LOOK IS ITS HOST: face, accent colour and masthead are hashed off the address, so
## the same site looks the same wherever the chapter visits it, and two sites never match by
## accident. Two kinds are recognised from what the chapter does with them, not declared: a
## page searched FROM that has little else on it is a search engine's home (a centred logo and
## a box), and a page reached BY a search is its results.

const BODY := 30
const SIZES := {1: 52, 2: 40, 3: 34}
const LEAD := 1.48            # line height over font size
const MARGIN := 56.0
const MAX_COLUMN := 1060.0    # sites have a max width; landscape centres the column

const SANS := ["Inter", "Roboto", "Helvetica Neue", "Helvetica", "Arial", "Liberation Sans",
	"DejaVu Sans"]

var script_doc: Dictionary
var index := -1
var width := 1200.0
var height := 0.0
var min_height := 0.0
var items: Array = []         # in page coordinates, in drawing order
var word_rect := {}           # global word index -> Rect2
var search_rect := Rect2()
var block_rect := {}          # block index -> Rect2, for the pictures a skim stops on
var bg := Color.WHITE
var ink := Color(0.1, 0.1, 0.12)
var accent := Color(0.15, 0.35, 0.8)
var serif := false
var title := ""               # what the tab says

var _x0 := 0.0
var _col := 0.0
var _y := 0.0
var _line: Array = []         # word items waiting for their line to be finished
var _line_x := 0.0
var _line_h := 0.0
var _center := false

static var _faces := {}


## The face for [param serif_face] at [param level] (bit 1 italic, bit 2 bold).
static func face(serif_face: bool, level: int) -> Font:
	var k := "%s%d" % [serif_face, level]
	if _faces.has(k):
		return _faces[k]
	var f := SystemFont.new()
	f.font_names = PackedStringArray(BookLayout.SERIFS if serif_face else SANS)
	f.font_italic = (level & 1) != 0
	f.font_weight = 700 if (level & 2) != 0 else 400
	f.antialiasing = TextServer.FONT_ANTIALIASING_GRAY
	_faces[k] = f
	return f


func _h(salt: String) -> int:
	return hash([String((script_doc["pages"][index] as Dictionary)["host"]), salt])


func _f(salt: String) -> float:
	return float(_h(salt) & 0xFFFF) / 65535.0


## Typeset page [param pi] of [param doc] at [param w] pixels wide.
func build(doc: Dictionary, pi: int, w: float, min_h: float) -> void:
	script_doc = doc
	index = pi
	width = w
	min_height = min_h
	items = []
	word_rect = {}
	block_rect = {}
	var page: Dictionary = doc["pages"][pi]
	serif = _f("serif") < 0.35 and not bool(page["results"])
	accent = Color.from_hsv(_f("hue"), lerpf(0.55, 0.85, _f("sat")), lerpf(0.45, 0.7, _f("val")))
	bg = Color.from_hsv(_f("hue") + 0.08, 0.04, 0.99) if _f("paper") < 0.4 else Color.WHITE
	_col = minf(w - MARGIN * 2.0, MAX_COLUMN)
	_x0 = (w - _col) * 0.5
	_y = 34.0
	var blocks: Array = page["blocks"]
	title = String(page["host"])
	if bool(page.get("real", false)):
		_real(page, w, min_h)
		return
	var engine := bool(page["search_box"]) and not bool(page["results"]) and _written(blocks) <= 3
	if engine:
		_engine_home(blocks)
		height = maxf(_y + 120.0, min_h * 0.6)
		return
	if bool(page["results"]):
		_search_box(String(page["query"]))
		_y += 18.0
		title = String(page["query"])
	var boxed := not bool(page["search_box"]) or bool(page["results"])
	for i in blocks.size():
		var b: Dictionary = blocks[i]
		match String(b["kind"]):
			"heading":
				var lvl := int(b["level"])
				if i == 0 and lvl == 1 and not bool(page["results"]) and _site_name().is_empty():
					_masthead(b)
				else:
					if i == 0 and not bool(page["results"]) and not _site_name().is_empty():
						_band(_site_name())
					if bool(page["results"]) and _is_link(b):
						_result_url(b)
					_text(b, mini(lvl, 3), 2)
				if title == String(page["host"]) and lvl <= 2:
					title = _plain(b)
				if bool(b.get("stub", false)):
					_squiggles(3, BODY, ink.lerp(bg, 0.72), "stub%d" % i)
					_y += 26.0
				if not boxed and i == 0:
					_search_box("")
					boxed = true
			"para":
				_text(b, 0, 0)
				_y += BODY * 0.7
			"image":
				_image(b, i)
			"filler":
				for k in int(b["n"]):
					_filler("f%d_%d" % [i, k])
			"rule":
				items.append({"kind": "rule", "rect": Rect2(_x0, _y + 10.0, _col, 2.0)})
				_y += 24.0
	if not boxed:
		_search_box("")
	var k := 0
	while _y < min_h:
		_filler("pad%d" % k)
		k += 1
	_y += 40.0
	items.append({"kind": "foot", "rect": Rect2(0.0, _y, w, 160.0)})
	_y += 160.0
	height = _y


static func _written(blocks: Array) -> int:
	var n := 0
	for b in blocks:
		if (b as Dictionary)["kind"] != "filler":
			n += 1
	return n


func _is_link(b: Dictionary) -> bool:
	var ws: PackedInt32Array = b.get("words", PackedInt32Array())
	return not ws.is_empty() and not String((script_doc["words"][ws[0]] as Dictionary)["link"]).is_empty()


func _plain(b: Dictionary) -> String:
	var out := PackedStringArray()
	for wi in (b.get("words", PackedInt32Array()) as PackedInt32Array):
		out.append(String((script_doc["words"][wi] as Dictionary)["text"]))
	return " ".join(out)


# --- text ----------------------------------------------------------------------------

## Set block [param b]'s words: [param level] picks the size (0 body), [param emph] is added
## to every word's own emphasis (2 = bold, for headings).
func _text(b: Dictionary, level: int, emph: int, col := Color(0, 0, 0, 0), centred := false) -> void:
	var fs: int = SIZES.get(level, BODY)
	_center = centred
	_y += fs * 0.25 if level > 0 else 0.0
	_line_x = 0.0
	_line = []
	_line_h = fs * LEAD
	for wi in (b.get("words", PackedInt32Array()) as PackedInt32Array):
		var w: Dictionary = script_doc["words"][wi]
		var lvl := int(w["emph"]) | emph
		var f := face(serif, lvl)
		var ww := f.get_string_size(String(w["text"]), HORIZONTAL_ALIGNMENT_LEFT, -1, fs).x
		var sp := f.get_string_size(" ", HORIZONTAL_ALIGNMENT_LEFT, -1, fs).x
		if _line_x > 0.0 and _line_x + ww > _col:
			_end_line()
		var link := not String(w["link"]).is_empty()
		var c := col if col.a > 0.0 else (accent if link else ink)
		_line.append({"kind": "word", "wi": wi, "x": _line_x, "w": ww, "fs": fs, "lvl": lvl,
			"col": c, "link": link})
		_line_x += ww + sp
	_end_line()


func _end_line() -> void:
	if _line.is_empty():
		return
	var used: float = (_line.back() as Dictionary)["x"] + (_line.back() as Dictionary)["w"]
	var shift := (_col - used) * 0.5 if _center else 0.0
	for it in _line:
		var d: Dictionary = it
		var fs := int(d["fs"])
		var x := _x0 + shift + float(d["x"])
		d["pos"] = Vector2(x, _y + fs * 1.08)
		var r := Rect2(x - 4.0, _y + fs * 0.12, float(d["w"]) + 8.0, fs * 1.24)
		d["rect"] = r
		word_rect[int(d["wi"])] = r
		items.append(d)
	_y += _line_h
	_line = []
	_line_x = 0.0


# --- the furniture ---------------------------------------------------------------------

## A REAL page: its capture, edge to edge at the tablet's width, as long as the capture is. Not
## captured yet, a plain page saying so - with the address, so the panel row is easy to find.
func _real(page: Dictionary, w: float, min_h: float) -> void:
	bg = Color.WHITE
	var key := String(page["snap"])
	var path := Illustrations.path_for(key)
	var h := min_h
	if not path.is_empty():
		var img := Image.load_from_file(path)
		if img != null and not img.is_empty():
			h = w * float(img.get_height()) / float(img.get_width())
	items.append({"kind": "snap", "rect": Rect2(0.0, 0.0, w, h), "key": key,
		"url": String(page["url"])})
	height = h


## The name an EARLIER page on this host gave the site (its opening H1, set as the masthead),
## or "" when this is the first page of the host seen - whose own H1 then becomes the masthead.
func _site_name() -> String:
	var host := String((script_doc["pages"][index] as Dictionary)["host"])
	for k in index:
		var pg: Dictionary = script_doc["pages"][k]
		if String(pg["host"]) != host or bool(pg["results"]) or (pg["blocks"] as Array).is_empty():
			continue
		var b0: Dictionary = (pg["blocks"] as Array)[0]
		if b0["kind"] == "heading" and int(b0["level"]) == 1:
			return _plain(b0)
	return ""


## The site's name, remembered from an earlier page, in a thinner band of its colour - not
## words of this page, so nothing to read or highlight.
func _band(name: String) -> void:
	var top := _y - 34.0
	items.append({"kind": "band", "rect": Rect2(0.0, top, width, 96.0), "col": accent.darkened(0.25)})
	items.append({"kind": "label", "pos": Vector2(MARGIN, top + 62.0), "text": name, "fs": 34,
		"col": Color.WHITE, "bold": true})
	_y = top + 96.0
	items.append({"kind": "nav", "rect": Rect2(0.0, _y, width, 58.0)})
	_y += 58.0 + 30.0


## A news site's name, in a band of its colour.
func _masthead(b: Dictionary) -> void:
	var top := _y - 34.0
	var n0 := items.size()
	_y += 26.0
	var x0 := _x0
	var col := _col
	_x0 = MARGIN
	_col = width - MARGIN * 2.0
	_text(b, 1, 2, Color.WHITE)
	_x0 = x0
	_col = col
	_y += 26.0
	items.insert(n0, {"kind": "band",
		"rect": Rect2(0.0, top, width, _y - top), "col": accent.darkened(0.25)})
	# a row of section names under it, as squiggles
	items.append({"kind": "nav", "rect": Rect2(0.0, _y, width, 58.0)})
	_y += 58.0 + 30.0


## A search engine's front page: its name as a logo, centred, and the box under it.
func _engine_home(blocks: Array) -> void:
	_y = 230.0
	for i in blocks.size():
		var b: Dictionary = blocks[i]
		if b["kind"] == "heading" and i == 0:
			var n0 := items.size()
			_text(b, 1, 2, accent, true)
			for k in range(n0, items.size()):
				var d: Dictionary = items[k]
				d["fs"] = 76
				d["logo"] = true
			# re-set at logo size, centred
			_relayout_logo(n0)
			_y += 50.0
			title = _plain(b)
			_search_box("", true)
			_y += 40.0
		elif b["kind"] == "para" or b["kind"] == "heading":
			_text(b, 0, 0, ink.lerp(bg, 0.35), true)
			_y += 12.0


func _relayout_logo(n0: int) -> void:
	var total := 0.0
	var f := face(false, 2)
	for k in range(n0, items.size()):
		var d: Dictionary = items[k]
		d["w"] = f.get_string_size(String(script_doc["words"][int(d["wi"])]["text"]),
			HORIZONTAL_ALIGNMENT_LEFT, -1, 76).x
		total += float(d["w"]) + 18.0
	total -= 18.0
	var x := (width - total) * 0.5
	var y: float = ((items[n0] as Dictionary)["rect"] as Rect2).position.y if n0 < items.size() else _y
	for k in range(n0, items.size()):
		var d: Dictionary = items[k]
		d["pos"] = Vector2(x, y + 76.0 * 1.0)
		d["rect"] = Rect2(x - 4.0, y, float(d["w"]) + 8.0, 76.0 * 1.25)
		word_rect[int(d["wi"])] = d["rect"]
		x += float(d["w"]) + 18.0
	_y = y + 76.0 * 1.3


func _search_box(query: String, wide := false) -> void:
	var w := _col * (0.86 if wide else 1.0)
	var r := Rect2((width - w) * 0.5, _y, w, 76.0)
	search_rect = r
	items.append({"kind": "search", "rect": r, "query": query})
	_y += 76.0 + 26.0


func _result_url(b: Dictionary) -> void:
	var ws: PackedInt32Array = b["words"]
	var url := String((script_doc["words"][ws[0]] as Dictionary)["link"])
	items.append({"kind": "label", "pos": Vector2(_x0, _y + 24.0), "text": TabletScript.url_key(url),
		"fs": 22, "col": Color(0.13, 0.5, 0.25)})
	_y += 30.0


func _image(b: Dictionary, bi: int) -> void:
	var path := Illustrations.path_for(String(b["key"]))
	var aspect := 9.0 / 16.0
	if not path.is_empty():
		var img := Image.load_from_file(path)
		if img != null and not img.is_empty():
			aspect = float(img.get_height()) / float(img.get_width())
	var r := Rect2(_x0, _y + 8.0, _col, _col * clampf(aspect, 0.4, 1.3))
	items.append({"kind": "image", "rect": r, "key": String(b["key"]), "prompt": String(b["prompt"])})
	block_rect[bi] = r
	_y = r.end.y + 34.0


## [param n] lines of squiggles, the last one short.
func _squiggles(n: int, fs: int, col: Color, salt: String, thick := 0.34) -> void:
	for i in n:
		var frac := 1.0 if i < n - 1 else lerpf(0.35, 0.8, _f(salt + str(i)))
		items.append({"kind": "squiggle", "rect": Rect2(_x0, _y + fs * 0.5, _col * frac, fs * thick),
			"seed": _h(salt + str(i)), "col": col})
		_y += fs * LEAD


## A story nobody wrote: a squiggled headline, sometimes a grey picture, a squiggled stub.
func _filler(salt: String) -> void:
	_y += 10.0
	_squiggles(1, 40, ink.lerp(bg, 0.45), salt + "h", 0.42)
	if _f(salt + "img") < 0.3:
		var r := Rect2(_x0, _y + 6.0, _col, _col * 0.42)
		items.append({"kind": "image", "rect": r, "key": "", "prompt": ""})
		_y = r.end.y + 22.0
	_squiggles(2 + (_h(salt + "n") & 3), BODY, ink.lerp(bg, 0.72), salt + "b")
	items.append({"kind": "rule", "rect": Rect2(_x0, _y + 14.0, _col, 1.5)})
	_y += 44.0


# --- drawing ----------------------------------------------------------------------------

## Draw the page onto [param ci] with its top at [param top] - [param scroll], keeping to the
## band [param clip_y0]..[param clip_y1]. [param lit] is the reading's highlight
## ([method TabletMedium._lit]).
func draw(ci: CanvasItem, top: float, scroll: float, clip_y0: float, clip_y1: float,
		lit: Dictionary, now: float, textures: Callable) -> void:
	ci.draw_rect(Rect2(0.0, clip_y0, width, clip_y1 - clip_y0), bg)
	var dy := top - scroll
	for it in items:
		var d: Dictionary = it
		var kind := String(d["kind"])
		if kind == "word":
			var p: Vector2 = d["pos"]
			if p.y + dy < clip_y0 - 10.0 or p.y + dy - float(d["fs"]) > clip_y1 + 10.0:
				continue
			_draw_word(ci, d, dy, lit)
			continue
		var r: Rect2 = d.get("rect", Rect2(d.get("pos", Vector2.ZERO), Vector2(1, 1)))
		var rr := Rect2(r.position + Vector2(0.0, dy), r.size)
		if rr.end.y < clip_y0 - 40.0 or rr.position.y > clip_y1 + 40.0:
			continue
		match kind:
			"band":
				ci.draw_rect(rr, d["col"])
			"nav":
				ci.draw_rect(rr, accent.darkened(0.45))
				var x := MARGIN
				var k := 0
				while x < width - MARGIN - 60.0:
					var sw := 60.0 + float((_h("nav%d" % k)) & 63)
					squiggle(ci, Vector2(x, rr.get_center().y), sw, 9.0, Color(1, 1, 1, 0.55), _h("nv%d" % k))
					x += sw + 34.0
					k += 1
			"squiggle":
				squiggle(ci, rr.position, rr.size.x, rr.size.y, d["col"], int(d["seed"]))
			"rule":
				ci.draw_rect(rr, ink.lerp(bg, 0.86))
			"label":
				ci.draw_string(face(serif, 2) if d.has("bold") else face(false, 0), d["pos"] + Vector2(0.0, dy), String(d["text"]),
					HORIZONTAL_ALIGNMENT_LEFT, -1, int(d["fs"]), d["col"])
			"image":
				_draw_image(ci, rr, d, textures)
			"search":
				_draw_search(ci, rr, String(d["query"]))
			"snap":
				var tex: Texture2D = textures.call(String(d["key"]))
				if tex != null:
					ci.draw_texture_rect(tex, rr, false)
				else:
					var f := face(false, 0)
					var c := Vector2(width * 0.5, clip_y0 + 260.0)
					ci.draw_string(face(false, 2), c - Vector2(width * 0.5, 0.0), "A real page, not captured yet",
						HORIZONTAL_ALIGNMENT_CENTER, width, 34, ink.lerp(bg, 0.4))
					ci.draw_string(f, c + Vector2(-width * 0.5, 56.0), String(d["url"]),
						HORIZONTAL_ALIGNMENT_CENTER, width, 26, accent)
					ci.draw_string(f, c + Vector2(-width * 0.5, 110.0), "Capture it, or import a screenshot, in the Illustrations panel.",
						HORIZONTAL_ALIGNMENT_CENTER, width, 22, ink.lerp(bg, 0.55))
			"foot":
				ci.draw_rect(rr, ink.lerp(bg, 0.93))
				for j in 3:
					squiggle(ci, rr.position + Vector2(MARGIN, 40.0 + j * 34.0),
						width * (0.3 - j * 0.06), 8.0, ink.lerp(bg, 0.7), _h("foot%d" % j))


func _draw_word(ci: CanvasItem, d: Dictionary, dy: float, hl: Dictionary) -> void:
	var wi := int(d["wi"])
	var fs := int(d["fs"])
	var f := face(serif, int(d["lvl"])) if not d.has("logo") else face(false, 2)
	var p: Vector2 = d["pos"] + Vector2(0.0, dy)
	var col: Color = d["col"]
	var text := String(script_doc["words"][wi]["text"])
	if d.has("logo"):
		# the engine's name in a run of colours, a letter at a time
		var x := p.x
		for c in text.length():
			var lc := Color.from_hsv(fposmod(_f("logo") + float(c) * 0.11, 1.0), 0.7, 0.85)
			ci.draw_string(f, Vector2(x, p.y), text[c], HORIZONTAL_ALIGNMENT_LEFT, -1, fs, lc)
			x += f.get_string_size(text[c], HORIZONTAL_ALIGNMENT_LEFT, -1, fs).x
		return
	# EVERY WORD IS DRAWN GLYPH BY GLYPH, lit or not, exactly as the book does it: two drawing
	# paths snap glyphs to the pixel grid differently, and the letters would jump as the colour
	# reached them. The colour is the book's: each letter its own hue by its place in the text,
	# lit as the voice reaches it and cooling back to ink behind it.
	var times: Dictionary = hl.get("times", {})
	var lit := times.has(wi)
	var cur := int(hl.get("cur", -1))
	var now := float(hl.get("now", 0.0))
	var frac := float(hl.get("frac", 0.0))
	var tt: Vector2 = times.get(wi, Vector2.ZERO)
	var c0 := 0
	if lit:
		var ch: PackedInt32Array = hl["char0"]
		c0 = ch[wi] if wi < ch.size() else 0
	var n := text.length()
	for gl in glyphs(f, text, fs):
		var k := int(gl["start"])
		var gc := col
		if lit:
			var at := (float(k) + 0.5) / float(maxi(n, 1))
			var g := 0.0
			if wi != cur:
				g = exp(-maxf(now - lerpf(tt.x, tt.y, at), 0.0) / BookMedium.TRAIL_TAU)
			elif at <= frac:
				g = exp(-maxf((frac - at) * maxf(tt.y - tt.x, 0.05), 0.0) / BookMedium.TRAIL_TAU)
			if g > 0.01:
				var ci_i := c0 + k
				var hue := fposmod(float(hl["hue0"]) + float(ci_i) * BookMedium.HUE_STEP, 1.0)
				var sw := 0.5 + 0.35 * sin(float(ci_i) * 0.21 - now * 0.9) \
					+ 0.15 * sin(float(ci_i) * 0.36 + now * 0.45)
				var sat := lerpf(0.45, 0.9, clampf(sw, 0.0, 1.0))
				gc = col.lerp(Color.from_hsv(hue, sat, 0.55), g)
		_ts.font_draw_glyph(gl["rid"], ci.get_canvas_item(), fs, p + (gl["pos"] as Vector2), int(gl["index"]), gc)
	if bool(d["link"]):
		ci.draw_line(p + Vector2(0.0, fs * 0.16), p + Vector2(float(d["w"]), fs * 0.16), Color(col, 0.6), 2.0)


## A word's shaped glyphs, `[{index, rid, pos, start}]` - the book's [method BookMedium._glyphs],
## cached the same way.
static var _glyph_cache := {}
static var _ts: TextServer = TextServerManager.get_primary_interface()

static func glyphs(font: Font, text: String, fs: int) -> Array:
	var key := "%d|%d|%s" % [font.get_instance_id(), fs, text]
	if _glyph_cache.has(key):
		return _glyph_cache[key]
	var line := TextLine.new()
	line.add_string(text, font, fs)
	var out: Array = []
	var x := 0.0
	for g in _ts.shaped_text_get_glyphs(line.get_rid()):
		var gd: Dictionary = g
		for _r in maxi(1, int(gd.get("repeat", 1))):
			var off: Vector2 = gd.get("offset", Vector2.ZERO)
			if int(gd.get("index", 0)) != 0 or not gd.has("font_rid"):
				out.append({"index": int(gd.get("index", 0)), "rid": gd.get("font_rid", RID()),
					"pos": Vector2(x, 0.0) + off, "start": int(gd.get("start", 0))})
			x += float(gd.get("advance", 0.0))
	if _glyph_cache.size() > 20000:
		_glyph_cache = {}
	_glyph_cache[key] = out
	return out


func _draw_image(ci: CanvasItem, r: Rect2, d: Dictionary, textures: Callable) -> void:
	var tex: Texture2D = textures.call(String(d["key"])) if not String(d["key"]).is_empty() else null
	if tex == null:
		# NOT YET PAINTED: a grey plate with a picture glyph, and what it will be, faintly
		ci.draw_rect(r, ink.lerp(bg, 0.9))
		var c := r.get_center() - Vector2(0.0, 18.0 if not String(d["prompt"]).is_empty() else 0.0)
		var g := ink.lerp(bg, 0.7)
		var s := minf(r.size.x, r.size.y) * 0.16
		ci.draw_rect(Rect2(c - Vector2(s, s * 0.75), Vector2(s * 2.0, s * 1.5)), g, false, 3.0)
		ci.draw_colored_polygon(PackedVector2Array([c + Vector2(-s * 0.85, s * 0.6),
			c + Vector2(-s * 0.2, -s * 0.15), c + Vector2(s * 0.25, s * 0.3),
			c + Vector2(s * 0.5, s * 0.05), c + Vector2(s * 0.85, s * 0.6)]), g)
		ci.draw_circle(c + Vector2(s * 0.45, -s * 0.38), s * 0.14, g)
		if not String(d["prompt"]).is_empty():
			ci.draw_multiline_string(face(false, 1), Vector2(r.position.x + 40.0, c.y + s + 46.0),
				String(d["prompt"]), HORIZONTAL_ALIGNMENT_CENTER, r.size.x - 80.0, 20, 3,
				ink.lerp(bg, 0.55))
		return
	var ts := Vector2(tex.get_size())
	var want := r.size.x / r.size.y
	var src := Rect2(Vector2.ZERO, ts)
	if ts.x / ts.y > want:
		src = Rect2((ts.x - ts.y * want) * 0.5, 0.0, ts.y * want, ts.y)
	else:
		src = Rect2(0.0, (ts.y - ts.x / want) * 0.5, ts.x, ts.x / want)
	ci.draw_texture_rect_region(tex, r, src)


var _box: StyleBoxFlat

func _draw_search(ci: CanvasItem, r: Rect2, query: String) -> void:
	if _box == null:
		_box = StyleBoxFlat.new()
	var sb := _box
	sb.bg_color = Color.WHITE
	sb.border_color = ink.lerp(bg, 0.7)
	sb.set_border_width_all(2)
	sb.set_corner_radius_all(int(r.size.y * 0.5))
	sb.anti_aliasing = true
	sb.shadow_color = Color(0, 0, 0, 0.08)
	sb.shadow_size = 6
	ci.draw_style_box(sb, r)
	var c := Vector2(r.position.x + 40.0, r.get_center().y)
	ci.draw_arc(c - Vector2(2, 2), 11.0, 0.0, TAU, 24, ink.lerp(bg, 0.45), 3.0, true)
	ci.draw_line(c + Vector2(6, 6), c + Vector2(14, 14), ink.lerp(bg, 0.45), 3.5, true)
	if not query.is_empty():
		ci.draw_string(face(false, 0), Vector2(r.position.x + 74.0, r.get_center().y + 11.0), query,
			HORIZONTAL_ALIGNMENT_LEFT, r.size.x - 100.0, 30, ink)


## A line of placeholder writing: wavy strokes in runs the length of words, so it has the
## shape of text and none of its content. Deterministic in [param seed].
static func squiggle(ci: CanvasItem, at: Vector2, w: float, h: float, col: Color, seed: int) -> void:
	var r := RandomNumberGenerator.new()
	r.seed = seed
	var x := 0.0
	var amp := h * 0.42
	while x < w - 8.0:
		var run := minf(r.randf_range(34.0, 130.0) * (h / 10.0 if h > 10.0 else 1.0), w - x)
		if run < 12.0:
			break
		var pts := PackedVector2Array()
		var ph := r.randf() * TAU
		var fq := r.randf_range(0.16, 0.26) * (10.0 / maxf(h, 6.0))
		var steps := maxi(3, int(run / 5.0))
		for i in steps + 1:
			var px := x + run * float(i) / float(steps)
			var env := sin(PI * float(i) / float(steps)) * 0.4 + 0.6
			pts.append(at + Vector2(px, sin(px * fq + ph) * amp * env))
		ci.draw_polyline(pts, col, maxf(2.0, h * 0.42), true)
		x += run + r.randf_range(12.0, 20.0) * maxf(1.0, h / 12.0)
