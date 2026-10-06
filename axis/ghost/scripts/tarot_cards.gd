extends RefCounted
class_name TarotCards

## TarotCards - a card's face, its back and its booklet page, composed in 2D for the table.
##
## THE PAINTER PAINTS PICTURES; THE DECK PRINTS CARDS. A generated picture is the card's
## illustration and nothing else (see [TarotPrompts.card_image]) - the stock, the frame, the
## numeral and the name are printed around it here, from the episode's look, so every card of a
## deck shares one frame exactly and every name is spelled right. The back is the same: the
## painted design inside the deck's frame.
##
## EACH IS A STOPPED SUBVIEWPORT, the comic's trick: drawn once (UPDATE_ONCE), then held as a
## texture in VRAM at no further cost, and drawn again only when something it shows changes - a
## picture landing mid-reading. Never read back to the CPU.

## A tarot card's proportions: 70 x 120 mm.
const ASPECT := 70.0 / 120.0
const FACE_PX := Vector2i(560, 960)
## The booklet page: a little white booklet's page, a touch narrower than the card is tall.
const PAGE_PX := Vector2i(660, 940)
const PAGE_ASPECT := float(PAGE_PX.x) / float(PAGE_PX.y)


## A viewport holding [param canvas], sized [param px], drawn once.
static func viewport(host: Node, canvas: Node2D, px: Vector2i) -> SubViewport:
	var vp := SubViewport.new()
	vp.size = px
	vp.disable_3d = true
	vp.transparent_bg = true
	vp.render_target_update_mode = SubViewport.UPDATE_ONCE
	vp.add_child(canvas)
	host.add_child(vp)
	return vp


## Draw [param vp] again, once (its content changed).
static func redraw(vp: SubViewport) -> void:
	if vp == null:
		return
	for c in vp.get_children():
		if c is CanvasItem:
			(c as CanvasItem).queue_redraw()
	vp.render_target_update_mode = SubViewport.UPDATE_ONCE


## A rounded rectangle, filled or outlined.
static func _box(ci: CanvasItem, r: Rect2, col: Color, radius: float, fill := true, width := 2.0) -> void:
	var sb := StyleBoxFlat.new()
	sb.bg_color = col if fill else Color(0, 0, 0, 0)
	sb.draw_center = fill
	sb.set_corner_radius_all(int(radius))
	sb.anti_aliasing = true
	if not fill:
		sb.set_border_width_all(int(maxf(1.0, width)))
		sb.border_color = col
	ci.draw_style_box(sb, r)


## [param tex] drawn into [param dst] cropped to cover it - the picture's center kept.
static func _cover(ci: CanvasItem, tex: Texture2D, dst: Rect2, flip := false) -> void:
	var ts := Vector2(tex.get_size())
	var want := dst.size.x / dst.size.y
	var have := ts.x / ts.y
	var src := Rect2(Vector2.ZERO, ts)
	if have > want:
		src.size.x = ts.y * want
		src.position.x = (ts.x - src.size.x) * 0.5
	else:
		src.size.y = ts.x / want
		src.position.y = (ts.y - src.size.y) * 0.5
	if flip:
		ci.draw_set_transform(dst.get_center(), PI, Vector2.ONE)
		ci.draw_texture_rect_region(tex, Rect2(-dst.size * 0.5, dst.size), src)
		ci.draw_set_transform(Vector2.ZERO, 0.0, Vector2.ONE)
	else:
		ci.draw_texture_rect_region(tex, dst, src)


## A picture not painted yet: a soft field of the deck's own colors, seeded per card - a
## picture still arriving, never a gray broken plate.
static func _placeholder(ci: CanvasItem, r: Rect2, palette: Array, seed: int) -> void:
	if palette.is_empty():
		palette = TarotTable.FALLBACK_PALETTE
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	var a := TarotTable.color(String(palette[0])) if not palette.is_empty() else Color(0.2, 0.2, 0.3)
	ci.draw_rect(r, a.darkened(0.2))
	for i in 9:
		var c := TarotTable.color(String(palette[rng.randi_range(0, maxi(0, palette.size() - 1))]))
		c.a = 0.22
		var p := r.position + Vector2(rng.randf(), rng.randf()) * r.size
		var rad := rng.randf_range(0.12, 0.38) * r.size.x
		for k in 5:
			ci.draw_circle(p, rad * (1.0 - float(k) * 0.17), c)


## The largest font size up to [param size] at which [param text] fits [param width].
static func _fit(font: Font, text: String, size: int, width: float) -> int:
	var s := size
	while s > 10 and font.get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, s).x > width:
		s -= 1
	return s


## THE PRINTED CARD - face or back. Everything it draws comes from the episode's look and the
## card; the picture is optional (a placeholder until it lands).
class Face:
	extends Node2D

	var look: Dictionary = {}
	var card: Dictionary = {}
	var art: Texture2D = null
	var back := false
	var seed := 0

	func _draw() -> void:
		var sz := Vector2(TarotCards.FACE_PX)
		var frame: Dictionary = look.get("frame", {})
		var style := String(frame.get("style", "line"))
		var stock := TarotTable.color(String(frame.get("stock", "#efe6d2")))
		var ink := TarotTable.color(String(frame.get("ink", "#1d1a2b")))
		var accent := TarotTable.color(String(frame.get("accent", "#c9a227")))
		var radius := sz.x * 0.06
		TarotCards._box(self, Rect2(Vector2.ZERO, sz), stock, radius)
		var m := sz.x * (0.018 if style == "bleed" else 0.06)
		var top := 0.0 if back or style == "bleed" else sz.y * 0.075
		var bottom := 0.0 if back or style == "bleed" else sz.y * 0.105
		var win := Rect2(Vector2(m, m + top), Vector2(sz.x - 2.0 * m, sz.y - 2.0 * m - top - bottom))
		if art != null:
			# a reversed card is printed the right way up: the TABLE turns it over
			TarotCards._cover(self, art, win)
		else:
			TarotCards._placeholder(self, win, look.get("palette", []),
				hash([seed, String(card.get("key", "back"))]))
		_frame(style, win, ink, accent, sz)
		if not back:
			_lettering(style, win, ink, stock, sz)

	func _frame(style: String, win: Rect2, ink: Color, accent: Color, sz: Vector2) -> void:
		var r := sz.x * 0.02
		match style:
			"double":
				TarotCards._box(self, win.grow(sz.x * 0.012), ink, r, false, sz.x * 0.009)
				TarotCards._box(self, win.grow(sz.x * 0.026), accent, r, false, sz.x * 0.004)
			"corners":
				TarotCards._box(self, win, ink, 0.0, false, sz.x * 0.005)
				var L := sz.x * 0.11
				var off := sz.x * 0.022
				var w := sz.x * 0.012
				for corner in [win.position, Vector2(win.end.x, win.position.y), win.end,
						Vector2(win.position.x, win.end.y)]:
					var dx := -1.0 if (corner as Vector2).x > win.get_center().x else 1.0
					var dy := -1.0 if (corner as Vector2).y > win.get_center().y else 1.0
					var c := (corner as Vector2) + Vector2(-dx, -dy) * off
					draw_polyline(PackedVector2Array([c + Vector2(0, dy * L), c, c + Vector2(dx * L, 0)]),
						accent, w, true)
					draw_circle(c, w * 1.4, accent)
			"deco":
				var g := sz.x * 0.018
				for i in 3:
					var rr := win.grow(g * float(i))
					var st := sz.x * (0.05 + 0.03 * float(i))
					var pts := PackedVector2Array([
						rr.position + Vector2(st, 0), Vector2(rr.end.x - st, rr.position.y),
						Vector2(rr.end.x - st, rr.position.y + st * 0.0), Vector2(rr.end.x, rr.position.y + st),
						Vector2(rr.end.x, rr.end.y - st), Vector2(rr.end.x - st, rr.end.y),
						Vector2(rr.position.x + st, rr.end.y), Vector2(rr.position.x, rr.end.y - st),
						Vector2(rr.position.x, rr.position.y + st), rr.position + Vector2(st, 0)])
					draw_polyline(pts, accent if i == 1 else ink, sz.x * (0.006 if i == 1 else 0.004), true)
			"bleed":
				TarotCards._box(self, win, ink, r, false, sz.x * 0.003)
			_:
				TarotCards._box(self, win, ink, r, false, sz.x * 0.006)
				TarotCards._box(self, win.grow(sz.x * 0.02), ink, r * 1.4, false, sz.x * 0.002)

	func _lettering(style: String, win: Rect2, ink: Color, stock: Color, sz: Vector2) -> void:
		var face := TarotTable.font(String(look.get("title_face", "roman")))
		var caps := String(look.get("title_face", "roman")) in ["roman", "deco", "sign", "typed"]
		var name := String(card.get("name", ""))
		if caps:
			name = name.to_upper()
		var num := String(card.get("numeral", ""))
		var band_top := Rect2(Vector2(win.position.x, sz.y * 0.06 * 0.5), Vector2(win.size.x, win.position.y - sz.y * 0.03))
		var band_bot := Rect2(Vector2(win.position.x, win.end.y), Vector2(win.size.x, sz.y - win.end.y - sz.x * 0.05))
		if style == "bleed":
			band_top = Rect2(win.position + Vector2(0, sz.y * 0.012), Vector2(win.size.x, sz.y * 0.07))
			band_bot = Rect2(Vector2(win.position.x, win.end.y - sz.y * 0.1), Vector2(win.size.x, sz.y * 0.085))
			var plate := stock
			plate.a = 0.86
			draw_rect(band_bot.grow_individual(-sz.x * 0.08, 0, -sz.x * 0.08, 0), plate)
			if not num.is_empty():
				var w := sz.x * 0.2
				draw_rect(Rect2(Vector2(sz.x * 0.5 - w * 0.5, band_top.position.y), Vector2(w, band_top.size.y)), plate)
		var pad := sz.x * 0.06
		var size := TarotCards._fit(face, name, int(sz.y * 0.052), band_bot.size.x - pad * 2.0)
		var asc := face.get_ascent(size)
		var desc := face.get_descent(size)
		var y := band_bot.get_center().y + (asc - desc) * 0.5
		draw_string(face, Vector2(band_bot.position.x, y), name, HORIZONTAL_ALIGNMENT_CENTER,
			band_bot.size.x, size, ink)
		if not num.is_empty():
			var ns := int(sz.y * 0.045)
			var ny := band_top.get_center().y + (face.get_ascent(ns) - face.get_descent(ns)) * 0.5
			draw_string(face, Vector2(band_top.position.x, ny), num, HORIZONTAL_ALIGNMENT_CENTER,
				band_top.size.x, ns, ink)


## THE BOOKLET'S COLORS are the card's: its stock for the paper, its ink and accent for the type,
## each kept readable on that paper (the running text at [constant TarotTable.TEXT_CONTRAST]).
## [param shade] (0..1) takes a page a hair darker, so two pages are not one sheet.
static func page_colors(look: Dictionary, shade: float) -> Dictionary:
	var frame: Dictionary = look.get("frame", {}) if look.get("frame") is Dictionary else {}
	var stock := TarotTable.color(String(frame.get("stock", "#efe6d2")))
	var paper := stock.lerp(stock.darkened(0.04), shade)
	return {"paper": paper,
		"ink": TarotTable.legible_ink(TarotTable.color(String(frame.get("ink", "#1d1a2b"))), paper,
			TarotTable.TEXT_CONTRAST),
		"accent": TarotTable.legible_ink(TarotTable.color(String(frame.get("accent", "#7a2e3a"))), paper)}


## THE BOOKLET PAGE beside a drawn card: its entry in the deck's little booklet, printed in the
## card's colors ([method page_colors]). Shown, never read aloud.
class Page:
	extends Node2D

	var look: Dictionary = {}
	var card: Dictionary = {}
	var seed := 0

	func _draw() -> void:
		var sz := Vector2(TarotCards.PAGE_PX)
		var rng := RandomNumberGenerator.new()
		rng.seed = hash([seed, "page"])
		var col := TarotCards.page_colors(look, rng.randf())
		var paper: Color = col["paper"]
		var ink: Color = col["ink"]
		var accent: Color = col["accent"]
		TarotCards._box(self, Rect2(Vector2.ZERO, sz), paper, sz.x * 0.012)
		# the page's own grain: a few soft blotches, never a texture that fights the type
		for i in 24:
			var c := paper.darkened(rng.randf_range(0.004, 0.012))
			draw_circle(Vector2(rng.randf(), rng.randf()) * sz, rng.randf_range(20.0, 90.0), c)
		var m := sz.x * 0.1
		TarotCards._box(self, Rect2(Vector2(m * 0.5, m * 0.5), sz - Vector2(m, m)), accent, 4.0, false, 2.0)
		var face := TarotTable.font(String(look.get("title_face", "roman")))
		var book := TarotTable.font(TarotTable.BOOK_FACE)
		var italic := TarotTable.font(TarotTable.BOOK_ITALIC)
		var x := m
		var w := sz.x - 2.0 * m
		var y := m * 1.25
		var num := String(card.get("numeral", ""))
		if not num.is_empty():
			draw_string(face, Vector2(x, y + 28.0), num, HORIZONTAL_ALIGNMENT_CENTER, w, 30, accent)
			y += 44.0
		var name := String(card.get("name", ""))
		var ns := TarotCards._fit(face, name, 52, w)
		y += face.get_ascent(ns)
		draw_string(face, Vector2(x, y), name, HORIZONTAL_ALIGNMENT_CENTER, w, ns, ink)
		y += face.get_descent(ns) + 10.0
		var rev := bool(card.get("reversed", false))
		var b: Dictionary = card.get("booklet", {}) if card.get("booklet") is Dictionary else {}
		if rev:
			y += 30.0
			draw_string(italic, Vector2(x, y), "Reversed", HORIZONTAL_ALIGNMENT_CENTER, w, 28, accent)
		var kw := TarotPrompts.strings(b.get("keywords", []))
		if not kw.is_empty():
			y += 40.0
			var line := "  ·  ".join(kw)
			var ks := TarotCards._fit(italic, line, 32, w)
			draw_string(italic, Vector2(x, y), line, HORIZONTAL_ALIGNMENT_CENTER, w, ks, ink.lerp(paper, 0.12))
		# the ornament between the head and the text
		y += 30.0
		var cx := sz.x * 0.5
		draw_line(Vector2(cx - w * 0.22, y), Vector2(cx - 12.0, y), accent, 2.0, true)
		draw_line(Vector2(cx + 12.0, y), Vector2(cx + w * 0.22, y), accent, 2.0, true)
		var d := PackedVector2Array([Vector2(cx, y - 7.0), Vector2(cx + 7.0, y), Vector2(cx, y + 7.0), Vector2(cx - 7.0, y)])
		draw_colored_polygon(d, accent)
		y += 26.0
		var text := String(b.get("reversed" if rev and b.has("reversed") else "upright", ""))
		if text.strip_edges().is_empty():
			text = "The booklet is silent on this card."
		var room := sz.y - m * 1.1 - y
		var size := 38
		while size > 18:
			var h := book.get_multiline_string_size(text, HORIZONTAL_ALIGNMENT_FILL, w, size).y
			if h <= room:
				break
			size -= 1
		# justified, except the paragraph's last line - stretched, it read "others   or   oneself."
		draw_multiline_string(book, Vector2(x, y + book.get_ascent(size)), text,
			HORIZONTAL_ALIGNMENT_FILL, w, size, -1, ink,
			TextServer.BREAK_MANDATORY | TextServer.BREAK_WORD_BOUND,
			TextServer.JUSTIFICATION_KASHIDA | TextServer.JUSTIFICATION_WORD_BOUND
			| TextServer.JUSTIFICATION_SKIP_LAST_LINE | TextServer.JUSTIFICATION_DO_NOT_SKIP_SINGLE_LINE)
