extends BookMedium
class_name NotebookMedium

## NotebookMedium - the reading as a research notebook, written by hand, open on a desk.
##
## Everything that makes the book work is the book's - the 3D spread, the turning leaf, the
## camera, the voice followed word by word - and only what a notebook does differently is here:
## ruled paper with a red margin, the chapter in a handwriting face ([NotebookLayout]), a
## marbled composition cover with a paper label, emphasis underlined, photos clipped on with a
## paper clip at an angle, and sketches drawn onto the page in the same ink as the writing.
##
## THE TONE IS THE AUTHOR'S. A chapter drafted as a lab notebook or a field journal - dated
## entries under headings, times in the margin, "<!-- sketch: ... -->" beside the note it goes
## with - reads as one here; the same chapter in the Novel medium reads as a book.

const RULE_COL := Color(0.50, 0.66, 0.86, 0.78)
const MARGIN_COL := Color(0.86, 0.36, 0.40, 0.8)
const PHOTO_PAPER := Color(0.97, 0.965, 0.95)
const STEEL := Color(0.60, 0.62, 0.66)

var _hand := "kalam"
var _plate: MeshInstance3D
static var _marble: Texture2D


func _make_layout() -> BookLayout:
	var l := NotebookLayout.new()
	l.hand = _hand
	l.hand_seed = _seed
	l.body_fs = int(NotebookLayout.HANDS[_hand]["size"])
	return l


func _cover_font() -> Font:
	return NotebookLayout.hand_face(_hand, 0)


func mount(st: SubViewport) -> void:
	super.mount(st)
	# THE SPINE TAPE: the black cloth strip a composition book is bound with, on both halves.
	var tm := BoxMesh.new()
	tm.size = Vector3(0.16, 0.0316, PAGE_H + 0.102)
	var tmat := StandardMaterial3D.new()
	tmat.albedo_color = Color(0.05, 0.05, 0.055)
	tmat.roughness = 0.8
	for side in [1.0, -1.0]:
		var tape := MeshInstance3D.new()
		tape.mesh = tm
		tape.material_override = tmat
		tape.position = Vector3(0.08 * side, -0.058, 0.0)
		if side > 0.0:
			_root3.add_child(tape)
		else:
			_pivot.add_child(tape)
	# THE LABEL: a paper plate on the front cover's underside (which faces up once folded - see
	# BookMedium's cover label), with the name and the author written on it.
	_plate = MeshInstance3D.new()
	var pm := PlaneMesh.new()
	pm.size = Vector2(0.62, 0.34)
	_plate.mesh = pm
	var pmat := StandardMaterial3D.new()
	pmat.albedo_texture = _label_texture()
	pmat.roughness = 0.9
	_plate.material_override = pmat
	_plate.transform = Transform3D(Basis(Vector3.RIGHT, PI), Vector3(-0.53, -0.0738, -0.2))
	_pivot.add_child(_plate)
	_cover_label.position = Vector3(-0.53, -0.0745, -0.25)
	_cover_label.font_size = 64
	_cover_author.position = Vector3(-0.53, -0.0745, -0.1)
	_cover_author.font_size = 40


func begin_session() -> void:
	super.begin_session()
	var r := RandomNumberGenerator.new()
	r.seed = _seed ^ 0x40E7
	var keys := NotebookLayout.HANDS.keys()
	keys.sort()
	var h := String(keys[r.randi() % keys.size()])
	if h != _hand:
		_hand = h
		_source = ""             # typeset again, in this hand
	# Paper is white with the faintest warmth; the pen is blue or black ballpoint, or a
	# blue-black fountain ink.
	_paper = Color.from_hsv(r.randf_range(0.10, 0.14), r.randf_range(0.02, 0.06),
		r.randf_range(0.955, 0.985))
	var inks := [Color(0.10, 0.16, 0.46), Color(0.08, 0.08, 0.11), Color(0.12, 0.14, 0.32)]
	_ink = (inks[r.randi() % inks.size()] as Color).lightened(r.randf_range(0.0, 0.06))
	var cm := _cover.material_override as StandardMaterial3D
	cm.albedo_color = Color.WHITE
	cm.albedo_texture = _marble_texture()
	cm.uv1_scale = Vector3(1.3, 1.3, 1.0)
	cm.roughness = 0.7
	for st in [_stack_l, _stack_r]:
		((st as MeshInstance3D).material_override as StandardMaterial3D).albedo_color = Color(0.95, 0.95, 0.93)
	var hf := NotebookLayout.hand_face(_hand, 0)
	_cover_label.font = hf
	_cover_author.font = hf
	_cover_label.modulate = _ink
	_cover_author.modulate = Color(_ink, 0.9)


## The composition book's black-and-white marble: warped noise cut hard into black ground and
## pale blotches. Built once, synchronously, for the reason [method BookMedium._grime] gives.
static func _marble_texture() -> Texture2D:
	if _marble != null:
		return _marble
	var n := FastNoiseLite.new()
	n.seed = 0xC0DE
	n.noise_type = FastNoiseLite.TYPE_SIMPLEX_SMOOTH
	n.frequency = 0.035
	n.fractal_type = FastNoiseLite.FRACTAL_FBM
	n.fractal_octaves = 4
	n.domain_warp_enabled = true
	n.domain_warp_amplitude = 40.0
	var img := n.get_seamless_image(512, 512)
	img.convert(Image.FORMAT_L8)
	var d := img.get_data()
	var out := PackedByteArray()
	out.resize(d.size() * 3)
	for i in d.size():
		var v := float(d[i]) / 255.0
		var w := smoothstep(0.58, 0.63, v) + smoothstep(0.30, 0.26, v) * 0.85
		var c := int(lerpf(18.0, 228.0, clampf(w, 0.0, 1.0)))
		out[i * 3] = c
		out[i * 3 + 1] = c
		out[i * 3 + 2] = c
	var m := Image.create_from_data(512, 512, false, Image.FORMAT_RGB8, out)
	m.generate_mipmaps()
	_marble = ImageTexture.create_from_image(m)
	return _marble


static func _label_texture() -> Texture2D:
	var img := Image.create(310, 170, false, Image.FORMAT_RGB8)
	img.fill(Color(0.95, 0.94, 0.90))
	for e in [Rect2i(0, 0, 310, 4), Rect2i(0, 166, 310, 4), Rect2i(0, 0, 4, 170), Rect2i(306, 0, 4, 170)]:
		img.fill_rect(e, Color(0.1, 0.1, 0.12))
	img.generate_mipmaps()
	return ImageTexture.create_from_image(img)


## The name is written to fit the label, however long it is.
func _set_cover(title: String, author: String) -> void:
	super._set_cover(title, author)
	var f := NotebookLayout.hand_face(_hand, 0)
	var w := f.get_string_size(title, HORIZONTAL_ALIGNMENT_LEFT, -1, 64).x * _cover_label.pixel_size
	_cover_label.font_size = int(clampf(64.0 * 0.54 / maxf(w, 0.001), 24.0, 64.0))


# --- the page -------------------------------------------------------------------

func _draw_paper(ci: CanvasItem, pg: Dictionary) -> void:
	var size := BookLayout.PAGE
	var y := NotebookLayout.HEADER
	var last: float = NotebookLayout.HEADER + NotebookLayout.RULE * floor((size.y - 80.0 - NotebookLayout.HEADER)
		/ NotebookLayout.RULE)
	var first := true
	while y <= last + 0.5:
		ci.draw_line(Vector2(0.0, y), Vector2(size.x, y), RULE_COL, 2.6 if first else 1.8, true)
		first = false
		y += NotebookLayout.RULE
	ci.draw_line(Vector2(NotebookLayout.MARGIN_X, 0.0), Vector2(NotebookLayout.MARGIN_X, size.y),
		MARGIN_COL, 2.2, true)
	super._draw_paper(ci, pg)


## A hand has no italic: an emphasised word is underlined, and the line runs on under the next
## word when that one is emphasised too, so a phrase gets one stroke rather than a dashed one.
func _decorate_word(ci: CanvasItem, i: int, w: Dictionary, ink_col: Color) -> void:
	if (int(w["emph"]) & 1) == 0:
		return
	var rect: Rect2 = w["rect"]
	var base: Vector2 = w["base"]
	var width := rect.size.x
	if i + 1 < _layout.words.size():
		var nx: Dictionary = _layout.words[i + 1]
		if (int(nx["emph"]) & 1) != 0 and int(nx["page"]) == int(w["page"]) \
				and absf((nx["base"] as Vector2).y - base.y) < 6.0:
			width = (nx["base"] as Vector2).x - base.x
	_underline(ci, Vector2(base.x, base.y + 6.0), width, ink_col, i)


## A pen line: not quite straight, not quite level.
func _underline(ci: CanvasItem, at: Vector2, width: float, col: Color, salt: int) -> void:
	var pts := PackedVector2Array()
	for k in 5:
		var f := float(k) / 4.0
		var jy := (float(hash([_seed, salt, k]) & 0xFF) / 255.0 - 0.5) * 2.4
		pts.append(at + Vector2(width * f, jy + f * 1.2))
	ci.draw_polyline(pts, col, 2.0, true)


func _draw_image(ci: CanvasItem, im: Dictionary) -> void:
	if not bool(im.get("photo", false)):
		super._draw_image(ci, im)
		return
	var rect: Rect2 = im["rect"]
	var c := rect.get_center()
	var hs := rect.size * 0.5
	var ang := float(im.get("angle", 0.0))
	# A soft shadow down and to the right of the lamp: the print is on the page, not in it.
	for k in 4:
		ci.draw_set_transform(c + Vector2(6.0, 9.0), ang, Vector2.ONE)
		var g := float(k) * 3.0
		ci.draw_rect(Rect2(-hs - Vector2(g, g), rect.size + Vector2(g, g) * 2.0), Color(0, 0, 0, 0.06))
	ci.draw_set_transform(c, ang, Vector2.ONE)
	ci.draw_rect(Rect2(-hs, rect.size), PHOTO_PAPER)
	var b := NotebookLayout.PHOTO_BORDER
	var inner := Rect2(-hs + Vector2(b, b), rect.size - Vector2(b, b) * 2.0)
	var tex := _texture_for(String(im["key"]))
	if tex == null:
		# NOT YET TAKEN: a blank print with what it will be written on it.
		ci.draw_rect(inner, Color(0.80, 0.80, 0.79))
		ci.draw_multiline_string(_layout.face(0), inner.position + Vector2(18.0, 40.0),
			String(im["prompt"]), HORIZONTAL_ALIGNMENT_LEFT, inner.size.x - 36.0, 24,
			maxi(1, int((inner.size.y - 40.0) / 30.0)), Color(_ink, 0.6))
	else:
		var ts := Vector2(tex.get_size())
		var want := inner.size.x / inner.size.y
		var src := Rect2(Vector2.ZERO, ts)
		if ts.x / ts.y > want:
			src = Rect2((ts.x - ts.y * want) * 0.5, 0.0, ts.y * want, ts.y)
		else:
			src = Rect2(0.0, (ts.y - ts.x / want) * 0.5, ts.x, ts.x / want)
		ci.draw_texture_rect_region(tex, inner, src)
	ci.draw_rect(Rect2(-hs, rect.size), Color(0, 0, 0, 0.18), false, 1.2)
	ci.draw_set_transform(Vector2.ZERO, 0.0, Vector2.ONE)


## The clips go on after every photo on the page, so a fanned stack sits under its clip.
func _draw_overlay(ci: CanvasItem, pg: Dictionary) -> void:
	for im in pg["images"]:
		var clip: Dictionary = (im as Dictionary).get("clip", {})
		if not clip.is_empty():
			_draw_clip(ci, clip["pos"], float(clip["angle"]))


## A Gem clip seen from above: a wire in three turns, the outer loop gripping the page edge at
## [param at] and its length pointing in along [param ang] (0 = straight down the page).
func _draw_clip(ci: CanvasItem, at: Vector2, ang: float) -> void:
	var l := NotebookLayout.CLIP_LEN
	var a := 14.0
	var g := 5.0
	var xl_i := -a + g
	var xr_i := a - g
	var ri := (xr_i - xl_i) * 0.5
	var y_ti := l * 0.22
	var rb := (xr_i + a) * 0.5
	var pts := PackedVector2Array()
	pts.append(Vector2(xl_i, l * 0.62))
	_clip_arc(pts, Vector2(0.0, y_ti + ri), ri, PI, TAU)
	_clip_arc(pts, Vector2((xr_i - a) * 0.5, l - rb), rb, 0.0, PI)
	_clip_arc(pts, Vector2(0.0, a), a, PI, TAU)
	pts.append(Vector2(a, l * 0.78))
	ci.draw_set_transform(at + Vector2(3.0, 4.0), ang, Vector2.ONE)
	ci.draw_polyline(pts, Color(0, 0, 0, 0.22), 4.2, true)
	ci.draw_set_transform(at, ang, Vector2.ONE)
	ci.draw_polyline(pts, STEEL, 3.4, true)
	ci.draw_set_transform(at + Vector2(-0.7, -0.7), ang, Vector2.ONE)
	ci.draw_polyline(pts, Color(0.93, 0.94, 0.96), 1.1, true)
	ci.draw_set_transform(Vector2.ZERO, 0.0, Vector2.ONE)


static func _clip_arc(pts: PackedVector2Array, c: Vector2, r: float, from: float, to: float) -> void:
	for k in 13:
		var t := lerpf(from, to, float(k) / 12.0)
		pts.append(c + Vector2(cos(t), sin(t)) * r)
