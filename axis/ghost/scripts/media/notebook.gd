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
## THE PEEK: a clipped photo lifts off the page, curling back from its free edge toward the
## clip, while the words it hides are being read, and settles again once they have been.
## `_peel[stack]` runs 0..1 at a steady rate and is drawn through a smoothstep, so it eases in
## and out; a stack lifts together.
##
## IT LIFTS WELL AHEAD OF THE VOICE, as a reader moves a photo aside while still a few lines
## above what it hides - not as the voice arrives ("a real reader would turn the photo much
## earlier, like 5 seconds or even 10"). It is fully up [constant PEEL_EARLY] seconds before the
## first hidden word is spoken, timed off that word's own start where the take has reached it;
## [constant PEEL_LEAD] words early (about the same, at speaking pace) where it has not yet.
const PEEL_TIME := 0.9
const PEEL_EARLY := 6.0
const PEEL_LEAD := 20
## ONCE UP, A PHOTO STAYS UP until the reading has passed the last word it hides, and never
## comes down within this many seconds of lifting. The "early" test switches from a word count
## to the take's own timings as they arrive, and the two can disagree for a moment: the photo
## curled, dropped and curled again.
const PEEL_HOLD := 4.0
var _raised := {}                # stack -> when it was lifted (subtitle clock)
## How much of a photo the clip PINS, from its clipped edge: that part stays flat under the clip
## and the rest folds back over it. About where a clip's inner end reaches.
const PIN := 76.0
## The roller a lifted print is turned over: wide enough to read as a curl rather than a crease,
## small enough that the loop does not lie back over the writing beside the page edge.
const CURL_RADIUS := 80.0
var _peel := {}
var _peel_dt := 0.0
var _prints := {}
var _prints_rev := -1
var _plate: MeshInstance3D
static var _marble: Texture2D


func _make_layout() -> BookLayout:
	var l := NotebookLayout.new()
	l.hand = _hand
	l.hand_seed = _seed
	l.body_fs = int(NotebookLayout.HANDS[_hand]["size"])
	return l


## Room past the page edge for what hangs off it: a clip's outer loop, and a photo folded back
## over its clip while the words under it are read. The widest photo (0.56 of the page) folded
## at its pin reaches ~430 px past a side edge; a landscape photo less than that past the top.
## Nothing folds downward, so the bottom needs only the clips' room.
func _page_pad() -> float:
	return 480.0


func _page_pad_top() -> float:
	return 480.0


func _page_pad_bottom() -> float:
	return 64.0


func _cover_font() -> Font:
	return NotebookLayout.hand_face(_hand, 0)


func advance(features, delta: float, bookend: float) -> void:
	_peel_dt = delta
	super.advance(features, delta, bookend)


func _refresh_pages() -> void:
	_tick_peel()
	super._refresh_pages()


func _tick_peel() -> void:
	if _layout == null:
		return
	var r := _reading()
	var li := int(r["layout"]) if not r.is_empty() else -1
	var now: float = _subs.now() if _subs != null and is_instance_valid(_subs) else 0.0
	var step := _peel_dt / PEEL_TIME
	for page in _needed_pages():
		if page < 0 or page >= _layout.pages.size():
			continue
		for im in (_layout.pages[page] as Dictionary)["images"]:
			var cover: Array = (im as Dictionary).get("cover", [-1, -1])
			if int(cover[0]) < 0:
				continue
			var st := int(im["stack"])
			var lo := int(cover[0])
			var t0 := _spoken_at(lo)
			var early := now >= t0 - PEEL_TIME - PEEL_EARLY if t0 >= 0.0 else li >= lo - PEEL_LEAD
			var at: float = _raised.get(st, NAN)
			at = peel_latch(at, now, li, lo, int(cover[1]), early)
			if is_nan(at):
				_raised.erase(st)
			else:
				_raised[st] = at
			var v := float(_peel.get(st, 0.0))
			_peel[st] = move_toward(v, 0.0 if is_nan(at) else 1.0, step)


## The lift decision for one stack: when it was lifted ([param at], NAN while it is down) after
## this frame. It LIFTS when the reading is within its words or [param early] says it is time;
## it DROPS only once the reading has left its words - past the last, or well back before the
## first (a restart) - and at least [constant PEEL_HOLD] seconds after it lifted. Pure, so the
## gate can drive it.
static func peel_latch(at: float, now: float, li: int, lo: int, hi: int, early: bool) -> float:
	if is_nan(at):
		return now if li >= 0 and li <= hi and (li >= lo or early) else NAN
	var gone := li > hi or li < lo - PEEL_LEAD * 3
	return NAN if gone and absf(now - at) >= PEEL_HOLD else at


## When layout word [param li] starts being spoken, from the take's own timings, or -1 while
## the voice has not reached it in what has arrived so far.
func _spoken_at(li: int) -> float:
	if _layout == null:
		return -1.0
	if float(_lay_t0.get(li, -1.0)) >= 0.0:
		return float(_lay_t0[li])
	return -1.0


func _reset_reading() -> void:
	super._reset_reading()
	_raised = {}
	_peel = {}


func _page_state(page: int) -> String:
	if _layout == null or page < 0 or page >= _layout.pages.size():
		return ""
	var out := ""
	for im in (_layout.pages[page] as Dictionary)["images"]:
		var st := int((im as Dictionary).get("stack", -1))
		if st >= 0:
			out += "%d:%.2f," % [st, snappedf(float(_peel.get(st, 0.0)), 0.02)]
	return out


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
	# Paper is white with the faintest warmth. The pen is BLACK unless a voice names its own
	# ink (see [method _ink_for]) - a colour per speaker is how a page tells voices apart
	# without labels, so the default must be the one colour no voice is likely to choose.
	_paper = Color.from_hsv(r.randf_range(0.10, 0.14), r.randf_range(0.02, 0.06),
		r.randf_range(0.955, 0.985))
	_ink = NotebookLayout.INKS["black"]
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


## Each voice writes in its own ink, from `ink:` on the voice in the frontmatter, carried beside
## the text as the document's `inks` ({speaker: ink}); a voice naming none writes in black.
func _ink_for(w: Dictionary) -> Color:
	var who := String(w.get("speaker", ""))
	if _subs == null or not is_instance_valid(_subs) or who.is_empty():
		return _ink
	var inks: Variant = (_subs.document as Dictionary).get("inks", {})
	if not (inks is Dictionary) or not (inks as Dictionary).has(who):
		return _ink
	var v := String((inks as Dictionary)[who])
	return _ink if v.strip_edges().is_empty() else NotebookLayout.ink_color(v)


## A hand has no italic: an emphasised word is underlined, and the line runs on under the next
## word when that one is emphasised too, so a phrase gets one stroke rather than a dashed one.
func _decorate_word(ci: CanvasItem, i: int, w: Dictionary, ink_col: Color) -> void:
	if not _underlined(w):
		return
	var rect: Rect2 = w["rect"]
	var base: Vector2 = w["base"]
	var width := rect.size.x
	if i + 1 < _layout.words.size():
		var nx: Dictionary = _layout.words[i + 1]
		if _underlined(nx) and int(nx["page"]) == int(w["page"]) \
				and absf((nx["base"] as Vector2).y - base.y) < 6.0:
			width = (nx["base"] as Vector2).x - base.x
	_underline(ci, Vector2(base.x, base.y + 6.0), width, ink_col, i)


## Emphasis, or a heading marked for it.
static func _underlined(w: Dictionary) -> bool:
	return (int(w["emph"]) & 1) != 0 or bool(w.get("underline", false))


## A pen line: not quite straight, and riding the same drift as the writing above it.
func _underline(ci: CanvasItem, at: Vector2, width: float, col: Color, salt: int) -> void:
	var pts := PackedVector2Array()
	for k in 7:
		var f := float(k) / 6.0
		var x := at.x + width * f
		var jy := (float(hash([_seed, salt, k]) & 0xFF) / 255.0 - 0.5) * 1.2
		pts.append(Vector2(x, at.y + _drift(_draw_page, x, at.y) + jy + f * 0.8))
	ci.draw_polyline(pts, col, 2.0, true)


# --- the hand ---------------------------------------------------------------------

## HOW THE WRITING WANDERS OFF THE LINE, as a smooth field over the page rather than a jitter
## per word: along a line the letters rise gradually toward its end (a hand drifts up as it goes
## right), by a slope that changes only slowly from line to line, so neighbouring lines lean
## together in clusters; a very slow wave rides along each line; the slant wanders slowly down
## the page; and each letter differs by a hair in size and slant and sits a fraction off, which
## is the tremor of a pen. Every term is a function of place and the session seed, so a page
## draws the same way every time and a render writes what the live reading wrote.
const DRIFT_SLOPE := -0.005        # mean rise per px along a line (negative is up)
const DRIFT_SLOPE_VARY := 0.0045   # ...and how far a cluster of lines strays from it
const DRIFT_CLUSTER := 5.0         # lines over which the slope changes
const DRIFT_WAVE := 1.1            # px of the slow wave along a line
const SLANT := -0.05               # mean slant (radians of skew)
const SLANT_VARY := 0.06
const LETTER_SCALE := 0.035        # a letter's size varies by this much either way
const LETTER_SLANT := 0.03
const TREMOR := 0.45               # px
var _draw_page := -1


func draw_page(ci: CanvasItem, page: int, hl: Dictionary) -> void:
	_draw_page = page
	super.draw_page(ci, page, hl)


## Smooth 1D value noise in 0..1 at [param t], lattice seeded by [param salt].
func _vnoise(t: float, salt: int) -> float:
	var i := floori(t)
	var f := t - float(i)
	var a := float(hash([_seed, salt, i]) & 0xFFFF) / 65535.0
	var b := float(hash([_seed, salt, i + 1]) & 0xFFFF) / 65535.0
	return lerpf(a, b, f * f * (3.0 - 2.0 * f))


## How far the writing sits off the line at page point ([param x], [param y]), in px.
func _drift(page: int, x: float, y: float) -> float:
	var line := y / NotebookLayout.RULE
	var slope := DRIFT_SLOPE + DRIFT_SLOPE_VARY * (_vnoise(line / DRIFT_CLUSTER, page * 17 + 1) - 0.5) * 2.0
	var dx := maxf(0.0, x - (NotebookLayout.MARGIN_X + 16.0))
	var phase := TAU * _vnoise(line / 3.0, page * 17 + 2)
	return slope * dx + DRIFT_WAVE * sin(x / 170.0 + phase)


func _glyph_xform(i: int, k: int, at: Vector2, page: int) -> Transform2D:
	var h := hash([_seed, i, k])
	var u1 := float(h & 0xFF) / 255.0 - 0.5
	var u2 := float((h >> 8) & 0xFF) / 255.0 - 0.5
	var u3 := float((h >> 16) & 0xFF) / 255.0 - 0.5
	var slant := SLANT + SLANT_VARY * (_vnoise(at.y / NotebookLayout.RULE / 7.0, page * 17 + 3) - 0.5) * 2.0
	var s := 1.0 + LETTER_SCALE * u1 * 2.0
	return Transform2D(0.0, Vector2(s, s), slant + LETTER_SLANT * u2 * 2.0,
		at + Vector2(0.0, _drift(page, at.x, at.y) + TREMOR * u3 * 2.0))


func _draw_image(ci: CanvasItem, im: Dictionary) -> void:
	if not bool(im.get("photo", false)):
		super._draw_image(ci, im)
		return
	var peel := smoothstep(0.0, 1.0, float(_peel.get(int(im.get("stack", -1)), 0.0)))
	if peel > 0.01:
		_draw_curled(ci, im, peel)
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


## A photo lifted by [param peel] (0..1), ROLLED BACK over a wide curve the way a person turns a
## print they do not want to crease. The part the clip pins ([constant PIN]) stays flat; past it
## the print wraps round a roller of radius [constant CURL_RADIUS] through an angle that grows to
## a half turn, and runs on straight from where it leaves the roller - so its length is exactly
## kept, and at full peel it lies back past the edge with a rounded loop over the clip. Drawn as
## strips, each lit by how far it faces from the page and showing the back once past upright.
## Three earlier cuts were each wrong in a way a still frame showed: a curl at the free end left
## the clipped half on the text; a lean to 135 degrees looked foreshortened; and a hinge fold
## "looks like a flat, linear fold - which is not what a real person would do". Past upright the
## clip is under the print, so it is drawn first.
func _draw_curled(ci: CanvasItem, im: Dictionary, peel: float) -> void:
	var rect: Rect2 = im["rect"]
	var sz := rect.size
	var c := rect.get_center()
	var ang := float(im.get("angle", 0.0))
	var a := Vector2(0, 1)          # clipped edge -> free edge, in the photo's own frame
	var o := Vector2(0, -sz.y * 0.5)
	var length := sz.y
	var width := sz.x
	match String(im.get("hinge", "top")):
		"right":
			a = Vector2(-1, 0)
			o = Vector2(sz.x * 0.5, 0)
			length = sz.x
			width = sz.y
		"left":
			a = Vector2(1, 0)
			o = Vector2(-sz.x * 0.5, 0)
			length = sz.x
			width = sz.y
	var side := Vector2(-a.y, a.x) * width * 0.5
	var t0 := clampf(PIN / length, 0.0, 0.5)
	var theta := peel * PI
	var rem := (1.0 - t0) * length
	var r := minf(CURL_RADIUS, rem / PI)
	var fold := o + a * t0 * length
	var tex := _print_for(im)
	var clip: Dictionary = im.get("clip", {})
	if theta > PI * 0.5 and not clip.is_empty():
		_draw_clip(ci, clip["pos"], float(clip["angle"]))
	# along the print from the fold: projected distance u, height z and the facing angle phi
	const N := 40
	var us := PackedFloat32Array()
	var zs := PackedFloat32Array()
	var ph := PackedFloat32Array()
	for i in N + 1:
		var sv := rem * float(i) / float(N)
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
	ci.draw_set_transform(c, ang, Vector2.ONE)
	# ONE shadow under what lies over the page, reaching as far in as the roll does
	var reach := 0.0
	var top := 0.0
	for i in N + 1:
		reach = maxf(reach, us[i])
		top = maxf(top, zs[i])
	var lift := minf(top / maxf(rem, 1.0), 1.0)
	var off := Vector2(6.0, 9.0) + Vector2(0.06, 0.09) * minf(top, 160.0)
	var shadow_end := fold + a * reach
	ci.draw_colored_polygon(PackedVector2Array([o + side + off, o - side + off,
		shadow_end - side + off, shadow_end + side + off]), Color(0, 0, 0, 0.14 * (1.0 - 0.5 * lift)))
	# the pinned part, flat
	_print_quad(ci, tex, o, fold, side, a, 0.0, t0, length, sz, 1.0)
	# the rolled part, strip by strip in order along the print: what comes later lies over what
	# came before, which is the right way round for a print rolled back on itself
	for i in N:
		var p0 := fold + a * us[i]
		var p1 := fold + a * us[i + 1]
		if p0.distance_to(p1) < 0.4:
			continue                 # edge-on at the top of the roll
		var q := PackedVector2Array([p0 + side, p0 - side, p1 - side, p1 + side])
		var fa := 0.5 * (ph[i] + ph[i + 1])
		var cf := cos(fa)
		if cf >= 0.0 and tex != null:
			var ta := t0 + (1.0 - t0) * float(i) / float(N)
			var tb := t0 + (1.0 - t0) * float(i + 1) / float(N)
			var uv := PackedVector2Array()
			for pt in [o + a * ta * length + side, o + a * ta * length - side,
					o + a * tb * length - side, o + a * tb * length + side]:
				uv.append(((pt as Vector2) + sz * 0.5) / sz)
			var sh := 0.62 + 0.38 * cf
			ci.draw_polygon(q, PackedColorArray([Color(sh, sh, sh), Color(sh, sh, sh),
				Color(sh, sh, sh), Color(sh, sh, sh)]), uv, tex)
		else:
			# the back of the print: plain paper, lit less the more it faces away from the lamp
			ci.draw_colored_polygon(q, PHOTO_PAPER.darkened(0.03 + 0.16 * (1.0 + cf) * 0.5 + 0.1 * sin(fa)))
	ci.draw_set_transform(Vector2.ZERO, 0.0, Vector2.ONE)


## One quad of the print, from [param p0] to [param p1] across [param side], showing the print
## between [param t0] and [param t1] along its clipped-to-free axis, lit by [param shade].
func _print_quad(ci: CanvasItem, tex: Texture2D, p0: Vector2, p1: Vector2, side: Vector2, a: Vector2,
		t0: float, t1: float, length: float, sz: Vector2, shade: float) -> void:
	var q := PackedVector2Array([p0 + side, p0 - side, p1 - side, p1 + side])
	if tex == null:
		ci.draw_colored_polygon(q, PHOTO_PAPER.darkened(1.0 - shade))
		return
	var o := p0 - a * t0 * length
	var uv := PackedVector2Array()
	for pt in [o + a * t0 * length + side, o + a * t0 * length - side,
			o + a * t1 * length - side, o + a * t1 * length + side]:
		uv.append(((pt as Vector2) + sz * 0.5) / sz)
	var col := Color(shade, shade, shade)
	ci.draw_polygon(q, PackedColorArray([col, col, col, col]), uv, tex)


## The whole print - white border and picture - as one texture, so a curled strip can be cut
## from it. Composed once per picture and size.
func _print_for(im: Dictionary) -> Texture2D:
	var rect: Rect2 = im["rect"]
	var key := String(im["key"])
	var src_tex := _texture_for(key)
	var ck := "%s|%d|%d|%s" % [key, int(rect.size.x), int(rect.size.y), src_tex != null]
	if _prints_rev == Illustrations.revision and _prints.has(ck):
		return _prints[ck]
	var w := maxi(8, int(rect.size.x))
	var h := maxi(8, int(rect.size.y))
	var img := Image.create(w, h, false, Image.FORMAT_RGBA8)
	img.fill(PHOTO_PAPER)
	var b := int(NotebookLayout.PHOTO_BORDER)
	var iw := maxi(1, w - b * 2)
	var ih := maxi(1, h - b * 2)
	if src_tex == null:
		img.fill_rect(Rect2i(b, b, iw, ih), Color(0.80, 0.80, 0.79))
	else:
		var pic := src_tex.get_image().duplicate()
		pic.clear_mipmaps()
		pic.convert(Image.FORMAT_RGBA8)
		var ts := Vector2(pic.get_size())
		var want := float(iw) / float(ih)
		var cut := Rect2i(0, 0, int(ts.x), int(ts.y))
		if ts.x / ts.y > want:
			cut = Rect2i(int((ts.x - ts.y * want) * 0.5), 0, int(ts.y * want), int(ts.y))
		else:
			cut = Rect2i(0, int((ts.y - ts.x / want) * 0.5), int(ts.x), int(ts.x / want))
		pic = pic.get_region(cut)
		pic.resize(iw, ih, Image.INTERPOLATE_BILINEAR)
		img.blit_rect(pic, Rect2i(0, 0, iw, ih), Vector2i(b, b))
	img.generate_mipmaps()
	var tex := ImageTexture.create_from_image(img)
	if _prints_rev != Illustrations.revision:
		_prints = {}                 # a reroll is a new picture under the same key
		_prints_rev = Illustrations.revision
	_prints[ck] = tex
	return tex


## The clips go on after every photo on the page, so a fanned stack sits under its clip. Then
## the clips on the OTHER SIDE of this sheet: a clip grips the sheet's edge, so the page behind
## a clipped photo shows it too - mirrored, and only what shows from behind (see [method
## _clip_path]). That is how a right-hand page shows the clips of the pages still to come.
func _draw_overlay(ci: CanvasItem, pg: Dictionary) -> void:
	for im in pg["images"]:
		var clip: Dictionary = (im as Dictionary).get("clip", {})
		# past upright the photo lies over its clip, which _draw_curled has already put under it
		var peel := smoothstep(0.0, 1.0, float(_peel.get(int((im as Dictionary).get("stack", -1)), 0.0)))
		if not clip.is_empty() and peel <= 0.5:
			_draw_clip(ci, clip["pos"], float(clip["angle"]))
	var p := _layout.pages.find(pg)
	var other := _sheet_partner(p)
	if other < 0 or other >= _layout.pages.size():
		return
	for im in (_layout.pages[other] as Dictionary)["images"]:
		var clip: Dictionary = (im as Dictionary).get("clip", {})
		if not clip.is_empty():
			_draw_clip(ci, clip["pos"], float(clip["angle"]), true, true)


## The page on the other side of page [param p]'s sheet: a right-hand page (odd) is backed by
## the next left-hand one. Page 0 is the inside of the cover and has no sheet. -1 for none.
static func _sheet_partner(p: int) -> int:
	if p < 1:
		return -1
	return p + 1 if p % 2 == 1 else p - 1


## A Gem clip, as its wire's path in its own frame: the outer loop's end at the origin, gripping
## the page edge, its length along +y. FRONT is the whole clip, lying over the sheet. BACK is what
## the other side of the sheet shows - the inner tongue, which is the part of a clip that goes
## behind the paper, and the outer loop's end where it stands past the edge.
static func _clip_path(back: bool) -> Array:
	var l := NotebookLayout.CLIP_LEN
	var a := 14.0
	var g := 5.0
	var xl_i := -a + g
	var xr_i := a - g
	var ri := (xr_i - xl_i) * 0.5
	var y_ti := l * 0.22
	var rb := (xr_i + a) * 0.5
	if back:
		var tongue := PackedVector2Array([Vector2(xl_i, l * 0.62)])
		_clip_arc(tongue, Vector2(0.0, y_ti + ri), ri, PI, TAU)
		tongue.append(Vector2(xr_i, l * 0.66))
		var loop := PackedVector2Array([Vector2(-a, NotebookLayout.CLIP_OVERHANG)])
		_clip_arc(loop, Vector2(0.0, a), a, PI, TAU)
		loop.append(Vector2(a, NotebookLayout.CLIP_OVERHANG))
		return [tongue, loop]
	var pts := PackedVector2Array()
	pts.append(Vector2(xl_i, l * 0.62))
	_clip_arc(pts, Vector2(0.0, y_ti + ri), ri, PI, TAU)
	_clip_arc(pts, Vector2((xr_i - a) * 0.5, l - rb), rb, 0.0, PI)
	_clip_arc(pts, Vector2(0.0, a), a, PI, TAU)
	pts.append(Vector2(a, l * 0.78))
	return [pts]


## Draw a clip at [param at] along [param ang] (0 = straight down the page) - its [param back]
## view, and [param mirror]ed across the page when it belongs to the sheet's other side.
func _draw_clip(ci: CanvasItem, at: Vector2, ang: float, back := false, mirror := false) -> void:
	var xf := Transform2D(ang, at)
	if mirror:
		xf = Transform2D(Vector2(-1, 0), Vector2(0, 1), Vector2(BookLayout.PAGE.x, 0)) * xf
	for path in _clip_path(back):
		ci.draw_set_transform_matrix(Transform2D(0.0, Vector2(3.0, 4.0)) * xf)
		ci.draw_polyline(path, Color(0, 0, 0, 0.22), 4.2, true)
		ci.draw_set_transform_matrix(xf)
		ci.draw_polyline(path, STEEL, 3.4, true)
		ci.draw_set_transform_matrix(Transform2D(0.0, Vector2(-0.7, -0.7)) * xf)
		ci.draw_polyline(path, Color(0.93, 0.94, 0.96), 1.1, true)
	ci.draw_set_transform(Vector2.ZERO, 0.0, Vector2.ONE)


# --- clips in the stacks ------------------------------------------------------------

## THE CLIPS ON PAGES THAT ARE NOT OPEN. Every clip in the chapter is a flat piece of the stack
## it is in, at its own sheet's height: under the stack's top, so the block hides all of it but
## what stands past the edge - which is exactly what shows on a real notebook with photos
## clipped through it. Read sheets are in the left stack and show the face that is up there
## (the back of an odd page's clip); unread ones in the right. The open leaves draw their own.
var _stack_clips: Array = []      # [{node, page, clip}]
var _stack_layout: BookLayout = null
var _clip_tex := {}               # back (bool) -> ViewportTexture


func _place_leaves() -> void:
	super._place_leaves()
	if _layout != _stack_layout:
		_build_stack_clips()
	_place_stack_clips()


## One drawing of the clip, front or back, for the stack pieces to show. Drawn once.
func _clip_texture(back: bool) -> Texture2D:
	if _clip_tex.has(back):
		return _clip_tex[back]
	var vp := SubViewport.new()
	vp.size = Vector2i(72, 320)
	vp.transparent_bg = true
	vp.disable_3d = true
	vp.render_target_update_mode = SubViewport.UPDATE_ONCE
	var cv := ClipCanvas.new()
	cv.medium = self
	cv.back = back
	vp.add_child(cv)
	add_child(vp)
	_clip_tex[back] = vp.get_texture()
	return _clip_tex[back]


func _build_stack_clips() -> void:
	for c in _stack_clips:
		(c["node"] as Node).queue_free()
	_stack_clips = []
	_stack_layout = _layout
	for p in _layout.pages.size():
		for im in (_layout.pages[p] as Dictionary)["images"]:
			var clip: Dictionary = (im as Dictionary).get("clip", {})
			if clip.is_empty() or p < 1:
				continue
			var m := MeshInstance3D.new()
			var q := QuadMesh.new()
			q.size = Vector2.ONE
			m.mesh = q
			m.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			_root3.add_child(m)
			_stack_clips.append({"node": m, "page": p, "clip": clip})


func _place_stack_clips() -> void:
	var d := _stack_depth
	var turning := _turn_t >= 0.0
	for c in _stack_clips:
		var m: MeshInstance3D = c["node"]
		var p := int(c["page"])
		var sheet := (p - 1) / 2            # pages 2k+1 and 2k+2 are sheet k
		# the sheets in view are drawn by their leaves: the left's (spread-1), the right's (spread),
		# and while a leaf turns, the one it uncovers
		var shown := [_spread - 1, _spread]
		if turning:
			shown.append(_turn_to)
		m.visible = not shown.has(sheet)
		if not m.visible:
			continue
		var left := sheet < _spread
		var up_page := sheet * 2 + 2 if left else sheet * 2 + 1   # the face that is up
		var back := up_page != p
		var clip: Dictionary = c["clip"]
		var ang := float(clip["angle"])
		var pos: Vector2 = clip["pos"]
		var ax := Vector2(cos(ang), sin(ang))            # the clip's x in page space
		var ay := Vector2(-sin(ang), cos(ang))           # ...and its length
		var centre := pos + ay * NotebookLayout.CLIP_LEN * 0.5
		if back:
			centre.x = BookLayout.PAGE.x - centre.x
			ax.x = -ax.x
			ay.x = -ay.x
		# page px -> world: one page width is 1 world unit, x from the spine outward
		var k := 1.0 / BookLayout.PAGE.x
		var wx := -(1.0 - centre.x * k) if left else centre.x * k
		var wz := (centre.y / BookLayout.PAGE.y - 0.5) * PAGE_H
		var n := maxi(1, _spread - 1) if left else maxi(1, _layout.spreads() - _spread - 1)
		var depth := (float(_spread - 1 - sheet) if left else float(sheet - _spread)) / float(n)
		var top := d.x if left else d.z
		var thick := d.y if left else d.w
		var wy := top - clampf(depth, 0.0, 1.0) * (thick - 0.002) - 0.001
		var w := 72.0 / 2.0 * k
		var h := 320.0 / 2.0 * k
		# the quad faces +Z; lay it face up, its x along the clip's x and its v down its length
		var bx := Vector3(ax.x, 0.0, ax.y) * w
		var by := -Vector3(ay.x, 0.0, ay.y) * h
		var bz := Vector3(0.0, 1.0, 0.0) * 0.001
		m.transform = Transform3D(Basis(bx, by, bz), Vector3(wx, wy, wz))
		if m.material_override == null or bool(m.get_meta("back", not back)) != back:
			var mat := StandardMaterial3D.new()
			mat.albedo_texture = _clip_texture(back)
			mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA_SCISSOR
			mat.alpha_scissor_threshold = 0.5
			mat.cull_mode = BaseMaterial3D.CULL_DISABLED
			mat.metallic = 0.6
			mat.roughness = 0.35
			m.material_override = mat
			m.set_meta("back", back)
		# a read sheet lies in the left half, which folds with the front cover
		var parent: Node = _pivot if left else _root3
		if m.get_parent() != parent:
			m.reparent(parent, false)


## Draws one clip, front or back, into its texture: the clip's frame fitted to the target.
class ClipCanvas:
	extends Node2D
	var medium = null
	var back := false

	func _draw() -> void:
		var xf := Transform2D(0.0, Vector2(2, 2), 0.0, Vector2(36, 8))
		for path in medium._clip_path(back):
			draw_set_transform_matrix(xf)
			draw_polyline(path, NotebookMedium.STEEL, 3.4, true)
			draw_set_transform_matrix(Transform2D(0.0, Vector2(-0.7, -0.7)) * xf)
			draw_polyline(path, Color(0.93, 0.94, 0.96), 1.1, true)


static func _clip_arc(pts: PackedVector2Array, c: Vector2, r: float, from: float, to: float) -> void:
	for k in 13:
		var t := lerpf(from, to, float(k) / 12.0)
		pts.append(c + Vector2(cos(t), sin(t)) * r)


func debug_line() -> String:
	return super.debug_line() + " peel %s" % [_peel]
