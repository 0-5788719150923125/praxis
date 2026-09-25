extends BookLayout
class_name NotebookLayout

## NotebookLayout - a chapter written by hand into a ruled notebook, for [NotebookMedium].
##
## The same typesetter as [BookLayout] - the same blocks, the same word list the highlight
## follows - with the choices a hand makes instead of a compositor's:
##
##   THE WRITING SITS ON THE RULES. Every line box is a whole number of rules and a baseline
##   lies on the rule under it, so the grid is the page's, not the font's. College rule.
##   RAGGED, NOT JUSTIFIED, and every word strays a little from where a typesetter would put
##   it - a line drifts off level along its length, a word sits a hair high or low. Hashed off
##   the session seed, so an export writes the same page as the live reading.
##   A TIMESTAMP GOES IN THE MARGIN. A paragraph that opens on a time ("09:40", "[14:05]",
##   "7:15 pm") has it set to the left of the red line, where a researcher logs one.
##   EMPHASIS IS UNDERLINED, because a hand has no italic - [NotebookMedium] draws the line.
##   PHOTOS ARE CLIPPED ON, NOT PRINTED IN. They lie over the writing, held by a paper clip at
##   the page's outer edge (or its top), at an angle; the text runs on underneath, heard and
##   not seen. Pictures that follow one another with no writing between share one clip, fanned.
##   A SKETCH IS DRAWN INTO THE WRITING: rows are left for it, and it is ink on the paper.

## College rule on this sheet: 7.1 mm on a 9.75 in page.
const RULE := 46.0
## The heavier top rule. The strip above it is the page's header; the chapter title goes there.
const HEADER := 196.0
## The red margin line, on the left of every page as a composition book has it.
const MARGIN_X := 172.0
## How close writing comes to the outer edge, and to the spine.
const PAD_OUTER := 64.0
const PAD_SPINE := 100.0
## A SKETCH IS CENTRED IN THE COLUMN and drawn as large as its shape allows: [constant
## SKETCH_WIDTH] of the column's width, with as many rows as that takes at its own aspect (the
## drawing's ink, cropped - see [method BookMedium.ink_texture]), between [constant SKETCH_ROWS]
## and [constant SKETCH_ROWS_MAX]. It used to sit at a random place along a fixed 8 rows and
## came out small, and always somewhere different to no purpose.
const SKETCH_ROWS := 6
const SKETCH_ROWS_MAX := 13
const SKETCH_WIDTH := 0.9
## A photo's white border, in page pixels.
const PHOTO_BORDER := 18.0
## A paper clip's length, and how far its outer loop stands off the page edge it grips - a clip
## is pushed on from the edge, so its end is always past it. Drawn in the page texture's margin
## ([method NotebookMedium._page_pad]), which must be wider than this.
const CLIP_LEN := 150.0
const CLIP_OVERHANG := 6.0

## THE HANDS, all OFL, bundled under fonts/hands/ (licences beside them). One is chosen per
## session. `size` is the body size that fills a rule in that hand - they differ a lot (Caveat's
## letters are small for their em). `bold` is a file, or empty to thicken the regular one.
const HANDS := {
	"caveat": {"regular": "res://fonts/hands/Caveat-Variable.ttf", "bold": "", "size": 40},
	"kalam": {"regular": "res://fonts/hands/Kalam-Regular.ttf",
		"bold": "res://fonts/hands/Kalam-Bold.ttf", "size": 30},
	"patrick": {"regular": "res://fonts/hands/PatrickHand-Regular.ttf", "bold": "", "size": 33},
}

## THE PENS a voice may write in, by name - what `ink:` on a voice in the frontmatter takes
## (a hex colour works too). Ink colours, not pure ones: "blue" is a ballpoint's, not #0000FF.
## Black is the default for every voice that names none.
const INKS := {
	"black": Color(0.08, 0.08, 0.10),
	"blue": Color(0.10, 0.18, 0.55),
	"red": Color(0.68, 0.10, 0.12),
	"green": Color(0.08, 0.38, 0.20),
	"purple": Color(0.36, 0.14, 0.50),
	"brown": Color(0.36, 0.20, 0.10),
}


## A voice's `ink:` as a colour: a name from [constant INKS], a hex colour, or black.
static func ink_color(v: String) -> Color:
	var k := v.strip_edges().to_lower()
	if INKS.has(k):
		return INKS[k]
	if Color.html_is_valid(k):
		return Color.html(k)
	return INKS["black"]


## Which of [constant HANDS] writes this chapter.
var hand := "kalam"
## Seeds every wobble, angle and placement.
var hand_seed := 0

var _time_re: RegEx
var _ampm_re: RegEx
var _last_photo := {}         # the photo the next one would fan onto
var _last_photo_words := -1   # ...and how many words had been written when it was clipped
var _last_photo_p := -1
var _stacks := 0              # photos under one clip share a stack number, which lifts as one

static var _hand_faces := {}


func _init() -> void:
	body_fs = int(HANDS["kalam"]["size"])


## A hand's face at an emphasis level. Italic is the regular face (it is underlined instead);
## bold is the hand's bold file, or its variable weight, or the regular thickened.
static func hand_face(key: String, level: int) -> Font:
	if not HANDS.has(key):
		key = "kalam"
	var bold := (level & 2) != 0
	var ck := "%s|%s" % [key, bold]
	if _hand_faces.has(ck):
		return _hand_faces[ck]
	var h: Dictionary = HANDS[key]
	var base: Font = load(String(h["regular"]))
	var f: Font = base
	if bold:
		if not String(h["bold"]).is_empty():
			f = load(String(h["bold"]))
		else:
			var v := FontVariation.new()
			v.base_font = base
			if String(h["regular"]).contains("Variable"):
				var ts := TextServerManager.get_primary_interface()
				v.variation_opentype = {ts.name_to_tag("wght"): 700}
			else:
				v.variation_embolden = 0.6
			f = v
	_hand_faces[ck] = f
	return f


func face(level: int) -> Font:
	return hand_face(hand, level)


func _h(n: int) -> float:
	return float(hash([hand_seed, n]) & 0xFFFF) / 65535.0


# --- the grid ---------------------------------------------------------------

func _top() -> float:
	return HEADER


## The last rule a line can sit on.
func _bottom() -> float:
	return HEADER + RULE * floor((PAGE.y - 80.0 - HEADER) / RULE)


## Whole rules, as many as the size needs. Measured against the size itself, not a leading
## factor: a hand's body size is already chosen to fill one rule (Caveat's 40 is small letters
## on a big em), and 1.25x of it double-spaced every line in that hand.
func _lh(fs: int) -> float:
	return RULE * maxf(1.0, ceil(float(fs) * 0.95 / RULE))


## Whole rules only: a gap of half a line would take the writing off the grid for the rest of
## the page.
func _space(lines: float) -> void:
	if _y > _top() + 1.0:
		_y += RULE * maxf(1.0, round(lines))


func column(p: int) -> Vector2:
	if _side(p) == 0:
		return Vector2(MARGIN_X + 16.0, PAGE.x - PAD_SPINE)
	return Vector2(MARGIN_X + 16.0, PAGE.x - PAD_OUTER)


func _justify() -> bool:
	return false


## On the rule, with the descenders below it, as handwriting sits.
func _baseline(y: float, lh: float) -> float:
	return y + lh - 8.0


## Words are set exactly on the line: how the writing wanders off it is drawn letter by letter
## by [method NotebookMedium._glyph_xform], as a smooth drift. Word-by-word jitter - each word a
## little high or low - is not how a hand writes, and was reported as looking strange.
func _ink_offset(_i: int, _line_y: float, _dx: float) -> Vector2:
	return Vector2.ZERO


# --- blocks -----------------------------------------------------------------

## The title in the header strip, larger, underlined.
func _title_block() -> void:
	var c := column(_p)
	var fs := int(body_fs * 1.45)
	var f := face(2)
	while fs > body_fs and f.get_string_size(title, HORIZONTAL_ALIGNMENT_LEFT, -1, fs).x > c.y - c.x:
		fs -= 2
	# WORDS in the header strip, not a label: the title is read aloud first.
	var lh := _lh(fs)
	_y = HEADER - 22.0 - lh + 8.0
	_set_line_words(title, fs, 2, false, {"underline": true})
	_y = _top()


## An entry heading - a date, a subject - written bold and underlined at the margin, with a
## blank rule above it unless it opens the page.
func _heading(text: String, level: int) -> void:
	var fs := int(body_fs * (1.2 if level <= 1 else 1.08))
	var lh := _lh(fs)
	_space(1.0)
	_keep_with_next(lh)
	_set_line_words(text, fs, 2, false, {"underline": level <= 2, "heading": true})


func _rule() -> void:
	if _y + RULE * 3.0 > _bottom():
		_open_page()
	_space(1.0)
	var c := column(_p)
	(pages[_p]["labels"] as Array).append({"text": "~", "pos": Vector2(c.x, _baseline(_y, RULE)),
		"fs": body_fs, "emph": 0, "align_w": c.y - c.x, "tone": 0.6})
	_y += RULE


# --- the margin ---------------------------------------------------------------

func _margin_lead(toks: Array) -> int:
	if toks.is_empty():
		return 0
	if _time_re == null:
		_time_re = RegEx.new()
		_time_re.compile("^[\\[(]?\\d{1,2}[:.h]\\d{2}(?::\\d{2})?(?:[ap]\\.?m\\.?)?[\\])]?[,:;.]?$")
		_ampm_re = RegEx.new()
		_ampm_re.compile("^(?:[ap]\\.?m\\.?|AM|PM)[\\])]?[,:;.]?$")
	if _time_re.search(String(toks[0]["text"]).to_lower()) == null:
		return 0
	if toks.size() > 2 and _ampm_re.search(String(toks[1]["text"])) != null:
		return 2
	return 1 if toks.size() > 1 else 0


## The time, a little smaller, right-aligned against the red line.
func _set_margin(toks: Array, _widths: Array, n: int, base_y: float, fs: int, lh: float) -> void:
	var mfs := int(float(fs) * 0.82)
	var sp := face(0).get_string_size(" ", HORIZONTAL_ALIGNMENT_LEFT, -1, mfs).x
	var ws: Array = []
	var total := 0.0
	for k in n:
		var w := face(int(toks[k]["emph"])).get_string_size(String(toks[k]["text"]),
			HORIZONTAL_ALIGNMENT_LEFT, -1, mfs).x
		ws.append(w)
		total += w + (sp if k > 0 else 0.0)
	var x := maxf(22.0, MARGIN_X - 14.0 - total)
	for k in n:
		var t: Dictionary = toks[k]
		var w := float(ws[k])
		words.append({"page": _p, "text": String(t["text"]), "norm": norm(String(t["text"])),
			"emph": int(t["emph"]), "fs": mfs, "base": Vector2(x, base_y), "speaker": _speaker,
			"rect": Rect2(x, base_y - lh * 0.64, w, lh * 0.86)})
		(pages[_p]["words"] as Array).append(words.size() - 1)
		x += w + sp


# --- pictures -----------------------------------------------------------------

## WHICH WORDS EACH CLIPPED PHOTO HIDES, as `cover: [first, last]` on every photo of its stack
## (or [-1, -1]), so the medium can lift it while they are read. A word is hidden when its
## centre or either end lies inside the photo's turned rectangle.
func _finish() -> void:
	var lo := {}
	var hi := {}
	for pg in pages:
		for im in pg["images"]:
			if not bool(im.get("photo", false)) or int(im.get("stack", -1)) < 0:
				continue
			var st := int(im["stack"])
			var r: Rect2 = im["rect"]
			var c := r.get_center()
			var ang := float(im.get("angle", 0.0))
			for wi in pg["words"]:
				var wr: Rect2 = words[int(wi)]["rect"]
				for p in [wr.get_center(), Vector2(wr.position.x + 4.0, wr.get_center().y),
						Vector2(wr.end.x - 4.0, wr.get_center().y)]:
					var q := ((p as Vector2) - c).rotated(-ang)
					if absf(q.x) < r.size.x * 0.5 and absf(q.y) < r.size.y * 0.5:
						lo[st] = mini(int(lo.get(st, wi)), int(wi))
						hi[st] = maxi(int(hi.get(st, wi)), int(wi))
						break
	for pg in pages:
		for im in pg["images"]:
			var st := int(im.get("stack", -1))
			im["cover"] = [int(lo.get(st, -1)), int(hi.get(st, -1))]


## An inline picture: a photo clipped over the writing, where the marker falls.
func _float(b: Dictionary) -> void:
	_clip_photo(b)


## A full-page picture: a large photo clipped to the top of a page of its own. It does not go
## over a page of writing, because it would hide all of it.
func _image_page(b: Dictionary) -> void:
	var pg := _new_page("image")
	pg["folio"] = 0
	var r := hash([hand_seed, String(b.get("key", "")), "full"])
	var w := PAGE.x * 0.8
	var sz := _photo_size(b, w)
	var ang := deg_to_rad(((r & 0xFF) / 255.0 - 0.5) * 6.0)
	var centre := Vector2(PAGE.x * 0.5 + (((r >> 8) & 0xFF) / 255.0 - 0.5) * 60.0, 70.0 + sz.y * 0.5)
	pg["images"].append(_photo(b, centre, sz, ang, true,
		{"pos": Vector2(centre.x + (((r >> 16) & 0xFF) / 255.0 - 0.5) * w * 0.3, -CLIP_OVERHANG),
		"angle": (((r >> 24) & 0xFF) / 255.0 - 0.5) * 0.14}))
	_last_photo = {}


## Rows left in the writing for a drawing, the drawing somewhere along them.
func _sketch(b: Dictionary) -> void:
	_space(1.0)
	var c := column(_p)
	var w := (c.y - c.x) * SKETCH_WIDTH
	var aspect := 1.5
	if _images_at.is_valid():
		var got: Vector2 = _images_at.call(String(b.get("key", "")))
		if got.x > 0.0 and got.y > 0.0:
			aspect = got.x / got.y
	var rows := clampi(int(ceil((w / aspect + 12.0) / RULE)), SKETCH_ROWS, SKETCH_ROWS_MAX)
	if _y + RULE * rows > _bottom():
		_open_page()
		c = column(_p)
	(pages[_p]["images"] as Array).append({"rect": Rect2(c.x + (c.y - c.x - w) * 0.5, _y + 6.0, w,
		RULE * rows - 12.0), "key": String(b.get("key", "")), "prompt": String(b.get("prompt", "")),
		"full": false, "sketch": true})
	_y += RULE * rows


## The photo's outer size (border included) at width [param w], from the picture's own aspect.
func _photo_size(b: Dictionary, w: float) -> Vector2:
	var aspect := DEFAULT_ASPECT
	if _images_at.is_valid():
		var got: Vector2 = _images_at.call(String(b.get("key", "")))
		if got.x > 0.0 and got.y > 0.0:
			aspect = got.x / got.y
	var inner := w - PHOTO_BORDER * 2.0
	return Vector2(w, inner / aspect + PHOTO_BORDER * 2.0)


func _photo(b: Dictionary, centre: Vector2, sz: Vector2, ang: float, full: bool, clip: Dictionary) -> Dictionary:
	return {"rect": Rect2(centre - sz * 0.5, sz), "angle": ang, "key": String(b.get("key", "")),
		"prompt": String(b.get("prompt", "")), "full": full, "photo": true, "clip": clip}


## Clip a photo on beside the writing at the current line. At the page's OUTER edge, with the
## clip gripping that edge, or - near the top of a page, now and then - hanging from the top
## edge. A photo following another with nothing written between fans out under the same clip.
func _clip_photo(b: Dictionary) -> void:
	if _p < 0:
		_open_page()
	var r := hash([hand_seed, String(b.get("key", "")), int(b.get("ordinal", 0))])
	var u := func(shift: int) -> float:
		return float((r >> shift) & 0xFF) / 255.0
	var turn := 1.0 if (r & 1) == 0 else -1.0
	var ang := turn * deg_to_rad(lerpf(2.0, 8.0, u.call(8)))
	var images: Array = pages[_p]["images"]
	if not _last_photo.is_empty() and _last_photo_p == _p and _last_photo_words == words.size():
		# FANNED: the same clip, the photo underneath slid a little and turned the other way
		var prev: Dictionary = _last_photo
		var sz := _photo_size(b, (prev["rect"] as Rect2).size.x * lerpf(0.9, 1.05, u.call(16)))
		var centre := (prev["rect"] as Rect2).get_center() + Vector2((u.call(24) - 0.5) * 70.0,
			lerpf(24.0, 60.0, u.call(4)))
		var ph := _photo(b, centre, sz, -signf(float(prev["angle"])) * absf(ang), false, {})
		ph["stack"] = prev["stack"]
		ph["hinge"] = prev["hinge"]
		# under the one already there: the clip was put on over the stack
		images.insert(images.find(prev), ph)
		return
	var w := PAGE.x * lerpf(0.44, 0.56, u.call(16))
	var sz := _photo_size(b, w)
	var side := _side(_p)
	var near_top := _y < _top() + RULE * 3.0
	var centre: Vector2
	var clip := {}
	if near_top and u.call(24) < 0.5:
		centre = Vector2(lerpf(sz.x * 0.5 + 60.0, PAGE.x - sz.x * 0.5 - 60.0, u.call(4)), 34.0 + sz.y * 0.5)
		clip = {"pos": Vector2(centre.x + (u.call(12) - 0.5) * sz.x * 0.4, -CLIP_OVERHANG),
			"angle": (u.call(20) - 0.5) * 0.14}
	else:
		var edge := 34.0 + sz.x * 0.5
		var cx := PAGE.x - edge if side == 1 else edge
		var cy := clampf(_y + sz.y * 0.3, sz.y * 0.5 + 40.0, PAGE.y - sz.y * 0.5 - 40.0)
		centre = Vector2(cx, cy)
		var gy := cy - sz.y * lerpf(0.1, 0.35, u.call(12))
		# the clip's long axis points in from the edge it grips
		clip = {"pos": Vector2(PAGE.x + CLIP_OVERHANG if side == 1 else -CLIP_OVERHANG, gy),
			"angle": (PI * 0.5 if side == 1 else -PI * 0.5) + (u.call(20) - 0.5) * 0.14}
	var ph := _photo(b, centre, sz, ang, false, clip)
	# THE HINGE is the edge the clip holds; the photo lifts from the other one
	ph["hinge"] = "top" if clip["pos"].y < 0.0 else ("right" if side == 1 else "left")
	ph["stack"] = _stacks
	_stacks += 1
	images.append(ph)
	_last_photo = ph
	_last_photo_p = _p
	_last_photo_words = words.size()
