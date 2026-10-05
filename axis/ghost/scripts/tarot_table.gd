extends RefCounted
class_name TarotTable

## TarotTable - what an episode's look may name, and how a look is made safe to draw.
##
## The producer invents a whole deck for every episode (see [TarotPrompts.producer]), but some
## of a look is drawn by ghost rather than painted: the card's frame and lettering, the booklet's
## type, the things standing on the table. Those are REGISTRIES - the producer picks keys from
## them, the table draws them - so a look can be anything an agent imagines and still always be
## something the table knows how to draw. Adding one is an entry here and a branch where it is
## drawn ([TarotCards] for faces and frames, [TarotMedium] for props).

## THE TYPE ON A CARD: the face its name and numeral are set in. One string literal per entry,
## and the description is what the producer reads when it chooses (OFL fonts, `fonts/tarot/`).
const FACES := {
	"roman": {"file": "res://fonts/tarot/Cinzel-Variable.ttf", "weight": 600,
		"about": "inscriptional Roman capitals - classical, carved, the traditional tarot title"},
	"fell": {"file": "res://fonts/tarot/IMFellEnglish-Regular.ttf",
		"about": "seventeenth-century English printing type, a little rough - old almanacs and broadsides"},
	"black": {"file": "res://fonts/tarot/UnifrakturMaguntia-Book.ttf",
		"about": "blackletter - Gothic manuscripts, German woodcuts"},
	"uncial": {"file": "res://fonts/tarot/UncialAntiqua-Regular.ttf",
		"about": "insular uncial - illuminated Celtic manuscripts"},
	"deco": {"file": "res://fonts/tarot/Limelight-Regular.ttf",
		"about": "high-contrast Art Deco display - 1920s cinema, cocktail menus"},
	"sign": {"file": "res://fonts/tarot/Bungee-Regular.ttf",
		"about": "chunky signage capitals - storefronts, arcades, roadside neon"},
	"typed": {"file": "res://fonts/tarot/CourierPrime-Bold.ttf",
		"about": "a typewriter - carbon copies, case files, office memos"},
}

## The booklet's text face, whatever the deck: little white booklets are set in a book face.
const BOOK_FACE := "res://fonts/tarot/EBGaramond-Variable.ttf"
const BOOK_ITALIC := "res://fonts/tarot/EBGaramond-Italic-Variable.ttf"

## THE FRAME around a card's picture - the border the deck prints on every face and back.
const FRAMES := {
	"line": "a plain border with one thin rule inside it",
	"double": "a border with a double rule, thick and thin",
	"corners": "a border with ornamental brackets in the four corners",
	"deco": "a border with stepped Art Deco corners",
	"bleed": "the picture runs nearly to the edge, with only a hairline rule",
}

## THINGS ON THE TABLE. The candles are the table's own (a flame that lights the cards is not
## something a picture can do); everything else is PAINTED - the look names two to four objects,
## each is painted alone, cut out of its background ([TarotCutout]) and stood on the table facing
## the camera, which never moves. A size is how tall the object stands, in meters.
const OBJECT_SIZES := {"small": 0.075, "medium": 0.12, "large": 0.19}
const MAX_OBJECTS := 4
const MAX_CANDLES := 4

## A look's colors when the producer's are missing or not colors at all.
const FALLBACK_PALETTE := ["#1d1a2b", "#c9a227", "#e8dcc0", "#7a2e3a", "#2f5d62"]


## THE LOOK MADE SAFE: every key the table reads present, every color a color, every registry
## key one the table knows. Whatever an agent wrote, the table can draw what this returns.
static func sanitize_look(look: Dictionary) -> Dictionary:
	var out := look.duplicate(true)
	var pal: Array = []
	for c in (look.get("palette", []) if look.get("palette") is Array else []):
		if _is_color(String(c)):
			pal.append(String(c))
	if pal.size() < 3:
		pal = FALLBACK_PALETTE.duplicate()
	out["palette"] = pal
	var frame: Dictionary = look.get("frame", {}) if look.get("frame") is Dictionary else {}
	var f := {"style": String(frame.get("style", "")).strip_edges().to_lower()}
	if not FRAMES.has(f["style"]):
		f["style"] = "line"
	f["stock"] = String(frame.get("stock", "")) if _is_color(String(frame.get("stock", ""))) else "#efe6d2"
	f["ink"] = String(frame.get("ink", "")) if _is_color(String(frame.get("ink", ""))) else String(pal[0])
	f["accent"] = String(frame.get("accent", "")) if _is_color(String(frame.get("accent", ""))) else String(pal[1])
	out["frame"] = f
	var face := String(look.get("title_face", "")).strip_edges().to_lower()
	out["title_face"] = face if FACES.has(face) else "roman"
	# the candles: a count - a look from before the objects named props, and its candles carry over
	var candles := 1
	var nc: Variant = look.get("candles")
	if nc is int or nc is float:
		candles = clampi(int(nc), 0, MAX_CANDLES)
	elif look.get("props") is Array:
		var old: Array = look["props"]
		candles = 3 if old.has("candles") else (1 if old.has("candle") else 0)
	out["candles"] = candles
	out.erase("props")
	# the objects: what each is, and how big
	var objects: Array = []
	for o in (look.get("objects", []) if look.get("objects") is Array else []):
		var what := ""
		var size := "medium"
		if o is Dictionary:
			what = str((o as Dictionary).get("what", "")).strip_edges()
			size = str((o as Dictionary).get("size", "medium")).strip_edges().to_lower()
		elif o is String:
			what = (o as String).strip_edges()
		if what.is_empty():
			continue
		objects.append({"what": what.substr(0, 240), "size": size if OBJECT_SIZES.has(size) else "medium"})
	out["objects"] = objects.slice(0, MAX_OBJECTS)
	var light: Dictionary = look.get("light", {}) if look.get("light") is Dictionary else {}
	out["light"] = {"kind": String(light.get("kind", "candlelight")),
		"color": String(light.get("color", "")) if _is_color(String(light.get("color", ""))) else "#ffb36b",
		"warmth": "cool" if String(light.get("warmth", "")).to_lower().begins_with("cool") else "warm"}
	for k in ["deck_name", "deck_style", "card_back", "surface", "setting"]:
		out[k] = String(look.get(k, ""))
	out["foil"] = clampf(float(look.get("foil", 0.6)) if (look.get("foil") is float or look.get("foil") is int) else 0.6, 0.0, 1.0)
	return out


## A card's footprint on the table at [param pos], turned by [param yaw]: a rectangle in x by z.
static func footprint(pos: Vector3, yaw: float, card: Vector2) -> Rect2:
	var hx := absf(cos(yaw)) * card.x * 0.5 + absf(sin(yaw)) * card.y * 0.5
	var hz := absf(sin(yaw)) * card.x * 0.5 + absf(cos(yaw)) * card.y * 0.5
	return Rect2(pos.x - hx, pos.z - hz, hx * 2.0, hz * 2.0)


## HOW FAR A SPREAD MUST MOVE to lie clear of [param keep_out] (a rectangle on the table, x by z -
## the deck): the smaller of stepping back (-z, away from the reader) and stepping aside, away
## from it. An offset (x, z), zero when nothing touches. [param slots] are `{pos, yaw}`.
static func clear_of(slots: Array, card: Vector2, keep_out: Rect2) -> Vector2:
	var back := 0.0
	var aside := 0.0
	var right := keep_out.get_center().x > 0.0         # step aside AWAY from it
	# a card that a step would carry THROUGH it rules that step out: one nearer the reader than it
	# (stepping back), one past its far side (stepping aside)
	var front := false
	var beyond := false
	for sl in slots:
		var r := footprint((sl as Dictionary)["pos"], float((sl as Dictionary)["yaw"]), card)
		var x_over := r.position.x < keep_out.end.x and r.end.x > keep_out.position.x
		var z_over := r.position.y < keep_out.end.y and r.end.y > keep_out.position.y
		if x_over and z_over:
			back = maxf(back, r.end.y - keep_out.position.y)
			aside = maxf(aside, r.end.x - keep_out.position.x if right else keep_out.end.x - r.position.x)
		elif x_over and r.position.y >= keep_out.end.y:
			front = true
		elif z_over and (r.position.x >= keep_out.end.x if right else r.end.x <= keep_out.position.x):
			beyond = true
	if back <= 0.0:
		return Vector2.ZERO
	var go_aside := (aside < back or front) and not (beyond and not front)
	return Vector2(-aside if right else aside, 0.0) if go_aside else Vector2(0.0, -back)


## What the viewer sees on the table besides the cloth, the deck and the cards, in words.
static func on_the_table(look: Dictionary) -> String:
	var things := PackedStringArray()
	var n := int(look.get("candles", 0))
	if n > 0:
		things.append("%s lit candle%s" % [["", "one", "two", "three", "four"][mini(n, 4)], "" if n == 1 else "s"])
	for o in look.get("objects", []):
		things.append(String((o as Dictionary).get("what", "")))
	return "; ".join(things) if not things.is_empty() else "nothing else"


static func _is_color(s: String) -> bool:
	return Manuscript._rx("^#[0-9a-fA-F]{6}$").search(s.strip_edges()) != null


static func color(s: String, fallback := Color(0.5, 0.5, 0.5)) -> Color:
	return Color.html(s.strip_edges()) if _is_color(s) else fallback


static var _fonts := {}

## The [Font] for a face key (or a font path), loaded once. A variable face is set at the
## weight the registry names.
static func font(key: String) -> Font:
	if _fonts.has(key):
		return _fonts[key]
	var entry: Dictionary = FACES.get(key, {})
	var path := String(entry.get("file", key))
	var f: Font = null
	if ResourceLoader.exists(path) or FileAccess.file_exists(path):
		var ff := FontFile.new()
		if ff.load_dynamic_font(path) == OK:
			f = ff
			if entry.has("weight"):
				var fv := FontVariation.new()
				fv.base_font = ff
				fv.variation_opentype = {"wght": int(entry["weight"])}
				f = fv
	if f == null:
		f = ThemeDB.fallback_font
	_fonts[key] = f
	return f
