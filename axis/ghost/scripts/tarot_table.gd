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

## THE CANDLES: the most flames on a table. Each lights the table, and the one nearest the middle
## throws its shadows (see [method TarotMedium._light_the_table]).
const MAX_CANDLES := 4

## THE THINGS ON THE TABLE are modeled, not painted (see [Props]): the set dresser describes each
## in parts and materials and names where it stands; the table builds and places it. A painted
## cut-out could neither take the candlelight nor cast a shadow. At most this many.
const MAX_THINGS := 12

## WHERE A THING CAN STAND, as the set dresser names it: stretches of the cloth the cards leave
## free. `aim` is a zone's middle on the table (x across, z toward the reader, meters); the table
## finds the spot itself ([method TarotMedium._place_things]). "by the deck" is aimed per episode.
const ZONES := {
	"back left": {"aim": Vector2(-0.3, -0.27), "about": "the far left of the cloth, behind where the cards are laid"},
	"back": {"aim": Vector2(0.0, -0.3), "about": "the far middle of the cloth, behind where the cards are laid"},
	"back right": {"aim": Vector2(0.3, -0.27), "about": "the far right of the cloth, behind where the cards are laid"},
	"left": {"aim": Vector2(-0.31, -0.1), "about": "the left side, beside where the cards are laid"},
	"right": {"aim": Vector2(0.31, -0.1), "about": "the right side, beside where the cards are laid"},
	"by the deck": {"aim": Vector2.ZERO, "about": "beside the deck, near the front of the cloth on the side the reader keeps it"},
}

## A look's colors when the producer's are missing or not colors at all.
const FALLBACK_PALETTE := ["#1d1a2b", "#c9a227", "#e8dcc0", "#7a2e3a", "#2f5d62"]

## The least contrast (WCAG ratio) a card's ink may have against its stock: the name is printed
## in it. 3 is WCAG's floor for large text.
const INK_CONTRAST := 3.0


## THE LENS: the camera's vertical field of view, degrees, before an episode's own small turn of it.
const VFOV := 42.0


## THE EPISODE'S LAYOUT, drawn from [param rng] in a fixed order: the camera (`camera`, `fov`,
## `pitch`), the lamp (`lamp`, `lamp_energy`), where the deck is kept (`deck`) and shuffled
## (`mid`). The table draws it as it is built; [method layout_of] lets a prompt know it first.
static func sample_layout(rng: RandomNumberGenerator) -> Dictionary:
	# shallow enough that the room always shows past the table's far edge (at 48 it was a sliver)
	var pitch := rng.randf_range(34.0, 42.0)
	var dist := rng.randf_range(0.5, 0.56)
	var yaw := deg_to_rad(rng.randf_range(-3.5, 3.5))
	var target := Vector3(rng.randf_range(-0.01, 0.01), 0.0, -0.05)
	var p := deg_to_rad(pitch)
	var eye := target + Vector3(sin(yaw) * cos(p), sin(p), cos(yaw) * cos(p)) * dist
	var out := {"pitch": pitch, "camera": Transform3D(Basis.looking_at(target - eye, Vector3.UP), eye),
		"fov": VFOV + rng.randf_range(-1.5, 1.5)}
	# a draw kept where the far blur's distance was sampled, so every draw after it - the lamp,
	# the deck, the spread - lands where it did for episodes already made
	rng.randf_range(0.45, 0.75)
	var side := -1.0 if rng.randf() < 0.6 else 1.0
	# low enough, and far enough to one side, that what stands on the table throws a shadow you see
	out["lamp"] = Vector3(side * rng.randf_range(0.4, 0.6), rng.randf_range(0.75, 0.95), rng.randf_range(-0.35, -0.1))
	out["lamp_energy"] = rng.randf_range(1.3, 1.9)
	out["deck"] = Vector3(rng.randf_range(0.17, 0.23) * (1.0 if rng.randf() < 0.7 else -1.0), 0.0, rng.randf_range(0.04, 0.07))
	out["mid"] = Vector3(rng.randf_range(-0.015, 0.015), 0.0, rng.randf_range(-0.05, -0.02))
	return out


## Episode [param seed]'s layout (see [method sample_layout]).
static func layout_of(seed: int) -> Dictionary:
	var rng := RandomNumberGenerator.new()
	rng.seed = hash([seed, "tarot-table"])
	return sample_layout(rng)


## WHERE [param at] LANDS IN THE PICTURE of a camera at [param cam] with [param fov] (16:9),
## 0..1 across and down; null behind the camera.
static func project(cam: Transform3D, fov: float, at: Vector3) -> Variant:
	var l: Vector3 = cam.affine_inverse() * at
	if l.z > -0.001:
		return null
	var k := tan(deg_to_rad(fov * 0.5))
	return Vector2(0.5 + l.x / (-l.z * k * (16.0 / 9.0)) * 0.5, 0.5 - l.y / (-l.z * k) * 0.5)


## HOW TALL A THING CAN STAND AND BE SEEN WHOLE, centimeters, in each of [constant ZONES] for
## episode [param seed]: the frame's top edge passes lower over the far cloth than the near, so
## the back holds only short things. For the set dresser's prompt.
static func headroom(seed: int) -> Dictionary:
	var lay := layout_of(seed)
	var out := {}
	for z in ZONES:
		var aim := zone_aim(String(z), lay["deck"])
		var h := 0.0
		while h < 0.6:
			var top: Variant = project(lay["camera"], float(lay["fov"]), Vector3(aim.x, h + 0.01, aim.y))
			if top == null or (top as Vector2).y < 0.03:
				break
			h += 0.005
		out[z] = roundi(h * 100.0)
	return out


## The middle of zone [param zone] on the table (x, z), the deck kept at [param deck].
static func zone_aim(zone: String, deck: Vector3) -> Vector2:
	if zone == "by the deck":
		return Vector2(deck.x + signf(deck.x) * 0.11, deck.z - 0.07)
	return (ZONES.get(zone, ZONES["back"]) as Dictionary)["aim"]


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
	# the stock can be anything from black to white (TarotPrompts.dice): the name must read on it
	var ink := color(f["ink"])
	var readable := legible_ink(ink, color(f["stock"]))
	if readable != ink:
		f["ink"] = "#" + readable.to_html(false)
	out["frame"] = f
	var face := String(look.get("title_face", "")).strip_edges().to_lower()
	out["title_face"] = face if FACES.has(face) else "roman"
	# the candles: a count - an older look named props, and its candles carry over
	var candles := 1
	var nc: Variant = look.get("candles")
	if nc is int or nc is float:
		candles = clampi(int(nc), 0, MAX_CANDLES)
	elif look.get("props") is Array:
		var old: Array = look["props"]
		candles = 3 if old.has("candles") else (1 if old.has("candle") else 0)
	out["candles"] = candles
	out.erase("props")
	out.erase("objects")
	var light: Dictionary = look.get("light", {}) if look.get("light") is Dictionary else {}
	out["light"] = {"kind": String(light.get("kind", "candlelight")),
		"color": String(light.get("color", "")) if _is_color(String(light.get("color", ""))) else "#ffb36b",
		"warmth": "cool" if String(light.get("warmth", "")).to_lower().begins_with("cool") else "warm"}
	for k in ["deck_name", "deck_style", "card_back", "surface", "setting"]:
		out[k] = String(look.get(k, ""))
	out["foil"] = clampf(float(look.get("foil", 0.6)) if (look.get("foil") is float or look.get("foil") is int) else 0.6, 0.0, 1.0)
	return out


## THE TABLE MADE SAFE: the set dresser's reply as [Props] can build it, every thing standing in a
## zone the table knows ("back" when it named none), things sharing a `group` kept together, and
## no more lit wicks than [constant MAX_CANDLES] - the first written keep their flames, the rest
## stand unlit.
static func sanitize_table(spec: Dictionary, look: Dictionary) -> Dictionary:
	var out := Props.sanitize(spec, look.get("palette", FALLBACK_PALETTE) if look.get("palette") is Array else FALLBACK_PALETTE)
	var things: Array = []
	var flames := 0
	for t in (out["things"] as Array).slice(0, MAX_THINGS):
		var thing: Dictionary = t
		var place := String(thing.get("place", "")).strip_edges().to_lower()
		thing["place"] = place if ZONES.has(place) else "back"
		var g: Variant = thing.get("group", "")
		thing["group"] = str(int(g)) if (g is float or g is int) else String(g if g is String else "").strip_edges()
		thing["turn"] = Props._num(thing.get("turn"), 0.0, -180.0, 180.0)
		for p in thing["parts"]:
			var part: Dictionary = p
			if not bool(part.get("wick", false)):
				continue
			var copies: Dictionary = part.get("copies", {})
			var n := int(copies.get("count", 1)) if not copies.is_empty() else 1
			if flames + n > MAX_CANDLES:
				part["wick"] = false
			else:
				flames += n
		things.append(thing)
	out["things"] = things
	return out


## THE TABLE BEFORE THE SET DRESSER HAS SET ONE: the look's candles alone, plain pillars of wax a
## shade of its palette each, flanking the cloth.
static func default_table(look: Dictionary, seed: int) -> Dictionary:
	var rng := RandomNumberGenerator.new()
	rng.seed = hash([seed, "tarot-default-candles"])
	var pal: Array = look.get("palette", FALLBACK_PALETTE) if look.get("palette") is Array else FALLBACK_PALETTE
	var things: Array = []
	var mats := {}
	for i in clampi(int(look.get("candles", 1)), 0, MAX_CANDLES):
		var r := rng.randf_range(1.7, 2.6)
		var h := rng.randf_range(7.0, 12.0)
		var wax := Color(0.8, 0.76, 0.68).lerp(color(String(pal[rng.randi_range(0, pal.size() - 1)])), rng.randf_range(0.1, 0.5))
		mats["wax %d" % i] = {"kind": "wax", "color": "#" + wax.to_html(false)}
		things.append({"name": "a candle", "place": "back left" if i % 2 == 0 else "back right",
			"parts": [{"shape": "lathe", "profile": [[0, 0], [r, 0], [r * 0.98, h], [0, h]], "material": "wax %d" % i,
				"wick": true, "drips": rng.randf_range(0.0, 0.4)}]})
	return sanitize_table({"things": things, "materials": mats}, look)


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


## What the reader is told stands on the table besides the cloth, the deck and the cards: the
## look's candles and the reader's own things (the set dresser's, which no passage depends on).
static func on_the_table(look: Dictionary) -> String:
	var n := int(look.get("candles", 0))
	if n <= 0:
		return "the reader's own things"
	return "%s lit candle%s and the reader's own things" % [["", "one", "two", "three", "four"][mini(n, 4)], "" if n == 1 else "s"]


static func _is_color(s: String) -> bool:
	return Manuscript._rx("^#[0-9a-fA-F]{6}$").search(s.strip_edges()) != null


static func color(s: String, fallback := Color(0.5, 0.5, 0.5)) -> Color:
	return Color.html(s.strip_edges()) if _is_color(s) else fallback


## WCAG's contrast ratio between two colors: 1 is the same, 21 is black on white.
static func contrast(a: Color, b: Color) -> float:
	var la := _luminance(a)
	var lb := _luminance(b)
	return (maxf(la, lb) + 0.05) / (minf(la, lb) + 0.05)


static func _luminance(c: Color) -> float:
	var l := c.srgb_to_linear()
	return 0.2126 * l.r + 0.7152 * l.g + 0.0722 * l.b


## [param ink] as it is when it reads on [param stock] (see [constant INK_CONTRAST]); otherwise
## moved toward white or black, whichever stands further from the stock, until it does.
static func legible_ink(ink: Color, stock: Color) -> Color:
	if contrast(ink, stock) >= INK_CONTRAST:
		return ink
	var to := Color.WHITE if contrast(Color.WHITE, stock) > contrast(Color.BLACK, stock) else Color.BLACK
	for i in range(1, 11):
		var c := ink.lerp(to, float(i) / 10.0)
		if contrast(c, stock) >= INK_CONTRAST:
			return c
	return to


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
