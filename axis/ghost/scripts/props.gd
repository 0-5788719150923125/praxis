extends RefCounted
class_name Props

## Props - things BUILT FROM A DESCRIPTION. A thing is a few PARTS, each one SHAPE with real sizes
## and one MATERIAL whose surface is procedural, with an ORNAMENT worked into it if it has one - or a
## GROUP of parts, placed and repeated as one (a candelabra's arm, cup and taper, copied round). An
## agent writes the description - centimeters, the thing's base on y = 0, its front toward +z - and
## [method sanitize] makes whatever it wrote buildable; [method build] makes the meshes, in meters.
##
## SHAPES, MATERIALS, PLAYS and ORNAMENTS are registries, and their words are what an agent reads
## when it chooses ([method describe]), so the prompt and the builder cannot disagree about what
## exists.
## Geometry is built here rather than from CSG nodes because a part's lobes, twist and wax drips
## displace it as it is built, which a spun polygon cannot do.

## The most parts a thing has (every part in every group counted), points a profile or path has,
## copies a part makes, groups deep a part may sit, parts a thing makes in all (every copy of every
## part), and wicks one wax part has.
const MAX_PARTS := 16
const MAX_POINTS := 40
const MAX_COPIES := 24
const MAX_DEPTH := 3
const MAX_INSTANCES := 96
const MAX_WICKS := 6
## The largest anything may be, centimeters: a thing past it is scaled down whole.
const MAX_SIZE := 60.0
## How high above the cloth a thing's FOOT is measured (meters): what a card sliding across the
## cloth would run into. A chalice's bowl is above it; its foot and stem are not.
const FOOT_H := 0.02
## A crystal this clear or clearer is drawn as glass.
const CLEAR := 0.8

## THE SHAPES a part can be, and what they are for.
const SHAPES := {
	"lathe": "a solid turned about the vertical axis, as on a lathe: vessels, candles, candlesticks, cups, bowls, bottles, vases, stems, bells, finials, coins. `profile`: [radius, height] points from the middle of the bottom outward and up the outside; a HOLLOW thing goes on over its rim and back down the inside to the middle of its inner floor - a bowl is [[0,0],[3,0],[6,4],[6.4,4.2],[5.8,4],[2.8,0.6],[0,0.6]]. Optional: `smooth` true curves through the points instead of joining them straight; `sides` 3-12 cuts it flat-sided (6 is hexagonal); `lobes` bulges round it (a scalloped bowl, a melon, a fluted column) with `lobe_depth` 0.05-0.4; `twist` in degrees from bottom to top (a twisted taper). A candle's wax has `wick` true - its flame is lit at the middle of its top, which is melted into a shallow pool - or `wicks` 2-6, set round the top (three make a triangle), or `wicks` [[x, z], ...] placing each, centimeters from the middle; and `drips` 0-1.",
	"box": "a block with rounded edges: books, boxes, tins, slabs, trays, a plinth. `size` [width, height, depth]; `round` the edges' radius; `taper` 0-0.9 narrows it toward the top.",
	"ball": "a sphere, stretched to `size` [width, height, depth]: crystal balls, beads, fruit, eggs, orbs, pebbles, tumbled stones. `lumpy` 0-1 makes it irregular like a tumbled stone or a fruit; `facets` true cuts it into flat faces like a rough, raw stone.",
	"point": "one crystal: a prism with a pointed end, standing up from its base - a tower, a single point. `radius`, `length` (the prism), `tip` (the point), `sides` (6 for quartz).",
	"cluster": "a crystal cluster: points growing out of one rough rock. `count` 3-24, `radius` (the rock), `length` [shortest, longest], `spread` (degrees the outer points lean out), `thickness` (a point's width over its length, 0.1-0.3), `base` (the rock's material, if not the crystal's).",
	"geode": "a geode broken open: a rough round rock, hollow, the hollow lined with crystal points growing in toward its middle. It lies open side up; `turn` [30, 0, 0] tips the opening toward the reader. `radius` (the rock's), `rind` (the thickness of its shell), `length` (the crystals'), `base` (the rock's material; plain gray rock if not given). The part's own material is the crystals'.",
	"ring": "a ring lying flat round the vertical axis, resting on whatever is under it: rims, rings, bangles, a coiled rope. `radius` (to the middle of the band), `thickness` (the band's), `arc` (degrees, less than 360 for an open ring).",
	"tube": "a round rod along a path: handles, stems, incense sticks, wands, branches, wire, a snake, a feather's quill. `path` [[x,y,z], ...] in the thing's own space, `radius` (or `radii`, one per point), `smooth` (default true).",
	"sheet": "a thin flat piece lying on the cloth unless turned, placed by its middle: a leaf, a petal, a feather, a page, a scrap of cloth. `outline` (leaf, petal, feather, oval or rect - or `points` [[x,z], ...] for any outline), `size` [width, length] (its length runs front to back), `bend` -1..1 (the tip curls up), `fold` 0-1 (the sides lift).",
	"bloom": "a flower head, petals in rings round a small middle: `petals`, `layers` 1-4, `radius`, `cup` 0 (open flat) to 1 (a closed bud), `width` (a petal's width over its length).",
	"extrude": "an outline raised straight up: trays, tiles, plaques, tablets, boxes, dishes and candles of any plan - a star, a hexagon, a heart. `outline` (polygon with `sides` 3-12, star with `sides` points, circle, rect or heart - or `points` [[x, z], ...] for any outline, which may go in and out), `size` [width, depth], `height`; optional `taper` 0-0.9 (narrower at the top), `bevel` (centimeters cut off the top edge), `wall` (centimeters: hollow, as a tray or a dish is, its floor as thick as its wall). A wax extrude takes `wick` or `wicks` as a turned candle does.",
}

## AN EXTRUDE's named outlines.
const EXTRUDE_OUTLINES := ["polygon", "star", "circle", "rect", "heart"]

## THE MATERIALS, each with its shader `code` and its defaults. `polish` is how smooth it is,
## `wear` how used, `pattern` how strong its natural pattern, `clarity` how far light goes into
## it, `grain` the pattern's size in centimeters.
const MATERIALS := {
	"metal": {"code": 1, "polish": 0.62, "wear": 0.2, "pattern": 0.0, "clarity": 0.5, "grain": 1.5,
		"about": "brass, copper, bronze, silver, iron, pewter, gold; `wear` is tarnish or patina in `color2`, `pattern` a hammered surface"},
	"wax": {"code": 2, "polish": 0.2, "wear": 0.0, "pattern": 0.0, "clarity": 0.5, "grain": 2.0,
		"about": "candle wax - beeswax, paraffin, tallow; `clarity` how much the light glows through it"},
	"glass": {"code": 3, "polish": 0.92, "wear": 0.0, "pattern": 0.0, "clarity": 0.85, "grain": 2.0,
		"about": "see-through; `color` tints it, a low `clarity` frosts it, an ornament on glass is etched"},
	"crystal": {"code": 4, "polish": 0.85, "wear": 0.0, "pattern": 0.2, "clarity": 0.5, "grain": 1.2,
		"about": "quartz, amethyst, citrine, obsidian, selenite, any mineral; `clarity` from milky 0 to water-clear 1, `pattern` banding in `color2` - curving round in rings with `rings` true (agate, malachite, onyx)"},
	"ceramic": {"code": 5, "polish": 0.6, "wear": 0.0, "pattern": 0.15, "clarity": 0.5, "grain": 1.0,
		"about": "glazed or bare clay - porcelain, stoneware, terracotta, faience; `polish` is the glaze, `pattern` speckle, `wear` crazing, `color2` where the glaze pools"},
	"stone": {"code": 6, "polish": 0.3, "wear": 0.1, "pattern": 0.4, "clarity": 0.5, "grain": 2.5,
		"about": "marble, slate, granite, soapstone, alabaster, jasper; `pattern` veins in `color2` - curving round in rings with `rings` true (banded jasper) - and `wear` flecks"},
	"wood": {"code": 7, "polish": 0.4, "wear": 0.0, "pattern": 0.55, "clarity": 0.5, "grain": 1.2,
		"about": "any timber, carved or turned; `pattern` its grain in `color2`, `polish` oil or varnish"},
	"paper": {"code": 8, "polish": 0.05, "wear": 0.2, "pattern": 0.0, "clarity": 0.5, "grain": 3.0,
		"about": "paper, card, parchment; `wear` stains in `color2`"},
	"leather": {"code": 9, "polish": 0.35, "wear": 0.3, "pattern": 0.0, "clarity": 0.5, "grain": 0.8,
		"about": "hide and leather bindings; `wear` scuffs in `color2`"},
	"fabric": {"code": 10, "polish": 0.3, "wear": 0.0, "pattern": 0.0, "clarity": 0.5, "grain": 2.0,
		"about": "cloth, velvet, silk, cord; `polish` is a velvet's sheen"},
	"bone": {"code": 11, "polish": 0.4, "wear": 0.4, "pattern": 0.0, "clarity": 0.5, "grain": 2.0,
		"about": "bone, ivory, horn, shell; `wear` yellows it toward `color2`"},
	"plant": {"code": 12, "polish": 0.2, "wear": 0.0, "pattern": 0.5, "clarity": 0.5, "grain": 1.0,
		"about": "leaves, petals, stems, herbs; `wear` dries them toward `color2`"},
	"painted": {"code": 0, "polish": 0.5, "wear": 0.0, "pattern": 0.0, "clarity": 0.5, "grain": 1.0,
		"about": "lacquer, enamel or paint over anything"},
}

## THE ORNAMENTS a part's surface can carry, with their shader codes.
const ORNAMENTS := {
	"bands": {"code": 1, "about": "raised rings round it, `count` of them"},
	"flutes": {"code": 2, "about": "grooves running up it, `count` round"},
	"dots": {"code": 3, "about": "a field of dots, `count` round"},
	"stars": {"code": 4, "about": "five-pointed stars, `count` round"},
	"moons": {"code": 5, "about": "crescent moons, `count` round"},
	"zigzag": {"code": 6, "about": "one zigzag band, `count` peaks round"},
	"scales": {"code": 7, "about": "overlapping scales, `count` round"},
	"spiral": {"code": 8, "about": "lines winding up it, `count` of them"},
	"lattice": {"code": 9, "about": "a diamond lattice, `count` diamonds round"},
	"hammered": {"code": 10, "about": "the dimpled face of hand-hammered metal, all over"},
}

## THE PLAY OF LIGHT a material can have - what moves in a stone, a shell or a metal as it turns,
## or as the flames catch it - with its shader code. Clear glass and crystal show only a rainbow.
const PLAYS := {
	"silk": {"code": 1, "about": "a silky band of light lying across its fibers, sliding over it as it turns; `pattern` stripes it in `color2` - tiger's eye, hawk's eye, satin spar selenite"},
	"flash": {"code": 2, "about": "broad patches that flash `color2` and the colors beside it where they catch the light, dark elsewhere - labradorite, spectrolite"},
	"fire": {"code": 3, "about": "small patches of every color, changing as it turns - opal"},
	"glitter": {"code": 4, "about": "tiny flakes of `color2` that sparkle in the flames - goldstone, sunstone, aventurine, the pyrite in lapis"},
	"rainbow": {"code": 5, "about": "an oily rainbow film over the surface - aura quartz, rainbow obsidian, bismuth, paua, abalone, mother of pearl"},
	"glow": {"code": 6, "about": "a soft light of `color2` floating under the surface - moonstone"},
}

## A sheet's outlines: its half-width along its length, 0 at the base to 1 at the tip.
const OUTLINES := ["leaf", "petal", "feather", "oval", "rect"]

const _SHADER := preload("res://shaders/prop.gdshader")
const _GLASS := preload("res://shaders/prop_glass.gdshader")
const _LENS := preload("res://shaders/prop_lens.gdshader")


## THE VOCABULARY, as an agent reads it: every shape, material and ornament with its words.
static func describe() -> String:
	var lines := PackedStringArray()
	lines.append("SHAPES (a part's `shape`):")
	for k in SHAPES:
		lines.append("- %s: %s" % [k, SHAPES[k]])
	lines.append("")
	lines.append("MATERIALS: name each one used under `materials`, then give a part its name. A material is {\"kind\", \"color\", \"color2\", \"polish\", \"wear\", \"pattern\", \"clarity\", \"grain\", \"play\", \"rings\"} - all but kind and color optional, numbers 0-1 except grain (centimeters). Kinds:")
	for k in MATERIALS:
		lines.append("- %s: %s" % [k, String((MATERIALS[k] as Dictionary)["about"])])
	lines.append("")
	lines.append("PLAY OF LIGHT: a material's `play` is the light that moves in it, most often a stone's, a shell's or a metal's (clear glass and crystal show only rainbow):")
	for k in PLAYS:
		lines.append("- %s: %s" % [k, String((PLAYS[k] as Dictionary)["about"])])
	lines.append("")
	lines.append("ORNAMENT: a part's surface can carry one: {\"kind\", \"count\", \"depth\" 0-1 (relief), \"color\" (paint or inlay; leave it out for relief alone), \"from\" and \"to\" (the band of the part's height it covers, 0 the bottom, 1 the top)}. Kinds:")
	for k in ORNAMENTS:
		lines.append("- %s: %s" % [k, String((ORNAMENTS[k] as Dictionary)["about"])])
	lines.append("")
	lines.append("COPIES: a part's `copies` repeats it - \"copies\": {\"ring\": {\"count\": 5, \"radius\": 4}} round the vertical axis (`arc` less than 360 makes it an arc, from `start` degrees: 0 the thing's right, 90 its front, 270 its back), {\"line\": {\"count\": 3, \"step\": [x, y, z]}}, {\"scatter\": {\"count\": 9, \"radius\": 6}} strewn about, never touching (stones set out on the cloth), or {\"heap\": {\"count\": 12, \"radius\": 3}} piled up, each resting on the ones under it (stones heaped in a dish or a shell, a pile of coins) - with `jitter` 0-1 so the copies differ a little. A copied part's `material` can be a LIST of names, which its copies take in turn: a handful of stones of five kinds, books in three bindings. A copied candle's wax lights a flame on every copy.")
	lines.append("")
	lines.append("GROUPS: a part can be a GROUP instead of a shape - {\"parts\": [...], \"at\", \"turn\", \"copies\"} - whose own parts are placed in its space as a thing's are in the thing's; repeat the group and all of it repeats. A candelabra's arm, its cup and its taper are one group copied round in a ring; a group can hold groups, %d deep. A list of materials on a part inside a copied group goes to the group's copies in turn. Every part in every group counts toward the %d a thing may have." % [MAX_DEPTH, MAX_PARTS])
	lines.append("")
	lines.append("FLAMES: a thing's flames burn as ONE light, however many it has - a candelabra's tapers, a pillar's three wicks, a dish of tea lights - up to %d on one thing." % TarotTable.MAX_FLAMES)
	return "\n".join(lines)


# --- sanitizing -----------------------------------------------------------------------------

## WHATEVER AN AGENT WROTE, as something [method build] can make: every part a known shape with its
## numbers in range, every material named and known, colors colors. A part with no shape the
## builder knows is dropped, and a thing with no part left is dropped. Keys this file does not own
## (where a thing stands, say) are passed through for the caller to judge. [param palette] gives
## the colors an agent left out.
static func sanitize(spec: Dictionary, palette: Array) -> Dictionary:
	var pal: Array = palette if not palette.is_empty() else ["#8a6a3a", "#c9b38a", "#3a3030"]
	var mats := {}
	var raw: Dictionary = spec.get("materials", {}) if spec.get("materials") is Dictionary else {}
	var i := 0
	for k in raw:
		mats[String(k).strip_edges()] = sanitize_material(raw[k], String(pal[i % pal.size()]))
		i += 1
	var things: Array = []
	for t in (spec.get("things", []) if spec.get("things") is Array else []):
		if not (t is Dictionary):
			continue
		var thing := _sanitize_thing(t as Dictionary, mats, pal)
		if not (thing["parts"] as Array).is_empty():
			things.append(thing)
	return {"idea": _text(spec.get("idea", ""), 600), "things": things, "materials": mats}


static func sanitize_material(m: Variant, fallback: String) -> Dictionary:
	var d: Dictionary = m if m is Dictionary else {}
	var kind := String(d.get("kind", "")).strip_edges().to_lower()
	if not MATERIALS.has(kind):
		kind = "painted"
	var base: Dictionary = MATERIALS[kind]
	var c := _color(d.get("color", ""), fallback)
	var out := {"kind": kind, "color": c,
		"color2": _color(d.get("color2", ""), "#" + Color.html(c).darkened(0.45).to_html(false)),
		"grain": _num(d.get("grain"), float(base["grain"]), 0.2, 20.0), "rings": _flag(d.get("rings"))}
	for k in ["polish", "wear", "pattern", "clarity"]:
		out[k] = _num(d.get(k), float(base[k]), 0.0, 1.0)
	var play := String(d["play"]).strip_edges().to_lower() if d.get("play") is String else ""
	if PLAYS.has(play):
		out["play"] = play
	return out


static func _sanitize_thing(t: Dictionary, mats: Dictionary, pal: Array) -> Dictionary:
	var out := t.duplicate(true)
	out["name"] = _text(t.get("name", "a thing"), 100)
	out["why"] = _text(t.get("why", ""), 200)
	out["parts"] = _sanitize_parts(t.get("parts"), mats, pal, 0, [MAX_PARTS])
	return out


## A list of parts made safe, a GROUP's own parts with it ({"parts": [...]} placed, turned and
## repeated as one), [constant MAX_DEPTH] groups deep at most - a group past that is dropped, with
## everything in it. [param budget] is how many shaped parts the thing has left ([constant
## MAX_PARTS] in all), spent as they are kept. A group left with no part is dropped.
static func _sanitize_parts(raw: Variant, mats: Dictionary, pal: Array, depth: int, budget: Array) -> Array:
	var out: Array = []
	for p in (raw if raw is Array else []):
		if int(budget[0]) <= 0:
			break
		if not (p is Dictionary):
			continue
		var d: Dictionary = p
		if d.get("parts") is Array:
			if depth >= MAX_DEPTH:
				continue
			var kids := _sanitize_parts(d["parts"], mats, pal, depth + 1, budget)
			if not kids.is_empty():
				out.append({"parts": kids, "at": _vec3(d.get("at"), Vector3.ZERO, MAX_SIZE), "turn": _turn(d.get("turn")),
					"wick": false, "copies": _copies_of(d)})
			continue
		var part := _sanitize_part(d, mats, pal)
		if not part.is_empty():
			out.append(part)
			budget[0] = int(budget[0]) - 1
	return out


static func _sanitize_part(p: Dictionary, mats: Dictionary, pal: Array) -> Dictionary:
	var shape := String(p.get("shape", "")).strip_edges().to_lower()
	if not SHAPES.has(shape):
		return {}
	var out := {"shape": shape, "at": _vec3(p.get("at"), Vector3.ZERO, MAX_SIZE),
		"turn": _turn(p.get("turn")), "wick": _flag(p.get("wick"))}
	# WICKS: `wick` true is one, at the middle of the top; `wicks` a number set round the top, or
	# each placed [[x, z], ...]
	var wicks := 1 if bool(out["wick"]) else 0
	var spots: Array = []
	if p.get("wicks") is Array:
		spots = _points2(p["wicks"], Vector2(-MAX_SIZE, -MAX_SIZE), Vector2(MAX_SIZE, MAX_SIZE)).slice(0, MAX_WICKS)
		wicks = spots.size()
	elif p.get("wicks") is float or p.get("wicks") is int:
		wicks = int(_num(p["wicks"], 1.0, 0.0, float(MAX_WICKS)))
	out["wick"] = wicks > 0
	out["wicks"] = wicks
	out["wick_spots"] = spots
	# THE MATERIAL: a name under `materials` or one written inline - or a LIST of them, which the
	# part's copies take in turn
	var names: Array = []
	for m in ((p["material"] as Array).slice(0, MAX_COPIES) if p.get("material") is Array else [p.get("material", "")]):
		names.append(_material_key(m, mats, pal))
	if names.is_empty():
		names.append(_material_key("", mats, pal))
	out["material"] = names[0]
	if names.size() > 1:
		out["materials"] = names
	match shape:
		"lathe":
			var prof := _points2(p.get("profile"), Vector2(0.0, -1.0), Vector2(MAX_SIZE * 0.5, MAX_SIZE))
			if prof.size() < 2:
				return {}
			out["profile"] = prof
			out["smooth"] = _flag(p.get("smooth"))
			var sides := int(_num(p.get("sides"), 0.0, 0.0, 16.0))
			out["sides"] = sides if sides >= 3 else 0
			out["lobes"] = int(_num(p.get("lobes"), 0.0, 0.0, 32.0))
			out["lobe_depth"] = _num(p.get("lobe_depth"), 0.15, 0.0, 0.45)
			out["twist"] = _num(p.get("twist"), 0.0, -1080.0, 1080.0)
			out["drips"] = _num(p.get("drips"), 0.0, 0.0, 1.0)
		"box":
			out["size"] = _vec3(p.get("size"), Vector3(6, 3, 4), MAX_SIZE, 0.1)
			var s: Vector3 = out["size"]
			out["round"] = _num(p.get("round"), minf(0.3, minf(s.x, minf(s.y, s.z)) * 0.1), 0.0, minf(s.x, minf(s.y, s.z)) * 0.5)
			out["taper"] = _num(p.get("taper"), 0.0, 0.0, 0.9)
		"ball":
			var r := _num(p.get("radius"), -1.0, 0.1, MAX_SIZE * 0.5)
			out["size"] = _vec3(p.get("size"), Vector3(r, r, r) * 2.0 if r > 0.0 else Vector3(4, 4, 4), MAX_SIZE, 0.1)
			out["lumpy"] = _num(p.get("lumpy"), 0.0, 0.0, 1.0)
			out["facets"] = _flag(p.get("facets"))
		"point":
			out["radius"] = _num(p.get("radius"), 1.0, 0.1, 12.0)
			out["length"] = _num(p.get("length"), 5.0, 0.2, 40.0)
			out["tip"] = _num(p.get("tip"), float(out["radius"]) * 1.4, 0.0, 20.0)
			out["sides"] = int(_num(p.get("sides"), 6.0, 3.0, 12.0))
		"cluster":
			out["count"] = int(_num(p.get("count"), 7.0, 2.0, 24.0))
			out["radius"] = _num(p.get("radius"), 4.0, 0.5, 20.0)
			var ln := _points2([p.get("length")], Vector2(0.3, 0.3), Vector2(30.0, 30.0))
			var lv: Vector2 = ln[0] if not ln.is_empty() else Vector2(2.0, 6.0)
			if p.get("length") is float or p.get("length") is int:
				lv = Vector2(float(p["length"]) * 0.45, float(p["length"]))
			out["length"] = Vector2(minf(lv.x, lv.y), maxf(lv.x, lv.y)).clamp(Vector2(0.3, 0.3), Vector2(30.0, 30.0))
			out["spread"] = _num(p.get("spread"), 35.0, 0.0, 80.0)
			out["thickness"] = _num(p.get("thickness"), 0.17, 0.06, 0.4)
			var b := String(p.get("base", "")).strip_edges()
			out["base"] = b if mats.has(b) else ""
		"geode":
			var r := _num(p.get("radius"), 6.0, 1.0, 25.0)
			out["radius"] = r
			out["rind"] = _num(p.get("rind"), r * 0.18, r * 0.06, r * 0.45)
			var hollow := r - float(out["rind"])
			out["length"] = _num(p.get("length"), hollow * 0.35, 0.2, hollow * 0.55)
			# enough points to line the hollow, each standing in about its own width
			var each := pow(float(out["length"]) * 0.45, 2.0)
			out["count"] = int(_num(p.get("count"), clampf(TAU * hollow * hollow * 0.6 / each, 12.0, 90.0), 6.0, 160.0))
			var b := String(p.get("base", "")).strip_edges()
			if not mats.has(b):
				b = "geode rock"
				if not mats.has(b):
					mats[b] = sanitize_material({"kind": "stone", "color": "#7a7168", "color2": "#b3aa9c", "polish": 0.1,
						"wear": 0.5, "pattern": 0.15}, "#7a7168")
			out["base"] = b
		"extrude":
			var outline := String(p.get("outline", "polygon")).strip_edges().to_lower()
			out["outline"] = outline if EXTRUDE_OUTLINES.has(outline) else "polygon"
			out["points"] = _points2(p.get("points"), Vector2(-MAX_SIZE, -MAX_SIZE), Vector2(MAX_SIZE, MAX_SIZE))
			var sz := _points2([p.get("size")], Vector2(0.2, 0.2), Vector2(MAX_SIZE, MAX_SIZE))
			out["size"] = sz[0] if not sz.is_empty() else Vector2(6.0, 6.0)
			out["height"] = _num(p.get("height"), 2.0, 0.05, MAX_SIZE)
			out["sides"] = int(_num(p.get("sides"), 6.0, 3.0, 12.0))
			out["taper"] = _num(p.get("taper"), 0.0, 0.0, 0.9)
			out["bevel"] = _num(p.get("bevel"), 0.0, 0.0, 5.0)
			out["wall"] = _num(p.get("wall"), 0.0, 0.0, 10.0)
		"ring":
			out["radius"] = _num(p.get("radius"), 3.0, 0.2, MAX_SIZE * 0.5)
			out["thickness"] = _num(p.get("thickness"), 0.4, 0.05, 10.0)
			out["arc"] = _num(p.get("arc"), 360.0, 10.0, 360.0)
		"tube":
			var path := _points3(p.get("path"), MAX_SIZE)
			if path.size() < 2:
				return {}
			out["path"] = path
			out["radius"] = _num(p.get("radius"), 0.3, 0.03, 10.0)
			var radii := PackedFloat32Array()
			if p.get("radii") is Array:
				for x in (p["radii"] as Array):
					radii.append(_num(x, float(out["radius"]), 0.0, 10.0))
			out["radii"] = radii
			out["smooth"] = _flag(p.get("smooth"), true)
		"sheet":
			var outline := String(p.get("outline", "leaf")).strip_edges().to_lower()
			out["outline"] = outline if OUTLINES.has(outline) else "leaf"
			out["points"] = _points2(p.get("points"), Vector2(-MAX_SIZE, -MAX_SIZE), Vector2(MAX_SIZE, MAX_SIZE))
			var sz := _points2([p.get("size")], Vector2(0.2, 0.2), Vector2(40.0, 40.0))
			out["size"] = sz[0] if not sz.is_empty() else Vector2(3.0, 6.0)
			out["bend"] = _num(p.get("bend"), 0.0, -1.0, 1.0)
			out["fold"] = _num(p.get("fold"), 0.0, 0.0, 1.0)
			out["thickness"] = _num(p.get("thickness"), 0.08, 0.02, 1.0)
		"bloom":
			out["petals"] = int(_num(p.get("petals"), 8.0, 3.0, 24.0))
			out["layers"] = int(_num(p.get("layers"), 2.0, 1.0, 4.0))
			out["radius"] = _num(p.get("radius"), 3.0, 0.5, 15.0)
			out["cup"] = _num(p.get("cup"), 0.4, 0.0, 1.0)
			out["width"] = _num(p.get("width"), 0.55, 0.2, 1.2)
	out["ornament"] = _sanitize_ornament(p.get("ornament"))
	out["copies"] = _copies_of(p)
	return out


## A part's (or a group's) repeat: its `copies` - or a REPEAT WRITTEN BESIDE IT ({"line": {...}} on
## the part itself), which is the same repeat: a set dresser wrote a skein's turns that way, and one
## turn of each color was built, the top one floating where the turns under it should have been.
static func _copies_of(p: Dictionary) -> Dictionary:
	var c := _sanitize_copies(p.get("copies"))
	if not c.is_empty():
		return c
	var loose := {}
	for kind in ["ring", "line", "scatter", "heap"]:
		if p.get(kind) is Dictionary:
			loose[kind] = p[kind]
	if loose.is_empty():
		return {}
	if p.has("jitter"):
		loose["jitter"] = p["jitter"]
	return _sanitize_copies(loose)


## How many flames [param part] lights: its wicks on every copy of it - a group's parts' on every
## copy of the group.
static func flames_of(part: Dictionary) -> int:
	var n := 0
	if part.get("parts") is Array:
		for q in part["parts"]:
			n += flames_of(q as Dictionary)
	elif part.get("wick") == true:
		n = maxi(int(part.get("wicks", 1)), 1)
	var c: Dictionary = part.get("copies", {})
	return n * (int(c.get("count", 1)) if not c.is_empty() else 1)


## [param part] standing unlit - a group with every part in it.
static func unlit(part: Dictionary) -> void:
	part["wick"] = false
	for q in (part.get("parts", []) if part.get("parts") is Array else []):
		unlit(q as Dictionary)


## The key of the material a part names: one under `materials`, one written inline (kept under a
## key of its own), or a bare kind; anything else is painted in the look's own colors.
static func _material_key(m: Variant, mats: Dictionary, pal: Array) -> String:
	if m is Dictionary:
		var key := "inline %d" % mats.size()
		mats[key] = sanitize_material(m, String(pal[mats.size() % pal.size()]))
		return key
	var name := String(m).strip_edges() if m is String else ""
	if mats.has(name) and not name.is_empty():
		return name
	if MATERIALS.has(name.to_lower()):
		var key := name.to_lower()
		if not mats.has(key):
			mats[key] = sanitize_material({"kind": key}, String(pal[mats.size() % pal.size()]))
		return key
	if not mats.has("painted"):
		mats["painted"] = sanitize_material({"kind": "painted"}, String(pal[0]))
	return "painted"


static func _sanitize_ornament(o: Variant) -> Dictionary:
	if not (o is Dictionary):
		return {}
	var d: Dictionary = o
	var kind := String(d.get("kind", "")).strip_edges().to_lower()
	if not ORNAMENTS.has(kind):
		return {}
	var out := {"kind": kind, "count": _num(d.get("count"), 8.0, 1.0, 64.0),
		"depth": _num(d.get("depth"), 0.5, 0.0, 1.0), "from": _num(d.get("from"), 0.0, 0.0, 1.0),
		"to": _num(d.get("to"), 1.0, 0.0, 1.0)}
	if _is_color(d.get("color", "")):
		out["color"] = String(d["color"]).strip_edges()
	return out


static func _sanitize_copies(c: Variant) -> Dictionary:
	if not (c is Dictionary):
		return {}
	var d: Dictionary = c
	var jitter := _num(d.get("jitter"), 0.0, 0.0, 1.0)
	for kind in ["ring", "line", "scatter", "heap"]:
		if not (d.get(kind) is Dictionary):
			continue
		var k: Dictionary = d[kind]
		var out := {"kind": kind, "count": int(_num(k.get("count"), 3.0, 1.0, float(MAX_COPIES))),
			"jitter": _num(k.get("jitter"), jitter, 0.0, 1.0)}
		match kind:
			"ring":
				out["radius"] = _num(k.get("radius"), 3.0, 0.0, MAX_SIZE * 0.5)
				out["start"] = _num(k.get("start"), 0.0, -360.0, 360.0)
				out["arc"] = _num(k.get("arc"), 360.0, 10.0, 360.0)
				out["face"] = _flag(k.get("face"), true)
			"line":
				out["step"] = _vec3(k.get("step"), Vector3(2, 0, 0), MAX_SIZE)
			"scatter", "heap":
				out["radius"] = _num(k.get("radius"), 4.0, 0.0, MAX_SIZE * 0.5)
		return out
	return {}


static func _num(v: Variant, fallback: float, lo: float, hi: float) -> float:
	var x := fallback
	if v is float or v is int:
		x = float(v)
	elif v is String and (v as String).is_valid_float():
		x = (v as String).to_float()
	if is_nan(x) or is_inf(x):
		x = fallback
	return clampf(x, lo, hi)


## A yes or no an agent wrote: itself when it is one, [param fallback] when it is anything else (a
## "yes" compared with true is an error, not false).
static func _flag(v: Variant, fallback := false) -> bool:
	return bool(v) if v is bool else fallback


static func _text(v: Variant, most: int) -> String:
	var s := (v as String) if v is String else ("" if v == null else str(v))
	return s.strip_edges().substr(0, most)


static func _is_color(v: Variant) -> bool:
	return v is String and Manuscript._rx("^#[0-9a-fA-F]{6}$").search((v as String).strip_edges()) != null


static func _color(v: Variant, fallback: String) -> String:
	if _is_color(v):
		return String(v).strip_edges()
	return fallback if _is_color(fallback) else "#808080"


static func _vec3(v: Variant, fallback: Vector3, lim: float, lo := -INF) -> Vector3:
	if not (v is Array) or (v as Array).size() < 3:
		return fallback
	var a: Array = v
	var out := Vector3(_num(a[0], fallback.x, -lim, lim), _num(a[1], fallback.y, -lim, lim), _num(a[2], fallback.z, -lim, lim))
	if lo > -INF:
		out = out.clamp(Vector3(lo, lo, lo), Vector3(lim, lim, lim))
	return out


static func _turn(v: Variant) -> Vector3:
	if v is float or v is int:
		return Vector3(0.0, clampf(float(v), -720.0, 720.0), 0.0)
	return _vec3(v, Vector3.ZERO, 720.0)


static func _points2(v: Variant, lo: Vector2, hi: Vector2) -> Array:
	var out: Array = []
	if not (v is Array):
		return out
	for q in (v as Array):
		if out.size() >= MAX_POINTS:
			break
		if q is Array and (q as Array).size() >= 2 and (q[0] is float or q[0] is int) and (q[1] is float or q[1] is int):
			out.append(Vector2(float(q[0]), float(q[1])).clamp(lo, hi))
	return out


static func _points3(v: Variant, lim: float) -> Array:
	var out: Array = []
	if not (v is Array):
		return out
	for q in (v as Array):
		if out.size() >= MAX_POINTS:
			break
		if q is Array and (q as Array).size() >= 3:
			out.append(_vec3(q, Vector3.ZERO, lim))
	return out


# --- building -----------------------------------------------------------------------------

## A THING, BUILT: `node` (its meshes, in meters: base on y = 0, centered over what it stands on,
## front toward +z), `size` (its bounds), `foot` (the convex outline of where it meets the cloth,
## up to [constant FOOT_H], x by z), `outline` (its whole outline seen from above), `wicks`
## (where its flames are lit), `glows` (each wick's wax material, whose `flame` the table sets as
## the flame flickers) and `meshes` (every MeshInstance3D in it). [param materials] are the
## sanitized materials; [param seed] varies drips, scatter, lumps and the patterns' noise.
static func build(thing: Dictionary, materials: Dictionary, seed: int) -> Dictionary:
	var inner := Node3D.new()
	var all := PackedVector3Array()
	var wicks: Array = []
	var glows: Array = []
	var meshes: Array = []
	# EVERY PART, its groups opened: its geometry and every place it stands - its groups' repeats and
	# its own together - and no more of them in all than a thing may make
	var leaves := _assemble(thing.get("parts", []), seed, [])
	var budget := MAX_INSTANCES
	for leaf in leaves:
		var xs: Array = leaf["xforms"]
		leaf["xforms"] = xs.slice(0, maxi(budget, 0))
		budget -= xs.size()
	for leaf in leaves:
		var part: Dictionary = leaf["part"]
		var geos: Array = leaf["geos"]
		var xforms: Array = leaf["xforms"]
		var salt: Variant = leaf["salt"]
		if xforms.is_empty():
			continue
		# A FLAME FOR EVERY WICK on the top of a candle that is not turned (a turned one's are set
		# where its pool is, in [method _lathe])
		var g0: Tris = (geos[0] as Dictionary)["geo"]
		if part.get("wick") == true and g0.wicks.is_empty():
			for sp in _wick_spots(part, _top_radius(g0)):
				g0.wicks.append(Vector3(g0.top.x + (sp as Vector2).x, g0.top.y, g0.top.z + (sp as Vector2).y))
		# A LIST OF MATERIALS is taken by the copies in turn: a mesh for each material's copies
		var names: Array = part.get("materials", [])
		if names.is_empty():
			names = [String(part.get("material", "painted"))]
		for gi in geos.size():
			var g: Tris = (geos[gi] as Dictionary)["geo"]
			if g.v.is_empty():
				continue
			var fixed := String((geos[gi] as Dictionary).get("material", ""))
			var takers := {}
			for ci in xforms.size():
				var nm := fixed if not fixed.is_empty() else String(names[ci % names.size()])
				if not takers.has(nm):
					takers[nm] = []
				(takers[nm] as Array).append(ci)
			for nm in takers:
				var ids: Array = takers[nm]
				var mine: Array = []
				for ci in ids:
					mine.append(xforms[ci])
				var merged := g.placed(mine, ids)
				var m: Dictionary = materials.get(nm, sanitize_material({}, "#808080"))
				var mi := MeshInstance3D.new()
				mi.mesh = merged.mesh()
				var mat := material(m, part.get("ornament", {}), g.girth, g.height, hash([seed, salt, gi, nm]), _solid(part))
				mi.material_override = mat
				if see_through(m):
					mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
				inner.add_child(mi)
				meshes.append(mi)
				all.append_array(merged.v)
				if gi == 0 and part.get("wick") == true:
					for xf in mine:
						for w in g.wicks:
							wicks.append((xf as Transform3D) * w)
							glows.append(mat)
	var root := Node3D.new()
	root.add_child(inner)
	if all.is_empty():
		return {"node": root, "size": AABB(), "foot": PackedVector2Array(), "outline": PackedVector2Array(),
			"wicks": [], "glows": [], "meshes": []}
	var box := AABB(all[0], Vector3.ZERO)
	for p in all:
		box = box.expand(p)
	# TOO BIG IS SCALED DOWN WHOLE: an agent's 2-meter vase is a mistake in units, not a vase
	var k := minf(1.0, MAX_SIZE * 0.01 / maxf(box.size.x, maxf(box.size.y, box.size.z)))
	var shift := Vector3(-box.get_center().x, -box.position.y, -box.get_center().z)
	inner.scale = Vector3(k, k, k)
	inner.position = shift * k
	var placed := PackedVector3Array()
	for p in all:
		placed.append((p + shift) * k)
	for i in wicks.size():
		wicks[i] = ((wicks[i] as Vector3) + shift) * k
	var size := AABB((box.position + shift) * k, box.size * k)
	return {"node": root, "size": size, "foot": _foot(placed, FOOT_H), "outline": _hull(placed, INF),
		"wicks": wicks, "glows": glows, "meshes": meshes}


## A thing's parts, groups opened: `[{part, geos, xforms, salt}]` - each shaped part with its
## geometry and every place it stands in the thing (its own repeats inside its groups', a group's
## copies sized, for a scatter or a heap, by everything in it). [param path] is where these parts
## sit among the groups; `salt` is a part's place, the same as it always was for a part in no group.
static func _assemble(parts: Array, seed: int, path: Array) -> Array:
	var out: Array = []
	for pi in parts.size():
		var part: Dictionary = parts[pi]
		var here: Variant = pi if path.is_empty() else path + [pi]
		var rng := RandomNumberGenerator.new()
		rng.seed = hash([seed, here, "part"])
		if part.get("parts") is Array:
			var inner := _assemble(part["parts"], seed, path + [pi])
			if inner.is_empty():
				continue
			var sized: Array = []
			if String((part.get("copies", {}) as Dictionary).get("kind", "")) in ["scatter", "heap"]:
				for leaf in inner:
					for e in leaf["geos"]:
						sized.append({"geo": ((e as Dictionary)["geo"] as Tris).placed(leaf["xforms"])})
			var outer := _placements(part, rng, sized)
			for leaf in inner:
				var xs: Array = []
				for o in outer:
					for x in leaf["xforms"]:
						xs.append((o as Transform3D) * (x as Transform3D))
				leaf["xforms"] = xs
				out.append(leaf)
			continue
		var geos := _geometry(part, rng)
		if geos.is_empty():
			continue
		out.append({"part": part, "geos": geos, "xforms": _placements(part, rng, geos), "salt": here})
	return out


## Where a candle's wicks stand on a top of radius [param R] (meters), from its middle: each where
## it was placed (kept on the top), or so many set round it - one in the middle, two to six on a
## ring half way out (more a little further).
static func _wick_spots(p: Dictionary, R: float) -> Array:
	var out: Array = []
	var spots: Array = p.get("wick_spots", [])
	if not spots.is_empty():
		for sp in spots:
			var v := (sp as Vector2) * 0.01
			if v.length() > R * 0.85:
				v = v.normalized() * R * 0.85
			out.append(v)
		return out
	var n := maxi(int(p.get("wicks", 1)), 1)
	if n == 1:
		return [Vector2.ZERO]
	var r := R * (0.5 if n <= 3 else 0.62)
	for i in n:
		var a := -PI * 0.5 + TAU * float(i) / float(n)
		out.append(Vector2(cos(a), sin(a)) * r)
	return out


## How far a part's top reaches from its middle (meters): what of it lies within a millimeter of its
## highest point.
static func _top_radius(g: Tris) -> float:
	var r := 0.0
	for q in g.v:
		if q.y >= g.top.y - 0.001:
			r = maxf(r, Vector2(q.x - g.top.x, q.z - g.top.z).length())
	return maxf(r, 0.004)


## Whether a material is drawn see-through: glass, and crystal clear enough to see through.
static func see_through(m: Dictionary) -> bool:
	var kind := String(m.get("kind", ""))
	return kind == "glass" or (kind == "crystal" and float(m.get("clarity", 0.5)) >= CLEAR)


## Whether a part is SOLID rather than a vessel: a ball, a crystal, a rod - or a turned part whose
## profile never goes back down (a hollow one climbs its outside and comes down its inside).
static func _solid(part: Dictionary) -> bool:
	if String(part.get("shape", "")) == "extrude":
		return float(part.get("wall", 0.0)) <= 0.0
	if String(part.get("shape", "")) != "lathe":
		return String(part.get("shape", "")) in ["ball", "point", "cluster", "geode", "tube", "ring"]
	var prof: Array = part.get("profile", [])
	var top := -INF
	for q in prof:
		top = maxf(top, (q as Vector2).y)
		if (q as Vector2).y < top - 0.5:
			return false
	return true


## The material for one part: the named material's look with this part's ornament, laid out for
## its own girth and height (so a motif keeps its shape on a tall vase and a squat bowl alike).
## SEEN-THROUGH material is glass for a vessel - a flame inside it shows - and a lens for a solid
## thing, which bends what is behind it.
static func material(m: Dictionary, orn: Variant, girth: float, height: float, salt: int, solid := false) -> ShaderMaterial:
	var kind := String(m.get("kind", "painted"))
	var mat := ShaderMaterial.new()
	# CLEAR CRYSTAL IS SEEN THROUGH: an opaque shader made a crystal ball a pearl
	var clear := see_through(m)
	mat.shader = (_LENS if solid else _GLASS) if clear else _SHADER
	if not clear:
		mat.set_shader_parameter("kind", int((MATERIALS.get(kind, MATERIALS["painted"]) as Dictionary)["code"]))
	mat.set_shader_parameter("color", Color.html(String(m.get("color", "#808080"))))
	mat.set_shader_parameter("color2", Color.html(String(m.get("color2", "#404040"))))
	for k in ["polish", "wear", "pattern", "clarity", "grain"]:
		mat.set_shader_parameter(k, float(m.get(k, 0.5)))
	mat.set_shader_parameter("seed", float(salt & 0xFFF) / 64.0)
	mat.set_shader_parameter("dims", Vector2(maxf(girth, 0.5), maxf(height, 0.5)))
	var play := String(m.get("play", ""))
	mat.set_shader_parameter("play", int((PLAYS[play] as Dictionary)["code"]) if PLAYS.has(play) else 0)
	mat.set_shader_parameter("rings", 1.0 if m.get("rings") == true else 0.0)
	var o: Dictionary = orn if orn is Dictionary else {}
	if not o.is_empty():
		mat.set_shader_parameter("orn", int((ORNAMENTS[String(o["kind"])] as Dictionary)["code"]))
		mat.set_shader_parameter("orn_count", float(o.get("count", 8.0)))
		mat.set_shader_parameter("orn_depth", float(o.get("depth", 0.5)))
		mat.set_shader_parameter("orn_zone", Vector2(float(o.get("from", 0.0)), float(o.get("to", 1.0))))
		if o.has("color"):
			mat.set_shader_parameter("orn_color", Color.html(String(o["color"])))
			mat.set_shader_parameter("orn_paint", 1.0)
	return mat


## Where each copy of a part goes, in the thing's space: the part turned about its own origin,
## then laid out as its copies ask, then moved to its `at`. [param geos] are the part's geometry
## ([method _geometry]), which strewn copies need the size of.
static func _placements(part: Dictionary, rng: RandomNumberGenerator, geos: Array) -> Array:
	var turn: Vector3 = part.get("turn", Vector3.ZERO)
	var own := Transform3D(Basis.from_euler(Vector3(deg_to_rad(turn.x), deg_to_rad(turn.y), deg_to_rad(turn.z))), Vector3.ZERO)
	var at: Vector3 = (part.get("at", Vector3.ZERO) as Vector3) * 0.01
	var c: Dictionary = part.get("copies", {})
	var out: Array = []
	if c.is_empty():
		return [Transform3D(Basis(), at) * own]
	var n := int(c["count"])
	var jit := float(c.get("jitter", 0.0))
	if String(c["kind"]) in ["scatter", "heap"]:
		return _strewn(String(c["kind"]) == "heap", n, jit, float(c["radius"]) * 0.01, own, at, _bounds(geos, own), rng)
	for i in n:
		var xf := Transform3D.IDENTITY
		match String(c["kind"]):
			"ring":
				# round the whole circle evenly, or along an arc from one end to the other
				var arc := float(c.get("arc", 360.0))
				var a := deg_to_rad(float(c.get("start", 0.0)) + (arc * float(i) / float(n) if arc >= 359.9 else arc * float(i) / float(maxi(n - 1, 1))))
				var r := float(c["radius"]) * 0.01
				xf = Transform3D(Basis(Vector3.UP, -a) if bool(c.get("face", true)) else Basis(), Vector3(cos(a), 0.0, sin(a)) * r)
			"line":
				xf = Transform3D(Basis(), (c["step"] as Vector3) * 0.01 * float(i))
		if jit > 0.0:
			var s := 1.0 + rng.randf_range(-0.15, 0.15) * jit
			xf = xf * Transform3D(Basis(Vector3.UP, rng.randf_range(-0.45, 0.45) * jit).scaled(Vector3(s, s, s)), Vector3.ZERO)
		out.append(Transform3D(Basis(), at) * xf * own)
	return out


## STREWN COPIES. Scattered ones never touch: each takes a spot in the circle clear of the rest,
## and a handful too crowded for it spreads wider rather than pass through itself. HEAPED ones
## settle as a dropped handful does: each tries a few spots in the circle and falls into the lowest
## - on the floor, or resting on the copies under it there (each taken as the rounded lump it most
## often is), a little tipped where it lies on others - so the floor fills before the heap rises.
static func _strewn(heap: bool, n: int, jit: float, radius: float, own: Transform3D, at: Vector3, bounds: Dictionary,
		rng: RandomNumberGenerator) -> Array:
	var reach := float(bounds["reach"])
	var girth := float(bounds["girth"])
	var low := float(bounds["low"])
	var tall := float(bounds["high"]) - low
	var laid: Array = []     # [where (x, z), reach, middle's height, half height, half-width]
	var out: Array = []
	for i in n:
		var s := 1.0 + rng.randf_range(-0.15, 0.15) * jit
		var r := reach * s
		var hh := tall * s * 0.5
		var spot := Vector2.ZERO
		var mid := hh
		if heap:
			var lowest := INF
			for k in 30:
				var q := Vector2.from_angle(rng.randf() * TAU) * radius * sqrt(rng.randf())
				var y := hh
				var held := 0.0    # how far off the middle of the lump holding it up it sits, as a share of their reach
				for e in laid:
					# two lumps meet across their half-widths, a little less, as rounded things nestle; a
					# lump taken as wide as its longest reach stood on air off the narrow side of another
					var touch := (girth * s + float(e[4])) * 0.9
					var d := q.distance_to(e[0])
					if d < touch:
						var up := float(e[2]) + (hh + float(e[3])) * sqrt(1.0 - pow(d / touch, 2.0)) * 0.9
						if up > y:
							y = up
							held = d / touch
				# a lump resting on another's shoulder would roll off it: it sits on the floor or over the middle
				if held > 0.5 and y > hh * 1.05:
					continue
				if y < lowest:
					lowest = y
					spot = q
				if k >= 5 and lowest < INF:
					break
			if lowest == INF:
				lowest = hh
				spot = Vector2.from_angle(rng.randf() * TAU) * radius * 1.3
			mid = lowest
		else:
			var wide := radius
			for k in 240:
				if k > 0 and k % 24 == 0:
					wide = wide * 1.15 + r * 0.3
				var q := Vector2.from_angle(rng.randf() * TAU) * wide * sqrt(rng.randf())
				var clear := true
				for e in laid:
					if q.distance_to(e[0]) < r + float(e[1]) + 0.001:
						clear = false
						break
				spot = q
				if clear:
					break
		laid.append([spot, r, mid, hh, girth * s])
		var b := Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3(s, s, s))
		if mid > hh * 1.2:
			b = Basis(Vector3(rng.randf_range(-1.0, 1.0), 0.0, rng.randf_range(-1.0, 1.0)).normalized(), rng.randf_range(0.1, 0.35)) * b
		out.append(Transform3D(Basis(), at) * Transform3D(b, Vector3(spot.x, mid - hh - low * s, spot.y)) * own)
	return out


## How far a part reaches from its own origin seen from above (`reach`, its farthest; `girth`, the
## mean of its half-width and half-depth), and how low and high it goes - turned by [param own] as
## each copy is before it is laid out.
static func _bounds(geos: Array, own: Transform3D) -> Dictionary:
	var reach := 0.0
	var hx := 0.0
	var hz := 0.0
	var low := INF
	var high := -INF
	for e in geos:
		for q in ((e as Dictionary)["geo"] as Tris).v:
			var w := own.basis * q
			reach = maxf(reach, Vector2(w.x, w.z).length())
			hx = maxf(hx, absf(w.x))
			hz = maxf(hz, absf(w.z))
			low = minf(low, w.y)
			high = maxf(high, w.y)
	if low > high:
		return {"reach": 0.01, "girth": 0.01, "low": 0.0, "high": 0.01}
	return {"reach": maxf(reach, 0.0005), "girth": maxf((hx + hz) * 0.5, 0.0005), "low": low, "high": maxf(high, low + 0.0005)}


## The part's geometry, in its own space (meters): `[{geo, material?}]` - a cluster's rock is a
## second entry when it has a material of its own.
static func _geometry(p: Dictionary, rng: RandomNumberGenerator) -> Array:
	match String(p["shape"]):
		"lathe":
			return [{"geo": _lathe(p, rng)}]
		"box":
			return [{"geo": _box(p)}]
		"ball":
			return [{"geo": _ball(p["size"] as Vector3, float(p["lumpy"]), bool(p["facets"]), rng)}]
		"point":
			var g := Tris.new()
			_crystal(g, float(p["radius"]) * 0.01, float(p["length"]) * 0.01, float(p["tip"]) * 0.01,
				int(p["sides"]), Transform3D.IDENTITY, rng)
			g.measure()
			return [{"geo": g}]
		"cluster":
			return _cluster(p, rng)
		"geode":
			return _geode(p, rng)
		"ring":
			return [{"geo": _ring(float(p["radius"]) * 0.01, float(p["thickness"]) * 0.01, float(p["arc"]))}]
		"tube":
			return [{"geo": _tube(p)}]
		"sheet":
			# placed by its middle, as a leaf laid down is
			var g := Tris.new()
			var mid := Vector3.ZERO if not (p.get("points", []) as Array).is_empty() else Vector3(0.0, 0.0, -float((p["size"] as Vector2).y) * 0.005)
			_sheet(g, p, Transform3D(Basis(), mid))
			g.measure()
			return [{"geo": g}]
		"bloom":
			return [{"geo": _bloom(p, rng)}]
		"extrude":
			return [{"geo": _extrude(p)}]
	return []


# --- the shapes -------------------------------------------------------------------------------

## A TURNED SOLID. The profile is joined straight (or curved through, `smooth`), closed at the axis
## top and bottom, and split into runs where it turns a corner, so a rim or a foot stays crisp and a
## belly stays smooth. Each ring of it can then be displaced as it goes round: LOBES bulge it,
## TWIST turns it with height, and wax DRIPS run down from the top - which is melted into a pool
## when the part carries a wick.
static func _lathe(p: Dictionary, rng: RandomNumberGenerator) -> Tris:
	var pts: Array = []
	for q in (p["profile"] as Array):
		pts.append((q as Vector2) * 0.01)
	if bool(p.get("smooth", false)) and pts.size() >= 3:
		pts = _smooth2(pts)
	if (pts[0] as Vector2).x > 0.0002:
		pts.insert(0, Vector2(0.0, (pts[0] as Vector2).y))
	if (pts[-1] as Vector2).x > 0.0002:
		pts.append(Vector2(0.0, (pts[-1] as Vector2).y))
	# THE POOL a burning candle melts round its wicks, and where its rim is
	var rim := Vector2.ZERO
	if bool(p.get("wick", false)):
		if pts.size() >= 3 and (pts[-2] as Vector2).x >= 0.002 and absf((pts[-2] as Vector2).y - (pts[-1] as Vector2).y) <= 0.002:
			rim = pts[-2]
		pts = _pool(pts)
	var twist := deg_to_rad(float(p.get("twist", 0.0)))
	var drips := float(p.get("drips", 0.0))
	if absf(twist) > 0.01 or drips > 0.0:
		pts = _subdivide(pts, 0.004)
	var y0 := INF
	var y1 := -INF
	var rmax := 0.0
	for q in pts:
		y0 = minf(y0, (q as Vector2).y)
		y1 = maxf(y1, (q as Vector2).y)
		rmax = maxf(rmax, (q as Vector2).x)
	var height := maxf(y1 - y0, 0.0005)
	var sides := int(p.get("sides", 0))
	var lobes := int(p.get("lobes", 0))
	var depth := float(p.get("lobe_depth", 0.15))
	var around := sides if sides >= 3 else clampi(roundi(rmax * 100.0 * 6.0 + 16.0), 24, 72)
	if lobes > 0:
		around = maxi(around, mini(lobes * 8, 160))
	# the drips: where round the top each runs, how wide, how far down, how proud
	var rtop := 0.0
	for q in pts:
		if (q as Vector2).y > y1 - height * 0.15:
			rtop = maxf(rtop, (q as Vector2).x)
	var drops: Array = []
	if drips > 0.0:
		for i in roundi(lerpf(2.0, 10.0, drips)):
			drops.append([rng.randf() * TAU, rng.randf_range(0.07, 0.17), rng.randf_range(0.12, 0.55) * (0.4 + 0.6 * drips),
				rtop * rng.randf_range(0.08, 0.15)])
	# arclength along the profile, for V
	var arc := PackedFloat32Array([0.0])
	for i in range(1, pts.size()):
		arc.append(arc[i - 1] + (pts[i] as Vector2).distance_to(pts[i - 1]))
	var total := maxf(arc[arc.size() - 1], 0.0001)
	# RUNS: split where the profile turns a corner
	var runs: Array = []
	var start := 0
	for i in range(1, pts.size() - 1):
		var a: Vector2 = (pts[i] as Vector2) - (pts[i - 1] as Vector2)
		var b: Vector2 = (pts[i + 1] as Vector2) - (pts[i] as Vector2)
		if a.length() < 1e-6 or b.length() < 1e-6:
			continue
		if absf(a.angle_to(b)) > deg_to_rad(35.0):
			runs.append([start, i])
			start = i
	runs.append([start, pts.size() - 1])
	var g := Tris.new()
	var surf := func(th: float, q: Vector2, n2: Vector2) -> Vector3:
		var r := q.x
		if lobes > 0:
			r *= 1.0 - depth * 0.5 * (1.0 - cos(float(lobes) * th))
		if not drops.is_empty() and absf(n2.x) > 0.5 and r > 0.0001:
			r += _drip(drops, th, (q.y - y0) / height)
		var tt := th + twist * (q.y - y0) / height
		return Vector3(cos(tt) * r, q.y, sin(tt) * r)
	for run in runs:
		var a := int(run[0])
		var b := int(run[1])
		if b <= a:
			continue
		# each point's outward normal in the profile's plane, within its run
		var n2s: Array = []
		for j in range(a, b + 1):
			var sum := Vector2.ZERO
			if j > a:
				var d: Vector2 = (pts[j] as Vector2) - (pts[j - 1] as Vector2)
				sum += Vector2(d.y, -d.x).normalized()
			if j < b:
				var d2: Vector2 = (pts[j + 1] as Vector2) - (pts[j] as Vector2)
				sum += Vector2(d2.y, -d2.x).normalized()
			n2s.append(sum.normalized() if sum.length() > 1e-6 else Vector2(1, 0))
		var rows: Array = []      # per profile point: [positions, normals] round the circle
		for j in range(a, b + 1):
			var q: Vector2 = pts[j]
			var n2: Vector2 = n2s[j - a]
			var ps := PackedVector3Array()
			var ns := PackedVector3Array()
			for i in around + 1:
				var th := TAU * float(i % around) / float(around)
				var pos: Vector3 = surf.call(th, q, n2)
				ps.append(pos)
				var tt := th + twist * (q.y - y0) / height
				var n0 := Vector3(cos(tt) * n2.x, n2.y, sin(tt) * n2.x)
				var nrm := n0
				if (lobes > 0 or not drops.is_empty() or absf(twist) > 0.01) and q.x > 0.0005 and sides < 3:
					var e := 0.002
					var dth: Vector3 = (surf.call(th + e, q, n2) as Vector3) - (surf.call(th - e, q, n2) as Vector3)
					var qa: Vector2 = pts[maxi(j - 1, a)]
					var qb: Vector2 = pts[mini(j + 1, b)]
					var ds: Vector3 = (surf.call(th, qb, n2) as Vector3) - (surf.call(th, qa, n2) as Vector3)
					var c := ds.cross(dth)
					if c.length() > 1e-12:
						nrm = c.normalized()
						if nrm.dot(n0) < 0.0:
							nrm = -nrm
				ns.append(nrm)
			rows.append([ps, ns])
		for j in range(a, b):
			var r0: Array = rows[j - a]
			var r1: Array = rows[j - a + 1]
			var v0 := arc[j] / total
			var v1 := arc[j + 1] / total
			var h0 := ((pts[j] as Vector2).y - y0) / height
			var h1 := ((pts[j + 1] as Vector2).y - y0) / height
			for i in around:
				var p00: Vector3 = (r0[0] as PackedVector3Array)[i]
				var p10: Vector3 = (r0[0] as PackedVector3Array)[i + 1]
				var p01: Vector3 = (r1[0] as PackedVector3Array)[i]
				var p11: Vector3 = (r1[0] as PackedVector3Array)[i + 1]
				var n00: Vector3 = (r0[1] as PackedVector3Array)[i]
				var n10: Vector3 = (r0[1] as PackedVector3Array)[i + 1]
				var n01: Vector3 = (r1[1] as PackedVector3Array)[i]
				var n11: Vector3 = (r1[1] as PackedVector3Array)[i + 1]
				if sides >= 3:
					# FLAT FACES: the face's own normal, kept facing out
					var mid := (n00 + n10 + n01 + n11)
					var fn := (p10 - p00).cross(p01 - p00)
					if fn.length() < 1e-12:
						fn = (p11 - p01).cross(p10 - p11)
					fn = fn.normalized() if fn.length() > 1e-12 else mid.normalized()
					if fn.dot(mid) < 0.0:
						fn = -fn
					n00 = fn
					n10 = fn
					n01 = fn
					n11 = fn
				var u0 := float(i) / float(around)
				var u1 := float(i + 1) / float(around)
				g.quad(p00, p10, p11, p01, n00, n10, n11, n01, Vector2(u0, v0), Vector2(u1, v0), Vector2(u1, v1), Vector2(u0, v1),
					h0, h0, h1, h1)
	g.girth = TAU * rmax * 100.0
	g.height = height * 100.0
	g.top = Vector3(0.0, (pts[-1] as Vector2).y, 0.0)
	# THE WICKS stand on the pool, as deep in it as it is where each stands (one in the middle is at
	# its bottom); a top with no pool has them on its top
	if bool(p.get("wick", false)):
		var R := rim.x if rim.x > 0.0 else maxf(rtop, 0.004)
		var sink := minf(0.005, rim.x * 0.3) if rim.x > 0.0 else 0.0
		var top_y := rim.y if rim.x > 0.0 else (pts[-1] as Vector2).y
		for sp in _wick_spots(p, R):
			var f := minf((sp as Vector2).length() / maxf(R, 1e-6), 1.0)
			g.wicks.append(Vector3((sp as Vector2).x, top_y - sink * (1.0 - f * f), (sp as Vector2).y))
	return g


## How far drip [param drops] push the wall out at angle [param th], [param yf] up the part (0..1):
## each a narrow ridge from the top down to its own length, ending in a bead.
static func _drip(drops: Array, th: float, yf: float) -> float:
	var out := 0.0
	for d in drops:
		var dd: Array = d
		var end := 1.0 - float(dd[2])
		if yf < end - 0.04:
			continue
		var dt := wrapf(th - float(dd[0]), -PI, PI)
		var across := exp(-pow(dt / float(dd[1]), 2.0))
		if across < 0.01:
			continue
		var s := (yf - end) / maxf(float(dd[2]), 0.01)
		var along := smoothstep(-0.08, 0.06, s) * (1.0 + 0.9 * exp(-pow(s / 0.07, 2.0)))
		out += float(dd[3]) * across * along
	return out


## A burned candle's top: the flat top of the wax melted into a shallow pool round the wick.
static func _pool(pts: Array) -> Array:
	if pts.size() < 3:
		return pts
	var last: Vector2 = pts[-1]
	var rim: Vector2 = pts[-2]
	if rim.x < 0.002 or absf(rim.y - last.y) > 0.002:
		return pts
	var depth := minf(0.005, rim.x * 0.3)
	var out := pts.slice(0, pts.size() - 1)
	for k in range(1, 7):
		var f := float(k) / 6.0
		var r := rim.x * (1.0 - f)
		out.append(Vector2(r, rim.y - depth * (1.0 - pow(r / rim.x, 2.0))))
	return out


## Points along a profile no further apart than [param step].
static func _subdivide(pts: Array, step: float) -> Array:
	var out: Array = [pts[0]]
	for i in range(1, pts.size()):
		var a: Vector2 = pts[i - 1]
		var b: Vector2 = pts[i]
		var n := maxi(1, ceili(a.distance_to(b) / step))
		for k in range(1, n + 1):
			out.append(a.lerp(b, float(k) / float(n)))
	return out


## A curve through [param pts] (Catmull-Rom), its ends kept.
static func _smooth2(pts: Array) -> Array:
	var out: Array = []
	var n := pts.size()
	for i in n - 1:
		var p0: Vector2 = pts[maxi(i - 1, 0)]
		var p1: Vector2 = pts[i]
		var p2: Vector2 = pts[i + 1]
		var p3: Vector2 = pts[mini(i + 2, n - 1)]
		var steps := 6
		for k in steps:
			var t := float(k) / float(steps)
			out.append(_cr(p0, p1, p2, p3, t))
	out.append(pts[n - 1])
	for i in out.size():
		out[i] = Vector2(maxf((out[i] as Vector2).x, 0.0), (out[i] as Vector2).y)
	return out


static func _cr(p0: Variant, p1: Variant, p2: Variant, p3: Variant, t: float) -> Variant:
	var t2 := t * t
	var t3 := t2 * t
	return 0.5 * ((2.0 * p1) + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)


## A BLOCK WITH ROUNDED EDGES, its base at y = 0: a cube's six faces, each a grid that crowds toward
## its edges, pushed onto the rounded shape (a point's nearest point on the inner box, plus the
## radius toward it), then narrowed toward the top by the taper.
static func _box(p: Dictionary) -> Tris:
	var s: Vector3 = (p["size"] as Vector3) * 0.01
	var h := s * 0.5
	var r := minf(float(p["round"]) * 0.01, minf(h.x, minf(h.y, h.z)))
	var taper := float(p["taper"])
	var g := Tris.new()
	var axis_samples := func(half: float) -> PackedFloat32Array:
		var out := PackedFloat32Array([-half])
		if r > 0.0001:
			for k in [0.15, 0.4, 0.7]:
				out.append(-half + r * (1.0 - cos(float(k) * PI * 0.5)) / (1.0 - cos(PI * 0.5)) * 1.0)
			out.append(-half + r)
			out.append(half - r)
			for k in [0.7, 0.4, 0.15]:
				out.append(half - r * (1.0 - cos(float(k) * PI * 0.5)))
		out.append(half)
		return out
	var inner := h - Vector3(r, r, r)
	var place := func(q: Vector3) -> Array:
		var c := q.clamp(-inner, inner)
		var d := q - c
		var nrm := d.normalized() if d.length() > 1e-9 else Vector3.ZERO
		var pos := c + nrm * r if d.length() > 1e-9 else q
		var yf := (pos.y + h.y) / maxf(s.y, 1e-6)
		var k := 1.0 - taper * yf
		pos = Vector3(pos.x * k, pos.y + h.y, pos.z * k)
		return [pos, nrm]
	# each face: its normal, and the two axes it spans
	var faces := [[Vector3.RIGHT, Vector3.BACK, Vector3.UP], [Vector3.LEFT, Vector3.FORWARD, Vector3.UP],
		[Vector3.UP, Vector3.RIGHT, Vector3.BACK], [Vector3.DOWN, Vector3.RIGHT, Vector3.FORWARD],
		[Vector3.BACK, Vector3.LEFT, Vector3.UP], [Vector3.FORWARD, Vector3.RIGHT, Vector3.UP]]
	for f in faces:
		var fn: Vector3 = f[0]
		var ua: Vector3 = f[1]
		var va: Vector3 = f[2]
		var us: PackedFloat32Array = axis_samples.call(absf(ua.dot(h)))
		var vs: PackedFloat32Array = axis_samples.call(absf(va.dot(h)))
		var grid: Array = []
		for j in vs.size():
			var row: Array = []
			for i in us.size():
				var q := fn * absf(fn.dot(h)) + ua * us[i] + va * vs[j]
				row.append(place.call(q))
			grid.append(row)
		for j in vs.size() - 1:
			for i in us.size() - 1:
				var c00: Array = grid[j][i]
				var c10: Array = grid[j][i + 1]
				var c11: Array = grid[j + 1][i + 1]
				var c01: Array = grid[j + 1][i]
				var n00: Vector3 = c00[1] if (c00[1] as Vector3) != Vector3.ZERO else fn
				var n10: Vector3 = c10[1] if (c10[1] as Vector3) != Vector3.ZERO else fn
				var n11: Vector3 = c11[1] if (c11[1] as Vector3) != Vector3.ZERO else fn
				var n01: Vector3 = c01[1] if (c01[1] as Vector3) != Vector3.ZERO else fn
				var uv := func(ii: int, jj: int) -> Vector2:
					return Vector2((us[ii] + absf(ua.dot(h))) / maxf(2.0 * absf(ua.dot(h)), 1e-6),
						(vs[jj] + absf(va.dot(h))) / maxf(2.0 * absf(va.dot(h)), 1e-6))
				var p00: Vector3 = c00[0]
				var p10: Vector3 = c10[0]
				var p11: Vector3 = c11[0]
				var p01: Vector3 = c01[0]
				g.quad(p00, p10, p11, p01, n00, n10, n11, n01, uv.call(i, j), uv.call(i + 1, j), uv.call(i + 1, j + 1),
					uv.call(i, j + 1), p00.y / s.y, p10.y / s.y, p11.y / s.y, p01.y / s.y)
	g.girth = (s.x + s.z) * 100.0
	g.height = s.y * 100.0
	g.top = Vector3(0.0, s.y, 0.0)
	return g


## A SPHERE stretched to [param size] (centimeters), resting on y = 0: smooth, LUMPY (pushed in and
## out by noise, as a stone or a fruit is) or FACETED (a coarse, jittered, flat-faced one, as a
## rough stone is).
static func _ball(size: Vector3, lumpy: float, facets: bool, rng: RandomNumberGenerator) -> Tris:
	var s := size * 0.01
	var rings := 6 if facets else 18
	var segs := 8 if facets else 32
	var noise := FastNoiseLite.new()
	noise.noise_type = FastNoiseLite.TYPE_SIMPLEX_SMOOTH
	noise.seed = rng.randi()
	noise.frequency = 1.6
	var jit: Array = []
	for j in rings + 1:
		var row: Array = []
		for i in segs + 1:
			row.append(Vector3(rng.randf_range(-1, 1), rng.randf_range(-1, 1), rng.randf_range(-1, 1)) * (0.12 if facets else 0.0))
		jit.append(row)
	var grid: Array = []
	for j in rings + 1:
		var row: Array = []
		var phi := PI * float(j) / float(rings)
		for i in segs + 1:
			var th := TAU * float(i % segs) / float(segs)
			var d := Vector3(sin(phi) * cos(th), -cos(phi), sin(phi) * sin(th))
			if facets and j > 0 and j < rings:
				d = (d + (jit[j][i % segs] as Vector3)).normalized()
			# a few broad lumps, as a tumbled stone or a fruit has, and a little unevenness over them: lumps
			# as fine as the second alone crimped the edge of a small stone like a dumpling's
			var k := 1.0 + lumpy * (0.42 * noise.get_noise_3dv(d * 0.45) + 0.05 * noise.get_noise_3dv(d * 1.6 + Vector3(5.0, 0.0, 0.0)))
			row.append(Vector3(d.x * s.x * 0.5, d.y * s.y * 0.5, d.z * s.z * 0.5) * k)
		grid.append(row)
	# resting on its lowest point
	var low := INF
	for row in grid:
		for q in row:
			low = minf(low, (q as Vector3).y)
	for row in grid:
		for i in (row as Array).size():
			row[i] = (row[i] as Vector3) - Vector3(0.0, low, 0.0)
	var g := Tris.new()
	var mid := Vector3(0.0, -low, 0.0)
	var hgt := maxf(s.y * 1.3, 0.0001)
	for j in rings:
		for i in segs:
			var p00: Vector3 = grid[j][i]
			var p10: Vector3 = grid[j][i + 1]
			var p11: Vector3 = grid[j + 1][i + 1]
			var p01: Vector3 = grid[j + 1][i]
			var ns: Array = []
			for q in [[j, i], [j, i + 1], [j + 1, i + 1], [j + 1, i]]:
				ns.append(_grid_normal(grid, int(q[0]), int(q[1]), segs, mid))
			if facets:
				var fn := ((p10 - p00).cross(p01 - p00) + (p11 - p01).cross(p10 - p11)).normalized()
				if fn.dot((p00 + p11) * 0.5 - mid) < 0.0:
					fn = -fn
				ns = [fn, fn, fn, fn]
			var u0 := float(i) / float(segs)
			var u1 := float(i + 1) / float(segs)
			var v0 := float(j) / float(rings)
			var v1 := float(j + 1) / float(rings)
			g.quad(p00, p10, p11, p01, ns[0], ns[1], ns[2], ns[3], Vector2(u0, v0), Vector2(u1, v0), Vector2(u1, v1), Vector2(u0, v1),
				p00.y / hgt, p10.y / hgt, p11.y / hgt, p01.y / hgt)
	g.girth = PI * (s.x + s.z) * 50.0
	g.height = s.y * 100.0
	g.top = Vector3(0.0, s.y, 0.0)
	return g


## A grid surface's normal at row [param j], column [param i] (columns wrap), facing away from
## [param mid].
static func _grid_normal(grid: Array, j: int, i: int, segs: int, mid: Vector3) -> Vector3:
	var rows := grid.size()
	var p: Vector3 = grid[j][i]
	var a: Vector3 = grid[j][(i + 1) % segs] - grid[j][(i - 1 + segs) % segs]
	var b: Vector3 = grid[mini(j + 1, rows - 1)][i] - grid[maxi(j - 1, 0)][i]
	var n := a.cross(b)
	if n.length() < 1e-12:
		n = p - mid
	n = n.normalized()
	if n.dot(p - mid) < 0.0:
		n = -n
	return n


## ONE CRYSTAL POINT into [param g]: a prism of [param sides], each face a little off the last,
## with a pyramidal termination whose apex is a little off center, placed by [param xf].
static func _crystal(g: Tris, radius: float, length: float, tip: float, sides: int, xf: Transform3D,
		rng: RandomNumberGenerator) -> void:
	var ring: Array = []
	var off := rng.randf() * TAU
	for i in sides:
		var a := off + TAU * float(i) / float(sides)
		var r := radius * rng.randf_range(0.88, 1.12)
		ring.append(Vector2(cos(a), sin(a)) * r)
	var apex := Vector3(rng.randf_range(-0.15, 0.15) * radius, length + tip, rng.randf_range(-0.15, 0.15) * radius)
	var top_y := length
	var total := maxf(length + tip, 0.0001)
	var axis_mid := xf * Vector3(0.0, total * 0.5, 0.0)
	for i in sides:
		var a: Vector2 = ring[i]
		var b: Vector2 = ring[(i + 1) % sides]
		var a0 := xf * Vector3(a.x, 0.0, a.y)
		var b0 := xf * Vector3(b.x, 0.0, b.y)
		var a1 := xf * Vector3(a.x, top_y, a.y)
		var b1 := xf * Vector3(b.x, top_y, b.y)
		var ap := xf * apex
		var fn := _face_n(a0, b0, a1, axis_mid)
		var u0 := float(i) / float(sides)
		var u1 := float(i + 1) / float(sides)
		g.quad(a0, b0, b1, a1, fn, fn, fn, fn, Vector2(u0, 0.0), Vector2(u1, 0.0), Vector2(u1, 0.7), Vector2(u0, 0.7),
			0.0, 0.0, top_y / total, top_y / total)
		var tn := _face_n(a1, b1, ap, axis_mid)
		g.tri(a1, b1, ap, tn, tn, tn, Vector2(u0, 0.7), Vector2(u1, 0.7), Vector2((u0 + u1) * 0.5, 1.0),
			Vector2(top_y / total, 0.0), Vector2(top_y / total, 0.0), Vector2(1.0, 0.0))
		var bn := (xf.basis * Vector3.DOWN).normalized()
		g.tri(xf * Vector3.ZERO, b0, a0, bn, bn, bn, Vector2(0.5, 0.0), Vector2(u1, 0.0), Vector2(u0, 0.0),
			Vector2.ZERO, Vector2.ZERO, Vector2.ZERO)


static func _face_n(a: Vector3, b: Vector3, c: Vector3, inside: Vector3) -> Vector3:
	var n := (b - a).cross(c - a)
	n = n.normalized() if n.length() > 1e-12 else Vector3.UP
	if n.dot((a + b + c) / 3.0 - inside) < 0.0:
		n = -n
	return n


## A CLUSTER: points growing out of a rough rock, the middle ones longest and most upright, the
## outer ones shorter and leaning out.
static func _cluster(p: Dictionary, rng: RandomNumberGenerator) -> Array:
	var R := float(p["radius"]) * 0.01
	var rock := _ball(Vector3(R * 200.0, R * 55.0, R * 170.0), 0.45, false, rng)
	var pts := Tris.new()
	var n := int(p["count"])
	var lens: Vector2 = p["length"]
	var spread := deg_to_rad(float(p["spread"]))
	for i in n:
		var rad := R * 0.62 * sqrt(rng.randf()) * (0.2 if i == 0 else 1.0)
		var ang := rng.randf() * TAU
		var out01 := rad / maxf(R * 0.62, 0.0001)
		var length := lerpf(lens.y, lens.x, clampf(out01 * 0.75 + rng.randf() * 0.35, 0.0, 1.0)) * 0.01
		var lean := spread * out01 + rng.randf_range(-0.12, 0.12)
		# out of the rock's own surface (an ellipsoid R x 0.55 R x 0.85 R resting on the cloth), a little sunk
		var e := sqrt(maxf(0.0, 1.0 - pow(cos(ang) * rad / R, 2.0) - pow(sin(ang) * rad / (R * 0.85), 2.0)))
		var base := Vector3(cos(ang) * rad, R * 0.275 * (1.0 + e) - R * 0.06, sin(ang) * rad)
		var axis := Vector3(-sin(ang), 0.0, cos(ang))
		var xf := Transform3D(Basis(axis, -lean) if axis.length() > 0.0 else Basis(), base)
		var thick := length * float(p["thickness"]) * rng.randf_range(0.8, 1.2)
		_crystal(pts, thick, length * 0.72, length * 0.28, 6, xf, rng)
	pts.measure()
	var base_mat := String(p.get("base", ""))
	if base_mat.is_empty():
		pts.append(rock)
		pts.measure()
		return [{"geo": pts}]
	return [{"geo": pts}, {"geo": rock, "material": base_mat}]


## A GEODE broken open, lying open side up: the lower half of a rough round rock, its cut face a band
## of rind round a hollow lined with crystal points growing in toward the middle - `[{geo}]` the
## lining, then the rock in its own material. The hollow follows the rock's lumps, so the rind is
## as thick all round as it was asked to be.
static func _geode(p: Dictionary, rng: RandomNumberGenerator) -> Array:
	var R := float(p["radius"]) * 0.01
	var hollow := R - float(p["rind"]) * 0.01
	var L := float(p["length"]) * 0.01
	var noise := FastNoiseLite.new()
	noise.noise_type = FastNoiseLite.TYPE_SIMPLEX_SMOOTH
	noise.seed = rng.randi()
	noise.frequency = 1.6
	var lump := func(d: Vector3) -> float:
		return 1.0 + 0.3 * noise.get_noise_3dv(d * 1.3)
	var rings := 10
	var segs := 32
	# half a lumpy ball, from the bottom up to the cut (at y = 0 until the whole is stood up); the
	# cut stays flat, since a lump there only moves the edge in or out
	var half := func(radius: float, wobble: float) -> Array:
		var grid: Array = []
		for j in rings + 1:
			var phi := PI * 0.5 * float(j) / float(rings)
			var row: Array = []
			for i in segs + 1:
				var th := TAU * float(i % segs) / float(segs)
				var d := Vector3(sin(phi) * cos(th), -cos(phi), sin(phi) * sin(th))
				row.append(d * radius * (float(lump.call(d)) + wobble * noise.get_noise_3dv(d * 3.1 + Vector3(9.0, 0.0, 0.0))))
			grid.append(row)
		return grid
	var outside: Array = half.call(R, 0.0)
	var inside: Array = half.call(hollow, 0.03)
	var rock := Tris.new()
	var lining := Tris.new()
	for j in rings:
		for i in segs:
			for side in [[outside, rock, 1.0], [inside, lining, -1.0]]:
				var grid: Array = side[0]
				var ps: Array = [grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]]
				var ns: Array = []
				for q in [[j, i], [j, i + 1], [j + 1, i + 1], [j + 1, i]]:
					ns.append(_grid_normal(grid, int(q[0]), int(q[1]), segs, Vector3.ZERO) * float(side[2]))
				var hs: Array = []
				for q in ps:
					hs.append(clampf(1.0 + (q as Vector3).y / R, 0.0, 1.0))
				(side[1] as Tris).quad(ps[0], ps[1], ps[2], ps[3], ns[0], ns[1], ns[2], ns[3],
					Vector2(float(i) / segs, float(j) / rings), Vector2(float(i + 1) / segs, float(j) / rings),
					Vector2(float(i + 1) / segs, float(j + 1) / rings), Vector2(float(i) / segs, float(j + 1) / rings), hs[0], hs[1], hs[2], hs[3])
	# THE CUT FACE: the rind, from the rock's edge in to the hollow's
	for i in segs:
		rock.quad(outside[rings][i], outside[rings][i + 1], inside[rings][i + 1], inside[rings][i], Vector3.UP, Vector3.UP,
			Vector3.UP, Vector3.UP, Vector2(float(i) / segs, 0.0), Vector2(float(i + 1) / segs, 0.0), Vector2(float(i + 1) / segs, 1.0),
			Vector2(float(i) / segs, 1.0), 1.0, 1.0, 1.0, 1.0)
	# THE CRYSTALS, spread evenly over the hollow (a spiral) from its bottom to a little under the
	# cut, each rooted in the wall and pointing in, toward a point under the middle of the cut
	var n := int(p["count"])
	var golden := PI * (3.0 - sqrt(5.0))
	for k in n:
		var yk := -1.0 + (float(k) + 0.5) / float(n) * 0.75
		var across := sqrt(maxf(0.0, 1.0 - yk * yk))
		var th := golden * float(k) + rng.randf_range(-0.3, 0.3)
		var d := Vector3(cos(th) * across, yk, sin(th) * across)
		var wall := d * hollow * float(lump.call(d))
		var aim := (Vector3(0.0, -hollow * 0.35, 0.0) - wall).normalized()
		aim = (aim + Vector3(rng.randf_range(-1.0, 1.0), rng.randf_range(-1.0, 1.0), rng.randf_range(-1.0, 1.0)) * 0.22).normalized()
		var length := L * rng.randf_range(0.55, 1.1)
		var basis := Basis(Quaternion(Vector3.UP, aim)) * Basis(Vector3.UP, rng.randf() * TAU)
		_crystal(lining, length * rng.randf_range(0.2, 0.3), length * 0.62, length * 0.38, 6,
			Transform3D(basis, wall - aim * length * 0.12), rng)
	var low := 0.0
	for row in outside:
		for q in row:
			low = minf(low, (q as Vector3).y)
	rock.lift(-low)
	lining.lift(-low)
	rock.measure()
	lining.measure()
	return [{"geo": lining}, {"geo": rock, "material": String(p["base"])}]


## A RING lying flat round the vertical axis, resting on y = 0: a torus, or part of one.
static func _ring(radius: float, thick: float, arc_deg: float) -> Tris:
	var g := Tris.new()
	var segs := clampi(roundi(arc_deg / 360.0 * 64.0), 8, 64)
	var tsegs := 12
	var arc := deg_to_rad(arc_deg)
	for i in segs:
		for j in tsegs:
			var corners: Array = []
			for q in [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]]:
				var a := arc * float(q[0]) / float(segs)
				var b := TAU * float(q[1]) / float(tsegs)
				var c := Vector3(cos(a), 0.0, sin(a))
				var n := c * cos(b) + Vector3.UP * sin(b)
				var pos := c * radius + n * thick + Vector3(0.0, thick, 0.0)
				corners.append([pos, n, Vector2(float(q[0]) / float(segs), float(q[1]) / float(tsegs)), pos.y / maxf(thick * 2.0, 1e-6)])
			g.quad(corners[0][0], corners[1][0], corners[2][0], corners[3][0], corners[0][1], corners[1][1], corners[2][1],
				corners[3][1], corners[0][2], corners[1][2], corners[2][2], corners[3][2], corners[0][3], corners[1][3],
				corners[2][3], corners[3][3])
	g.girth = arc * radius * 100.0
	g.height = thick * 200.0
	g.top = Vector3(0.0, thick * 2.0, 0.0)
	return g


## A ROD along a path: curved through its points (Catmull-Rom) unless told not to, its frame carried
## along without twisting, its radius following `radii` if given, closed at both ends.
static func _tube(p: Dictionary) -> Tris:
	var raw: Array = []
	for q in (p["path"] as Array):
		raw.append((q as Vector3) * 0.01)
	var radii: PackedFloat32Array = p.get("radii", PackedFloat32Array())
	var base_r := float(p["radius"]) * 0.01
	var path: Array = []
	var rs := PackedFloat32Array()
	if bool(p.get("smooth", true)) and raw.size() >= 3:
		for i in raw.size() - 1:
			for k in 8:
				var t := float(k) / 8.0
				path.append(_cr(raw[maxi(i - 1, 0)], raw[i], raw[i + 1], raw[mini(i + 2, raw.size() - 1)], t))
				rs.append(lerpf(_radius_at(radii, i, base_r), _radius_at(radii, i + 1, base_r), t))
		path.append(raw[-1])
		rs.append(_radius_at(radii, raw.size() - 1, base_r))
	else:
		path = raw
		for i in raw.size():
			rs.append(_radius_at(radii, i, base_r))
	var g := Tris.new()
	var sides := 12
	var n := path.size()
	var tans: Array = []
	for i in n:
		var d: Vector3 = (path[mini(i + 1, n - 1)] as Vector3) - (path[maxi(i - 1, 0)] as Vector3)
		tans.append(d.normalized() if d.length() > 1e-9 else Vector3.UP)
	var t0: Vector3 = tans[0]
	var nrm := t0.cross(Vector3.UP)
	if nrm.length() < 0.1:
		nrm = t0.cross(Vector3.RIGHT)
	nrm = nrm.normalized()
	var frames: Array = []
	for i in n:
		var t: Vector3 = tans[i]
		if i > 0:
			var prev: Vector3 = tans[i - 1]
			var ax := prev.cross(t)
			if ax.length() > 1e-6:
				nrm = nrm.rotated(ax.normalized(), prev.angle_to(t))
		nrm = (nrm - t * nrm.dot(t)).normalized()
		frames.append([nrm, t.cross(nrm).normalized()])
	var arc := PackedFloat32Array([0.0])
	for i in range(1, n):
		arc.append(arc[i - 1] + (path[i] as Vector3).distance_to(path[i - 1]))
	var total := maxf(arc[n - 1], 1e-6)
	var ymin := INF
	var ymax := -INF
	for q in path:
		ymin = minf(ymin, (q as Vector3).y)
		ymax = maxf(ymax, (q as Vector3).y)
	var hgt := maxf(ymax - ymin, 1e-4)
	var ringpt := func(i: int, k: int) -> Array:
		var a := TAU * float(k % sides) / float(sides)
		var f: Array = frames[i]
		var dir: Vector3 = (f[0] as Vector3) * cos(a) + (f[1] as Vector3) * sin(a)
		return [(path[i] as Vector3) + dir * rs[i], dir]
	for i in n - 1:
		for k in sides:
			var a: Array = ringpt.call(i, k)
			var b: Array = ringpt.call(i, k + 1)
			var c: Array = ringpt.call(i + 1, k + 1)
			var d: Array = ringpt.call(i + 1, k)
			var v0 := arc[i] / total
			var v1 := arc[i + 1] / total
			g.quad(a[0], b[0], c[0], d[0], a[1], b[1], c[1], d[1], Vector2(float(k) / sides, v0), Vector2(float(k + 1) / sides, v0),
				Vector2(float(k + 1) / sides, v1), Vector2(float(k) / sides, v1), ((a[0] as Vector3).y - ymin) / hgt,
				((b[0] as Vector3).y - ymin) / hgt, ((c[0] as Vector3).y - ymin) / hgt, ((d[0] as Vector3).y - ymin) / hgt)
	for end in [0, n - 1]:
		var t: Vector3 = tans[end] * (-1.0 if end == 0 else 1.0)
		var center: Vector3 = path[end]
		for k in sides:
			var a: Array = ringpt.call(end, k)
			var b: Array = ringpt.call(end, k + 1)
			var yf := (center.y - ymin) / hgt
			g.tri(center, a[0], b[0], t, t, t, Vector2(0.5, 0.5), Vector2(0, 0), Vector2(1, 0), Vector2(yf, 0), Vector2(yf, 0), Vector2(yf, 0))
	var rmax := 0.0
	for r in rs:
		rmax = maxf(rmax, r)
	g.girth = TAU * rmax * 100.0
	g.height = total * 100.0
	g.top = path[-1]
	return g


static func _radius_at(radii: PackedFloat32Array, i: int, fallback: float) -> float:
	if radii.is_empty():
		return fallback
	return maxf(radii[mini(i, radii.size() - 1)] * 0.01, 0.0002)


## A SHEET into [param g]: its outline (a named one, or `points`), its base at the origin and its
## length along +z, a little thickness, the tip curled up by `bend` and the sides lifted by
## `fold`; placed by [param xf].
static func _sheet(g: Tris, p: Dictionary, xf: Transform3D) -> void:
	var sz: Vector2 = (p["size"] as Vector2) * 0.01
	var bend := float(p.get("bend", 0.0))
	var fold := float(p.get("fold", 0.0))
	var thick := float(p.get("thickness", 0.08)) * 0.01
	var lift := func(x: float, z: float, w: float, l: float) -> float:
		var t := clampf(z / maxf(l, 1e-6), 0.0, 1.0)
		var s := clampf(x / maxf(w * 0.5, 1e-6), -1.0, 1.0)
		return bend * l * 0.35 * t * t + fold * w * 0.3 * s * s
	var pts: Array = p.get("points", [])
	if pts.size() >= 3:
		var poly := PackedVector2Array()
		var lo := Vector2(INF, INF)
		var hi := Vector2(-INF, -INF)
		for q in pts:
			var v := (q as Vector2) * 0.01
			poly.append(v)
			lo = lo.min(v)
			hi = hi.max(v)
		var tris := Geometry2D.triangulate_polygon(poly)
		var w := hi.x - lo.x
		var l := hi.y - lo.y
		var top := func(v: Vector2) -> Vector3:
			return Vector3(v.x, thick + float(lift.call(v.x - (lo.x + hi.x) * 0.5, v.y - lo.y, w, l)), v.y)
		for t in range(0, tris.size() - 2, 3):
			var a := poly[tris[t]]
			var b := poly[tris[t + 1]]
			var c := poly[tris[t + 2]]
			for side in [1.0, -1.0]:
				var pa: Vector3 = top.call(a) + Vector3(0, 0 if side > 0.0 else -thick, 0)
				var pb: Vector3 = top.call(b) + Vector3(0, 0 if side > 0.0 else -thick, 0)
				var pc: Vector3 = top.call(c) + Vector3(0, 0 if side > 0.0 else -thick, 0)
				var nn := (xf.basis * Vector3(0, side, 0)).normalized()
				var uva := (a - lo) / Vector2(maxf(w, 1e-6), maxf(l, 1e-6))
				var uvb := (b - lo) / Vector2(maxf(w, 1e-6), maxf(l, 1e-6))
				var uvc := (c - lo) / Vector2(maxf(w, 1e-6), maxf(l, 1e-6))
				g.tri(xf * pa, xf * pb, xf * pc, nn, nn, nn, uva, uvb, uvc, Vector2(1, 0) if side > 0.0 else Vector2.ZERO,
					Vector2(1, 0) if side > 0.0 else Vector2.ZERO, Vector2(1, 0) if side > 0.0 else Vector2.ZERO)
		# ITS EDGE, all round: a slab, not a sheet of paper standing off the cloth
		var mid := Vector2((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5)
		for i in poly.size():
			var a := poly[i]
			var b := poly[(i + 1) % poly.size()]
			var ta: Vector3 = top.call(a)
			var tb: Vector3 = top.call(b)
			var out := Vector2(b.y - a.y, a.x - b.x).normalized()
			if out.dot((a + b) * 0.5 - mid) < 0.0:
				out = -out
			var nn := (xf.basis * Vector3(out.x, 0.0, out.y)).normalized()
			g.quad(xf * (ta - Vector3(0, thick, 0)), xf * (tb - Vector3(0, thick, 0)), xf * tb, xf * ta, nn, nn, nn, nn,
				Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), Vector2(0, 1), 0.0, 0.0, 1.0, 1.0)
		return
	var outline := String(p.get("outline", "leaf"))
	var nl := 14
	var nw := 6
	var grid: Array = []
	# A FEATHER is not a leaf: one vane narrower than the other, and the whole of it curving a little
	# to the narrow side
	var narrow := 0.5 if outline == "feather" else 1.0
	var sweep := 0.06 if outline == "feather" else 0.0
	for j in nl + 1:
		var t := float(j) / float(nl)
		var hw := sz.x * 0.5 * _outline_w(outline, t)
		var row: Array = []
		for i in nw + 1:
			var s := -1.0 + 2.0 * float(i) / float(nw)
			var x := s * hw * (narrow if s < 0.0 else 1.0) - sweep * sz.y * sin(PI * t)
			var z := t * sz.y
			row.append(Vector3(x, thick + float(lift.call(x, z, sz.x, sz.y)), z))
		grid.append(row)
	# ITS EDGE: down each side and across the base and the tip
	var rim: Array = []
	for j in nl + 1:
		rim.append(grid[j][0])
	for j in range(nl, -1, -1):
		rim.append(grid[j][nw])
	for k in rim.size() - 1:
		var a: Vector3 = rim[k]
		var b: Vector3 = rim[k + 1]
		if a.distance_to(b) < 1e-6:
			continue
		var out := Vector3(b.z - a.z, 0.0, a.x - b.x).normalized()
		if out.dot((a + b) * 0.5 - Vector3(0.0, 0.0, sz.y * 0.5)) < 0.0:
			out = -out
		var nn := (xf.basis * out).normalized()
		g.quad(xf * (a - Vector3(0, thick, 0)), xf * (b - Vector3(0, thick, 0)), xf * b, xf * a, nn, nn, nn, nn,
			Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), Vector2(0, 1), 0.0, 0.0, 1.0, 1.0)
	for side in [1.0, -1.0]:
		for j in nl:
			for i in nw:
				var cs: Array = []
				for q in [[j, i], [j, i + 1], [j + 1, i + 1], [j + 1, i]]:
					var jj := int(q[0])
					var ii := int(q[1])
					var pos: Vector3 = grid[jj][ii]
					var a: Vector3 = grid[jj][mini(ii + 1, nw)] - grid[jj][maxi(ii - 1, 0)]
					var b: Vector3 = grid[mini(jj + 1, nl)][ii] - grid[maxi(jj - 1, 0)][ii]
					var nn := b.cross(a)
					nn = nn.normalized() if nn.length() > 1e-12 else Vector3.UP
					if nn.y < 0.0:
						nn = -nn
					if side < 0.0:
						pos -= Vector3(0, thick, 0)
						nn = -nn
					cs.append([xf * pos, (xf.basis * nn).normalized(), Vector2(float(ii) / nw, float(jj) / nl)])
				g.quad(cs[0][0], cs[1][0], cs[2][0], cs[3][0], cs[0][1], cs[1][1], cs[2][1], cs[3][1], cs[0][2], cs[1][2],
					cs[2][2], cs[3][2], 0.0, 0.0, 0.0, 0.0)


## A named outline's half-width at [param t] along its length, 0 at the base and 1 at the tip.
static func _outline_w(kind: String, t: float) -> float:
	match kind:
		"petal":
			return clampf(1.25 * sqrt(t) * pow(sin(PI * clampf(t * 0.92 + 0.04, 0.0, 1.0)), 0.55), 0.0, 1.0)
		"feather":
			return pow(sin(PI * pow(t, 0.75)), 0.45) * (0.85 + 0.15 * t)
		"oval":
			return sqrt(maxf(0.0, 1.0 - pow(2.0 * t - 1.0, 2.0)))
		"rect":
			return 1.0
	return pow(sin(PI * t), 0.9)


## A FLOWER HEAD: rings of petals round a small middle, the inner rings shorter and more upright.
static func _bloom(p: Dictionary, rng: RandomNumberGenerator) -> Tris:
	var g := Tris.new()
	var R := float(p["radius"])
	var cup := float(p["cup"])
	var layers := int(p["layers"])
	var n := int(p["petals"])
	for k in layers:
		var length := R * (1.0 - 0.22 * float(k))
		var tilt := deg_to_rad(lerpf(8.0, 70.0, cup) + float(k) * lerpf(10.0, 22.0, cup))
		for i in n:
			var yaw := TAU * (float(i) + 0.5 * float(k)) / float(n) + rng.randf_range(-0.08, 0.08)
			var petal := {"size": Vector2(length * float(p["width"]), length), "bend": 0.25 - cup * 0.6, "fold": 0.35,
				"thickness": 0.05, "outline": "petal", "points": []}
			var xf := Transform3D(Basis(Vector3.UP, -yaw + PI * 0.5) * Basis(Vector3.RIGHT, -tilt), Vector3(0.0, 0.002 * float(k) + R * 0.04, 0.0))
			_sheet(g, petal, xf)
	var heart := _ball(Vector3(R * 0.36, R * 0.24, R * 0.36), 0.3, false, rng)
	heart.lift(R * 0.01 * 0.06)
	g.append(heart)
	g.measure()
	return g


## AN OUTLINE RAISED STRAIGHT UP, resting on y = 0: its walls - round where the outline is round,
## crisp where it turns a corner - narrowed toward the top by the taper, its top edge cut back by
## the bevel, its top and its bottom. With a WALL it is hollow, as a tray is: a rim round an inner
## floor as thick as the wall, the inside's walls facing in. The inside is the outline drawn smaller
## about its middle, so a star tray's wall is thinner at its points than in its notches.
static func _extrude(p: Dictionary) -> Tris:
	var g := Tris.new()
	var pts := _outline2(p)
	if pts.size() < 3:
		return g
	var h := float(p["height"]) * 0.01
	var taper := float(p.get("taper", 0.0))
	var wall := float(p.get("wall", 0.0)) * 0.01
	var mid := Vector2.ZERO
	for q in pts:
		mid += q
	mid /= float(pts.size())
	var reach := 0.0
	for q in pts:
		reach += (q as Vector2).distance_to(mid)
	reach = maxf(reach / float(pts.size()), 0.001)
	var bevel := 0.0 if wall > 0.0 else minf(float(p.get("bevel", 0.0)) * 0.01, minf(h * 0.5, reach * 0.4))
	# a point of the outline at height y, drawn smaller about the middle by k and by the taper there
	var ring := func(k: float, y: float) -> Array:
		var s := k * (1.0 - taper * y / maxf(h, 1e-6))
		var out: Array = []
		for q in pts:
			var v: Vector2 = mid + ((q as Vector2) - mid) * s
			out.append(Vector3(v.x, y, v.y))
		return out
	var n := pts.size()
	# along the outline, for laying out a surface's pattern round it
	var along := PackedFloat32Array([0.0])
	for i in n:
		along.append(along[i] + (pts[i] as Vector2).distance_to(pts[(i + 1) % n]))
	var perim := maxf(along[n], 1e-6)
	if wall <= 0.0:
		_extrude_wall(g, ring.call(1.0, 0.0), ring.call(1.0, h - bevel), along, perim, h, 1.0)
		if bevel > 0.0:
			_extrude_wall(g, ring.call(1.0, h - bevel), ring.call(1.0 - bevel / reach, h), along, perim, h, 1.0)
		_extrude_face(g, ring.call(1.0 - bevel / reach, h), 1.0, h)
		_extrude_face(g, ring.call(1.0, 0.0), -1.0, h)
	else:
		var k_in := clampf(1.0 - wall / reach, 0.15, 0.95)
		var floor_y := minf(wall, h * 0.5)
		_extrude_wall(g, ring.call(1.0, 0.0), ring.call(1.0, h), along, perim, h, 1.0)
		_extrude_wall(g, ring.call(k_in, floor_y), ring.call(k_in, h), along, perim, h, -1.0)
		_extrude_rim(g, ring.call(1.0, h), ring.call(k_in, h), h)
		_extrude_face(g, ring.call(k_in, floor_y), 1.0, h)
		_extrude_face(g, ring.call(1.0, 0.0), -1.0, h)
	g.girth = perim * 100.0
	g.height = h * 100.0
	g.top = Vector3(mid.x, h, mid.y)
	return g


## An extrude's outline in meters about its own middle, wound one way (its area positive, so an
## edge's outside is to its right): a named one at its `size`, or its own `points`.
static func _outline2(p: Dictionary) -> Array:
	var out: Array = []
	var own: Array = p.get("points", [])
	var half: Vector2 = (p.get("size", Vector2(6.0, 6.0)) as Vector2) * 0.005
	var n := int(p.get("sides", 6))
	if own.size() >= 3:
		for q in own:
			out.append((q as Vector2) * 0.01)
	else:
		match String(p.get("outline", "polygon")):
			"rect":
				out = [Vector2(-half.x, -half.y), Vector2(half.x, -half.y), Vector2(half.x, half.y), Vector2(-half.x, half.y)]
			"circle":
				for i in 48:
					var a := TAU * float(i) / 48.0
					out.append(Vector2(cos(a) * half.x, sin(a) * half.y))
			"star":
				for i in n * 2:
					var a := -PI * 0.5 + PI * float(i) / float(n)
					var r := 1.0 if i % 2 == 0 else 0.48
					out.append(Vector2(cos(a) * half.x * r, sin(a) * half.y * r))
			"heart":
				# the heart curve, its point toward the reader
				for i in 48:
					var t := TAU * float(i) / 48.0
					var x := 16.0 * pow(sin(t), 3.0)
					var y := 13.0 * cos(t) - 5.0 * cos(2.0 * t) - 2.0 * cos(3.0 * t) - cos(4.0 * t)
					out.append(Vector2(x / 16.0 * half.x, -(y + 2.5) / 14.5 * half.y))
			_:
				for i in n:
					var a := -PI * 0.5 + TAU * float(i) / float(n)
					out.append(Vector2(cos(a) * half.x, sin(a) * half.y))
	var clean: Array = []
	for q in out:
		if clean.is_empty() or (q as Vector2).distance_to(clean[-1]) > 1e-5:
			clean.append(q)
	while clean.size() > 1 and (clean[0] as Vector2).distance_to(clean[-1]) <= 1e-5:
		clean.pop_back()
	var area := 0.0
	for i in clean.size():
		var a2: Vector2 = clean[i]
		var b2: Vector2 = clean[(i + 1) % clean.size()]
		area += a2.x * b2.y - b2.x * a2.y
	if area < 0.0:
		clean.reverse()
	return clean


## A band of wall from ring [param lo] up to ring [param hi] (each a point per outline point), its
## faces turned out ([param side] 1) or in (-1): smooth across a gentle turn of the outline, crisp
## at a corner.
static func _extrude_wall(g: Tris, lo: Array, hi: Array, along: PackedFloat32Array, perim: float, h: float, side: float) -> void:
	var n := lo.size()
	var faces: Array = []
	for i in n:
		var a: Vector3 = lo[i]
		var b: Vector3 = lo[(i + 1) % n]
		var c: Vector3 = hi[i]
		var out2 := Vector3(b.z - a.z, 0.0, a.x - b.x) * side
		var fn := (b - a).cross(c - a)
		if fn.length() < 1e-12:
			fn = out2
		fn = fn.normalized()
		if fn.dot(out2) < 0.0:
			fn = -fn
		faces.append(fn)
	for i in n:
		var j := (i + 1) % n
		var na: Vector3 = faces[i]
		var nb: Vector3 = faces[i]
		if (faces[(i - 1 + n) % n] as Vector3).dot(faces[i]) > cos(deg_to_rad(35.0)):
			na = ((faces[(i - 1 + n) % n] as Vector3) + (faces[i] as Vector3)).normalized()
		if (faces[j] as Vector3).dot(faces[i]) > cos(deg_to_rad(35.0)):
			nb = ((faces[j] as Vector3) + (faces[i] as Vector3)).normalized()
		var u0 := along[i] / perim
		var u1 := along[i + 1] / perim
		var a: Vector3 = lo[i]
		var b: Vector3 = lo[j]
		var c: Vector3 = hi[j]
		var d: Vector3 = hi[i]
		g.quad(a, b, c, d, na, nb, nb, na, Vector2(u0, a.y / maxf(h, 1e-6)), Vector2(u1, b.y / maxf(h, 1e-6)),
			Vector2(u1, c.y / maxf(h, 1e-6)), Vector2(u0, d.y / maxf(h, 1e-6)), a.y / maxf(h, 1e-6), b.y / maxf(h, 1e-6),
			c.y / maxf(h, 1e-6), d.y / maxf(h, 1e-6))


## A flat face over ring [param r], facing up ([param up] 1) or down (-1): the outline cut into
## triangles, in and out as it goes - or, an outline that crosses itself, fanned from its middle.
static func _extrude_face(g: Tris, r: Array, up: float, h: float) -> void:
	var flat := PackedVector2Array()
	var lo := Vector2(INF, INF)
	var hi := Vector2(-INF, -INF)
	for q in r:
		var v := Vector2((q as Vector3).x, (q as Vector3).z)
		flat.append(v)
		lo = lo.min(v)
		hi = hi.max(v)
	var span := (hi - lo).max(Vector2(1e-6, 1e-6))
	var nrm := Vector3(0.0, up, 0.0)
	var y := (r[0] as Vector3).y
	var hf := y / maxf(h, 1e-6)
	var tris := Geometry2D.triangulate_polygon(flat)
	if tris.is_empty():
		var c := Vector3.ZERO
		for q in r:
			c += q
		c /= float(r.size())
		for i in r.size():
			tris.append_array([-1, i, (i + 1) % r.size()])
		for t in range(0, tris.size(), 3):
			var pa: Vector3 = c if tris[t] < 0 else r[tris[t]]
			var pb: Vector3 = r[tris[t + 1]]
			var pc: Vector3 = r[tris[t + 2]]
			g.tri(pa, pb, pc, nrm, nrm, nrm, (Vector2(pa.x, pa.z) - lo) / span, (Vector2(pb.x, pb.z) - lo) / span,
				(Vector2(pc.x, pc.z) - lo) / span, Vector2(hf, 0.0), Vector2(hf, 0.0), Vector2(hf, 0.0))
		return
	for t in range(0, tris.size(), 3):
		var pa: Vector3 = r[tris[t]]
		var pb: Vector3 = r[tris[t + 1]]
		var pc: Vector3 = r[tris[t + 2]]
		g.tri(pa, pb, pc, nrm, nrm, nrm, (flat[tris[t]] - lo) / span, (flat[tris[t + 1]] - lo) / span,
			(flat[tris[t + 2]] - lo) / span, Vector2(hf, 0.0), Vector2(hf, 0.0), Vector2(hf, 0.0))


## A hollow extrude's rim: the band across its top from ring [param outer] in to ring [param inner].
static func _extrude_rim(g: Tris, outer: Array, inner: Array, h: float) -> void:
	var n := outer.size()
	for i in n:
		var j := (i + 1) % n
		g.quad(outer[i], outer[j], inner[j], inner[i], Vector3.UP, Vector3.UP, Vector3.UP, Vector3.UP,
			Vector2(float(i) / n, 0.0), Vector2(float(i + 1) / n, 0.0), Vector2(float(i + 1) / n, 1.0), Vector2(float(i) / n, 1.0),
			1.0, 1.0, 1.0, 1.0)


# --- footprints ------------------------------------------------------------------------------

## The convex outline, seen from above, of everything in [param pts] up to [param below] meters
## high - and of where the surface crosses that height, so a straight wall's outline is its own
## and not only its corners'.
static func _foot(pts: PackedVector3Array, below: float) -> PackedVector2Array:
	var flat := PackedVector2Array()
	for t in range(0, pts.size() - 2, 3):
		var tri := [pts[t], pts[t + 1], pts[t + 2]]
		for i in 3:
			var a: Vector3 = tri[i]
			var b: Vector3 = tri[(i + 1) % 3]
			if a.y <= below:
				flat.append(Vector2(a.x, a.z))
			if (a.y - below) * (b.y - below) < 0.0:
				var f := (below - a.y) / (b.y - a.y)
				var c := a.lerp(b, f)
				flat.append(Vector2(c.x, c.z))
	return Geometry2D.convex_hull(flat) if flat.size() >= 3 else PackedVector2Array()


## The convex outline of every point in [param pts] lower than [param below], seen from above.
static func _hull(pts: PackedVector3Array, below: float) -> PackedVector2Array:
	var flat := PackedVector2Array()
	for q in pts:
		if q.y <= below:
			flat.append(Vector2(q.x, q.z))
	return Geometry2D.convex_hull(flat) if flat.size() >= 3 else PackedVector2Array()


## A GROWING TRIANGLE LIST - positions, normals and both UV channels, three per triangle - with
## the part's measure (its girth and height in centimeters, for laying out ornament) and the top
## of it where a wick goes.
class Tris:
	extends RefCounted
	var v := PackedVector3Array()
	var n := PackedVector3Array()
	var uv := PackedVector2Array()
	var uv2 := PackedVector2Array()
	## Once placed, four floats a vertex: the middle of its copy (x, y, z) and the copy's own number
	## (0..1), which a pattern centers on and varies by ([method placed])
	var copy := PackedFloat32Array()
	## Where a candle's flames stand on it, in its own space
	var wicks := PackedVector3Array()
	var girth := 10.0
	var height := 10.0
	var top := Vector3.ZERO

	## One triangle, wound so its front faces the way its normals point (Godot draws clockwise
	## fronts): no builder's index arithmetic has to get the winding right.
	func tri(a: Vector3, b: Vector3, c: Vector3, na: Vector3, nb: Vector3, nc: Vector3, ta: Vector2, tb: Vector2,
			tc: Vector2, sa: Vector2, sb: Vector2, sc: Vector2) -> void:
		if (b - a).cross(c - a).dot(na + nb + nc) > 0.0:
			v.append_array([a, c, b])
			n.append_array([na, nc, nb])
			uv.append_array([ta, tc, tb])
			uv2.append_array([sa, sc, sb])
		else:
			v.append_array([a, b, c])
			n.append_array([na, nb, nc])
			uv.append_array([ta, tb, tc])
			uv2.append_array([sa, sb, sc])

	## A quad as two triangles; [param ha]..[param hd] are each corner's height up the part (0..1).
	func quad(a: Vector3, b: Vector3, c: Vector3, d: Vector3, na: Vector3, nb: Vector3, nc: Vector3, nd: Vector3,
			ta: Vector2, tb: Vector2, tc: Vector2, td: Vector2, ha: float, hb: float, hc: float, hd: float) -> void:
		tri(a, b, c, na, nb, nc, ta, tb, tc, Vector2(ha, 0), Vector2(hb, 0), Vector2(hc, 0))
		tri(a, c, d, na, nc, nd, ta, tc, td, Vector2(ha, 0), Vector2(hc, 0), Vector2(hd, 0))

	func append(o: Tris) -> void:
		v.append_array(o.v)
		n.append_array(o.n)
		uv.append_array(o.uv)
		uv2.append_array(o.uv2)
		copy.append_array(o.copy)

	func lift(dy: float) -> void:
		for i in v.size():
			v[i] = v[i] + Vector3(0.0, dy, 0.0)

	## Girth, height and top from the triangles themselves, for shapes made of pieces.
	func measure() -> void:
		if v.is_empty():
			return
		var box := AABB(v[0], Vector3.ZERO)
		for q in v:
			box = box.expand(q)
		girth = PI * (box.size.x + box.size.z) * 50.0
		height = box.size.y * 100.0
		top = Vector3(box.get_center().x, box.end.y, box.get_center().z)

	## Every copy of this geometry at [param xforms], as one - each vertex carrying its copy's middle
	## and number ([member copy]; [param ids] are the copies' places among the part's own).
	func placed(xforms: Array, ids: Array = []) -> Tris:
		var out := Tris.new()
		out.girth = girth
		out.height = height
		out.top = top
		var mid := Vector3.ZERO
		if not v.is_empty():
			var box := AABB(v[0], Vector3.ZERO)
			for q in v:
				box = box.expand(q)
			mid = box.get_center()
		for k in xforms.size():
			var t: Transform3D = xforms[k]
			var nb := t.basis.inverse().transposed()
			for i in v.size():
				out.v.append(t * v[i])
				out.n.append((nb * n[i]).normalized())
			out.uv.append_array(uv)
			out.uv2.append_array(uv2)
			var c := t * mid
			var block := PackedFloat32Array([c.x, c.y, c.z, fposmod(float(ids[k] if k < ids.size() else k) * 0.618034 + 0.137, 1.0)])
			while block.size() < v.size() * 4:
				block.append_array(block.duplicate())
			block.resize(v.size() * 4)
			out.copy.append_array(block)
		return out

	func mesh() -> ArrayMesh:
		var arrays := []
		arrays.resize(Mesh.ARRAY_MAX)
		arrays[Mesh.ARRAY_VERTEX] = v
		arrays[Mesh.ARRAY_NORMAL] = n
		arrays[Mesh.ARRAY_TEX_UV] = uv
		arrays[Mesh.ARRAY_TEX_UV2] = uv2
		var flags := 0
		if copy.size() == v.size() * 4:
			arrays[Mesh.ARRAY_CUSTOM0] = copy
			flags = Mesh.ARRAY_CUSTOM_RGBA_FLOAT << Mesh.ARRAY_FORMAT_CUSTOM0_SHIFT
		var m := ArrayMesh.new()
		m.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays, [], {}, flags)
		return m
