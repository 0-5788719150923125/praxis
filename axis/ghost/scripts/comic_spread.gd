extends RefCounted
class_name ComicSpread

## ComicSpread - the two facing pages of an open comic book. What [ComicVehicle] flies over.
##
## A comic is not read a page at a time. It is read a SPREAD at a time: you open the book
## and two pages face you across the spine, and the leaf you turn carries a page on each of
## its sides. Drawing one page at a time was the first cut of this vehicle and it cost three
## things at once - a portrait sheet inside a landscape frame leaves a wedge of desk in shot
## however close the camera gets; a pan has only one page's width of content to travel
## across before it runs onto the trim edge; and a "page turn" of a single sheet has no
## hinge that means anything, because a real page hinges on a SPINE and the spine is the
## thing a single page does not have.
##
## THE SPREAD IS THE UNIT, therefore. Two [ComicPage] layouts side by side, sharing one
## print style, indexed as one reading order that runs the left page and then the right.
##
## SPREAD SPACE: x in [0, 2] with the spine at x = 1 (so the left page is x in [0, 1] and
## the right is x in [1, 2]), y in [0, aspect] and DOWN, exactly as [ComicPage] uses its own
## unit box. A page-local point becomes a spread point by adding the page's side index to x,
## and that is the whole mapping - which is why the pages can stay unaware they are in a
## spread at all.
##
## ONE PRINT STYLE ACROSS THE SPREAD, and that is not a simplification, it is what a printed
## book does. The gutter, the margin, the corner radius and the page proportion are
## properties of the EDITION; only the panel grid is a property of the page. Sampling them
## per page gave the left page rounded corners and the right page square ones, facing each
## other across two feet of paper, which reads as two different books rather than as one
## spread.

## How many pages face each other. Named rather than written as 2, because every mapping
## below is "+ side" and the reader deserves to know which 2 that is.
const PAGES := 2

## The most panels a spread can hold, and therefore how many render targets a pool needs.
## See [constant ComicVehicle.POOL].
const MAX_PANELS := ComicPage.MAX_PANELS * PAGES

## The print style, shared by both leaves. Sampled here and handed down, see the note above.
var aspect := 1.5
var gutter := 0.045
var margin := 0.06
var radius := 0.0

## The two [ComicPage] layouts, left first.
var pages: Array = []

## Every panel of the spread in READING ORDER - the left page's panels, then the right's.
## Each entry is `{rect: Rect2, cant: float, side: int, local: int}`, where `rect` is in
## SPREAD space and `side`/`local` say which page it came from and its index there.
var panels: Array = []


## Roll a spread. [param key] is folded into every draw, exactly as [ComicPage] does it, so
## the same session seed and spread index always produce the same two pages.
func _init(key: int) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = key
	aspect = rng.randf_range(ComicPage.ASPECT.x, ComicPage.ASPECT.y)
	gutter = rng.randf_range(ComicPage.GUTTER.x, ComicPage.GUTTER.y)
	margin = maxf(gutter, rng.randf_range(ComicPage.MARGIN.x, ComicPage.MARGIN.y))
	# Hard corners are the classic look, so weight toward zero - see ComicPage.RADIUS.
	radius = ComicPage.RADIUS.y * pow(rng.randf(), 2.2)
	var style := {"aspect": aspect, "gutter": gutter, "margin": margin, "radius": radius}
	for side in PAGES:
		# A SEPARATE DRAW PER PAGE, not a walk of one stream: the two grids must be
		# independent, or a spread whose left page happens to roll three rows biases the
		# right page toward matching it, and a book of symmetrical spreads is a book that
		# looks printed by a machine with one idea.
		var pg := ComicPage.new(hash([key, "spread-page", side]), style)
		pages.append(pg)
		for i in pg.panels.size():
			var p: Dictionary = pg.panels[i]
			var r: Rect2 = p["rect"]
			panels.append({
				"rect": Rect2(r.position + Vector2(float(side), 0.0), r.size),
				"cant": p["cant"],
				"side": side,
				"local": i,
			})


## The x of the spine in spread coordinates. The leaf hinges here and the mirror reflects
## about it, so it is named once rather than written as 1.0 in five places.
const SPINE := 1.0


## Which page panel [param i] is on: 0 for the left, 1 for the right.
func side_of(i: int) -> int:
	return int(panels[i]["side"])


## The global indices of every panel on page [param side], in reading order. What the leaf
## draw walks - a turning leaf carries ONE page, not the spread.
func page_panels(side: int) -> Array:
	var out: Array = []
	for i in panels.size():
		if int(panels[i]["side"]) == side:
			out.append(i)
	return out


## The four corners of page [param side] in spread coordinates, counter-clockwise from the
## top-left. The paper quad and the shadow walk these.
func page_corners(side: int) -> Array:
	var x := float(side)
	return [Vector2(x, 0.0), Vector2(x + 1.0, 0.0),
		Vector2(x + 1.0, aspect), Vector2(x, aspect)]


## The four corners of the WHOLE spread, for the covering solve.
func spread_corners() -> Array:
	return [Vector2(0, 0), Vector2(2, 0), Vector2(2, aspect), Vector2(0, aspect)]


## Reflect a spread point about the spine. This is what makes the leaf's BACK face work:
## the incoming left page, mirrored here and then rotated about the spine by pi, lands
## exactly back on itself.
##
##   mirror(q).x = 2 - q.x, so a left-page point q.x in [0, 1] becomes [1, 2]
##   rotating about the spine by pi maps world x -> -x, i.e. spread x -> 2 - x
##   composed: 2 - (2 - q.x) = q.x
##
## That identity is the whole reason the turn can be a real leaf with a printed back rather
## than a cross-dissolve: at the end of the swing the leaf IS the new left page, in exactly
## the place the new left page will be drawn flat the moment the turn ends, so there is
## nothing to blend and nothing that jumps.
static func mirror(p: Vector2) -> Vector2:
	return Vector2(2.0 * SPINE - p.x, p.y)


## Map a point in panel [param i]'s own unit square (0..1, y down) into SPREAD space, cant
## included. The textured grid walks this: the panel's UV grid IS its unit square, so the
## same call gives both the position and the texture coordinate.
func panel_point(i: int, uv: Vector2) -> Vector2:
	var p: Dictionary = panels[i]
	var r: Rect2 = p["rect"]
	var c := r.position + r.size * 0.5
	var ca: float = p["cant"]
	var v := r.position + r.size * uv - c
	if is_zero_approx(ca):
		return c + v
	var cs := cos(ca)
	var sn := sin(ca)
	return c + Vector2(v.x * cs - v.y * sn, v.x * sn + v.y * cs)


## The inverse of [method panel_point] for a point already in spread space: where
## [param p] falls in panel [param i]'s own unit square. Outside 0..1 means outside the
## panel, and by how much - which is what [method ComicVehicle._content_at] reads to decide
## whether the camera is looking at a picture or at the paper between pictures.
func panel_uv(i: int, p: Vector2) -> Vector2:
	var e: Dictionary = panels[i]
	var r: Rect2 = e["rect"]
	var c := r.position + r.size * 0.5
	var ca: float = e["cant"]
	var v := p - c
	if not is_zero_approx(ca):
		var cs := cos(-ca)
		var sn := sin(-ca)
		v = Vector2(v.x * cs - v.y * sn, v.x * sn + v.y * cs)
	return (v + r.size * 0.5) / Vector2(maxf(r.size.x, 1e-4), maxf(r.size.y, 1e-4))


## Panel [param i]'s four corners in spread space, cant applied. Counter-clockwise from the
## top-left, so a caller can walk them directly.
func corners(i: int) -> PackedVector2Array:
	var out := PackedVector2Array()
	for q in [Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), Vector2(0, 1)]:
		out.append(panel_point(i, q))
	return out


## Panel [param i]'s aspect (width / height). The vehicle sizes that panel's render target
## to it, so a scene is never squeezed or cropped to fit its frame.
func panel_aspect(i: int) -> float:
	var r: Rect2 = panels[i]["rect"]
	return r.size.x / maxf(r.size.y, 1e-4)


## The centre of panel [param i] in spread space - what the reading camera aims at.
func panel_center(i: int) -> Vector2:
	var r: Rect2 = panels[i]["rect"]
	return r.position + r.size * 0.5


## The middle of the spread - the point the aim is pulled toward so a shot on an outer panel
## keeps paper across the frame. See [constant ComicVehicle.AIM_PULL].
func center() -> Vector2:
	return Vector2(SPINE, aspect * 0.5)
