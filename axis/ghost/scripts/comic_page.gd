extends RefCounted
class_name ComicPage

## ComicPage - one seeded page of comic panels. The layout half of [ComicVehicle].
##
## Cattle, not pets, applied to page design: a page is not a hand-drawn template picked
## from a list of six, it is a RECIPE - a row split, a per-row column split, gutters, a
## corner radius, a cant - whose numbers are all sampled from ranges around sensible
## centres. So the show does not cycle through the same half-dozen grids; it draws from
## the space of comic pages, and a given song always draws the same ones.
##
## The page lives in PAGE SPACE: x and y both in [-1, 1], y DOWN, so a panel rect is
## directly usable as the u/v of a quad on the page plane and needs no aspect handed to
## it. [member aspect] is how tall the page is relative to its width, and panel rects are
## expressed in that same normalized box - the vehicle scales by it when it places the
## page in the world.
##
## WHY THE GRID IS ROWS-THEN-COLUMNS and not a free-form packing: a comic page reads in
## rows. A layout engine that can place a panel anywhere produces pages with no reading
## order, which is the one thing a comic page must have. Rows of varying height, split
## into a varying number of columns of varying width, is the whole classic vocabulary -
## including the full-width establishing panel (a row of one) and the tall splash.

## A page is between these many panels. Under three there is barely a page; over six the
## panels are too small for a scene to read at the wide shot.
const MIN_PANELS := 3
const MAX_PANELS := 6

## Page proportion (height / width). Around the American comic's 1.5, sampled.
const ASPECT := Vector2(1.34, 1.60)
## Gutter between panels, in page-width fractions. Comics run wider gutters than people
## remember; too tight and the panels read as one grid rather than as separate moments.
const GUTTER := Vector2(0.030, 0.062)
## Margin between the outermost panels and the page edge. Always at least the gutter,
## because a panel touching the trim reads as a printing error rather than a bleed.
const MARGIN := Vector2(0.045, 0.085)
## Corner radius, in units of the panel's shorter side. 0 for a page that wants hard
## corners - which most classic pages do, so the low end is weighted.
const RADIUS := Vector2(0.0, 0.055)
## How far a canted panel is rotated off square, in radians. A couple of degrees; this is
## a page with a tilted panel on it, not a scrapbook.
const CANT := 0.045
## The widest and narrowest a panel may be allowed to get. Outside this a subject sized
## to the panel's shorter side is cropped through, so the roll is retried.
const PANEL_ASPECT := Vector2(0.42, 2.6)

## Height / width of the page, sampled.
var aspect := 1.5
## Gutter and margin actually rolled, in page-width fractions.
var gutter := 0.045
var margin := 0.06
## Corner radius in units of the shorter panel side.
var radius := 0.0
## The panels, in READING ORDER. Each is `{rect: Rect2, cant: float}` where `rect` is in
## the normalized page box (x, y both 0..1, y down) and `cant` is its rotation off square.
var panels: Array = []


## Roll a page. [param key] is folded into every draw, so the same session seed and page
## index always produce the same page - and two different pages of one session are
## independent rolls rather than a walk of one stream.
func _init(key: int) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = key
	aspect = rng.randf_range(ASPECT.x, ASPECT.y)
	gutter = rng.randf_range(GUTTER.x, GUTTER.y)
	margin = maxf(gutter, rng.randf_range(MARGIN.x, MARGIN.y))
	# Hard corners are the classic look, so weight toward zero rather than sampling flat:
	# a squared roll pulls the mass down and leaves the rounded pages as the minority.
	radius = RADIUS.y * pow(rng.randf(), 2.2)
	_roll_grid(rng)


# Rows first, then columns inside each row. Retried until every panel's aspect is inside
# PANEL_ASPECT: it is far simpler to reject a bad roll than to constrain the split so it
# cannot happen, and the rejection rate is low enough not to matter (a handful of tries).
func _roll_grid(rng: RandomNumberGenerator) -> void:
	for _attempt in 24:
		panels = _try_grid(rng)
		if not panels.is_empty():
			return
	# Every roll was rejected - fall back to an even 2x2, which cannot be out of range.
	panels = _try_grid_fixed(2, [2, 2])


func _try_grid(rng: RandomNumberGenerator) -> Array:
	var want := rng.randi_range(MIN_PANELS, MAX_PANELS)
	# Split `want` panels into rows. Prefer 2-3 rows; a 4-row page of singles is a strip,
	# not a page.
	var rows := clampi(int(round(sqrt(float(want)))), 2, 3)
	var per_row: Array = []
	var left := want
	for r in rows:
		var remaining := rows - r - 1
		# leave at least one panel for each row still to come, and take at most three
		var hi := mini(3, left - remaining)
		var lo := 1
		if hi < lo:
			return []
		per_row.append(rng.randi_range(lo, hi))
		left -= int(per_row[r])
	if left != 0:
		per_row[rows - 1] = int(per_row[rows - 1]) + left
		if int(per_row[rows - 1]) < 1 or int(per_row[rows - 1]) > 3:
			return []
	return _lay_out(per_row, rng)


func _try_grid_fixed(_rows: int, per_row: Array) -> Array:
	var rng := RandomNumberGenerator.new()
	rng.seed = 1
	return _lay_out(per_row, rng)


# Turn a per-row panel count into rects. Row heights and, inside a row, column widths are
# each a normalized random split - so a page gets a tall row over a short one, and a wide
# panel beside a narrow one, without either being authored.
func _lay_out(per_row: Array, rng: RandomNumberGenerator) -> Array:
	var rows: int = per_row.size()
	var inner_w := 1.0 - 2.0 * margin
	var inner_h := aspect - 2.0 * margin
	var row_h := _split(rows, rng, 0.62)
	var out: Array = []
	var y := margin
	for r in rows:
		var h: float = (inner_h - gutter * float(rows - 1)) * row_h[r]
		var cols: int = int(per_row[r])
		var col_w := _split(cols, rng, 0.55)
		var x := margin
		for c in cols:
			var w: float = (inner_w - gutter * float(cols - 1)) * col_w[c]
			# reject the whole roll if any panel would crop a subject through
			var a := w / maxf(h, 1e-4)
			if a < PANEL_ASPECT.x or a > PANEL_ASPECT.y:
				return []
			# a cant is rare and small - about one panel in seven, and only when the page
			# has room for it (a panel on the trim edge would rotate off the paper)
			var canted := rng.randf() < 0.14 \
				and x > margin * 0.6 and y > margin * 0.6 \
				and x + w < 1.0 - margin * 0.6 and y + h < aspect - margin * 0.6
			out.append({
				"rect": Rect2(x, y, w, h),
				"cant": (rng.randf_range(-CANT, CANT) if canted else 0.0),
			})
			x += w + gutter
		y += h + gutter
	return out


# `n` fractions summing to 1, each at least `even_floor` of an even share - so a split is
# visibly uneven but never degenerate (a 4% sliver of a panel is not a panel).
func _split(n: int, rng: RandomNumberGenerator, even_floor: float) -> Array:
	if n <= 1:
		return [1.0]
	var raw: Array = []
	var total := 0.0
	for _i in n:
		var v := rng.randf_range(even_floor, 2.0 - even_floor)
		raw.append(v)
		total += v
	var out: Array = []
	for v in raw:
		out.append(float(v) / total)
	return out


## The panel's four corners in the normalized page box, with its cant applied about its
## own centre. Counter-clockwise from the top-left, so a caller can walk them directly.
func corners(i: int) -> PackedVector2Array:
	var p: Dictionary = panels[i]
	var r: Rect2 = p["rect"]
	var c := r.position + r.size * 0.5
	var ca: float = p["cant"]
	var cs := cos(ca)
	var sn := sin(ca)
	var out := PackedVector2Array()
	for q in [Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), Vector2(0, 1)]:
		var v: Vector2 = r.position + r.size * q - c
		out.append(c + Vector2(v.x * cs - v.y * sn, v.x * sn + v.y * cs))
	return out


## Map a point in a panel's own unit square (0..1, y down) into the normalized page box,
## cant included. This is what the textured grid walks: the panel's UV grid IS its unit
## square, so the same call gives both the position and (trivially) the texture coord.
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


## Panel [param i]'s aspect (width / height). The vehicle sizes that panel's render
## target to it, so a scene is never squeezed or cropped to fit its frame.
func panel_aspect(i: int) -> float:
	var r: Rect2 = panels[i]["rect"]
	return r.size.x / maxf(r.size.y, 1e-4)


## The centre of panel [param i] in the normalized page box - what the reading camera
## aims at.
func panel_center(i: int) -> Vector2:
	var r: Rect2 = panels[i]["rect"]
	return r.position + r.size * 0.5
