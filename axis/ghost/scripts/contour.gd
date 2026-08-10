extends RefCounted
class_name Contour

## Contour - isolines out of a sampled field, and the vocabulary a technical sheet is
## annotated with.
##
## Two halves that belong in one file because they only ever appear together. The first
## is MARCHING SQUARES: given a grid of scalars it returns, per level, an ordered list of
## polylines - not a soup of loose segments, which is the difference between something a
## pen can draw and something only a scatter plot can. The second is the ANNOTATION
## VOCABULARY - graticule ticks, a survey cross, a leader line with an elbow, a scale
## bar - because a contour plot is not a map. What makes a sheet read as a printed survey
## is the paperwork around the lines, and every one of those marks is a handful of
## strokes that a later cross-section or plan view wants in exactly the same form.
##
## ONE PASS OVER THE GRID, ALL LEVELS AT ONCE. The naive loop is level-major: for each of
## sixteen levels, walk every cell. That is sixteen full passes and, at a 144x81 grid,
## 186k cell visits to find the two or three levels that actually cross each cell. Here
## the cell is visited once, its four corners give a min and a max, and because the
## interval is FIXED the crossing levels are an arithmetic range - `floor((min-lo)/step)+1`
## through `floor((max-lo)/step)` - so only the levels genuinely present are tested. A
## smooth field puts one to three levels in a cell, which is an order of magnitude less
## work and is what lets the extraction sit inside a single worker job.
##
## STITCHING WITHOUT FLOAT KEYS. Segments are joined by EDGE IDENTITY, never by comparing
## endpoint coordinates: a crossing lies on a named grid edge (horizontal or vertical,
## indexed by the cell it starts at), so the join key is an integer and two cells sharing
## an edge agree on it exactly. The alternative - quantizing floats to a hash - fails at
## exactly the shallow gradients where a crossing sits arbitrarily close to a corner, and
## a contour that silently breaks into two polylines there is a contour that cannot be
## dashed, labelled or drawn with a consistent pen. Every segment is also emitted with
## the high ground on the same side, so each interior crossing is the END of exactly one
## segment and the START of exactly one other, and the walk is a lookup rather than a
## search.
##
## SMOOTHING IS NOT COSMETIC HERE. Marching squares on a grid whose cells are a dozen
## pixels across produces a visibly faceted line, and a faceted contour reads as a
## computer graphic rather than as ink. Two rounds of Chaikin corner cutting turn it into
## a curve for the cost of one linear pass, and a following simplify pass gives the
## points back by dropping everything that has become collinear. That pair is why a grid
## coarse enough to sample in a few tens of milliseconds still draws like a pen.

## The marching-squares case table, four entries per case: the two edges the first
## segment runs between, then the two of a second segment (-1 for none). Edges are
## numbered 0 top, 1 right, 2 bottom, 3 left, and every pair is directed so that the
## HIGH ground lies on the same side of the direction of travel - which is the property
## the stitch pass depends on, because it makes each interior crossing the end of exactly
## one segment and the start of exactly one other.
##
## The single-corner cases are the whole table and everything else follows from them: two
## adjacent corners are the union of two of them with the shared edge cancelled, three
## corners are the complement of one reversed, and the two DIAGONAL cases are ambiguous -
## the contour either wraps the two high corners separately or joins them through the
## middle. Rows 16 and 17 are those saddles resolved the other way, chosen at runtime on
## the cell's own centre value; a table with only one reading of a saddle produces
## isolines that cross each other, which no amount of smoothing will hide.
const CASES := [
	-1, -1, -1, -1,     # 0  nothing above
	3, 0, -1, -1,       # 1  corner 0
	0, 1, -1, -1,       # 2  corner 1
	3, 1, -1, -1,       # 3  corners 0,1
	1, 2, -1, -1,       # 4  corner 2
	3, 0, 1, 2,         # 5  saddle 0+2, centre low
	0, 2, -1, -1,       # 6  corners 1,2
	3, 2, -1, -1,       # 7  corners 0,1,2
	2, 3, -1, -1,       # 8  corner 3
	2, 0, -1, -1,       # 9  corners 0,3
	0, 1, 2, 3,         # 10 saddle 1+3, centre low
	2, 1, -1, -1,       # 11 corners 0,1,3
	1, 3, -1, -1,       # 12 corners 2,3
	1, 0, -1, -1,       # 13 corners 0,2,3
	0, 3, -1, -1,       # 14 corners 1,2,3
	-1, -1, -1, -1,     # 15 all above
	1, 0, 3, 2,         # 16 saddle 0+2, centre high
	0, 3, 2, 1,         # 17 saddle 1+3, centre high
]

## Working buffers for one extraction, one segment per slot: the two edge ids it joins,
## its level, and its two endpoints. They are members rather than locals because the
## stitch walk reads them from a helper, and members keep that to a plain indexed read.
var _sa := PackedInt32Array()
var _sb := PackedInt32Array()
var _sk := PackedInt32Array()
var _ax := PackedFloat32Array()
var _ay := PackedFloat32Array()
var _bx := PackedFloat32Array()
var _by := PackedFloat32Array()


## Extract every level from a grid of [param nx] x [param ny] samples. Level k is
## `lo + k * step`. Returns an Array of [param count] entries, each an Array of
## [PackedVector2Array] polylines in GRID coordinates (x in 0..nx-1, y in 0..ny-1); a
## closed loop repeats its first point at the end, which is how [method smooth] tells
## the two apart.
func extract(h: PackedFloat32Array, nx: int, ny: int,
		lo: float, step: float, count: int) -> Array:
	var out: Array = []
	out.resize(count)
	for k in count:
		out[k] = []
	if nx < 2 or ny < 2 or count <= 0 or step <= 0.000001 or h.size() < nx * ny:
		return out
	_sa.clear()
	_sb.clear()
	_sk.clear()
	_ax.clear()
	_ay.clear()
	_bx.clear()
	_by.clear()

	# The four edge crossings of the current cell, hoisted out of both loops: a fresh
	# local packed array per cell would be tens of thousands of allocations a pass.
	var ex := PackedFloat32Array()
	ex.resize(4)
	var ey := PackedFloat32Array()
	ey.resize(4)
	var eid := PackedInt32Array()
	eid.resize(4)

	for y in ny - 1:
		var r0 := y * nx
		var r1 := r0 + nx
		var fy := float(y)
		for x in nx - 1:
			var v0 := h[r0 + x]              # corner 0: (x, y)
			var v1 := h[r0 + x + 1]          # corner 1: (x+1, y)
			var v2 := h[r1 + x + 1]          # corner 2: (x+1, y+1)
			var v3 := h[r1 + x]              # corner 3: (x, y+1)
			var cmin := minf(minf(v0, v1), minf(v2, v3))
			var cmax := maxf(maxf(v0, v1), maxf(v2, v3))
			# The levels this cell can possibly carry, straight out of the fixed interval.
			var k0 := int(floor((cmin - lo) / step)) + 1
			var k1 := int(floor((cmax - lo) / step))
			if k0 < 0:
				k0 = 0
			if k1 > count - 1:
				k1 = count - 1
			if k1 < k0:
				continue
			var fx := float(x)
			# Edge ids. A horizontal edge is keyed by the cell it starts in, a vertical
			# edge likewise, and the two cells either side of an edge derive the same id.
			eid[0] = (r0 + x) * 2                # top,    between corners 0 and 1
			eid[1] = (r0 + x + 1) * 2 + 1        # right,  between corners 1 and 2
			eid[2] = (r1 + x) * 2                # bottom, between corners 3 and 2
			eid[3] = (r0 + x) * 2 + 1            # left,   between corners 0 and 3
			for k in range(k0, k1 + 1):
				var lv := lo + float(k) * step
				var c := 0
				if v0 >= lv:
					c |= 1
				if v1 >= lv:
					c |= 2
				if v2 >= lv:
					c |= 4
				if v3 >= lv:
					c |= 8
				if c == 0 or c == 15:
					continue
				var d := v1 - v0
				ex[0] = fx + (0.5 if absf(d) < 1e-9 else clampf((lv - v0) / d, 0.0, 1.0))
				ey[0] = fy
				d = v2 - v1
				ex[1] = fx + 1.0
				ey[1] = fy + (0.5 if absf(d) < 1e-9 else clampf((lv - v1) / d, 0.0, 1.0))
				d = v2 - v3
				ex[2] = fx + (0.5 if absf(d) < 1e-9 else clampf((lv - v3) / d, 0.0, 1.0))
				ey[2] = fy + 1.0
				d = v3 - v0
				ex[3] = fx
				ey[3] = fy + (0.5 if absf(d) < 1e-9 else clampf((lv - v0) / d, 0.0, 1.0))
				# A saddle takes its alternative reading (rows 16/17) when the cell's own
				# centre stands above the level.
				var ci := c
				if c == 5 or c == 10:
					if (v0 + v1 + v2 + v3) * 0.25 >= lv:
						ci = 16 if c == 5 else 17
				var row := ci * 4
				for si in 2:
					var ea: int = CASES[row + si * 2]
					if ea < 0:
						break
					var eb: int = CASES[row + si * 2 + 1]
					_sa.append(eid[ea])
					_sb.append(eid[eb])
					_sk.append(k)
					_ax.append(ex[ea])
					_ay.append(ey[ea])
					_bx.append(ex[eb])
					_by.append(ey[eb])

	var total := _sk.size()
	if total == 0:
		return out
	# Bucket the segments by level with a counting sort - one pass to count, one to
	# place. Appending into a per-level packed array instead would copy that array on
	# every append, since a packed array read out of an Array is a value.
	var counts := PackedInt32Array()
	counts.resize(count)
	counts.fill(0)
	for i in total:
		counts[_sk[i]] += 1
	var base := PackedInt32Array()
	base.resize(count)
	var acc := 0
	for k in count:
		base[k] = acc
		acc += counts[k]
	var cur := base.duplicate()
	var order := PackedInt32Array()
	order.resize(total)
	for i in total:
		var k := _sk[i]
		order[cur[k]] = i
		cur[k] += 1

	for k in count:
		var n := counts[k]
		if n == 0:
			continue
		var b0 := base[k]
		var b1 := b0 + n
		var start_of := {}
		var end_of := {}
		for j in range(b0, b1):
			var s := order[j]
			start_of[_sa[s]] = s
			end_of[_sb[s]] = s
		var used := {}
		var polys: Array = []
		# Open chains first: a segment whose start is nobody's end is a line that runs
		# off the edge of the grid, and it must be walked from that end or it would be
		# picked up mid-way and drawn as two pieces.
		for j in range(b0, b1):
			var s2 := order[j]
			if used.has(s2) or end_of.has(_sa[s2]):
				continue
			polys.append(_chain(s2, start_of, used))
		# Whatever is left is a closed loop, and closes itself: the walk ends on the
		# segment that leads back to where it started, whose far endpoint IS that start.
		for j in range(b0, b1):
			var s3 := order[j]
			if used.has(s3):
				continue
			polys.append(_chain(s3, start_of, used))
		out[k] = polys
	return out


# Follow one chain of segments from `s0` to wherever it ends, marking each visited.
func _chain(s0: int, start_of: Dictionary, used: Dictionary) -> PackedVector2Array:
	var pl := PackedVector2Array()
	var s := s0
	pl.append(Vector2(_ax[s], _ay[s]))
	while true:
		used[s] = true
		pl.append(Vector2(_bx[s], _by[s]))
		var nid := _sb[s]
		if not start_of.has(nid):
			break
		var nxt: int = start_of[nid]
		if used.has(nxt):
			break
		s = nxt
	return pl


## Chaikin corner cutting, [param iters] rounds. Closed polylines (first point equal to
## last) stay closed; open ones keep both endpoints pinned, so a line that runs off the
## sheet still meets the edge it left through.
static func smooth(lines: Array, iters: int) -> Array:
	var cur := lines
	for _r in maxi(0, iters):
		var nxt: Array = []
		for pl in cur:
			var p: PackedVector2Array = pl
			var n := p.size()
			if n < 3:
				nxt.append(p)
				continue
			var closed: bool = p[0].is_equal_approx(p[n - 1])
			var q := PackedVector2Array()
			if not closed:
				q.append(p[0])
			for i in n - 1:
				var a: Vector2 = p[i]
				var b: Vector2 = p[i + 1]
				var d := b - a
				q.append(a + d * 0.25)
				q.append(a + d * 0.75)
			if closed:
				q.append(q[0])
			else:
				q.append(p[n - 1])
			nxt.append(q)
		cur = nxt
	return cur


## Drop points that lie within [param eps] of the chord their neighbours span. Run after
## [method smooth]: cutting corners leaves long straight runs finely subdivided, and this
## hands those points back before anything has to draw them.
static func simplify(lines: Array, eps: float) -> Array:
	if eps <= 0.0:
		return lines
	var e2 := eps * eps
	var out: Array = []
	for pl in lines:
		var p: PackedVector2Array = pl
		var n := p.size()
		if n < 3:
			out.append(p)
			continue
		var keep := PackedVector2Array()
		keep.append(p[0])
		var a: Vector2 = p[0]
		for i in range(1, n - 1):
			var c: Vector2 = p[i]
			var b: Vector2 = p[i + 1]
			var ab := b - a
			var l2 := ab.length_squared()
			var d2 := 0.0
			if l2 < 1e-9:
				d2 = (c - a).length_squared()
			else:
				var t := clampf((c - a).dot(ab) / l2, 0.0, 1.0)
				d2 = (c - (a + ab * t)).length_squared()
			if d2 >= e2:
				keep.append(c)
				a = c
		keep.append(p[n - 1])
		out.append(keep)
	return out


## Draw polylines in grid coordinates onto [param tb], mapping grid to screen by
## `org + g * cell`. [param soft] carries [TriBatch]'s feather - on for a line that is
## the picture, off for bulk ruling where three quads per segment is not affordable.
static func stroke(tb: TriBatch, lines: Array, col: Color, width: float,
		org: Vector2, cell: Vector2, soft := true) -> void:
	for pl in lines:
		var p: PackedVector2Array = pl
		var n := p.size()
		if n < 2:
			continue
		var a := org + Vector2(p[0].x * cell.x, p[0].y * cell.y)
		for i in range(1, n):
			var b := org + Vector2(p[i].x * cell.x, p[i].y * cell.y)
			tb.line(a, b, col, width, soft)
			a = b


## Fill everything below [param level] - the water body, or any sub-level plate.
##
## Per cell, the region below the level is the cell polygon clipped by the same crossing
## points marching squares already knows how to find, so the shoreline follows the
## contour exactly rather than stepping cell by cell. The one case that needs care is the
## saddle, where the two below-corners are diagonal: walking the boundary there produces
## a bowtie, so it is emitted as the two separate triangles it actually is.
static func fill_below(tb: TriBatch, h: PackedFloat32Array, nx: int, ny: int,
		level: float, col: Color, org: Vector2, cell: Vector2) -> void:
	if nx < 2 or ny < 2 or h.size() < nx * ny:
		return
	var poly := PackedVector2Array()
	for y in ny - 1:
		var r0 := y * nx
		var r1 := r0 + nx
		var fy := float(y)
		for x in nx - 1:
			var v0 := h[r0 + x]
			var v1 := h[r0 + x + 1]
			var v2 := h[r1 + x + 1]
			var v3 := h[r1 + x]
			var b0 := v0 < level
			var b1 := v1 < level
			var b2 := v2 < level
			var b3 := v3 < level
			var below := int(b0) + int(b1) + int(b2) + int(b3)
			if below == 0:
				continue
			var fx := float(x)
			var c0 := org + Vector2(fx * cell.x, fy * cell.y)
			var c1 := org + Vector2((fx + 1.0) * cell.x, fy * cell.y)
			var c2 := org + Vector2((fx + 1.0) * cell.x, (fy + 1.0) * cell.y)
			var c3 := org + Vector2(fx * cell.x, (fy + 1.0) * cell.y)
			if below == 4:
				tb.quad(c0, c1, c2, c3, col)
				continue
			var e0 := c0.lerp(c1, _cut(v0, v1, level))
			var e1 := c1.lerp(c2, _cut(v1, v2, level))
			var e2 := c3.lerp(c2, _cut(v3, v2, level))
			var e3 := c0.lerp(c3, _cut(v0, v3, level))
			if below == 2 and b0 == b2 and b1 == b3:
				# Saddle: two opposite corners under water, connected only through the
				# cell's middle, which this resolution deliberately does not join.
				if b0:
					tb.tri(e3, c0, e0, col)
					tb.tri(e1, c2, e2, col)
				else:
					tb.tri(e0, c1, e1, col)
					tb.tri(e2, c3, e3, col)
				continue
			poly.clear()
			if b0:
				poly.append(c0)
			if b0 != b1:
				poly.append(e0)
			if b1:
				poly.append(c1)
			if b1 != b2:
				poly.append(e1)
			if b2:
				poly.append(c2)
			if b2 != b3:
				poly.append(e2)
			if b3:
				poly.append(c3)
			if b3 != b0:
				poly.append(e3)
			tb.poly(poly, col)


static func _cut(a: float, b: float, level: float) -> float:
	var d := b - a
	if absf(d) < 1e-9:
		return 0.5
	return clampf((level - a) / d, 0.0, 1.0)


## Fine diagonal ruling clipped to the band `[band_lo, band_hi)` of the same grid - the
## hatched band between two index contours.
##
## Hatching by clipping each ruled line against a polygon would mean building the band's
## polygon first; marching along the line and reading the grid gives the same picture for
## far less machinery. The march is coarse (about half a cell, which is a few pixels) and
## every run END is then refined by three bisections, so the band's edges land on the
## contour instead of on the march grid - the ragged alternative is the one thing that
## would immediately read as sampled rather than drawn.
static func hatch(tb: TriBatch, h: PackedFloat32Array, nx: int, ny: int,
		band_lo: float, band_hi: float, angle: float, spacing: float, march: float,
		col: Color, width: float, org: Vector2, cell: Vector2) -> void:
	if nx < 2 or ny < 2 or spacing <= 0.001 or march <= 0.001 or band_hi <= band_lo:
		return
	var d := Vector2(cos(angle), sin(angle))
	var nrm := Vector2(-d.y, d.x)
	var w := float(nx - 1)
	var hgt := float(ny - 1)
	# The grid's four corners projected onto both axes of the ruling frame.
	var dmin := INF
	var dmax := -INF
	var tmin := INF
	var tmax := -INF
	for i in 4:
		var p := Vector2(w if (i == 1 or i == 2) else 0.0, hgt if i >= 2 else 0.0)
		var dv := p.dot(nrm)
		var tv := p.dot(d)
		dmin = minf(dmin, dv)
		dmax = maxf(dmax, dv)
		tmin = minf(tmin, tv)
		tmax = maxf(tmax, tv)
	# Anchor the ruling to the grid origin rather than to dmin, so the pattern does not
	# crawl sideways when the band moves.
	var k0 := int(ceil(dmin / spacing))
	var k1 := int(floor(dmax / spacing))
	for k in range(k0, k1 + 1):
		var base := nrm * (float(k) * spacing)
		var t0 := tmin
		var t1 := tmax
		if absf(d.x) > 1e-6:
			var ta := (0.0 - base.x) / d.x
			var tb2 := (w - base.x) / d.x
			t0 = maxf(t0, minf(ta, tb2))
			t1 = minf(t1, maxf(ta, tb2))
		elif base.x < 0.0 or base.x > w:
			continue
		if absf(d.y) > 1e-6:
			var ta2 := (0.0 - base.y) / d.y
			var tb3 := (hgt - base.y) / d.y
			t0 = maxf(t0, minf(ta2, tb3))
			t1 = minf(t1, maxf(ta2, tb3))
		elif base.y < 0.0 or base.y > hgt:
			continue
		if t1 - t0 < march:
			continue
		var inside := false
		var run := t0
		var t := t0
		while t <= t1:
			var v := _samp(h, nx, ny, base + d * t)
			var ins := v >= band_lo and v < band_hi
			if ins and not inside:
				run = _refine(h, nx, ny, base, d, maxf(t0, t - march), t, band_lo, band_hi)
			elif inside and not ins:
				var e := _refine(h, nx, ny, base, d, t, maxf(t0, t - march), band_lo, band_hi)
				_rule(tb, base + d * run, base + d * e, col, width, org, cell)
			inside = ins
			t += march
		if inside:
			_rule(tb, base + d * run, base + d * t1, col, width, org, cell)


# Bisect between a t that is OUTSIDE the band and one that is inside, returning the
# crossing. Three rounds takes a half-cell march down to about a sixteenth of a cell,
# which at these grids is inside a pixel.
static func _refine(h: PackedFloat32Array, nx: int, ny: int, base: Vector2, d: Vector2,
		t_out: float, t_in: float, band_lo: float, band_hi: float) -> float:
	var a := t_out
	var b := t_in
	for _i in 3:
		var m := (a + b) * 0.5
		var v := _samp(h, nx, ny, base + d * m)
		if v >= band_lo and v < band_hi:
			b = m
		else:
			a = m
	return b


static func _rule(tb: TriBatch, a: Vector2, b: Vector2, col: Color, width: float,
		org: Vector2, cell: Vector2) -> void:
	# Hatching is the most numerous mark on the sheet and a printed rule is crisp, so it
	# goes down hard-edged: one quad a stroke instead of [TriBatch]'s three.
	tb.line(org + Vector2(a.x * cell.x, a.y * cell.y),
		org + Vector2(b.x * cell.x, b.y * cell.y), col, width, false)


static func _samp(h: PackedFloat32Array, nx: int, ny: int, p: Vector2) -> float:
	var x := clampf(p.x, 0.0, float(nx - 1) - 0.0001)
	var y := clampf(p.y, 0.0, float(ny - 1) - 0.0001)
	var xi := int(x)
	var yi := int(y)
	var fx := x - float(xi)
	var fy := y - float(yi)
	var r0 := yi * nx + xi
	var r1 := r0 + nx
	return lerpf(lerpf(h[r0], h[r0 + 1], fx), lerpf(h[r1], h[r1 + 1], fx), fy)


# ---------------------------------------------------------------------------------
# The annotation vocabulary. Everything below is strokes on a [TriBatch] in screen
# space, and every one of them is a mark a draughtsman would recognise by name.
# ---------------------------------------------------------------------------------

## A run of graticule ticks along the neatline from [param a] to [param b], each growing
## along [param inward]. Every [param every]-th tick is a long one, which is what gives a
## margin a countable rhythm instead of a comb.
static func graticule(tb: TriBatch, a: Vector2, b: Vector2, inward: Vector2, divs: int,
		minor: float, major: float, every: int, col: Color, w: float) -> void:
	if divs < 1:
		return
	var ev := maxi(1, every)
	for i in divs + 1:
		var p := a.lerp(b, float(i) / float(divs))
		tb.line(p, p + inward * (major if (i % ev) == 0 else minor), col, w, true)


## A survey cross: four strokes leaving a clean gap at the centre. The gap is the whole
## mark - a plus sign points at a place, a cross with a void in it points at a POINT, and
## that is the difference between a decoration and a station.
static func cross(tb: TriBatch, p: Vector2, r: float, gap: float,
		col: Color, w: float) -> void:
	var g := minf(gap, r * 0.7)
	tb.line(p + Vector2(-r, 0.0), p + Vector2(-g, 0.0), col, w, true)
	tb.line(p + Vector2(g, 0.0), p + Vector2(r, 0.0), col, w, true)
	tb.line(p + Vector2(0.0, -r), p + Vector2(0.0, -g), col, w, true)
	tb.line(p + Vector2(0.0, g), p + Vector2(0.0, r), col, w, true)


## A leader line: a diagonal away from the station, then a horizontal shelf for the label
## to sit on. Returns the far end of the shelf - the label's anchor.
static func leader(tb: TriBatch, p: Vector2, dir: Vector2, run: float, shelf: float,
		col: Color, w: float) -> Vector2:
	var knee := p + dir.normalized() * run
	var s := shelf if dir.x >= 0.0 else -shelf
	var end := knee + Vector2(s, 0.0)
	tb.line(p, knee, col, w, true)
	tb.line(knee, end, col, w, true)
	return end


## A scale bar: alternating filled and open cells on a baseline, with a tick at every
## division. [param paper] is the ground colour, painted into the open cells so the bar
## stays legible over hatching or water.
static func scale_bar(tb: TriBatch, org: Vector2, length: float, cells: int, h: float,
		ink: Color, paper: Color, w: float) -> void:
	var n := maxi(1, cells)
	var step := length / float(n)
	for i in n:
		var x0 := org.x + float(i) * step
		var x1 := x0 + step
		tb.quad(Vector2(x0, org.y), Vector2(x1, org.y),
			Vector2(x1, org.y + h), Vector2(x0, org.y + h),
			ink if (i % 2) == 0 else paper)
	for i in n + 1:
		var x := org.x + float(i) * step
		tb.line(Vector2(x, org.y - h * 0.55), Vector2(x, org.y), ink, w, true)
	tb.line(org, Vector2(org.x + length, org.y), ink, w, true)
	tb.line(Vector2(org.x, org.y + h), Vector2(org.x + length, org.y + h), ink, w, true)
	tb.line(org, Vector2(org.x, org.y + h), ink, w, true)
	tb.line(Vector2(org.x + length, org.y), Vector2(org.x + length, org.y + h), ink, w, true)


## A rectangle as four rules - the neatline, and any box drawn beside it.
static func box(tb: TriBatch, a: Vector2, b: Vector2, col: Color, w: float) -> void:
	tb.line(Vector2(a.x, a.y), Vector2(b.x, a.y), col, w, true)
	tb.line(Vector2(b.x, a.y), Vector2(b.x, b.y), col, w, true)
	tb.line(Vector2(b.x, b.y), Vector2(a.x, b.y), col, w, true)
	tb.line(Vector2(a.x, b.y), Vector2(a.x, a.y), col, w, true)
