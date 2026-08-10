extends RefCounted
class_name PlateField

## PlateField - the standing-wave field of a driven plate, baked onto a grid.
##
## The physical half of a Chladni figure. A plate driven at one of its resonances does
## not move uniformly: it splits into cells that heave in opposite phase, separated by
## NODAL LINES where the surface is still. Sand poured onto the plate is thrown off the
## moving cells and accumulates on the still lines, so the sand ends up drawing the
## mode's own geometry. This class owns the field w(x, y) whose zero set those lines
## are, plus the gradient of |w| - which is the only thing a grain needs to know, since
## it says which way "quieter" is.
##
## Plate coordinates are -1..1 on both axes, whatever the plate's outline; a caller
## converts to pixels itself.
##
## WHY IT IS BAKED. A grain asks the field where to walk on every simulation step, and
## there are thousands of grains. Evaluating the closed form per grain costs four cos()
## calls for the gradient alone - around 160k trig calls a tick at 20k grains, which is
## not a budget any scene has. So the field is evaluated ONCE onto a res x res grid
## together with its gradient, and a grain step becomes one bilinear read: a dozen array
## lookups and a handful of multiplies.
##
## WHY IT IS BAKED IN SLABS. One full bake of a 72x72 grid is roughly 130k interpreted
## operations - about 10 ms, which would be a visible hitch every time the music changed
## the mode. [method advance] therefore does a few ROWS per call and the previous figure
## stays on screen until the new one is finished. Two grids are kept for exactly that
## reason and [member blend] crossfades between them. Because differentiation is LINEAR,
## lerping the two amplitude grids and lerping their two gradient grids gives exactly the
## gradient of the lerped field - the crossfade is consistent, not an approximation. (The
## same is emphatically not true of |w| itself, which is why the grids store |w| and its
## gradient rather than signed w.)
##
## THE FAMILIES. Three plates, each with its own eigenfunctions, and a rim condition that
## changes the figure completely rather than decorating it:
##   square - the classic. Free rim: cos(m pi x)cos(n pi y) - cos(n pi x)cos(m pi y),
##            Chladni's own plate, which is where the crosses, stars and grids come from.
##            Clamped rim: the same antisymmetric pair built from sin, pinning the edge
##            and pulling the structure inward. m must differ from n or the antisymmetric
##            combination is identically zero - the caller guarantees that.
##   circle - J_m(k r) cos(m theta). The rim condition picks the radial wavenumber from
##            one of two zero tables: zeros of J_m for a clamped rim (the rim is itself a
##            nodal circle) or zeros of J_m' for a free one (the rim is an antinode, so
##            sand never rests there). A genuinely free circular plate also carries a
##            modified-Bessel term this does not model; what is here is the right FAMILY
##            of figures, not a plate solver.
##   hex    - no closed-form eigenfunction exists, but three plane waves at 60 degrees
##            summed IS an exact Helmholtz solution with sixfold symmetry, and a second
##            triple rotated 30 degrees carries the other mode index. A quarter-period
##            phase shift turns its honeycomb of nodes into the triangular lattice, which
##            is what "clamped" means for this plate.

## Highest angular order the circular zero tables carry.
const MAX_ANG := 4
## Radial samples in the circular plate's Bessel lookup. 192 is well past what a 72-cell
## grid can resolve, and the table is built once per bake, so it is free accuracy.
const RAD := 192

## Zeros of J_m - the CLAMPED circular plate, whose held rim is a nodal circle.
const J_ZEROS := [
	[2.4048, 5.5201, 8.6537, 11.7915],
	[3.8317, 7.0156, 10.1735, 13.3237],
	[5.1356, 8.4172, 11.6198, 14.7960],
	[6.3802, 9.7610, 13.0152, 16.2235],
	[7.5883, 11.0647, 14.3725, 17.6160],
]
## Zeros of J_m' - the FREE rim, an antinode instead: the sand is flung off the edge.
const JP_ZEROS := [
	[3.8317, 7.0156, 10.1735, 13.3237],
	[1.8412, 5.3314, 8.5363, 11.7060],
	[3.0542, 6.7061, 9.9695, 13.1704],
	[4.2012, 8.0152, 11.3459, 14.5858],
	[5.3175, 9.2824, 12.6819, 15.9641],
]

## Grid resolution per axis.
var res := 72
## "square" / "circle" / "hex".
var shape := "square"
## "free" / "clamped".
var boundary := "free"
## 0 = the previous figure, 1 = the current one. Held at 0 while a bake is in progress,
## so a half-written grid is never visible.
var blend := 1.0

var _shape_id := 0                   # 0 square, 1 circle, 2 hex - the hot-path form
var _ang0 := 0.0                     # seeded orientation of the wave family
var _cells := 0

var _a0 := PackedFloat32Array()      # previous figure: |w|, d|w|/dx, d|w|/dy
var _gx0 := PackedFloat32Array()
var _gy0 := PackedFloat32Array()
var _a1 := PackedFloat32Array()      # current figure
var _gx1 := PackedFloat32Array()
var _gy1 := PackedFloat32Array()
var _raw := PackedFloat32Array()     # un-normalised |w|, the gradient pass's input

# Mode-independent per-cell geometry, built once.
var _rr := PackedFloat32Array()      # circle: radius 0..1
var _th := PackedFloat32Array()      # circle: angle, already offset by _ang0
var _dots: Array = []                # hex: six PackedFloat32Array of (p . u_k)

# Bake state.
var _phase := 0                      # 0 idle, 1 field rows, 2 gradient rows
var _row := 0
var _amax := 1.0
var _m := 1
var _n := 2
var _m2 := 1
var _n2 := 2
var _w2 := 0.0
var _t1 := PackedFloat32Array()      # square: the four separable 1D wave tables
var _t2 := PackedFloat32Array()
var _t3 := PackedFloat32Array()
var _t4 := PackedFloat32Array()
var _lut1 := PackedFloat32Array()    # circle: J_m(k r) sampled radially
var _lut2 := PackedFloat32Array()
var _k1 := 0.0                       # hex: the four plane-wave numbers
var _k2 := 0.0
var _k3 := 0.0
var _k4 := 0.0
var _hex_ph := 0.0


func _init(shape_ := "square", boundary_ := "free", res_ := 72, base_angle := 0.0) -> void:
	shape = shape_
	boundary = boundary_
	_ang0 = base_angle
	res = clampi(res_, 24, 160)
	_cells = res * res
	_shape_id = 1 if shape == "circle" else (2 if shape == "hex" else 0)
	_hex_ph = PI * 0.5 if boundary == "clamped" else 0.0
	_a0.resize(_cells)
	_gx0.resize(_cells)
	_gy0.resize(_cells)
	_a1.resize(_cells)
	_gx1.resize(_cells)
	_gy1.resize(_cells)
	_raw.resize(_cells)
	_build_geometry()


# The per-cell quantities that do NOT depend on the mode: polar coordinates for the
# circular plate, the six plane-wave projections for the hexagonal one. Paid once at
# scene build so a bake is pure table lookup afterwards.
func _build_geometry() -> void:
	var inv := 2.0 / float(res - 1)
	if _shape_id == 1:
		_rr.resize(_cells)
		_th.resize(_cells)
		for j in res:
			var y := -1.0 + float(j) * inv
			for i in res:
				var x := -1.0 + float(i) * inv
				var o := j * res + i
				_rr[o] = minf(1.0, sqrt(x * x + y * y))
				_th[o] = atan2(y, x) - _ang0
	elif _shape_id == 2:
		for k in 6:
			# Two triples: 0/60/120 degrees, and the same rotated by 30. Between them
			# they carry the plate's two mode indices.
			var a := _ang0 + (PI / 3.0) * float(k % 3) + (0.0 if k < 3 else PI / 6.0)
			var d := Vector2(cos(a), sin(a))
			var arr := PackedFloat32Array()
			arr.resize(_cells)
			for j in res:
				var y := -1.0 + float(j) * inv
				for i in res:
					arr[j * res + i] = (-1.0 + float(i) * inv) * d.x + y * d.y
			_dots.append(arr)


## Is this plate coordinate on the plate at all? The grain walk asks per step, so it is
## an analytic test rather than a grid lookup.
func inside(x: float, y: float) -> bool:
	if _shape_id == 1:
		return x * x + y * y <= 1.0
	if _shape_id == 2:
		# Regular hexagon, circumradius 1, vertices on the axes at 0 / 60 / ... degrees.
		var ax := absf(x)
		var ay := absf(y)
		return ay <= 0.8660254 and 0.8660254 * ax + 0.5 * ay <= 0.8660254
	return absf(x) <= 1.0 and absf(y) <= 1.0


## The plate's silhouette in plate coordinates, for the scene to draw.
func outline(steps := 64) -> PackedVector2Array:
	var out := PackedVector2Array()
	if _shape_id == 1:
		var n := maxi(12, steps)
		for i in n:
			var a := TAU * float(i) / float(n)
			out.append(Vector2(cos(a), sin(a)))
	elif _shape_id == 2:
		for i in 6:
			var a := TAU * float(i) / 6.0
			out.append(Vector2(cos(a), sin(a)))
	else:
		out.append(Vector2(-1.0, -1.0))
		out.append(Vector2(1.0, -1.0))
		out.append(Vector2(1.0, 1.0))
		out.append(Vector2(-1.0, 1.0))
	return out


## Start baking a new figure. The current grid is pushed back into the previous slot and
## [member blend] resets to 0, so the figure on screen holds - untouched - while the new
## one is written row by row into the slot nothing is reading yet.
func begin(m: int, n: int, m2: int, n2: int, second_weight: float) -> void:
	_a0 = _a1.duplicate()
	_gx0 = _gx1.duplicate()
	_gy0 = _gy1.duplicate()
	blend = 0.0
	_m = m
	_n = n
	_m2 = m2
	_n2 = n2
	_w2 = clampf(second_weight, 0.0, 0.6)
	_amax = 1e-6
	_phase = 1
	_row = 0
	_prepare()


## True while a figure is still being written.
func baking() -> bool:
	return _phase != 0


## Do at most [param rows] row-units of bake work. A whole bake is 2 * res row-units
## (the field pass, then the gradient pass).
func advance(rows: int) -> void:
	var todo := maxi(1, rows)
	while todo > 0 and _phase != 0:
		if _phase == 1:
			_field_row(_row)
		else:
			_grad_row(_row)
		_row += 1
		if _row >= res:
			_row = 0
			_phase = 2 if _phase == 1 else 0
		todo -= 1


## Finish the pending bake right now and make it the only figure - for the FIRST one,
## where there is nothing to crossfade from and the opening frame must already be a
## picture.
func complete() -> void:
	while _phase != 0:
		advance(res)
	blend = 1.0
	_a0 = _a1.duplicate()
	_gx0 = _gx1.duplicate()
	_gy0 = _gy1.duplicate()


## The baked field at plate coordinate (x, y): Vector3(|w| in 0..1, d|w|/dx, d|w|/dy),
## already crossfaded between the previous figure and the current one.
func sample(x: float, y: float) -> Vector3:
	var fx := (x * 0.5 + 0.5) * float(res - 1)
	var fy := (y * 0.5 + 0.5) * float(res - 1)
	var i := clampi(int(fx), 0, res - 2)
	var j := clampi(int(fy), 0, res - 2)
	var tx := clampf(fx - float(i), 0.0, 1.0)
	var ty := clampf(fy - float(j), 0.0, 1.0)
	var o := j * res + i
	var o2 := o + res
	if blend >= 0.999:
		return Vector3(_bl(_a1, o, o2, tx, ty), _bl(_gx1, o, o2, tx, ty), _bl(_gy1, o, o2, tx, ty))
	var b := blend
	var ia := 1.0 - b
	return Vector3(
		_bl(_a0, o, o2, tx, ty) * ia + _bl(_a1, o, o2, tx, ty) * b,
		_bl(_gx0, o, o2, tx, ty) * ia + _bl(_gx1, o, o2, tx, ty) * b,
		_bl(_gy0, o, o2, tx, ty) * ia + _bl(_gy1, o, o2, tx, ty) * b)


func _bl(arr: PackedFloat32Array, o: int, o2: int, tx: float, ty: float) -> float:
	var a: float = arr[o]
	var b: float = arr[o2]
	a += (arr[o + 1] - a) * tx
	b += (arr[o2 + 1] - b) * tx
	return a + (b - a) * ty


# ---------------------------------------------------------------------------------
# The bake itself.
# ---------------------------------------------------------------------------------

func _prepare() -> void:
	if _shape_id == 0:
		var clamped := boundary == "clamped"
		_t1 = _wave_table(_m, clamped)
		_t2 = _wave_table(_n, clamped)
		_t3 = _wave_table(_m2, clamped)
		_t4 = _wave_table(_n2, clamped)
	elif _shape_id == 1:
		_lut1 = _radial(_m, _n)
		_lut2 = _radial(_m2, _n2)
	else:
		_k1 = PI * float(maxi(1, _m))
		_k2 = PI * float(maxi(1, _n))
		_k3 = PI * float(maxi(1, _m2))
		_k4 = PI * float(maxi(1, _n2))


# One axis of the square plate's separable eigenfunction. The whole reason a square bake
# is cheap: 4 * res cos() calls stand in for 4 per cell.
func _wave_table(k: int, clamped: bool) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(res)
	var inv := 1.0 / float(res - 1)
	for i in res:
		var xp := PI * float(k) * float(i) * inv
		out[i] = sin(xp) if clamped else cos(xp)
	return out


# J_m(k r) sampled along the radius, so the per-cell cost is one lerp.
func _radial(m: int, n: int) -> PackedFloat32Array:
	var mm := clampi(m, 0, MAX_ANG)
	var nn := clampi(n, 1, 4)
	var row: Array = JP_ZEROS[mm] if boundary == "free" else J_ZEROS[mm]
	var k: float = float(row[nn - 1])
	var out := PackedFloat32Array()
	out.resize(RAD + 1)
	for s in RAD + 1:
		out[s] = bessel_j(mm, k * float(s) / float(RAD))
	return out


func _lookup(lut: PackedFloat32Array, r: float) -> float:
	var x := clampf(r, 0.0, 1.0) * float(RAD)
	var s := int(x)
	if s >= RAD:
		return lut[RAD]
	var a: float = lut[s]
	return a + (lut[s + 1] - a) * (x - float(s))


# One row of |w|, un-normalised, plus a running maximum for the normalisation the
# gradient pass applies.
func _field_row(j: int) -> void:
	var base := j * res
	var wa := 1.0 - _w2
	var second := _w2 > 0.001
	if _shape_id == 0:
		var aj: float = _t1[j]
		var bj: float = _t2[j]
		var cj: float = _t3[j]
		var dj: float = _t4[j]
		for i in res:
			var o := base + i
			var w := wa * (_t1[i] * bj - _t2[i] * aj)
			if second:
				w += _w2 * (_t3[i] * dj - _t4[i] * cj)
			var a := absf(w)
			_raw[o] = a
			if a > _amax:
				_amax = a
	elif _shape_id == 1:
		var m1 := float(clampi(_m, 0, MAX_ANG))
		var m2f := float(clampi(_m2, 0, MAX_ANG))
		for i in res:
			var o := base + i
			var th: float = _th[o]
			var w := wa * _lookup(_lut1, _rr[o]) * cos(m1 * th)
			if second:
				w += _w2 * _lookup(_lut2, _rr[o]) * cos(m2f * th)
			var a := absf(w)
			_raw[o] = a
			if a > _amax:
				_amax = a
	else:
		var d0: PackedFloat32Array = _dots[0]
		var d1: PackedFloat32Array = _dots[1]
		var d2: PackedFloat32Array = _dots[2]
		var e0: PackedFloat32Array = _dots[3]
		var e1: PackedFloat32Array = _dots[4]
		var e2: PackedFloat32Array = _dots[5]
		var ph := _hex_ph
		for i in res:
			var o := base + i
			var t0: float = d0[o]
			var t1: float = d1[o]
			var t2: float = d2[o]
			var u0: float = e0[o]
			var u1: float = e1[o]
			var u2: float = e2[o]
			var wa1 := cos(_k1 * t0 + ph) + cos(_k1 * t1 + ph) + cos(_k1 * t2 + ph)
			var wb1 := cos(_k2 * u0 + ph) + cos(_k2 * u1 + ph) + cos(_k2 * u2 + ph)
			var w := wa * (wa1 + wb1) / 6.0
			if second:
				var wa2 := cos(_k3 * t0 + ph) + cos(_k3 * t1 + ph) + cos(_k3 * t2 + ph)
				var wb2 := cos(_k4 * u0 + ph) + cos(_k4 * u1 + ph) + cos(_k4 * u2 + ph)
				w += _w2 * (wa2 + wb2) / 6.0
			var a := absf(w)
			_raw[o] = a
			if a > _amax:
				_amax = a


# One row of the finished grids: |w| normalised to 0..1 and its central-difference
# gradient. Reads _raw (whole rows above and below), writes _a1 / _gx1 / _gy1, so the
# two passes never tread on each other.
func _grad_row(j: int) -> void:
	var inv := 1.0 / maxf(_amax, 1e-6)
	var h := 2.0 / float(res - 1)
	var base := j * res
	var ju := maxi(j - 1, 0)
	var jd := mini(j + 1, res - 1)
	var up := ju * res
	var dn := jd * res
	var dy := float(jd - ju) * h
	for i in res:
		var o := base + i
		var il := maxi(i - 1, 0)
		var ir := mini(i + 1, res - 1)
		var dx := float(ir - il) * h
		_a1[o] = _raw[o] * inv
		_gx1[o] = (_raw[base + ir] - _raw[base + il]) * inv / dx
		_gy1[o] = (_raw[dn + i] - _raw[up + i]) * inv / dy


## Bessel function of the first kind, order [param m], by its power series. Accurate well
## past the largest argument the zero tables can ask for (about 17.6): the series peaks
## near k = x/2 at roughly 1e7 there, which costs about eight of a double's sixteen
## digits and leaves eight - far more than a display field needs.
static func bessel_j(m: int, x: float) -> float:
	var ax := absf(x)
	var half := ax * 0.5
	var t := 1.0
	for k in m:
		t *= half / float(k + 1)
	var s := t
	var k2 := 1
	while k2 <= 60:
		t *= -(half * half) / (float(k2) * float(k2 + m))
		s += t
		if absf(t) < 1e-13 * maxf(1.0, absf(s)):
			break
		k2 += 1
	return s
