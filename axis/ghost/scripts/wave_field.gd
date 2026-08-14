extends RefCounted
class_name WaveField

## WaveField - an analytic water surface: a sum of Gerstner components carrying its
## own closed-form gradient.
##
## The thing this exists to avoid is a grid solver. A water surface can be integrated
## on a lattice - a height field with a wave equation stepped every tick - but then the
## surface has a resolution, a stability limit, a CFL condition tying its timestep to
## its cell size, and a warm-up before it looks like anything. A sum of travelling
## sinusoids has none of that: it is a closed-form function of position and time, exact
## at any point, at any scale, with no state to carry between frames and nothing to
## pre-warm. You can ask it for the surface at t = 900 seconds without having simulated
## the first 899.
##
## GERSTNER, not plain sine. A sine surface has round crests and round troughs and reads
## as gelatin. A Gerstner (trochoidal) component moves each particle in a circle rather
## than up and down, which piles water into a sharp crest and flattens the trough, and
## that asymmetry is most of what makes water look like water. The steepness [code]q[/code]
## in 0..1 is how far toward the cusp each component leans.
##
## WHAT IT RETURNS, AND WHY THAT IS THE INTERESTING PART. Height alone is not enough for
## anything optical. Refraction needs the surface NORMAL, and the normal of a Gerstner
## surface is not simply (-dh/dx, -dh/dy, 1), because the horizontal displacement warps
## the parameter grid underneath the height. Writing the surface in parameter space as
##
##     H(u)  = SUM a_i cos(phase_i)
##     X(u)  = u - SUM q_i a_i d_i sin(phase_i)
##     phase_i = k_i (d_i . u) + w_i t + p_i
##
## the height gradient is [code]G = -SUM a_i k_i sin(phase_i) d_i[/code] and the Jacobian
## of the horizontal map is [code]I - SUM q_i a_i k_i cos(phase_i) d_i d_i^T[/code]. For
## the steepnesses water actually uses, that Jacobian is well approximated by the scalar
## [code]jz = 1 - SUM q_i a_i k_i cos(phase_i)[/code], and the normal is then simply
## proportional to [code](-G.x, -G.y, jz)[/code] - no inverse, no square roots per
## component. So a sample returns [code]Vector4(h, G.x, G.y, jz)[/code] and the caller
## normalizes once. This is the same decomposition GPU Gems used for vertex-shader water,
## restated for a CPU that has to be frugal.
##
## [member steep] scales every component's q at once, which is the physically right knob
## for "the sea got choppier": it pinches the crests without changing a single amplitude
## or wavelength, so nothing grows or shrinks. jz falling toward zero at a crest is what
## tilts the normal hard there, and a caller reading normals should clamp it off zero -
## past the point where jz goes negative a Gerstner surface has folded over itself, which
## is a physical breaking wave and not something a normal is defined on.
##
## THE SWEEP IS THE POINT OF THE CLASS. Sampling a grid point by point costs two
## trigonometric calls per component per point, and a scene wanting a caustic net wants
## thousands of points every frame. [method sweep] instead walks a row and advances each
## component's (sin, cos) pair by a fixed angle using the angle-addition identity - four
## multiplies and two adds, no trig at all - so the whole grid costs two trig calls per
## component per ROW. The rotation is orthonormal by construction, so the error after a
## hundred steps is still at the last bit of a double; there is no drift worth correcting.
##
## RINGS are the transient half: a drip is not a plane wave and cannot join the sweep, so
## [method sweep_rings] adds compact wave packets whose envelope has finite support. That
## finite support is what makes them cheap - the evaluation is bounded to the annulus the
## packet actually occupies, computed per row in closed form, so a ring costs work
## proportional to its circumference rather than to the grid. The caller owns their
## lifecycle; the field only evaluates what it is handed.

## Deep-water dispersion constant. Long waves travel faster than short ones by
## w = sqrt(g k), and keeping that relation is what makes a swell and a chop read as the
## same body of water rather than as two unrelated animations.
const GRAVITY := 9.8

var _dx := PackedFloat32Array()      # unit direction
var _dy := PackedFloat32Array()
var _k := PackedFloat32Array()       # wavenumber, TAU / wavelength
var _om := PackedFloat32Array()      # angular frequency
var _ph := PackedFloat32Array()      # phase offset
var _amp := PackedFloat32Array()     # amplitude (the audio-driven one)
var _q := PackedFloat32Array()       # per-component steepness, 0..1
var _n := 0

## Global multiplier on every component's steepness - the "chop" knob (see the class doc).
var steep := 1.0


## Add one travelling component. [param tempo] scales the dispersion-derived frequency:
## real ripples on a hand-sized pool cross it in well under a second, which is true and
## unwatchable, so a caller slows the whole field with one number rather than lying about
## the dispersion relation per component. Returns the component's index.
func add(angle: float, wavelength: float, amp: float, q: float, phase: float,
		tempo := 1.0) -> int:
	var k := TAU / maxf(0.0001, wavelength)
	_dx.append(cos(angle))
	_dy.append(sin(angle))
	_k.append(k)
	_om.append(sqrt(GRAVITY * k) * tempo)
	_ph.append(phase)
	_amp.append(maxf(0.0, amp))
	_q.append(clampf(q, 0.0, 1.0))
	_n += 1
	return _n - 1


func count() -> int:
	return _n


## This component's wavelength, in the same world units as its positions.
func wavelength_of(i: int) -> float:
	if i < 0 or i >= _n:
		return 0.0
	return TAU / _k[i]


func set_amp(i: int, a: float) -> void:
	if i >= 0 and i < _n:
		_amp[i] = maxf(0.0, a)


## Set every amplitude at once - the per-frame audio hand-off, and deliberately a packed
## array so it can cross a thread boundary inside a snapshot without being duplicated
## element by element.
func set_amps(a: PackedFloat32Array) -> void:
	for i in mini(_n, a.size()):
		_amp[i] = maxf(0.0, a[i])


## The surface at one point: Vector4(height, dh/dx, dh/dy, jz). The reference form of the
## field - correct, readable, and the wrong way to fill a grid (see [method sweep]).
func sample(p: Vector2, t: float) -> Vector4:
	var h := 0.0
	var gx := 0.0
	var gy := 0.0
	var jz := 1.0
	for c in _n:
		var a := _amp[c]
		if a <= 0.000001:
			continue
		var k := _k[c]
		var dxc := _dx[c]
		var dyc := _dy[c]
		var th := k * (dxc * p.x + dyc * p.y) + _om[c] * t + _ph[c]
		var s := sin(th)
		var co := cos(th)
		h += a * co
		gx -= a * k * dxc * s
		gy -= a * k * dyc * s
		jz -= _q[c] * steep * a * k * co
	return Vector4(h, gx, gy, jz)


## Accumulate the whole field onto a regular grid of [param gw] x [param gh] nodes whose
## node (ix, iy) sits at [code]org + Vector2(ix, iy) * step[/code], row-major.
##
## ADDITIVE, and it does not clear: the caller fills hh/gx/gy with 0 and jz with 1 first,
## which is what lets rings and any other contribution stack onto the same buffers.
## All four arrays must already be gw*gh long.
func sweep(t: float, org: Vector2, step: float, gw: int, gh: int,
		hh: PackedFloat32Array, gx: PackedFloat32Array,
		gy: PackedFloat32Array, jz: PackedFloat32Array) -> void:
	if gw <= 0 or gh <= 0:
		return
	for c in _n:
		var a := _amp[c]
		if a <= 0.000001:
			continue
		var k := _k[c]
		var dxc := _dx[c]
		var dyc := _dy[c]
		# The phase is affine in the grid indices, so stepping one cell in x or y is
		# stepping the phase by a CONSTANT - which is the whole reason this is cheap.
		var ax := k * dxc * step
		var ay := k * dyc * step
		var base := k * (dxc * org.x + dyc * org.y) + _om[c] * t + _ph[c]
		var ca := a
		var cgx := -a * k * dxc
		var cgy := -a * k * dyc
		var cj := _q[c] * steep * a * k
		var sax := sin(ax)
		var cax := cos(ax)
		var i := 0
		for row in gh:
			var th := base + float(row) * ay
			var s := sin(th)
			var co := cos(th)
			for _col in gw:
				hh[i] += ca * co
				gx[i] += cgx * s
				gy[i] += cgy * s
				jz[i] -= cj * co
				i += 1
				# rotate (s, co) by the constant per-cell angle: no trig in here
				var s2 := s * cax + co * sax
				co = co * cax - s * sax
				s = s2


## Add transient ring packets (drips) onto the same four buffers.
##
## Each entry is a Dictionary with world centre [code]x, y[/code], front radius
## [code]r[/code], packet half width [code]w[/code], amplitude [code]a[/code], wavenumber
## [code]k[/code], accumulated carrier phase [code]phi[/code], phase offset [code]ph[/code],
## and steepness [code]q[/code]. The caller advances r, phi and a; this only evaluates.
##
## The envelope is [code](1 - u^2)^2[/code] over u = (r - front)/width rather than a
## gaussian, for two reasons: it has EXACTLY compact support, so the row spans below are
## the true extent of the packet and nothing is evaluated that contributes nothing, and
## its derivative is a polynomial, so a ring costs no exp() anywhere. The carrier phase
## advances faster than the front, which is what makes crests appear at the back of the
## packet and die at its leading edge - the actual look of a drip ring.
##
## THE RINGS RIDE THE SWELL. Call this AFTER [method sweep] and the four buffers already hold
## the travelling field, so a ring can read the water it is spreading across instead of ignoring
## it. Two couplings, both from data already in the buffers and costing a handful of flops per
## point:
##
##   WARP (`warp` on the ring) displaces the packet's front and its carrier by the local swell
##   height, so a ring crossing a crest runs ahead there and lags in the trough. It stops being
##   a circle stamped over the water and becomes a front travelling THROUGH it - which is what
##   "droplets create a circular ripple effect which does not interact with the other waves at
##   all" was asking for. Physically it is the leading-order term of a shallow-water celerity
##   c = sqrt(g(d + h)) accumulating differently over crest and trough; the second-order piece
##   (the front's own gradient picking up the swell's) is dropped, which is worth a few percent
##   on the normal and nothing on the silhouette.
##
##   DAMP (`damp`) kills the packet where the swell is already steep. Ripples do not survive on
##   choppy water, and a drip that dies in the rough while spreading clean across glass is most
##   of what makes the two systems look like one body of water. Read off the squared gradient,
##   so no square root enters the inner loop.
static func sweep_rings(rings: Array, org: Vector2, step: float, gw: int, gh: int,
		hh: PackedFloat32Array, gx: PackedFloat32Array,
		gy: PackedFloat32Array, jz: PackedFloat32Array) -> void:
	if gw <= 0 or gh <= 0 or step <= 0.0:
		return
	for entry in rings:
		var rd: Dictionary = entry
		var a: float = rd["a"]
		if a <= 0.000001:
			continue
		var cx: float = rd["x"]
		var cy: float = rd["y"]
		var rad: float = rd["r"]
		var wdt: float = maxf(0.0005, float(rd["w"]))
		var kk: float = rd["k"]
		var phi: float = rd["phi"]
		var ph: float = rd["ph"]
		var qk: float = float(rd["q"]) * a * kk
		var wrp: float = float(rd.get("warp", 0.0))
		var dmp: float = float(rd.get("damp", 0.0))
		# The warp displaces the front by `wrp` times the LOCAL swell height, so the row spans
		# have to widen by the largest that can be, or the packet is clipped exactly where it
		# rides a crest. The caller knows the live amplitude sum and hands the bound in.
		var slack := absf(float(rd.get("warp_max", 0.0)))
		var r_out := rad + wdt + slack
		var r_in := rad - wdt - slack
		var iy0 := maxi(0, int(ceil((cy - r_out - org.y) / step)))
		var iy1 := mini(gh - 1, int(floor((cy + r_out - org.y) / step)))
		for iy in range(iy0, iy1 + 1):
			var dy := org.y + float(iy) * step - cy
			var d2 := r_out * r_out - dy * dy
			if d2 <= 0.0:
				continue
			var xo := sqrt(d2)
			# One span across the disc, or two across the annulus when this row clears
			# the hole in the middle. Held in locals so a row allocates nothing.
			var xa0 := cx - xo
			var xb0 := cx + xo
			var xa1 := 0.0
			var xb1 := -1.0
			if r_in > 0.0 and absf(dy) < r_in:
				var xi := sqrt(r_in * r_in - dy * dy)
				xb0 = cx - xi
				xa1 = cx + xi
				xb1 = cx + xo
			for span in 2:
				var xa := xa0 if span == 0 else xa1
				var xb := xb0 if span == 0 else xb1
				if xb < xa:
					continue
				var ix0 := maxi(0, int(ceil((xa - org.x) / step)))
				var ix1 := mini(gw - 1, int(floor((xb - org.x) / step)))
				var i := iy * gw + ix0
				for ix in range(ix0, ix1 + 1):
					var dx := org.x + float(ix) * step - cx
					var rr := sqrt(dx * dx + dy * dy)
					if rr < 0.00001:
						i += 1
						continue
					# The swell as it stands at this node - sweep() has already run, so these are
					# the travelling field's own height and gradient, not the ring's.
					var swell := hh[i]
					var rw := rr - wrp * swell            # the front, carried by the water
					var u := (rw - rad) / wdt
					if u <= -1.0 or u >= 1.0:
						i += 1
						continue
					var b := 1.0 - u * u
					var env := b * b
					if dmp > 0.0:
						var sgx := gx[i]
						var sgy := gy[i]
						env /= 1.0 + dmp * (sgx * sgx + sgy * sgy)   # chop eats ripples
					var denv := -4.0 * u * b / wdt
					var th := kk * rw - phi + ph
					var s := sin(th)
					var co := cos(th)
					var dhdr := a * (denv * co - env * kk * s)
					hh[i] += a * env * co
					gx[i] += dx / rr * dhdr
					gy[i] += dy / rr * dhdr
					jz[i] -= qk * env * co
					i += 1
