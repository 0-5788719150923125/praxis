extends RefCounted
class_name Field

## Field - a composable procedural scalar field; the universal "texture / modulation".
##
## The reusable surface under terrain, texture, and any spatial modulation. A Field is a
## function of position you sample with at(p) -> 0..1. Build a base kind (fbm rolling
## hills, ridged mountains, billow puffs, cells/cracks, strata bands, a gradient), then
## shape it: domain-WARP it by another field, push it through a Nonlinear CURVE, scale /
## offset it, and COMBINE fields (add / mul / mask / max / min / sub) into a tree. The
## same field drives a mountain's height, mottles a rock's colour, carves a fissure, or
## modulates any scalar elsewhere - textures abstracted into one primitive.
##
## Sampling is pure and stateless, so a scene samples a Field once into a grid at build
## (terrain is static) and renders the cheap grid thereafter.

# Leaf (a noise generator) -----------------------------------------------------------
var _noise: FastNoiseLite
var _kind := "fbm"
var _freq := 1.0
var _amp := 1.0
var _off := 0.0
var _bands := 0.0          # > 0 => fold the output into that many strata bands
var _invert := false       # 1 - v (cells -> cracks)
var _curve := ""           # optional Nonlinear curve name
var _curve_k := 1.0
var _grad := Vector2.ZERO  # gradient direction (for kind == "gradient"); ZERO = radial
# Domain warp ------------------------------------------------------------------------
var _warp: Field
var _warp_amt := 0.0
# Composition ------------------------------------------------------------------------
var _op := ""              # "" = leaf; else add / mul / mask / max / min / sub
var _a: Field
var _b: Field
var _w := 1.0              # weight on b


## A leaf field of the given kind, seeded. freq sets the feature scale (higher = finer).
static func make(kind: String, seed_value: int, freq := 1.0, octaves := 4) -> Field:
	var fld := Field.new()
	fld._kind = kind
	fld._freq = freq
	if kind == "gradient":
		return fld
	var n := FastNoiseLite.new()
	n.seed = seed_value
	n.frequency = 1.0
	n.fractal_octaves = octaves
	n.fractal_type = FastNoiseLite.FRACTAL_FBM
	match kind:
		"ridged":
			n.noise_type = FastNoiseLite.TYPE_SIMPLEX
			n.fractal_type = FastNoiseLite.FRACTAL_RIDGED
		"billow":
			n.noise_type = FastNoiseLite.TYPE_SIMPLEX
			n.fractal_type = FastNoiseLite.FRACTAL_PING_PONG
		"cells":
			n.noise_type = FastNoiseLite.TYPE_CELLULAR
			n.cellular_distance_function = FastNoiseLite.DISTANCE_EUCLIDEAN
			n.cellular_return_type = FastNoiseLite.RETURN_DISTANCE2_SUB   # ridged cell walls
		_:  # fbm
			n.noise_type = FastNoiseLite.TYPE_SIMPLEX
	fld._noise = n
	return fld


## Fluent shapers (return self, so you can chain): f.scale(2).offset(-0.1).strata(6).
func scale(a: float) -> Field:
	_amp = a
	return self
func offset(o: float) -> Field:
	_off = o
	return self
func strata(bands: float) -> Field:
	_bands = bands
	return self
func inverted() -> Field:
	_invert = true
	return self
func curve(name: String, k := 1.0) -> Field:
	_curve = name
	_curve_k = k
	return self
func gradient_dir(d: Vector2) -> Field:
	_grad = d
	return self
func warp(by: Field, amount: float) -> Field:
	_warp = by
	_warp_amt = amount
	return self


## Combine two fields by op (add / mul / mask / max / min / sub), b weighted by w.
static func combine(a: Field, op: String, b: Field, w := 1.0) -> Field:
	var fld := Field.new()
	fld._op = op
	fld._a = a
	fld._b = b
	fld._w = w
	return fld


## Sample the field at p, returning roughly 0..1.
func at(p: Vector2) -> float:
	if _op != "":
		var a := _a.at(p)
		var b := _b.at(p)
		var v: float
		match _op:
			"add": v = a + b * _w
			"mul": v = a * lerpf(1.0, b, _w)
			"mask": v = a * lerpf(1.0 - _w, 1.0, b)
			"max": v = maxf(a, b * _w)
			"min": v = minf(a, lerpf(1.0, b, _w))
			"sub": v = a - b * _w
			_: v = a
		return clampf(v, 0.0, 1.0)
	var v := _raw(p)
	if _invert:
		v = 1.0 - v
	if _bands > 0.0:
		v = 0.5 + 0.5 * sin(v * _bands * TAU)        # fold into strata bands
	if _curve != "":
		v = Nonlinear.apply(_curve, v, _curve_k)
	return clampf(_off + _amp * v, 0.0, 1.0)


func _raw(p: Vector2) -> float:
	var q := p * _freq
	if _warp != null and _warp_amt != 0.0:
		q += Vector2(_warp.at(p) - 0.5, _warp.at(p + Vector2(5.2, 1.3)) - 0.5) * _warp_amt
	if _kind == "gradient":
		if _grad == Vector2.ZERO:
			return clampf(1.0 - q.length(), 0.0, 1.0)      # radial (centre high)
		return clampf(0.5 + 0.5 * q.dot(_grad.normalized()), 0.0, 1.0)
	var n := _noise.get_noise_2d(q.x, q.y)                  # ~ -1..1
	return clampf(n * 0.5 + 0.5, 0.0, 1.0)
