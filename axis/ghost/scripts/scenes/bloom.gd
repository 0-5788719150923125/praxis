extends GhostScene

## Bloom - elegant procedural rosette curves (the koch replacement).
##
## The old snowflake hard-coded a star and stepped its recursion depth in integer
## jumps (which popped). This draws the shape from a *formula* instead: the
## superformula, whose symmetry and pinch are a handful of numbers, traces a single
## smooth closed contour that can be a star, a flower, a gear, or a soft polygon -
## and morphs continuously between them. A few are layered concentrically with a
## hue gradient and a gentle sway; the audio sharpens and brightens them (through a
## nonlinear curve) without ever stepping. Classy, fluid, different every seed.

var _f: AudioFeatures = AudioFeatures.new()
var _hue := 0.0
var _spin := 0.0
var _morph := 0.0
var _sharp := 0.0


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "plane"
	_hue = rng.randf()
	return {
		"m": float(rng.randi_range(3, 9)),          # symmetry (lobes)
		"n1": rng.randf_range(0.30, 0.9),
		"n2": rng.randf_range(0.4, 1.7),
		"n3": rng.randf_range(0.4, 1.7),
		"layers": rng.randi_range(2, 4),
		"radius": rng.randf_range(0.24, 0.34),
		"hue_step": rng.randf_range(0.03, 0.12),
		"width": rng.randf_range(1.5, 3.0),
		"spin_rate": rng.randf_range(-0.10, 0.10),
		"m_step": rng.randf() < 0.5,                 # neighbouring layers shift symmetry
		"samples": 260,
	}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.04, 0.06)
	# Bounded sway about rest (a lone plane shouldn't tumble) + a slow morph clock.
	_spin = float(params.spin_rate) * _life * 0.4 + 0.16 * mod.value("turn")
	_morph += delta * (0.25 + 0.7 * f.energy)
	# Energy sharpens the lobes (smoothly), echoing the old "crystallize" intent.
	_sharp = Nonlinear.flare(_sharp, clampf(0.3 + 0.7 * f.energy + 0.3 * f.beat, 0.0, 1.0), delta, 5.0, 1.5)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()
	var layers := int(params.layers)
	var base_r: float = float(params.radius) * u
	var m0: float = params.m
	var n1: float = params.n1
	var n2: float = params.n2
	var n3: float = params.n3
	# Sharper lobes = a smaller n1; energy pulls it down. A touch of breathing on n2/n3.
	var n1_eff := maxf(0.12, n1 * (1.0 - 0.45 * _sharp))
	var breathe := 0.25 * sin(_morph)
	for layer in layers:
		var m: float = m0 + (1.0 if params.m_step else 0.0) * float(layer) * 2.0
		var scale := base_r * (1.0 - 0.20 * layer)
		var rot := _spin + float(layer) * 0.22 + 0.1 * sin(_morph + layer)
		var curve := _superform(m, n1_eff, n2 + breathe, n3 - breathe, int(params.samples), scale, rot)
		var h := fposmod(_hue + float(params.hue_step) * layer + 0.05 * _f.treble, 1.0)
		var val := clampf(0.55 + 0.35 * _f.energy + 0.2 * _sharp, 0.2, 1.0)
		draw_polyline(curve, Color.from_hsv(h, 0.5, val, 0.92), float(params.width), true)


# A closed superformula contour: r(θ) = (|cos(mθ/4)|^n2 + |sin(mθ/4)|^n3)^(-1/n1).
func _superform(m: float, n1: float, n2: float, n3: float, samples: int, scale: float, rot: float) -> PackedVector2Array:
	var pts := PackedVector2Array()
	pts.resize(samples + 1)
	var rmax := 0.0
	var raw := []
	raw.resize(samples)
	for i in samples:
		var th := TAU * float(i) / float(samples)
		var t := m * th / 4.0
		var part := pow(absf(cos(t)), maxf(0.05, n2)) + pow(absf(sin(t)), maxf(0.05, n3))
		var r: float = pow(part, -1.0 / n1) if part > 1e-6 else 0.0
		raw[i] = r
		rmax = maxf(rmax, r)
	if rmax <= 0.0:
		rmax = 1.0
	for i in samples:
		var th := TAU * float(i) / float(samples)
		var r: float = float(raw[i]) / rmax * scale       # normalise so the shape fits `scale`
		pts[i] = Vector2(cos(th + rot), sin(th + rot)) * r
	pts[samples] = pts[0]                                  # close the loop
	return pts
