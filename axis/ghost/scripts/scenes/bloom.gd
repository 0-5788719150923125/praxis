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

## The contour's ARCHETYPE. One formula spans star, flower, cog and soft polygon, but only
## in specific corners of its parameter space, so the seed picks a FAMILY first and samples
## inside it - which is why two seeds now differ in silhouette rather than in the third
## decimal of the same shape.
##
## The two facts that decide a family: small n1 with small n2/n3 drives the radius to
## collapse between lobes (a star), while n1 = n2 = n3 well above 1 is a SUPERELLIPSE -
## flat sides, sharp corners, a polygon. `tie` marks the second case, because sampling the
## three apart lands near n2 = n3 = 2, which is a circle exactly, whatever n1 does.
const FORMS := {
	"star":    {"m": [5, 12],  "n1": [0.16, 0.38], "n2": [0.35, 0.85], "n3": [0.35, 0.85]},
	"flower":  {"m": [4, 9],   "n1": [0.45, 0.95], "n2": [0.30, 1.10], "n3": [0.30, 1.10]},
	"ripple":  {"m": [14, 26], "n1": [0.70, 1.40], "n2": [1.10, 1.60], "n3": [2.20, 3.20]},
	"cog":     {"m": [10, 18], "tie": [5.0, 14.0]},
	"polygon": {"m": [3, 7],   "tie": [4.0, 12.0]},
}

var _f: AudioFeatures = AudioFeatures.new()
var _sch: Scheme = null
var _spin := 0.0
var _morph := 0.0
var _sharp := 0.0


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "plane"
	# Nothing about a rosette says what colour it is, so any mood is fair - it was a bare
	# random hue before, which varied but never related to itself.
	_sch = Scheme.pick(rng)
	var keys := FORMS.keys()
	var form := String(keys[rng.randi() % keys.size()])
	var fm: Dictionary = FORMS[form]
	var m_r: Array = fm["m"]
	var m := float(rng.randi_range(int(m_r[0]), int(m_r[1])))
	var n1 := 0.0
	var n2 := 0.0
	var n3 := 0.0
	if fm.has("tie"):
		var tie_r: Array = fm["tie"]
		n1 = rng.randf_range(float(tie_r[0]), float(tie_r[1]))
		n2 = n1
		n3 = n1
	else:
		var n1_r: Array = fm["n1"]
		var n2_r: Array = fm["n2"]
		var n3_r: Array = fm["n3"]
		n1 = rng.randf_range(float(n1_r[0]), float(n1_r[1]))
		n2 = rng.randf_range(float(n2_r[0]), float(n2_r[1]))
		n3 = rng.randf_range(float(n3_r[0]), float(n3_r[1]))
	var layers := rng.randi_range(1, 5)
	# Either the stack sweeps base -> accent (a two-tone rosette) or it stays one colour
	# with each ring drifted slightly off it. Two quite different objects.
	var gradient := rng.randf() < 0.55
	var hues: Array = []
	for i in layers:
		hues.append(_sch.hue_at(i, layers) if gradient else _sch.vary(rng, 0.6))
	return {
		"form": form,
		"mood": _sch.name,
		"hue": _sch.hue,
		"m": m,                                      # symmetry (lobes)
		"n1": n1,
		"n2": n2,
		"n3": n3,
		"layers": layers,
		"hues": hues,
		"gradient": gradient,
		"radius": rng.randf_range(0.18, 0.40),
		"width": rng.randf_range(1.0, 4.0),
		"spin_rate": rng.randf_range(-0.10, 0.10),
		"m_step": rng.randf() < 0.5,                 # neighbouring layers shift symmetry
		# Enough samples to keep a high-lobe contour smooth - a 26-fold ripple traced at a
		# fixed 260 points showed its faceting.
		"samples": clampi(int(m) * 26, 240, 600),
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
		var hues: Array = params.hues
		var h := fposmod(float(hues[layer]) + 0.05 * _f.treble, 1.0)
		# Value still rides the audio; the mood supplies the character it rides on.
		var vmul := clampf(0.62 + 0.40 * _f.energy + 0.22 * _sharp, 0.25, 1.15)
		draw_polyline(curve, _sch.color(h, 0.85, vmul, 0.92), float(params.width), true)


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
