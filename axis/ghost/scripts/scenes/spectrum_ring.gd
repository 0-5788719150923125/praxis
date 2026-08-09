extends GhostScene

## Spectrum ring - the spectrum bent into a circle.
##
## The whole band array is laid around a ring; each band pushes a bar outward by
## its energy. The ring breathes with energy, spins at a seeded rate, flashes on
## the beat, and the whole frame drifts/tilts on its own through the view.
##
## The hue used to be a bare `randf()` - varied, but unrelated to anything, so the
## bars swept an arbitrary span from it. It now runs base -> accent along a [Scheme]
## mood, which also sets the ring's saturation and how bright it sits. Two other
## rolls decide what the ring IS: how a bar is drawn, and which way it grows off the
## rim - the same audio mapping, three silhouettes.

## How each band's bar is drawn. Energy -> length is identical in all three; only the
## shape differs, so the ring reads as a spoke wheel, a sunburst, or a dashed dial.
const FORMS := ["spoke", "spoke", "wedge", "tick"]      # the spoke is the familiar read

## Which way bars grow off the rim. Inward turns the ring inside out - a closing iris
## rather than a fringed rim - and "both" makes each band a two-ended spike.
const GROWTH := ["out", "out", "out", "in", "both"]

var _spin := 0.0
var _drift := 0.0
var _free := 0.0
var _f: AudioFeatures = AudioFeatures.new()
var _act: Activation
var _light: Lighting
# Optionally a real 3D human eye at the core instead of the pseudo-sphere (by seed).
var _use_eye := false
var _eye: EyeBody
var _lens: Lens3D


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	var sparsity := 0.0 if rng.randf() < 0.4 else rng.randf_range(0.3, 0.7)
	_act = Activation.new(Spectrum.BAND_COUNT, rng, sparsity)
	_light = Lighting.new(rng)
	_free = 0.0 if rng.randf() < 0.65 else rng.randf_range(0.3, 1.0)
	_use_eye = rng.randf() < 0.4
	if _use_eye:
		_eye = EyeBody.new(rng.randi())          # looks around in centre-biased saccades
		_lens = Lens3D.new()
		_lens.eye = Vector3(0.0, 0.0, 4.0)
		_lens.look = Vector3.ZERO
		_lens.fov = 48.0
	var sch := Scheme.pick(rng)                  # a ring of pure spectrum: any mood fits
	# The band sweep runs base -> accent the SHORT way round, so low and high bands are
	# two ends of one relationship rather than an arbitrary arc of the colour wheel.
	var span := wrapf(sch.accent - sch.hue, -0.5, 0.5) * rng.randf_range(0.35, 1.0)
	return {
		"hue": sch.hue,
		"hue_spread": span,
		"sat": sch.sat,
		"val": sch.val,
		"mood": sch.name,
		# Radius and reach were 0.16-0.26 and 0.12-0.30 - always a mid-size ring with a
		# modest fringe. Widened, the seed can build a tight hub with long spikes or a
		# broad rim that barely bristles.
		"radius": rng.randf_range(0.10, 0.34),
		"bar_len": rng.randf_range(0.07, 0.42),
		"spin_rate": rng.randf_range(-0.10, 0.10),
		"mirror": rng.randi_range(1, 2),
		"thickness": rng.randf_range(1.5, 9.0),
		"form": FORMS[rng.randi() % FORMS.size()],
		"growth": GROWTH[rng.randi() % GROWTH.size()],
	}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.05, 0.07, 0.06, 0.12)
	_act.update(f.energy + 0.5 * f.beat, delta)
	_light.update(f, delta)
	if _use_eye:
		_eye.update(delta, clampf(f.energy * 0.7 + f.beat * 0.4, 0.0, 1.0))
	# Bounded sway about rest; continuous slow drift only when free.
	_drift += float(params.spin_rate) * delta * _free
	_spin = _drift + 0.22 * mod.value("turn")
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()
	var base_r: float = u * params.radius          # fixed - the ring doesn't breathe
	var radius_frac: float = params.radius
	var max_len: float = u * params.bar_len
	var glow := _light.glow()
	var n := _f.bands.size()
	if n == 0:
		n = Spectrum.BAND_COUNT

	var mirror := int(params.mirror)
	var thickness: float = params.thickness
	var hue: float = params.hue
	var hue_spread: float = params.hue_spread
	var sat: float = params.sat
	var vmax: float = params.val
	var form: String = params.form
	var growth: String = params.growth
	# Inward bars must stop short of the core (eye or sphere - both land near 0.65 of
	# the ring radius on screen), or they swallow it.
	var in_room: float = maxf(0.0, base_r * 0.28)
	var count := n * mirror
	for i in count:
		var band_i := i % n if mirror == 1 else (i if i < n else count - 1 - i)
		band_i = clampi(band_i, 0, n - 1)
		var e := _f.bands[band_i] if _f.bands.size() > 0 else 0.0
		# Activation gates how far each bar reaches: rooted bands stay short, active
		# ones swell and decay. Each bar also drifts on its own radius/angle (fluid
		# behavior only); rigid behaviors leave wobble() at 0.
		e *= 0.25 + 0.75 * _act.level(band_i)
		var r_i := base_r * (1.0 + 0.10 * wobble("bar", i))
		var ang := _spin + TAU * float(i) / float(count) + 0.05 * wobble("ang", i)
		var dir := Vector2(cos(ang), sin(ang))
		# Colour, not size, carries the audio: a moving hotspot lights the bars it
		# sweeps over, and the global glow lifts everything on a beat.
		var lit := _light.at(dir * radius_frac)
		var h := fposmod(hue + hue_spread * float(band_i) / float(n) + 0.10 * lit + _light.hue_shift(), 1.0)
		var val := clampf(0.35 + 0.4 * e + 0.5 * lit + 0.4 * glow, 0.0, 1.0) * (0.55 + 0.45 * vmax)
		var col := Color.from_hsv(h, sat, clampf(val, 0.0, 1.0))
		var reach := max_len * e
		if growth != "in":
			var out_len := reach if growth == "out" else reach * 0.6
			_bar(dir * r_i, dir * (r_i + out_len), dir, col, form, thickness)
		if growth != "out":
			var in_len := minf(reach if growth == "in" else reach * 0.6, in_room)
			_bar(dir * r_i, dir * (r_i - in_len), dir, col, form, thickness)

	# The core: either a real 3D human eye (looking around) or the pseudo-3D sphere.
	if _use_eye:
		_lens.prepare()
		_eye.draw(self, _lens, u, Vector3.ZERO, float(params.radius) * 1.15)
	else:
		# A shaded sphere at the core: real curvature (dark rim -> bright, light-offset
		# highlight) and a soft halo, brightening on the beat. Not a flat disc.
		_draw_sphere(Vector2.ZERO, base_r * 0.62, hue, glow, sat / 0.7)


# One band's bar, in this ring's FORM. `a` sits on the rim and `b` is the tip; the
# audio still only decides how far apart they are - the shape between them is seeded.
func _bar(a: Vector2, b: Vector2, dir: Vector2, col: Color, form: String, thickness: float) -> void:
	var d := b - a
	var ln := d.length()
	if ln < 0.5:
		return
	if form == "wedge":
		var perp := Vector2(-dir.y, dir.x)
		var w := thickness * 0.9
		draw_colored_polygon(PackedVector2Array([a - perp * w, a + perp * w,
			b + perp * w * 0.18, b - perp * w * 0.18]), col)
	elif form == "tick":
		# Dash count is capped rather than derived from the bar length alone: a long
		# bar gets longer dashes, not hundreds of them.
		var want := clampi(int(ln / maxf(6.0, thickness * 2.6)), 1, 18)
		var step := ln / float(want)
		var dn := d / ln
		for k in want:
			var s := a + dn * (float(k) * step)
			draw_line(s, s + dn * (step * 0.6), col, thickness, true)
	else:
		draw_line(a, b, col, thickness, true)


# A pseudo-3D sphere: opaque circles stacked large-dark to small-bright, each nudged
# toward the light, so a highlight sits up-left and the lower-right falls into shadow.
# `s_mul` scales the saturation to the ring's mood - an ash ring gets a grey ball.
func _draw_sphere(c: Vector2, radius: float, hue: float, glow: float, s_mul := 1.0) -> void:
	# Soft outer halo (a couple of faint wide discs).
	for h in 3:
		var hr := radius * (1.4 + 0.5 * h)
		draw_circle(c, hr, Color.from_hsv(hue, clampf(0.4 * s_mul, 0.0, 1.0), 1.0,
			(0.05 + 0.10 * glow) * (1.0 - h / 3.0)))
	var light := Vector2(-0.36, -0.42)
	var layers := 16
	for i in layers:
		var t := float(i) / float(layers - 1)          # 0 rim .. 1 highlight
		var r := radius * (1.0 - 0.9 * t)
		var off := light * radius * 0.5 * t
		var val := clampf(0.12 + 0.95 * t * (0.55 + 0.55 * glow), 0.0, 1.0)
		var sat := clampf(0.55 * s_mul * (1.0 - 0.85 * t), 0.0, 1.0)   # desaturate toward the highlight
		draw_circle(c + off, r, Color.from_hsv(hue, sat, val))
