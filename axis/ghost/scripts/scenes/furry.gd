extends GhostScene

## Furry - dense, thick, long tufts of fur/hair, magnetized rather than random.
##
## Built on [Filament]'s new "fur" variant: long, mostly-unbranched strands whose
## lean is dominated by a per-tuft BIAS direction (not the free-wandering flow_follow
## root/tendril growth uses). That bias is computed per tuft from two procedural
## point sets - a few bright spots (attractors) and a few dense patches of the
## OPPOSING hue (repulsors, at fposmod(hue+0.5, 1.0) - ghost's standard complement
## idiom) - so red fur visibly recoils from the densest dark blue and leans toward
## the nearest light. No pixel readback: this codebase never samples its own
## rendered canvas per frame (the two existing readback call sites are both
## one-shot/throttled, not per-frame), so brightness/opposing-color are their own
## small procedural point sets, audio-driven, EMA-tracked - the same substrate
## every other weather/growth scene in the Layer registry uses, not something
## sampled from pixels.
##
## Each tuft is grown ONCE, in LOCAL space (root at the origin), then swayed/leaned
## at DRAW time via draw_set_transform - a rotation around its own root, not a
## rebuild of the strand data. That's what lets 150+ long strands recompute their
## lean every frame cheaply: only the transform changes, never the segment list.

const Flow2D := preload("res://scripts/flow.gd")

var _flow: Flow2D
var _hue := 0.02
var _comp_hue := 0.52
var _tufts: Array = []
var _bright: Array = []   # {pos, phase, band, r}
var _dark: Array = []     # {pos, phase, band, r} - the repelling, opposing-hue patches
var _f: AudioFeatures = AudioFeatures.new()
var _sway_gain := 0.0     # EMA'd energy, widens the sway with the music


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "canvas"
	framing = "field"
	_hue = rng.randf()
	_comp_hue = fposmod(_hue + 0.5, 1.0)
	_flow = Flow2D.new(rng.randi(), rng.randf_range(1.1, 2.0), 0.035)
	add_layer("bed", rng, {"hue": _comp_hue, "sat": 0.22, "val": 0.07, "pools": 2})

	for i in rng.randi_range(2, 4):
		_bright.append({
			"pos": Vector2(rng.randf_range(-0.55, 0.55), rng.randf_range(-0.5, 0.55)),
			"phase": rng.randf() * TAU, "band": rng.randi() % 4, "r": rng.randf_range(0.16, 0.30),
		})
	for i in rng.randi_range(2, 3):
		_dark.append({
			"pos": Vector2(rng.randf_range(-0.55, 0.55), rng.randf_range(-0.5, 0.55)),
			"phase": rng.randf() * TAU, "band": rng.randi() % 4, "r": rng.randf_range(0.20, 0.36),
		})

	var n := rng.randi_range(160, 230)      # dense
	for i in n:
		var root := Vector2(rng.randf_range(-0.8, 0.8), rng.randf_range(-0.65, 0.75))
		var bias0 := _bias_dir(root)
		var length := rng.randf_range(0.34, 0.62)          # long - winter coat / flowing hair
		var steps := rng.randi_range(14, 22)
		var width := rng.randf_range(1.3, 3.0)
		var fil := Filament.grow("fur", Vector2.ZERO, bias0, length, width, steps,
			_flow, rng, 0.0, bias0)
		_tufts.append({
			"root": root, "fil": fil, "base_ang": bias0, "lean_ema": bias0,
			"sway_phase": rng.randf() * TAU, "sway_rate": rng.randf_range(0.5, 1.3),
			"sway_amt": rng.randf_range(0.03, 0.09),
			"hue_off": rng.randf_range(-0.035, 0.035),
			"grown_delay": rng.randf_range(0.0, 1.1), "grown_dur": rng.randf_range(0.5, 1.1),
			"grown": 0.0, "depth": rng.randf(),
		})
	return {"hue": _hue}


## Weighted pull toward every bright point plus weighted push away from every dark
## (opposing-hue) point, both inverse-distance weighted so the NEAREST/strongest
## source dominates a tuft's lean rather than every source pulling equally hard
## regardless of how far away it is.
func _bias_dir(root: Vector2) -> float:
	var acc := Vector2.ZERO
	for b in _bright:
		var d: Vector2 = (b.pos as Vector2) - root
		var len: float = maxf(0.05, d.length())
		acc += d / len * (1.0 / len)
	for d2 in _dark:
		var d: Vector2 = root - (d2.pos as Vector2)
		var len: float = maxf(0.05, d.length())
		acc += d / len * (1.0 / len)
	if acc.length() < 1e-4:
		return 0.0
	return acc.angle()


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.015, 0.025)
	_flow.advance(delta)
	update_layers(f, delta)
	_sway_gain = lerpf(_sway_gain, f.energy, 1.0 - exp(-2.0 * delta))

	var bands := [f.bass, f.low_mid, f.mid, f.high]
	for b in _bright:
		b.pos = (b.pos as Vector2).rotated(delta * 0.035 * (0.4 + float(bands[b.band])))
	for d in _dark:
		d.pos = (d.pos as Vector2).rotated(-delta * 0.028 * (0.4 + float(bands[d.band])))

	# EMA the lean toward the CURRENT bias (recomputed from the now-moved attractor/
	# repulsor points) - attack a little faster than release, so a beat-driven
	# lurch of a hotspot reads promptly but the fur settles back unhurried, the same
	# attack/release asymmetry Nonlinear.flare formalizes elsewhere in this codebase.
	var k := 1.0 - exp(-(1.6 if f.beat > 0.3 else 0.8) * delta)
	for tf in _tufts:
		var target := _bias_dir(tf.root)
		tf.lean_ema = lerp_angle(float(tf.lean_ema), target, k)
		if float(tf.grown) < 1.0:
			tf.grown = clampf((_life - float(tf.grown_delay)) / maxf(0.05, float(tf.grown_dur)), 0.0, 1.0)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()
	_draw_hotspots(u)
	for tf in _tufts:
		var fil: Filament = tf.fil
		if fil == null:
			continue
		var sway := sin(_life * float(tf.sway_rate) + float(tf.sway_phase)) \
			* float(tf.sway_amt) * (0.5 + 0.9 * _sway_gain)
		var lean := angle_difference(float(tf.base_ang), float(tf.lean_ema))
		draw_set_transform((tf.root as Vector2) * u, lean + sway, Vector2.ONE)
		var jitter := 0.004 + 0.01 * _f.energy
		fil.draw_growing(self, u, float(tf.grown), _color_for.bind(tf), _tip_color(tf), jitter, _life)
	draw_set_transform(Vector2.ZERO, 0.0, Vector2.ONE)


## Soft glows for the attractor/repulsor points themselves - not just an invisible
## force, the dark patches the fur recoils from and the light it leans toward are
## both actually ON SCREEN, so the lean reads as caused by something, not random.
func _draw_hotspots(u: float) -> void:
	for b in _bright:
		var pulse := 0.55 + 0.45 * sin(_life * 0.6 + float(b.phase))
		var c := Color.from_hsv(fposmod(_hue - 0.06, 1.0), 0.25, 1.0, 0.10 * pulse)
		Layer.puff(self, (b.pos as Vector2) * u, float(b.r) * u * 1.4, c)
	for d in _dark:
		var pulse := 0.55 + 0.45 * sin(_life * 0.5 + float(d.phase) + 1.7)
		var c := Color.from_hsv(_comp_hue, 0.6, 0.22, 0.16 * pulse)
		Layer.puff(self, (d.pos as Vector2) * u, float(d.r) * u * 1.3, c)


## Root-to-tip gradient: darker/denser at the root (where it's rooted in shadow),
## brightening toward the tip (catching the light it's leaning toward) - so the
## "attracted to bright" read carries all the way into the strand's own shading,
## not just its overall angle.
func _color_for(depth: int, along: float, tf: Dictionary) -> Color:
	var h := fposmod(_hue + float(tf.hue_off) + 0.05 * along, 1.0)
	var v := clampf(0.22 + 0.5 * along + 0.30 * _f.bass + 0.25 * _sway_gain, 0.08, 1.0)
	var sat := clampf(0.62 - 0.18 * along, 0.15, 0.85)
	return Color.from_hsv(h, sat, v, 0.9)


func _tip_color(tf: Dictionary) -> Color:
	var h := fposmod(_hue + float(tf.hue_off) - 0.04, 1.0)
	return Color.from_hsv(h, 0.35, 1.0, 0.7)
