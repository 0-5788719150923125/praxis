extends VortexScene

## Koch snowflake - a fractal that crystallizes with energy.
##
## Koch snowflakes drawn as closed outlines; recursion depth eases up with the
## music, so edges sprout finer detail as the track swells and smooth back when it
## calms. Usually a single elegant flake held square-on (a lone plane shouldn't
## tumble); sometimes several smaller ones scattered. Rotation is a bounded sway
## about a rest angle, not a continuous spin.

var _f: AudioFeatures = AudioFeatures.new()
var _spin := 0.0
var _rest := 0.0
var _drift := 0.0
var _free := 0.0
var _level := 0.0      # eased fractal depth
var _inst: Array = []  # snowflake instances {pos, scale, phase}


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "plane"
	_rest = rng.randf_range(-PI, PI)
	_free = 0.0 if rng.randf() < 0.7 else rng.randf_range(0.3, 1.0)
	var count := 1 if rng.randf() < 0.6 else rng.randi_range(2, 4)
	for i in count:
		var pos := Vector2.ZERO
		var scl := 1.0
		if count > 1:
			pos = Vector2(rng.randf_range(-0.30, 0.30), rng.randf_range(-0.26, 0.26))
			scl = rng.randf_range(0.4, 0.7)
		_inst.append({"pos": pos, "scale": scl, "phase": rng.randf() * TAU})
	return {
		"rings": 1 if count > 1 else rng.randi_range(1, 3),
		"max_depth": rng.randi_range(3, 4),
		"radius": rng.randf_range(0.22, 0.32),
		"hue": rng.randf(),
		"hue_step": rng.randf_range(0.04, 0.16),
		"spin_rate": rng.randf_range(-0.08, 0.08),
		"width": rng.randf_range(1.5, 3.0),
	}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.04, 0.06, 0.05, 0.10)
	# Bounded sway about a rest angle; continuous drift only when free.
	_drift += float(params.spin_rate) * delta * _free
	_spin = _rest + _drift + 0.18 * mod.value("turn")
	# Rest at depth ~1 (a six-point star) and crystallize further with energy.
	var target := 1.2 + f.energy * (float(params.max_depth) - 1.0)
	_level = lerpf(_level, target, 0.05)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()
	var depth := clampi(int(round(_level)), 0, int(params.max_depth))
	var rings := int(params.rings)
	var hue: float = params.hue
	var hue_step: float = params.hue_step
	var width: float = params.width
	for inst: Dictionary in _inst:
		var ipos: Vector2 = Vector2(inst.pos) * u
		var iscale: float = float(inst.scale)
		var iphase: float = float(inst.phase)
		for rdx in rings:
			var scale: float = float(params.radius) * u * iscale * (1.0 - 0.22 * rdx)
			var pts := _snowflake(scale, depth)
			var rot := _spin + iphase + rdx * 0.3
			var out := PackedVector2Array()
			out.resize(pts.size() + 1)
			for i in pts.size():
				out[i] = ipos + pts[i].rotated(rot)
			out[pts.size()] = out[0]
			var h := fposmod(hue + hue_step * rdx, 1.0)
			draw_polyline(out, Color.from_hsv(h, 0.5, 0.7 + 0.3 * _f.energy, 0.9), width, true)


func _snowflake(scale: float, depth: int) -> PackedVector2Array:
	var pts := PackedVector2Array()
	for k in 3:
		var a := -PI * 0.5 + k * TAU / 3.0
		pts.append(Vector2(cos(a), sin(a)) * scale)
	return _koch(pts, depth)


# Each pass replaces every segment with the 4-segment Koch bump.
func _koch(points: PackedVector2Array, depth: int) -> PackedVector2Array:
	var pts := points
	for it in depth:
		var np := PackedVector2Array()
		var n := pts.size()
		for i in n:
			var a := pts[i]
			var b := pts[(i + 1) % n]
			var d := (b - a) / 3.0
			var p1 := a + d
			var p2 := a + d * 2.0
			var peak := p1 + d.rotated(-PI / 3.0)
			np.append(a)
			np.append(p1)
			np.append(peak)
			np.append(p2)
		pts = np
	return pts
