extends VortexScene

## Fog lights - soft lights breathing under a drifting cloud cover.
##
## A few glowing orbs sit in the dark, each tied to a slice of the spectrum so it
## pulses with its own frequencies. Over them drift several big, low-alpha blobs -
## fog - that diffuse and occlude the lights as they pass. The result is
## atmospheric rather than graphic: light bleeding through cloud, the whole field
## swelling on the louder passages.

var _f: AudioFeatures = AudioFeatures.new()
var _lights: Array = []
var _fog: Array = []


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "field"
	var base_hue := rng.randf()
	var warm := rng.randf() < 0.5
	var light_count := rng.randi_range(3, 6)
	for i in light_count:
		_lights.append({
			"pos": Vector2(rng.randf_range(-0.40, 0.40), rng.randf_range(-0.30, 0.30)),
			"hue": fposmod(base_hue + rng.randf_range(-0.12, 0.14), 1.0),
			"band": rng.randf(),                       # where it samples the spectrum
			"size": rng.randf_range(0.12, 0.26),
		})
	var fog_count := rng.randi_range(5, 9)
	for i in fog_count:
		_fog.append({
			"pos": Vector2(rng.randf_range(-0.6, 0.6), rng.randf_range(-0.5, 0.5)),
			"size": rng.randf_range(0.35, 0.75),
		})
	return {
		"base_hue": base_hue,
		"fog_tint": fposmod(base_hue + (0.04 if warm else -0.45), 1.0),
		"fog_alpha": rng.randf_range(0.030, 0.060),
	}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.03, 0.04, 0.03, 0.08)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()

	# Lights first, so the fog reads as cover above them.
	for i in _lights.size():
		var L: Dictionary = _lights[i]
		var base: Vector2 = L.pos
		var drift := Vector2(mod.value("lx%d" % i), mod.value("ly%d" % i)) * 0.04
		var pos := (base + drift) * u
		var bright: float = _f.sample(float(L.band)) * 0.8 + _f.energy * 0.3 + _f.beat * 0.25
		bright = clampf(0.15 + bright, 0.0, 1.3)
		var radius: float = u * float(L.size) * (0.6 + 0.6 * bright)
		var col := Color.from_hsv(float(L.hue), 0.55, 1.0, clampf(0.10 + 0.30 * bright, 0.0, 0.5))
		_glow(pos, radius, col)

	# Fog: big faint blobs drifting across, diffusing the lights below.
	var tint := Color.from_hsv(float(params.fog_tint), 0.12, 0.9, float(params.fog_alpha))
	for i in _fog.size():
		var Fb: Dictionary = _fog[i]
		var base: Vector2 = Fb.pos
		var drift := Vector2(mod.value("fx%d" % i), mod.value("fy%d" % i)) * 0.10
		var pos := (base + drift) * u
		var radius: float = u * float(Fb.size) * (0.9 + 0.2 * _f.low_mid)
		_soft_blob(pos, radius, tint)


# Layered concentric circles: bright tight center fading to a wide halo.
func _glow(c: Vector2, radius: float, color: Color, layers := 7) -> void:
	for i in layers:
		var frac := float(i) / float(layers - 1)
		var r := radius * (1.0 - 0.82 * frac)
		var al := color.a * (0.06 + 0.22 * frac)
		draw_circle(c, r, Color(color.r, color.g, color.b, al))


# A wide, very soft disc - one puff of fog.
func _soft_blob(c: Vector2, radius: float, color: Color, layers := 5) -> void:
	for i in layers:
		var frac := float(i) / float(layers - 1)
		var r := radius * (0.5 + 0.5 * frac)
		draw_circle(c, r, Color(color.r, color.g, color.b, color.a))
