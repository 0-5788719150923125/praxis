extends GhostScene

## Fog lights - soft lights breathing under a drifting cloud cover.
##
## A few glowing orbs sit in the dark, each tied to a slice of the spectrum so it
## pulses with its own frequencies. Over them drift several big, low-alpha blobs -
## fog - that diffuse and occlude the lights as they pass. The result is
## atmospheric rather than graphic: light bleeding through cloud, the whole field
## swelling on the louder passages.

## How the lamps are ARRANGED - the scene's silhouette, and the thing a viewer reads before
## any colour. Scattered is the original haze of orbs; a row is street lighting seen down a
## street, a ring is something enclosing you, a pair is two sources facing off across the
## frame. Same lights, same physics, four different pictures.
const LAYOUTS := ["scatter", "row", "ring", "pair"]

var _f: AudioFeatures = AudioFeatures.new()
var _lights: Array = []
var _fog: Array = []
var _swirl := 0.0        # field rotation angle
var _swirl_vel := 0.0    # angular velocity - kicked on beats, decays between them
var _beat_prev := 0.0


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "field"
	# Lights in fog can be any colour at all - a sodium street, a green exit sign, cold
	# moonlight - so the mood is unrestricted and the lamps drift around it by its spread.
	var sch := Scheme.pick(rng)
	var layout := String(LAYOUTS[rng.randi() % LAYOUTS.size()])
	var light_count := rng.randi_range(2, 9)
	if layout == "pair":
		light_count = 2
	elif layout == "row":
		light_count = rng.randi_range(3, 7)
	# An arranged layout wants lamps of a COMMON size (they read as the same fixture seen
	# several times); a scatter wants them wildly unequal, which is what reads as depth.
	var common := rng.randf_range(0.10, 0.24)
	var row_y := rng.randf_range(-0.20, 0.20)
	var ring_r := Vector2(rng.randf_range(0.28, 0.48), rng.randf_range(0.16, 0.38))
	var ring_ph := rng.randf() * TAU
	for i in light_count:
		var pos := Vector2(rng.randf_range(-0.40, 0.40), rng.randf_range(-0.30, 0.30))
		var size := rng.randf_range(0.08, 0.34)
		var frac := (float(i) + 0.5) / float(light_count)
		match layout:
			"row":
				pos = Vector2(lerpf(-0.46, 0.46, frac) + rng.randf_range(-0.03, 0.03),
					row_y + rng.randf_range(-0.05, 0.05))
				size = common * rng.randf_range(0.85, 1.2)
			"ring":
				var th := ring_ph + TAU * frac
				pos = Vector2(cos(th) * ring_r.x, sin(th) * ring_r.y)
				size = common * rng.randf_range(0.8, 1.25)
			"pair":
				var side := 1.0 if i == 0 else -1.0
				pos = Vector2(side * rng.randf_range(0.24, 0.44), rng.randf_range(-0.18, 0.18))
				size = rng.randf_range(0.20, 0.38)
		_lights.append({
			"pos": pos,
			"hue": sch.vary(rng),                      # a family around the mood, not a spray
			"band": rng.randf(),                       # where it samples the spectrum
			"size": size,
		})
	var fog_count := rng.randi_range(4, 12)
	for i in fog_count:
		_fog.append({
			"pos": Vector2(rng.randf_range(-0.6, 0.6), rng.randf_range(-0.5, 0.5)),
			"size": rng.randf_range(0.30, 0.90),
		})
	return {
		"base_hue": sch.hue,
		"mood": sch.name,
		"layout": layout,
		# The lamps keep the mood's saturation but stay bright - a light source that took a
		# dim mood's value would stop reading as a source at all.
		"light_sat": clampf(sch.sat * rng.randf_range(0.7, 1.1), 0.08, 1.0),
		# The cover sits on the scheme's own partner hue, so it veils rather than clashes.
		"fog_tint": sch.accent,
		"fog_sat": clampf(sch.sat * rng.randf_range(0.15, 0.4), 0.03, 0.5),
		"fog_alpha": rng.randf_range(0.025, 0.070),
	}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.03, 0.04, 0.03, 0.08)
	# Tempo-driven swirl: each beat kicks the angular velocity, which then decays back
	# toward a slow baseline - so the field lurches with the music and coasts down,
	# instead of rotating at one uniform rate.
	var beat_edge: bool = f.beat > 0.55 and _beat_prev <= 0.55
	_beat_prev = f.beat
	if beat_edge:
		_swirl_vel += 0.8 * (0.5 + f.energy)
	_swirl_vel = 0.06 + (_swirl_vel - 0.06) * exp(-1.3 * delta)   # decay toward baseline
	_swirl += _swirl_vel * delta
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()

	# Lights first, so the fog reads as cover above them.
	for i in _lights.size():
		var L: Dictionary = _lights[i]
		var base: Vector2 = L.pos
		var drift := Vector2(mod.value("lx%d" % i), mod.value("ly%d" % i)) * 0.04
		var pos := (base + drift).rotated(_swirl) * u
		var bright: float = _f.sample(float(L.band)) * 0.8 + _f.energy * 0.3 + _f.beat * 0.25
		bright = clampf(0.15 + bright, 0.0, 1.3)
		var radius: float = u * float(L.size) * (0.6 + 0.6 * bright)
		var col := Color.from_hsv(float(L.hue), float(params.light_sat), 1.0,
			clampf(0.10 + 0.30 * bright, 0.0, 0.5))
		_glow(pos, radius, col)

	# Fog: big faint blobs drifting across, diffusing the lights below.
	var tint := Color.from_hsv(float(params.fog_tint), float(params.fog_sat), 0.9,
		float(params.fog_alpha))
	for i in _fog.size():
		var Fb: Dictionary = _fog[i]
		var base: Vector2 = Fb.pos
		var drift := Vector2(mod.value("fx%d" % i), mod.value("fy%d" % i)) * 0.10
		var pos := (base + drift).rotated(_swirl * 0.6) * u
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
