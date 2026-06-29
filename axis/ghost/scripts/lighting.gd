extends RefCounted
class_name Lighting

## Lighting - audio-reactive colour, not scale.
##
## The preferred channel for sound to drive a scene. Pulsing geometry *size* with
## amplitude reads as cheap throbbing; driving *colour* reads as alive. Lighting
## provides moving bright **hotspots** that sweep the frame (region-aware lighting
## / gradient swipes), a global **glow** that flares on beats and decays slowly,
## and a slow **hue drift**. Geometry stays structurally stable; the light moves.
##
## A scene asks `at(pos)` for a local brightness boost (pos in unit-fraction space,
## roughly -0.5..0.5), and `glow()` / `hue_shift()` for the global channels. This
## is the shared modulation surface the renderer will eventually route everything
## through, 2D or 3D.

var _hot: Array = []
var _glow := 0.0
var _hue := 0.0
var _mod: ModBank


func _init(rng: RandomNumberGenerator, count := 3) -> void:
	_mod = ModBank.new(rng.randi())
	for i in count:
		_hot.append({
			"home": Vector2(rng.randf_range(-0.4, 0.4), rng.randf_range(-0.4, 0.4)),
			"radius": rng.randf_range(0.25, 0.5),
		})
	_hue = rng.randf()


func update(f: AudioFeatures, dt: float) -> void:
	_mod.advance(dt, f.energy)
	# Glow flares fast on energy/beat, fades slowly - the shared asymmetric envelope
	# (the same flare() every alive thing uses now, instead of a private EMA here).
	var target := clampf(f.energy * 0.8 + f.beat * 0.7, 0.0, 1.0)
	_glow = Nonlinear.flare(_glow, target, dt, 8.0, 1.5)
	_hue += dt * 0.01   # slow palette drift


## Local brightness boost at [param pos] (unit-fraction space) from nearby
## hotspots, which drift across the frame - so bright regions sweep over time.
func at(pos: Vector2) -> float:
	var b := 0.0
	for i in _hot.size():
		var h: Dictionary = _hot[i]
		var c: Vector2 = Vector2(h.home) + Vector2(
			_mod.value("hx%d" % i), _mod.value("hy%d" % i)) * 0.28
		var d := pos.distance_to(c)
		var r: float = float(h.radius)
		b += exp(-(d * d) / (2.0 * r * r))
	return b * (0.45 + 0.55 * _glow)


## Global glow 0..1 - flares on beats, decays slowly.
func glow() -> float:
	return _glow


## Slow hue offset to add to a scene's base hue.
func hue_shift() -> float:
	return _hue
