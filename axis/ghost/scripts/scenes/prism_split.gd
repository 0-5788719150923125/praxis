extends GhostScene

## Prism split - one prism becomes two (from "the-point").
##
## Starts as a single blue [PrismBody] at centre; a red one emerges from it and the
## pair separates left/right, shrinking a little as they part - one original
## splitting into two. The split advances over time, surged by energy (a drop pushes
## them apart). Both living cores keep flowing throughout.

var _f: AudioFeatures = AudioFeatures.new()
var _blue: PrismBody
var _red: PrismBody
var _split := 0.0     # 0 = one (together), 1 = two (apart)


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "scene3d"
	framing = "field"
	_blue = PrismBody.new(rng.randi())
	_red = PrismBody.new(rng.randi())
	return {"radius": rng.randf_range(0.26, 0.34)}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.01, 0.02)
	var drive := clampf(f.energy * 0.8 + f.beat * 0.6, 0.0, 1.0)
	_blue.update(delta, drive)
	_red.update(delta, drive)
	_split = minf(1.0, _split + delta * (0.12 + 0.45 * f.energy))   # parts faster on energy
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()
	var sep := _split * 0.30 * u                       # half-separation
	var sc := float(params.radius) * u * (1.0 - 0.15 * _split)
	# Blue is the original (always full); red emerges as the split grows.
	_blue.draw(self, Vector2(-sep, 0.0), sc, 0.6, 1.0)
	_red.draw(self, Vector2(sep, 0.0), sc, 0.0, smoothstep(0.0, 0.5, _split))
