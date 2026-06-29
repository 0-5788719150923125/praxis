extends GhostScene

## Prism - a single living wireframe tetrahedron (from "the-point").
##
## A see-through 4-point prism with a living neural core, ported from the browser
## Prism via [PrismBody]: glowing edges only, tendrils flowing from the centre,
## hovering and slowly "looking around". Blue or red by seed. The camera holds (the
## brief: static, forward-facing); the core comes to life with the audio.

var _f: AudioFeatures = AudioFeatures.new()
var _prism: PrismBody
var _hue := 0.6


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "scene3d"
	framing = "plane"                          # the prism hovers; keep the view square-on
	_hue = 0.6 if rng.randf() < 0.5 else 0.0   # electric blue or deep red
	_prism = PrismBody.new(rng.randi())
	return {"radius": rng.randf_range(0.30, 0.40)}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.01, 0.02)                  # nearly static, per the brief
	_prism.update(delta, clampf(f.energy * 0.8 + f.beat * 0.6, 0.0, 1.0))
	queue_redraw()


func _draw() -> void:
	begin_draw()
	_prism.draw(self, Vector2.ZERO, float(params.radius) * unit(), _hue)
