extends GhostScene

## Prism - a single living wireframe tetrahedron (from "the-point").
##
## A see-through 4-point prism with a living neural core, ported from the browser
## Prism via [PrismBody]: glowing edges only, tendrils flowing from the centre,
## hovering and slowly "looking around". The camera holds (the brief: static,
## forward-facing); the core comes to life with the audio.
##
## Colour is a [Scheme] mood rather than the old blue-or-red coin flip. Only the hue
## reaches the body (it draws its own saturation from the browser's palette), and
## every hue is a plausible crystal, so nothing is excluded.
##
## THE SHELL DOES NOT VARY. It is the browser tetrahedron, every time - see [PrismBody]'s
## `form` for the five-way shell roll that briefly lived there and why it was taken out. A
## prism is a tetrahedron; the variety belongs in its colour and its framing, not in its
## solid. The liberty this scene takes on top is PRESENCE.

## How much of the frame the prism claims. A solitary body has no count and no
## arrangement to vary, so scale IS its silhouette: a mote lost in the void and a
## shape that overruns the view are different shots of the same object, and the old
## 0.30-0.40 band could only ever give the middle one.
const PRESENCE := {
	"distant":  [0.15, 0.21],
	"portrait": [0.28, 0.38],
	"looming":  [0.48, 0.62],
}

var _f: AudioFeatures = AudioFeatures.new()
var _prism: PrismBody
var _hue := 0.6


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "scene3d"
	framing = "plane"                          # the prism hovers; keep the view square-on
	var sch := Scheme.pick(rng)
	_hue = sch.vary(rng)
	_prism = PrismBody.new(rng.randi())
	var pres := String(PRESENCE.keys()[rng.randi() % PRESENCE.size()])
	var band: Array = PRESENCE[pres]
	return {"radius": rng.randf_range(float(band[0]), float(band[1])),
		"mood": sch.name, "hue": _hue, "presence": pres, "form": _prism.form}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.01, 0.02)                  # nearly static, per the brief
	_prism.update(delta, clampf(f.energy * 0.8 + f.beat * 0.6, 0.0, 1.0))
	queue_redraw()


func _draw() -> void:
	begin_draw()
	_prism.draw(self, Vector2.ZERO, float(params.radius) * unit(), _hue)
