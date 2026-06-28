extends Scene3D

## Planes - the spectrum as a ring of real planes under a forced-perspective camera.
##
## The first scene on the unified 3D path, and a deliberate echo of `spectrum_ring`
## (which bends the spectrum into a 2D circle): here the bars are genuine [Plane3D]
## quads standing on a ground plane in a circle, with a [Mesh3D] crystal tumbling at
## the centre - all projected and depth-sorted through one [Lens3D]. A slow orbit
## with a wide lens makes the near bars loom over the far ones: true perspective and
## real occlusion (the core passes behind and in front of the bars), not a 2D shear.

const R := 2.3                # ring radius, world units
const BAR_W := 0.13           # bar half-width (tangential)
const BASE_H := 0.16          # bar half-height at rest

var _f: AudioFeatures = AudioFeatures.new()
var _bars: Array = []         # parallel to `planes`: {t} spectrum coordinate
var _core: Mesh3D
var _core_basis := Basis.IDENTITY
var _hue := 0.0
var _glow := 0.0
var _yaw := 0.0


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "field"                       # the lens is the camera; keep 2D shots gentle
	_hue = rng.randf()
	lens.fov = rng.randf_range(60.0, 74.0)  # wide lens => forced perspective
	var count := rng.randi_range(30, 46)
	for i in count:
		var a := float(i) / float(count) * TAU
		var t := float(i) / float(maxi(1, count - 1))
		var c := Vector3(cos(a) * R, BASE_H, sin(a) * R)
		var uax := Vector3(-sin(a), 0.0, cos(a)) * BAR_W      # tangential width
		var vax := Vector3.UP * BASE_H
		var pl := Plane3D.new(c, uax, vax, Color.from_hsv(_hue, 0.7, 0.8, 0.9))
		pl.edge = Color(1, 1, 1, 0.22)
		add_plane(pl)
		_bars.append({"t": t})
	_core = Mesh3D.rock("crystal", rng)
	_core_basis = Basis.from_euler(Vector3(rng.randf() * TAU, rng.randf() * TAU, 0.0))
	return {"count": count}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.02, 0.03)
	_glow = lerpf(_glow, clampf(0.3 * f.energy + 0.6 * f.beat, 0.0, 1.0), 1.0 - exp(-6.0 * delta))
	_yaw += delta * (0.16 + 0.30 * f.energy)
	var pitch := 0.30 + 0.08 * sin(_life * 0.2)
	lens.orbit(Vector3(0.0, BASE_H * 1.5, 0.0), 5.4, _yaw, pitch)

	for i in _bars.size():
		var t: float = _bars[i].t
		var band := f.sample(t)
		var h: float = BASE_H + band * 1.5 + f.beat * 0.12
		var pl: Plane3D = planes[i]
		pl.v_axis = Vector3.UP * h
		pl.center.y = h                      # stand on the ground (bottom at y = 0)
		var lit := clampf(0.30 + 0.70 * band + 0.45 * _glow, 0.0, 1.0)
		pl.color = Color.from_hsv(fposmod(_hue + 0.25 * t + 0.05 * _glow, 1.0), 0.7, lit, 0.9)

	# The central crystal turns slowly and reveals its volume (audio drives glow).
	_core_basis = _core_basis * Basis.from_euler(Vector3(0.08, 0.22, 0.0) * delta)
	bodies.clear()
	add_body(_core, _core_basis, Vector3.ZERO, 0.7,
		fposmod(_hue + 0.5, 1.0), 0.5, 2, 0.9, 0.3 + 0.5 * _glow)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	render_world()
