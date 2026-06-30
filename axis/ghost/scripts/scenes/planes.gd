extends Scene3D

## Planes - the spectrum as a ring of real planes under a forced-perspective camera.
##
## A deliberate echo of `spectrum_ring`: the bars are genuine [Plane3D] quads standing
## on a ground plane in a circle, with an optional [Mesh3D] body tumbling at the centre,
## projected and depth-sorted through one [Lens3D]. Nearly everything is sampled per
## scene so it is never the same shot twice: the lens / orbit (distance, pitch, yaw
## direction and speed), the ring radius and bar geometry, and the central body - which
## may be a rock, a hybrid, a platonic solid, or nothing at all, and is kept small so it
## no longer dominates the frame. It can also bleed a faint sky (stars / fog) behind the
## ring, composed from the shared [Layer] registry.

const BASE_H := 0.16          # bar half-height at rest
const PANEL_H := 0.42         # half-height of each panel (uniform: they SLIDE, they don't grow)
const SLIDE_AMP := 0.85       # how far a panel travels up/down along its single (vertical) axis
const SLIDE_MID := 0.5        # band level that sits a panel at the ring's resting height

var _f: AudioFeatures = AudioFeatures.new()
var _bars: Array = []         # parallel to `planes`: {t} spectrum coordinate
var _core: Mesh3D = null
var _core_basis := Basis.IDENTITY
var _core_scale := 0.45
var _core_spin := Vector3(0.08, 0.22, 0.0)
var _has_core := true
var _hue := 0.0
var _glow := 0.0
var _yaw := 0.0
# Sampled camera + ring (set in build_params).
var _R := 2.3
var _bar_w := 0.13
var _orbit_dist := 5.4
var _pitch_base := 0.30
var _pitch_amp := 0.08
var _yaw_dir := 1.0
var _yaw_base := 0.16
# The whole ring tumbles on a sampled axis (not just the camera's yaw) with a wind-up /
# wind-down angular speed, so the circle tilts and spins on other axes over time.
var _ring_basis := Basis.IDENTITY
var _ring_angle := 0.0
var _tumble_axis := Vector3.UP
var _tumble_base := 0.05
var _tumble_amp := 0.25
var _tumble_rate := 0.3
# A travelling TWIST in the band: around one point on the ring the panels rotate their slide axis
# from UP toward OUTWARD (and a little past it - an inversion), so the band's animation is
# redirected radially over a phase of the circle. The twist point drifts slowly around the ring.
var _twist_a := 0.0
var _twist_max := PI * 0.6
var _twist_w := PI * 0.45
var _twist_speed := 0.2


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "field"                       # the lens is the camera; keep 2D shots gentle
	_hue = rng.randf()
	# Sample the whole shot: lens, orbit, ring - so the angle and framing vary every time.
	lens.fov = rng.randf_range(54.0, 78.0)  # wide lens => forced perspective
	_R = rng.randf_range(1.9, 2.8)
	_bar_w = rng.randf_range(0.09, 0.17)
	_orbit_dist = rng.randf_range(4.6, 7.0)
	_pitch_base = rng.randf_range(0.10, 0.55)
	_pitch_amp = rng.randf_range(0.03, 0.12)
	_yaw_dir = 1.0 if rng.randf() < 0.5 else -1.0
	_yaw_base = rng.randf_range(0.06, 0.16)             # gentler camera spin; the ring tumbles too
	# Ring tumble: an axis biased strongly HORIZONTAL (so the circle visibly TILTS / rolls, not
	# just spins like the camera). The orientation is a slow net spin (_tumble_base) plus a big
	# wind-up / wind-down TILT that swings the ring back and forth (_tumble_amp radians).
	_tumble_axis = Vector3(rng.randf_range(-1, 1), rng.randf_range(-0.15, 0.15),
		rng.randf_range(-1, 1)).normalized()
	_tumble_base = rng.randf_range(-0.08, 0.08)        # net spin rate (rad/s)
	_tumble_amp = rng.randf_range(0.45, 1.15)          # tilt amplitude (rad) of the wind-up/down
	_tumble_rate = rng.randf_range(0.15, 0.4)          # how fast it winds up and back
	# The band twist: how far the panels rotate at the point (a bit past 90 deg = a flip), the
	# phase-width it eases over, and how fast the twist point travels round the ring.
	_twist_max = rng.randf_range(PI * 0.5, PI * 0.78)
	_twist_w = rng.randf_range(PI * 0.28, PI * 0.55)
	_twist_speed = rng.randf_range(0.08, 0.28) * (1.0 if rng.randf() < 0.5 else -1.0)
	_twist_a = rng.randf() * TAU

	var count := rng.randi_range(28, 50)
	for i in count:
		var a := float(i) / float(count) * TAU
		# Palindrome spectrum mapping: t runs 0 -> 1 -> 0 around the ring, so the value at the
		# wrap matches itself and the circle closes seamlessly. (It used to run 0 -> 1 once and
		# jump from treble straight back to bass between the last panel and the first - the hard
		# loop-edge.) Every panel is one continuous circle; each just slides on its own Y axis.
		var u := float(i) / float(count)
		var t := 1.0 - absf(2.0 * u - 1.0)
		var c := Vector3(cos(a) * _R, BASE_H, sin(a) * _R)
		var uax := Vector3(-sin(a), 0.0, cos(a)) * _bar_w      # tangential width
		var vax := Vector3.UP * PANEL_H
		var pl := Plane3D.new(c, uax, vax, Color.from_hsv(_hue, 0.7, 0.8, 0.9))
		pl.edge = Color(1, 1, 1, 0.22)
		add_plane(pl)
		# Store each panel's LOCAL ring position + tangent so the whole ring can be re-oriented
		# by _ring_basis each frame (the panel slides along the ring's local up).
		_bars.append({"t": t, "a": a, "bx": cos(a) * _R, "bz": sin(a) * _R, "ul": uax})

	_make_core(rng)
	# Cross-scene bleed: a faint sky behind the ring, sometimes.
	var bg := rng.randf()
	if bg < 0.40:
		add_layer("stars", rng, {"z": "back", "count": rng.randi_range(70, 130), "hue": 0.6})
	elif bg < 0.58:
		add_layer("fog", rng, {"z": "back", "hue": _hue, "sat": 0.25, "alpha": 0.03, "count": 5})
	return {"count": count}


# The central body: by seed a rock, a hybrid, a platonic solid, or nothing - and small,
# so it is an accent in the ring, not the whole screen.
func _make_core(rng: RandomNumberGenerator) -> void:
	_has_core = rng.randf() >= 0.22                       # ~1 in 5 has no centre at all
	if not _has_core:
		_core = null
		return
	var pick := rng.randf()
	if pick < 0.45:
		_core = Mesh3D.rock(["plain", "rough", "crystal"][rng.randi() % 3], rng)
	elif pick < 0.65:
		_core = Mesh3D.hybrid(rng)
	else:
		match rng.randi() % 4:
			0: _core = Mesh3D.cube()
			1: _core = Mesh3D.octahedron()
			2: _core = Mesh3D.tetrahedron()
			_: _core = Mesh3D.icosphere(1)
	_core_basis = Basis.from_euler(Vector3(rng.randf() * TAU, rng.randf() * TAU, 0.0))
	_core_scale = rng.randf_range(0.28, 0.55)            # small - no longer dominates
	_core_spin = Vector3(rng.randf_range(-0.20, 0.20),
		rng.randf_range(0.10, 0.30) * (1.0 if rng.randf() < 0.5 else -1.0),
		rng.randf_range(-0.15, 0.15))


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.02, 0.03)
	update_layers(f, delta)
	_glow = lerpf(_glow, clampf(0.3 * f.energy + 0.6 * f.beat, 0.0, 1.0), 1.0 - exp(-6.0 * delta))
	_yaw += delta * (_yaw_base + 0.30 * f.energy) * _yaw_dir
	var pitch := _pitch_base + _pitch_amp * sin(_life * 0.2)
	lens.orbit(Vector3(0.0, BASE_H * 1.5, 0.0), _orbit_dist, _yaw, pitch)

	# Tumble the whole ring: a slow net spin that accumulates (energy nudges it faster) plus a
	# big wind-up / wind-down TILT swinging it back and forth, so the circle visibly tilts and
	# rolls on axes the camera alone can't.
	_ring_angle += delta * (_tumble_base + 0.12 * f.energy)
	var angle := _ring_angle + _tumble_amp * sin(_life * _tumble_rate)
	_ring_basis = Basis(_tumble_axis, angle)
	_twist_a = wrapf(_twist_a + delta * _twist_speed, -PI, PI)     # the twist point travels round the ring

	for i in _bars.size():
		var bar: Dictionary = _bars[i]
		var band := f.sample(float(bar.t))
		var amp: float = band * 1.5 + f.beat * 0.12
		var pl: Plane3D = planes[i]
		# Twist: how much THIS panel is rotated, peaking at the travelling twist point and easing to
		# nothing a phase-window away. Its slide axis rotates from UP toward the OUTWARD radial (a
		# bit past, an inversion), so near the twist the panel stands out radially and its animation
		# is redirected outward instead of up.
		var d := absf(wrapf(float(bar.a) - _twist_a, -PI, PI))
		var w := 1.0 - smoothstep(0.0, _twist_w, d)
		var ang := w * _twist_max
		var radial := Vector3(float(bar.bx), 0.0, float(bar.bz)) / _R       # unit outward
		var axis := Vector3.UP * cos(ang) + radial * sin(ang)              # up -> outward (-> inverted)
		# Each panel SLIDES along its (possibly twisted) axis by its band, then the whole ring is
		# re-oriented by the tumble basis - so the undulating circle also rolls and tilts as a body.
		var local := Vector3(float(bar.bx), 0.0, float(bar.bz)) + axis * (SLIDE_AMP * (amp - SLIDE_MID))
		pl.center = _ring_basis * local
		pl.u_axis = _ring_basis * (bar.ul as Vector3)
		pl.v_axis = _ring_basis * (axis * PANEL_H)
		var lit := clampf(0.30 + 0.70 * band + 0.45 * _glow, 0.0, 1.0)
		pl.color = Color.from_hsv(fposmod(_hue + 0.25 * float(bar.t) + 0.05 * _glow, 1.0), 0.7, lit, 0.9)

	bodies.clear()
	if _has_core and _core != null:
		_core_basis = _core_basis * Basis.from_euler(_core_spin * delta)
		add_body(_core, _core_basis, Vector3.ZERO, _core_scale,
			fposmod(_hue + 0.5, 1.0), 0.5, 2, 0.9, 0.3 + 0.5 * _glow)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers("back")          # faint sky behind the ring (stars / fog), when present
	render_world()
	draw_layers("front")
