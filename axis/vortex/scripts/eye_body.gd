extends RefCounted
class_name EyeBody

## EyeBody - a real-3D human eyeball, drawn through a [Lens3D] with a real light.
##
## Discrete geometries, each lit differently (not faked in 2D):
##   sclera  - a matte white SPHERE ([Mesh3D] icosphere), diffuse.
##   iris    - a recessed CAP ([Mesh3D] dome, concave) on the front, with colour +
##             a little surface texture; it foreshortens as the eye turns.
##   pupil   - a small flat-black cap at the iris centre, dilating with the audio.
##   cornea  - a clear convex DOME over the iris, translucent and glossy, so it
##             catches a wet specular highlight from the light source.
## It looks around by ROTATING the whole eyeball in 3D (centre-biased saccades), so
## the iris/cornea genuinely swing across the front and shrink toward the limb. The
## wet catchlight is a real reflection point on the cornea, projected - it tracks the
## eye, the camera, and the light, because it is 3D, not a pasted-on dot.

var gaze := Vector2.ZERO       # yaw (x), pitch (y), radians - the eyeball's rotation
var target := Vector2.ZERO     # where the gaze is heading (set externally if not autonomous)
var autonomous := true         # true = self-saccades; false = the owner drives `target`
var hue := 0.55                # iris hue (public: a morph handoff can copy it)
var _sat := 0.65
var _dwell := 0.0
var _dilate := 0.3
var _rng := RandomNumberGenerator.new()

var _sclera: Mesh3D
var _iris: Mesh3D
var _pupil: Mesh3D
var _cornea: Mesh3D


func _init(seed_value := 0, hue_override := -1.0) -> void:
	_rng.seed = seed_value
	hue = hue_override if hue_override >= 0.0 else _rng.randf_range(0.05, 0.6)
	_sat = _rng.randf_range(0.55, 0.85)
	_dwell = _rng.randf_range(0.6, 1.5)
	_sclera = Mesh3D.icosphere(3)            # finer -> rounder silhouette; smooth-shaded
	_sclera.compute_normals()
	_iris = Mesh3D.dome(3, 28, -0.10)        # shallow concave bowl
	_iris.texturize(0.18, 6.0, _rng)         # faint fibre-ish surface variation
	_iris.compute_normals()
	_pupil = Mesh3D.dome(2, 20, -0.05)
	_cornea = Mesh3D.dome(4, 30, 0.34)       # clear bulging lens
	_cornea.compute_normals()


func update(dt: float, energy: float) -> void:
	_dilate = lerpf(_dilate, clampf(0.25 + 0.6 * energy, 0.0, 1.0), 1.0 - exp(-4.0 * dt))
	if autonomous:
		_dwell -= dt
		if _dwell <= 0.0:
			_saccade()
	gaze = gaze.lerp(target, 1.0 - exp(-30.0 * dt))   # fast snap = a jerky saccade


# A centre-biased saccade target (radians). Static helper so a multi-eye owner can
# reuse the same distribution to drive a SHARED gaze.
static func saccade_target(rng: RandomNumberGenerator) -> Vector2:
	if rng.randf() < 0.4:
		return Vector2.ZERO                             # prefer the neutral forward gaze
	var a := rng.randf() * TAU
	var r := rng.randf()
	r = r * r * 0.5                                     # squared -> centre bias
	return Vector2(cos(a), sin(a)) * r


func _saccade() -> void:
	target = saccade_target(_rng)
	_dwell = _rng.randf_range(0.35, 1.7)


## Draw the eyeball at world [param pos], world radius [param radius], through the
## [param lens]. [param fade] scales opacity (for the split morph's emergence).
func draw(ci: CanvasItem, lens: Lens3D, u: float, pos: Vector3, radius: float, fade := 1.0) -> void:
	var eyeb := Basis.from_euler(Vector3(gaze.y, gaze.x, 0.0))
	var front: Vector3 = eyeb * Vector3(0, 0, 1)

	# Sclera: matte near-white sphere, SMOOTH-shaded (no facets).
	_sclera.draw_through(ci, lens, u, Basis.IDENTITY, pos, radius, 0.09, 0.05, 0, fade,
		0.0, 0.0, 0.04, 0.95, Color(0, 0, 0, 0), true)
	# Iris: recessed cap, gaze-rotated (foreshortens in 3D), smooth.
	_iris.draw_through(ci, lens, u, eyeb, pos + front * radius * 0.80, radius * 0.5,
		hue, _sat, 0, fade, 0.0, 0.0, 0.2, 0.5, Color(0, 0, 0, 0), true)
	# Pupil: flat black, dilating (unlit, no smoothing needed).
	var prad := radius * 0.5 * (0.30 + 0.5 * _dilate)
	_pupil.draw_through(ci, lens, u, eyeb, pos + front * radius * 0.84, prad,
		0.0, 0.0, 0, fade, 0.0, 0.0, 0.0, 1.0, Color(0.02, 0.02, 0.03, 1.0))
	# Cornea: clear glossy dome over the iris (translucent + tight specular), smooth.
	var cornea_pos := pos + front * radius * 0.66
	_cornea.draw_through(ci, lens, u, eyeb, cornea_pos, radius * 0.62,
		hue, 0.12, 0, fade * 0.16, 0.0, 0.0, 0.95, 0.1, Color(0, 0, 0, 0), true)
	# Wet catchlight: a real 3D reflection point on the cornea toward the light.
	_draw_catchlight(ci, lens, u, cornea_pos, radius, front, fade)


func _draw_catchlight(ci: CanvasItem, lens: Lens3D, u: float, cornea_pos: Vector3,
		radius: float, front: Vector3, fade: float) -> void:
	var view := (lens.eye - cornea_pos).normalized()
	var facing := front.dot(view)
	if facing < 0.15:
		return                                          # cornea turned away - no glint
	var light := Mesh3D.LIGHT.normalized()
	var catch3 := cornea_pos + (light + view).normalized() * radius * 0.42
	var pr := lens.project(catch3)
	if pr.z <= lens.near:
		return
	var sp := Vector2(pr.x, pr.y) * u
	var er := _screen_radius(lens, cornea_pos, radius, u)
	var cs := er * 0.12
	var a := fade * clampf(facing, 0.0, 1.0)
	ci.draw_circle(sp, cs * 2.4, Color(1, 1, 1, 0.10 * a))     # soft bloom
	ci.draw_circle(sp, cs * 1.2, Color(1, 1, 1, 0.45 * a))
	ci.draw_circle(sp, cs * 0.55, Color(1, 1, 1, 0.95 * a))    # hot core
	ci.draw_circle(sp + Vector2(cs * 1.4, cs * 1.1), cs * 0.42, Color(1, 1, 1, 0.4 * a))  # 2nd reflection


# Projected on-screen radius of the eyeball (px), for sizing the catchlight.
func _screen_radius(lens: Lens3D, pos: Vector3, radius: float, u: float) -> float:
	var c := lens.project(pos)
	var e := lens.project(pos + Vector3(radius, 0.0, 0.0))
	return (Vector2(e.x, e.y) - Vector2(c.x, c.y)).length() * u
