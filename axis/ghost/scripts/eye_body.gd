extends RefCounted
class_name EyeBody

## EyeBody - a real-3D human eyeball, drawn through a [Lens3D] with a real light.
##
## A genuine 3D sclera sphere carries a high-detail iris that is placed *by* the eye's
## 3D orientation (so it foreshortens correctly as the eye turns) but drawn in 2D for
## fidelity the flat-shaded mesh can't reach:
##   sclera  - a glossy near-white SPHERE ([Mesh3D] icosphere, smooth-shaded), with a
##             wet specular sheen and a soft limbal shadow where the iris is set in.
##   iris    - a projected disc with procedural radial fibres (the vein-like detail), a
##             dark limbal ring, a ciliary->collarette colour gradient, and crypts.
##   pupil   - a black disc with a soft inner shadow, dilating with the audio AND
##             *accommodating* to focus depth (a near focus constricts it).
##   cornea  - a wet catchlight (a real reflection point toward the light) plus a faint
##             rim glaze on the light side.
##
## Gaze: the eye looks at a 3D point. [method look_at_point] aims it from its own world
## position, so two eyes sharing a focus *verge* - they toe in on a near point and run
## parallel on a far one - rather than just rotating to the same direction. When no
## owner drives it, it self-saccades (centre-biased).

# Ocular motility limits: a real eyeball rotates only so far in its orbit before the
# muscles stop it (roughly 35 deg). The gaze and its target are held inside this cone so
# the eye can never roll its iris to the pole or turn fully away - the "spinning wildly"
# failure - no matter what the focus geometry or the saccade spring asks for.
const YAW_LIMIT := 0.52        # ~30 deg, left/right
const PITCH_LIMIT := 0.42      # ~24 deg, up/down
const GAZE_VEL_MAX := 12.0     # rad/s ceiling: a frame hitch can't fling the eye

var gaze := Vector2.ZERO       # yaw (x), pitch (y), radians - the eyeball's rotation
var gaze_vel := Vector2.ZERO   # angular velocity of the gaze (the saccade's momentum)
var target := Vector2.ZERO     # where the gaze is heading (set externally if not autonomous)
var _zeta := 0.85              # THIS dart's landing damping (rolled per saccade; see update)
var _last_target := Vector2.ZERO
var autonomous := true         # true = self-saccades; false = the owner drives the gaze/focus
var hue := 0.55                # iris hue (public: a morph handoff can copy it)
var focus_dist := 6.0          # world distance to the thing being looked at (accommodation)
var _sat := 0.65
var _dwell := 0.0
var _dilate := 0.35            # 0..1 pupil openness (fraction of the iris radius)
var _t := 0.0                  # local clock for the gentle ambient-light drift
var _light := 0.6              # 0..1 ambient light ON the eye: drives BOTH the catchlight AND the pupil
var _rng := RandomNumberGenerator.new()

var _sclera: Mesh3D
var _fibres: Array = []        # seeded iris vein fibres (stable per eye)
var _crypts: Array = []        # seeded iris crypts (small dark lacunae)
var _flecks: Array = []        # seeded faint sclera veins near the periphery


func _init(seed_value := 0, hue_override := -1.0) -> void:
	_rng.seed = seed_value
	hue = hue_override if hue_override >= 0.0 else _rng.randf_range(0.05, 0.6)
	_sat = _rng.randf_range(0.55, 0.85)
	_dwell = _rng.randf_range(0.6, 1.5)
	_sclera = Mesh3D.icosphere(3)            # finer -> rounder silhouette; smooth-shaded
	_sclera.compute_normals()
	# Radial vein fibres: many thin strands from the collarette out to the limbus, each
	# at its own angle, length, curve, and shade (bright striae and dark furrows).
	var nf := _rng.randi_range(70, 110)
	for i in nf:
		_fibres.append({
			"a": TAU * float(i) / float(nf) + _rng.randf_range(-0.03, 0.03),
			"ri": _rng.randf_range(0.30, 0.40),
			"ro": _rng.randf_range(0.84, 0.99),
			"shade": _rng.randf_range(-0.34, 0.46),
			"wob": _rng.randf_range(-0.07, 0.07),
			"w": _rng.randf_range(0.6, 1.7),
		})
	for i in _rng.randi_range(4, 8):
		_crypts.append({"a": _rng.randf() * TAU, "r": _rng.randf_range(0.42, 0.72),
			"size": _rng.randf_range(0.05, 0.12), "dark": _rng.randf_range(0.3, 0.6)})
	for i in _rng.randi_range(3, 6):          # a few faint red sclera veins
		_flecks.append({"a": _rng.randf() * TAU, "len": _rng.randf_range(0.25, 0.5),
			"wob": _rng.randf_range(-0.3, 0.3)})


## Aim the eye from its own world position at a world focus point. Two eyes sharing one
## focus will verge (converge on near points, run parallel on far ones); also records
## the focus distance for pupil accommodation.
func look_at_point(eye_pos: Vector3, focus: Vector3) -> void:
	var d := focus - eye_pos
	var dist := d.length()
	if dist < 1e-4:
		return
	focus_dist = dist
	d /= dist
	target = Vector2(atan2(d.x, d.z), -asin(clampf(d.y, -1.0, 1.0)))


func update(dt: float, energy: float) -> void:
	# A gentle, slow drift of the ambient LIGHT falling on the eye, nudged a little by the music. This
	# is the VISIBLE cause of the pupil's size: the catchlight (the reflected highlight) brightens and
	# swells with it, and the pupil follows it INVERSELY - dim light OPENS the pupil, bright light CLOSES
	# it (the pupillary light reflex). So a dilation always coincides with the light visibly dimming,
	# instead of the pupil ballooning for no reason.
	_t += dt
	var light_t := clampf(0.58 + 0.30 * sin(_t * 0.33) + 0.16 * (energy - 0.35), 0.12, 1.0)
	_light = lerpf(_light, light_t, 1.0 - exp(-2.2 * dt))
	# Pupil = accommodation (near focus constricts) + the light reflex (dim -> dilate). The pupil eases
	# faster than the light drifts, so it visibly tracks the light rather than leading it.
	var accom := clampf((focus_dist - 1.2) / 18.0, 0.0, 1.0)        # 0 near .. 1 far
	var pupil_t := clampf(lerpf(0.17, 0.52, accom) + 0.42 * (1.0 - _light), 0.12, 0.80)
	_dilate = lerpf(_dilate, pupil_t, 1.0 - exp(-4.0 * dt))
	if autonomous:
		_dwell -= dt
		if _dwell <= 0.0:
			_saccade()
	# When a new dart begins (the target jumps), roll THIS landing's damping. The nonlinear
	# skew (randf^3) keeps MOST landings well-damped - a clean, subtle stop - and lets only
	# the occasional one come in loose and bouncy. Variance, not a wobble on every move.
	if target.distance_to(_last_target) > 0.02:
		_zeta = lerpf(0.97, 0.68, pow(_rng.randf(), 3.0))
	_last_target = target
	# Hold the aim point inside the motility cone before the spring chases it, so a hard
	# vergence or a divergence nudge can never command an impossible rotation.
	target.x = clampf(target.x, -YAW_LIMIT, YAW_LIMIT)
	target.y = clampf(target.y, -PITCH_LIMIT, PITCH_LIMIT)
	# Move the gaze with the spring: it carries momentum, and when _zeta is low it
	# overshoots and re-fixates - a real saccade, not a smooth ramp into a hard stop.
	var sdt := minf(dt, 0.04)                          # clamp so a frame hitch can't blow it up
	var omega := 22.0                                  # natural frequency (a calmer saccade)
	var accel := (target - gaze) * (omega * omega) - gaze_vel * (2.0 * _zeta * omega)
	gaze_vel += accel * sdt
	if gaze_vel.length() > GAZE_VEL_MAX:               # finite muscle: cap the angular speed
		gaze_vel = gaze_vel.normalized() * GAZE_VEL_MAX
	gaze += gaze_vel * sdt
	# Stop at the orbit's edge: clamp into the cone and kill the velocity into the wall, so
	# the eye settles against its limit instead of rolling past it (the spinning failure).
	if gaze.x < -YAW_LIMIT or gaze.x > YAW_LIMIT:
		gaze_vel.x = 0.0
	if gaze.y < -PITCH_LIMIT or gaze.y > PITCH_LIMIT:
		gaze_vel.y = 0.0
	gaze.x = clampf(gaze.x, -YAW_LIMIT, YAW_LIMIT)
	gaze.y = clampf(gaze.y, -PITCH_LIMIT, PITCH_LIMIT)


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

	# Sclera: glossy near-white sphere, SMOOTH-shaded, with a wet specular sheen.
	_sclera.draw_through(ci, lens, u, Basis.IDENTITY, pos, radius, 0.07, 0.05, 0, fade,
		0.0, 0.0, 0.5, 0.14, Color(0, 0, 0, 0), true)

	# The iris sits on the front of the eye; project it and build its on-screen ellipse
	# basis, so all the 2D detail foreshortens exactly as the eyeball turns.
	var iris_c3 := pos + front * radius * 0.80
	var view := (lens.eye - iris_c3).normalized()
	var facing := front.dot(view)
	var pc := lens.project(iris_c3)
	if facing < 0.04 or pc.z <= lens.near:
		return                                          # iris turned away - sclera only
	var e1 := front.cross(Vector3.UP)
	if e1.length() < 1e-4:
		e1 = front.cross(Vector3.RIGHT)
	e1 = e1.normalized()
	var e2 := front.cross(e1).normalized()
	var iris_world := radius * 0.52
	var center := Vector2(pc.x, pc.y) * u
	var ua := _axis(lens, u, iris_c3, e1, iris_world, center)   # screen image of local x
	var va := _axis(lens, u, iris_c3, e2, iris_world, center)   # screen image of local y
	var iris_px: float = maxf(ua.length(), va.length())          # for sizing line widths
	var a := fade

	_draw_sclera_veins(ci, center, ua, va, a * 0.5)
	# Soft limbal shadow: the sclera dips in around the iris - a dark ring just outside it.
	_ring(ci, center, ua, va, 1.06, Color(0.04, 0.03, 0.04, 0.5 * a), 44)
	_ring(ci, center, ua, va, 1.0, Color(0.05, 0.04, 0.05, 0.35 * a), 44)
	_draw_iris(ci, center, ua, va, iris_px, a)
	# Pupil: black disc with a soft inner shadow, sized by dilation/accommodation.
	var prad: float = clampf(_dilate, 0.14, 0.74) * 0.92
	_ring(ci, center, ua, va, prad + 0.06, Color(0, 0, 0, 0.5 * a), 36)
	ci.draw_colored_polygon(_disc(center, ua, va, prad, 36), Color(0.02, 0.02, 0.03, a))
	# Cornea: the wet catchlight + a faint rim glaze on the light side.
	_draw_cornea(ci, lens, u, pos, radius, front, view, center, ua, va, iris_px, fade)


# The screen vector that the iris-plane unit axis `e` maps to (foreshortened by the
# projection), measured from the iris centre.
func _axis(lens: Lens3D, u: float, c3: Vector3, e: Vector3, world_r: float, center: Vector2) -> Vector2:
	var p := lens.project(c3 + e * world_r)
	return Vector2(p.x, p.y) * u - center


# A filled disc / ring in the iris's projected frame, at local radius `r` (1 = limbus).
func _disc(center: Vector2, ua: Vector2, va: Vector2, r: float, segs: int) -> PackedVector2Array:
	var pts := PackedVector2Array()
	for i in segs:
		var th := TAU * float(i) / float(segs)
		pts.append(center + ua * (cos(th) * r) + va * (sin(th) * r))
	return pts


func _ring(ci: CanvasItem, center: Vector2, ua: Vector2, va: Vector2, r: float, col: Color, segs: int) -> void:
	var pts := _disc(center, ua, va, r, segs)
	pts.append(pts[0])
	ci.draw_polyline(pts, col, maxf(1.0, (ua.length() + va.length()) * 0.5 * 0.04), true)


# The iris body: a limbus->collarette colour gradient (concentric discs), then the
# radial vein fibres, crypts, and the collarette ridge.
func _draw_iris(ci: CanvasItem, center: Vector2, ua: Vector2, va: Vector2, iris_px: float, a: float) -> void:
	var layers := 8
	for i in layers:
		var t := float(i) / float(layers - 1)        # 0 outer (limbus) .. 1 inner
		var r := lerpf(1.0, 0.34, t)
		# Dark limbal ring outside, brighter ciliary body, lifting toward the collarette.
		var val := lerpf(0.16, 0.60, smoothstep(0.0, 1.0, t))
		var sat := clampf(_sat * lerpf(1.05, 0.82, t), 0.0, 1.0)
		var hh := fposmod(hue + 0.03 * t, 1.0)
		ci.draw_colored_polygon(_disc(center, ua, va, r, 44), Color.from_hsv(hh, sat, val, a))
	# Radial fibres - the vein-like detail. A gently curved 3-point strand each.
	var lw: float = maxf(0.8, iris_px * 0.014)
	for f in _fibres:
		var ca: float = cos(f.a)
		var sa: float = sin(f.a)
		var perp_a: float = f.a + PI * 0.5
		var midr: float = (float(f.ri) + float(f.ro)) * 0.5
		var wob: float = f.wob
		var p_in := center + ua * (ca * float(f.ri)) + va * (sa * float(f.ri))
		var p_mid := center + ua * (cos(f.a) * midr + cos(perp_a) * wob) \
			+ va * (sin(f.a) * midr + sin(perp_a) * wob)
		var p_out := center + ua * (ca * float(f.ro)) + va * (sa * float(f.ro))
		var v := clampf(0.42 + float(f.shade), 0.08, 0.96)
		var col := Color.from_hsv(fposmod(hue + 0.02, 1.0), clampf(_sat * 0.85, 0.0, 1.0), v,
			(0.30 + 0.4 * absf(float(f.shade))) * a)
		ci.draw_polyline(PackedVector2Array([p_in, p_mid, p_out]), col, lw * float(f.w), true)
	# Crypts: small dark notches around the collarette.
	for c in _crypts:
		var cc := center + ua * (cos(c.a) * float(c.r)) + va * (sin(c.a) * float(c.r))
		ci.draw_colored_polygon(_disc(cc, ua * float(c.size), va * float(c.size), 1.0, 12),
			Color(0.04, 0.03, 0.03, float(c.dark) * a))
	# Collarette ridge - the boundary of the pupillary zone.
	_ring(ci, center, ua, va, 0.36, Color.from_hsv(hue, clampf(_sat * 0.7, 0, 1), 0.72, 0.55 * a), 40)


func _draw_sclera_veins(ci: CanvasItem, center: Vector2, ua: Vector2, va: Vector2, a: float) -> void:
	for fl in _flecks:
		var r0: float = 1.25
		var r1: float = 1.25 + float(fl.len)
		var p0 := center + ua * (cos(fl.a) * r0) + va * (sin(fl.a) * r0)
		var pm := center + ua * (cos(fl.a + 0.1) * (r0 + r1) * 0.5 + float(fl.wob)) \
			+ va * (sin(fl.a + 0.1) * (r0 + r1) * 0.5)
		var p1 := center + ua * (cos(fl.a) * r1) + va * (sin(fl.a) * r1)
		ci.draw_polyline(PackedVector2Array([p0, pm, p1]),
			Color(0.7, 0.2, 0.2, 0.12 * a), maxf(1.0, (ua.length() + va.length()) * 0.012), true)


func _draw_cornea(ci: CanvasItem, lens: Lens3D, u: float, pos: Vector3, radius: float,
		front: Vector3, view: Vector3, center: Vector2, ua: Vector2, va: Vector2,
		iris_px: float, fade: float) -> void:
	var cornea_pos := pos + front * radius * 0.66
	var facing := front.dot(view)
	# A faint wet glaze across the cornea on the light side (a soft bright crescent).
	var light := Mesh3D.LIGHT.normalized()
	var lit := light - light.project(front)             # light direction in the iris plane
	# The catchlight/glaze track the ambient light `_light` - they brighten and swell as it rises, dim
	# and shrink as it falls - so the pupil's light reflex has a plainly visible source on the cornea.
	var glow := 0.45 + 0.75 * _light         # highlight brightness follows the light
	var swell := 0.68 + 0.5 * _light         # highlight size follows the light
	if lit.length() > 1e-3:
		var lx := lit.dot(e_basis(front, true))
		var ly := lit.dot(e_basis(front, false))
		var gc := center + (ua * lx + va * ly) * 0.5
		ci.draw_colored_polygon(_disc(gc, ua * 0.5, va * 0.5, 1.0, 20),
			Color(1, 1, 1, 0.06 * glow * fade * clampf(facing, 0, 1)))
	# The hot catchlight: a real reflection point toward the light, projected.
	if facing < 0.12:
		return
	var catch3 := cornea_pos + (light + view).normalized() * radius * 0.42
	var pr := lens.project(catch3)
	if pr.z <= lens.near:
		return
	var sp := Vector2(pr.x, pr.y) * u
	var cs: float = maxf(1.5, iris_px * 0.1) * swell
	var a := fade * clampf(facing, 0.0, 1.0)
	ci.draw_circle(sp, cs * 2.4, Color(1, 1, 1, clampf(0.10 * glow, 0.0, 0.3) * a))     # soft bloom
	ci.draw_circle(sp, cs * 1.2, Color(1, 1, 1, clampf(0.45 * glow, 0.0, 0.8) * a))
	ci.draw_circle(sp, cs * 0.55, Color(1, 1, 1, clampf(0.95 * glow, 0.0, 1.0) * a))    # hot core
	ci.draw_circle(sp + Vector2(cs * 1.5, cs * 1.1), cs * 0.4, Color(1, 1, 1, 0.4 * glow * a))  # 2nd glint


# One of the two in-plane unit axes used for the iris frame (kept consistent with draw).
func e_basis(front: Vector3, first: bool) -> Vector3:
	var e1 := front.cross(Vector3.UP)
	if e1.length() < 1e-4:
		e1 = front.cross(Vector3.RIGHT)
	e1 = e1.normalized()
	return e1 if first else front.cross(e1).normalized()
