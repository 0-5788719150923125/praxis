extends Scene3D

## Eye + prism - the right eye becomes its digital self (the-point, scenes 3-5).
##
## The two eyes' composition continues, but the RIGHT eye dissolves and CRYSTALLIZES into
## the glowing BLUE wireframe [PrismBody] while the LEFT human eye remains, watching. Then
## the prism "looks around" the void on its own; and toward the end the remaining eye begins
## to TREMBLE and vibrate, faster and faster, light building around it - the riser into the
## drop. Arrives by morph from `two_eyes` (`morph_in = "eyes"`): it reuses the SAME two
## EyeBody instances, so nothing jumps - the left eye simply keeps looking while the right
## one turns to crystal in place. Hands off to `two_prisms` (`morph_out = "eye2prism"`): the
## live blue PrismBody plus the trembling eye's slot, so the drop can burst a red prism there.

var _f: AudioFeatures = AudioFeatures.new()
var _rng := RandomNumberGenerator.new()
var _eye: EyeBody              # the surviving LEFT human eye
var _reye: EyeBody             # the RIGHT eye, dissolving as the prism forms
var _blue: PrismBody           # the crystallized right eye
var _off := 0.28               # world x of each slot (matches two_eyes)
var _eye_rad := 0.27           # world radius of the eye
var _t := 0.0                  # scene-local clock (reset on build AND morph)
var _crys := 0.0               # 0..1 crystallization of the right eye -> blue prism
var _tremble := 0.0            # 0..1 the surviving eye's vibration (the riser)
var _flash := 0.0              # the form-flash envelope at the crystallize moment
var _focus := Vector3(0, 0, 6.0)
var _focus_target := Vector3(0, 0, 6.0)
var _focus_dwell := 0.0
const HUE_BLUE := 0.6


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "scene3d"
	framing = "plane"
	morph_in = "eyes"
	morph_out = "eye2prism"
	_rng.seed = rng.randi()
	var h := rng.randf_range(0.05, 0.6)
	_eye = EyeBody.new(rng.randi(), h)
	_reye = EyeBody.new(rng.randi(), h)
	_eye.autonomous = false
	_reye.autonomous = false
	_blue = PrismBody.new(rng.randi())
	_off = rng.randf_range(0.24, 0.30)
	_eye_rad = rng.randf_range(0.24, 0.30)
	lens.eye = Vector3(0, 0, 4.0)
	lens.look = Vector3.ZERO
	lens.fov = 48.0
	_t = 0.0
	_crys = 0.0
	return {}


# Arrived from two_eyes: keep BOTH live eyes (the left survives, the right crystallizes),
# and continue their shared focus - so the swap is invisible and only the crystallize reads.
func begin_morph(from: GhostScene) -> void:
	var p := from.morph_payload()
	if p.is_empty():
		return
	if p.has("left"):
		_eye = p["left"]
	if p.has("right"):
		_reye = p["right"]
	_eye.autonomous = false
	_reye.autonomous = false
	_off = float(p.get("offset", _off))
	_eye_rad = float(p.get("radius", _eye_rad))
	_focus = p.get("focus", _focus)
	_focus_target = _focus
	_t = 0.0
	_crys = 0.0
	_tremble = 0.0


# Hand the live blue prism and the trembling eye's slot to two_prisms, so the drop bursts a
# red prism exactly where the eye was and the blue one simply keeps living. Slots are handed
# over as SCREEN unit-fractions (the 2D space the prism scenes draw in), projected from the
# 3D slots, so nothing shifts across the morph.
func morph_payload() -> Dictionary:
	lens.prepare()
	var bj := lens.project(Vector3(_off, 0, 0))
	var ej := lens.project(Vector3(-_off, 0, 0))
	var scale_frac := _eye_rad * lens._focal / maxf(0.1, bj.z) * 1.15   # unit-fraction, * unit() = px
	return {"blue": _blue, "blue_slot": Vector2(bj.x, bj.y), "eye_slot": Vector2(ej.x, ej.y),
		"scale": scale_frac, "eye_hue": _eye.hue}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	last_f = f
	tick(f, delta)
	drift_view(f, 0.004, 0.008)                 # nearly static, per the brief
	_t += delta
	var drive := clampf(f.energy * 0.7 + f.beat * 0.4, 0.0, 1.0)
	# Keyframes are paced as fractions of the scene's hold (phase_span), so the crystallize and the
	# riser always complete on screen however short the scene is; the ambient life (the eye's gaze,
	# the prism's spin) runs on the raw delta below and is never sped up.
	var H := phase_span(7.0)
	var frac := clampf(_t / H, 0.0, 1.0)

	# 1) Crystallize the right eye into the blue prism over the first ~18% of the hold (a form-flash
	#    at the hand-off point where the eye is half-gone and the wireframe half-there).
	var pre := _crys
	_crys = minf(1.0, _crys + delta / maxf(0.2, H * 0.18))
	if pre < 0.5 and _crys >= 0.5:
		_flash = 1.0
	_flash = maxf(0.0, _flash - delta * 2.4)

	# 2) The surviving eye's gaze drifts near -> far and lingers (same feel as `eye`).
	_focus_dwell -= delta
	if _focus_dwell <= 0.0:
		_new_focus()
	_eye.look_at_point(Vector3(-_off, 0, 0), _focus_target)

	# 3) The riser: the eye trembles harder as the build rises. A slow floor over _t makes it
	#    always crescendo, and the audio energy adds the live jitter on top; it will burst on
	#    the drop (the storyboard cuts to two_prisms there).
	var ramp := smoothstep(0.42, 1.0, frac)                  # riser builds over the back half of the hold
	var tt := clampf(ramp * (0.55 + 0.6 * drive), 0.0, 1.0)
	_tremble = lerpf(_tremble, tt, 1.0 - exp(-6.0 * delta))

	# The eye's own dilation reacts to the whole drive plus the tremble (it widens as it strains).
	_eye.update(delta, clampf(drive + _tremble * 0.5, 0.0, 1.0))
	if _crys < 1.0:
		_reye.look_at_point(Vector3(_off, 0, 0), _focus_target)
		_reye.update(delta, drive)
	# The blue prism comes to life a touch stronger as it finishes forming.
	_blue.update(delta, clampf(drive * (0.4 + 0.6 * _crys), 0.0, 1.0))
	queue_redraw()


func _new_focus() -> void:
	var tier := _rng.randf()
	var d: float
	var dwell: float
	if tier < 0.4:
		d = _rng.randf_range(2.2, 4.0)
		dwell = _rng.randf_range(2.2, 4.0)
	else:
		d = _rng.randf_range(8.0, 40.0)
		dwell = _rng.randf_range(2.0, 3.6)
	var sacc := EyeBody.saccade_target(_rng)
	var lat := _rng.randf_range(0.12, 0.4)
	_focus_target = Vector3(sacc.x * lat, sacc.y * lat * 0.7, d)
	_focus_dwell = dwell


func _draw() -> void:
	begin_draw()
	lens.prepare()
	var u := unit()

	# The blue prism sits exactly where the right eye was: project the right slot to the screen.
	var pj := lens.project(Vector3(_off, 0, 0))
	var pc := Vector2(pj.x, pj.y) * u
	var pscale := _eye_rad * lens._focal / maxf(0.1, pj.z) * u * 1.15   # ~eye footprint, a touch larger

	# The surviving left eye, trembling. High-frequency vibration jitters its world position and
	# its gaze; a soft light builds around it as the riser peaks.
	var epos := Vector3(-_off, 0, 0)
	if _tremble > 0.001:
		var jx := (sin(_t * 61.0) + 0.6 * sin(_t * 97.0 + 1.3)) * 0.5
		var jy := (sin(_t * 71.0 + 0.7) + 0.6 * sin(_t * 113.0)) * 0.5
		epos += Vector3(jx, jy, 0.0) * (0.05 * _tremble)
		_draw_eye_light(lens.project(epos), u, _tremble)
	_eye.draw(self, lens, u, epos, _eye_rad)

	# The crystallizing right eye fades out as the wireframe fades in; a white flash at the crossover.
	if _crys < 1.0:
		_reye.draw(self, lens, u, Vector3(_off, 0, 0), _eye_rad, clampf(1.0 - _crys, 0.0, 1.0))
	_blue.draw(self, pc, pscale, HUE_BLUE, smoothstep(0.0, 1.0, _crys))
	if _flash > 0.001:
		draw_circle(pc, pscale * (0.6 + 1.2 * _flash), Color(0.75, 0.86, 1.0, 0.5 * _flash))
		draw_circle(pc, pscale * (0.3 + 0.6 * _flash), Color(1, 1, 1, 0.6 * _flash))


# A soft radial glow gathering on the eye as it trembles - the "light building around it".
func _draw_eye_light(pj: Vector3, u: float, k: float) -> void:
	if pj.z <= lens.near:
		return
	var c := Vector2(pj.x, pj.y) * u
	var base := _eye_rad * lens._focal / maxf(0.1, pj.z) * u
	var rings := 6
	for i in rings:
		var fr := 1.0 - float(i) / float(rings - 1)      # 1 rim .. 0 core
		var rr := base * (1.2 + 3.0 * fr) * (0.7 + 0.5 * k)
		draw_circle(c, rr, Color(0.7, 0.82, 1.0, clampf(0.05 * k * (1.0 - 0.85 * fr), 0.0, 0.3)))
