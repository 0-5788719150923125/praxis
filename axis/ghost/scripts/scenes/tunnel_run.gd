extends Scene3D

## Tunnel run - the camera on a track, falling forward through a tube that twists, banks and
## sometimes turns right over.
##
## The one thing in this catalogue that is a RIDE. Everything else is looked AT - a body in a
## void, a field, a landscape under an orbiting camera - and the difference a first-person track
## makes is not a matter of degree: the frame stops being a picture and becomes a place the
## viewer is inside of. That is also why it is the scene with the strongest opinion about its own
## camera: no shot, no drift worth mentioning, because a tunnel is only a tunnel while the
## vanishing point holds still.
##
## THE TRACK IS INTEGRATED, NOT SPLINED, and that is what lets it loop. A curve fitted through
## control points can bend but it cannot roll, and a path defined as `(x(s), y(s), s)` - the
## obvious way - can never turn back on itself at all, because z increases by construction. Here
## a FRAME is carried along instead (right / up / forward, orthonormal) and turned each step by
## three seeded curvature functions of arclength: yaw about its own up, pitch about its own
## right, roll about its own forward. Integrate that and a sustained pitch is a loop, a sustained
## yaw is a spiral staircase, and roll is the barrel the whole tube turns through while it does
## either. The functions are sums of sinusoids at incommensurate wavelengths, so the track never
## repeats and never needs to be stored.
##
## NOTHING IS EVER STORED, in fact: stations are generated a few dozen ahead of the camera and
## dropped behind it, so the track is endless at constant memory and there is no seam to hide.
##
## WHAT IS IN THE TUNNEL. The wall's cross-section is a small Fourier series in the angle - one
## roll gives a circle, a lobed star, a flattened duct or a fluted column - and it TWISTS along
## the track, so the flutes spiral past instead of running straight. Its radius swells and closes
## on a long wavelength, which is what makes the tube open into chambers and neck back down. RIBS
## punctuate it at a fixed interval in arclength, so they arrive at a steady rate and read as
## distance covered; SEAMS run the whole length at a few fixed angles, which is the thing that
## makes the roll legible (a smooth tube rolling looks like a still tube). MOTES hang in the air
## and stream past. Everything fades into a fog at the view distance, so the tunnel recedes into
## haze rather than ending at a hole.
##
## WHAT THE MUSIC DOES. The SPEED, which is the whole feeling of the thing - loud is fast. And
## the wall's radius, per angle, from the spectrum: the cross-section carries the bands, wound
## into a spiral along the track so the camera flies through the music rather than past a bar
## chart of it. Beats brighten the ribs. The track's SHAPE is never touched: a tunnel whose
## curvature moved on the beat would be a camera being shaken, which is a different and much
## cheaper sensation.
##
## THE COST, AND WHY IT IS OFF-THREAD. A frame is around a thousand quads, each projected,
## culled, shaded and depth-sorted - five to fifteen milliseconds of GDScript, which is most of
## a frame budget. It is built inside a [FrameForge] job like the other scene3d scenes. The one
## frame of latency that costs is uniform (the job is kicked every frame and delivered in order),
## so it reads as a constant time offset rather than as judder.

## Arclength between stations. Fine enough that the wall's curvature is smooth at the near end,
## coarse enough that the visible track is a few dozen rings rather than a few hundred.
const STEP := 0.55

## How many stations are kept ahead of the camera, and how far behind. The view distance is
## STEP * AHEAD; the fog is sized from it so the far end is always haze.
##
## BEHIND IS LARGE BECAUSE THE CAMERA DOES NOT LOOK ALONG ITS OWN TANGENT. It aims at a point
## several stations up the track (see update), so on a bend the view axis tilts tens of degrees
## off the direction of travel - and then the frustum sweeps back over tube the camera has
## already passed. At three stations there was not enough of it, and the gap showed as BLACK
## WEDGES at the frame edges. Rings behind the eye are nearly free: they fail the near-plane
## test and cost one comparison each.
const AHEAD := 56
const BEHIND := 16

## How far up the track the camera aims, in stations. Enough that a bend reads as a bend being
## taken rather than as one permanently at the edge of frame; not so far that the view axis
## leaves the tube on the tightest tracks (which is also what BEHIND has to cover).
const LOOK_AHEAD := 5.0

## The cross-section's resolution. Sixteen was the first cut and the near wall came out as a
## handful of enormous flat plates: at a 90-degree lens the first ring subtends most of the
## frame, so its facets are the picture. Twenty-eight reads as a tube, and Gouraud shading in
## the job does the rest.
const SIDES := 28

## The seeded cross-sections. Each is a pair of Fourier terms on the angle - amplitude and
## harmonic - and the whole vocabulary of tube shapes this scene has.
const PROFILES := {
	"round": {"a1": [0.0, 0.04], "n1": [2, 3], "a2": [0.0, 0.03], "n2": [5, 7]},
	"fluted": {"a1": [0.07, 0.16], "n1": [6, 12], "a2": [0.0, 0.04], "n2": [3, 4]},
	"star": {"a1": [0.16, 0.30], "n1": [3, 6], "a2": [0.0, 0.06], "n2": [8, 12]},
	"duct": {"a1": [0.10, 0.20], "n1": [2, 2], "a2": [0.04, 0.10], "n2": [4, 4]},
	"ribbed": {"a1": [0.05, 0.10], "n1": [4, 5], "a2": [0.06, 0.12], "n2": [9, 14]},
}

## Track characters. The curvature amplitudes are radians per unit of arclength, so an amplitude
## near 0.2 held over a quarter wavelength is most of a full loop - which is why `coaster` always
## turns right over, and tests/tunnel_track_check.gd holds it to that.
##
## THESE ARE DEGREES OF THE SAME THING, not separate worlds, and it is worth being exact about
## why. Once ROLL has turned the frame, a YAW about the frame's own now-tilted up axis IS world
## pitch - so any character with a barrel roll can end up in any orientation whatever its own
## pitch column says. Two attempts to make `glide` provably level failed on that and its table
## was tightened twice for nothing; these numbers are about how HARD a track turns, which is the
## only thing they were ever able to express.
const TRACKS := {
	"glide": {"yaw": [0.03, 0.09], "pitch": [0.02, 0.06], "roll": [0.02, 0.10], "wave": [55.0, 110.0]},
	"weave": {"yaw": [0.10, 0.19], "pitch": [0.05, 0.12], "roll": [0.06, 0.18], "wave": [30.0, 60.0]},
	"corkscrew": {"yaw": [0.05, 0.11], "pitch": [0.05, 0.11], "roll": [0.22, 0.48], "wave": [40.0, 80.0]},
	"coaster": {"yaw": [0.06, 0.14], "pitch": [0.16, 0.30], "roll": [0.10, 0.28], "wave": [45.0, 95.0]},
}

var _forge := FrameForge.new()
var _sch: Scheme

# The track's shape, as three curvature functions of arclength. Two sinusoids each, at
# wavelengths that share no common multiple, so the ride never comes back round to itself.
var _k_yaw := PackedFloat32Array()      # [amp0, w0, ph0, amp1, w1, ph1]
var _k_pitch := PackedFloat32Array()
var _k_roll := PackedFloat32Array()

# The stations: the track, generated ahead and dropped behind.
var _pos: Array = []                    # Vector3
var _bas: Array = []                    # Basis (x right, y up, -z forward)
var _s0 := 0.0                          # arclength of station 0

# Where the camera is, and how fast.
var _s := 0.0
var _speed := 6.0
var _base_speed := 6.0

# The tube.
var _radius := 1.25
var _prof_name := "round"
var _a1 := 0.0
var _n1 := 3.0
var _o1 := 0.0
var _a2 := 0.0
var _n2 := 7.0
var _o2 := 0.0
var _twist := 0.0                       # radians of cross-section rotation per unit arclength
var _swell := 0.18                      # how much the radius opens into chambers
var _swell_w := 40.0
var _rib_every := 9
var _rib_depth := 0.16
var _seams := PackedInt32Array()
var _band_gain := 0.16
var _band_wind := 0.05                  # how fast the spectrum spirals along the track

# Look.
var _hue := 0.0
var _hue_travel := 0.004                # colour zones per unit of track (a period, not a rate)
var _fog := Color.BLACK
var _lit := 1.0
var _mote_n := 0
var _motes: Array = []
var _light_th := 1.2          # where the lit strip runs, as an angle in the cross-section
var _glow := 0.0
var _energy := 0.0
var _ch := Vector2.ZERO
var _f: AudioFeatures = AudioFeatures.new()


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "scene3d"
	# A tunnel fills the frame by definition, so it takes the gentle shots - and even those are
	# damped hard in update(). The vanishing point is the subject; anything that moves it is
	# the camera coming loose from the vehicle.
	framing = "field"
	_sch = Scheme.pick(rng)
	_hue = _sch.vary(rng)
	# One full swing between the scheme's two hues every 60 to 300 units of track.
	_hue_travel = rng.randf_range(0.0033, 0.0167)

	var track := String(TRACKS.keys()[rng.randi() % TRACKS.size()])
	var tr: Dictionary = TRACKS[track]
	_k_yaw = _roll_curve(rng, tr["yaw"], tr["wave"])
	_k_pitch = _roll_curve(rng, tr["pitch"], tr["wave"])
	_k_roll = _roll_curve(rng, tr["roll"], tr["wave"])

	_prof_name = String(PROFILES.keys()[rng.randi() % PROFILES.size()])
	var pf: Dictionary = PROFILES[_prof_name]
	_a1 = _pick(rng, pf["a1"])
	_n1 = float(rng.randi_range(int(pf["n1"][0]), int(pf["n1"][1])))
	_a2 = _pick(rng, pf["a2"])
	_n2 = float(rng.randi_range(int(pf["n2"][0]), int(pf["n2"][1])))
	_o1 = rng.randf() * TAU
	_o2 = rng.randf() * TAU
	# THE TWIST is what stops a shaped cross-section reading as an extrusion. Without it the
	# flutes are straight rails running to the vanishing point and the eye reads the tube as a
	# static pipe however much the track bends; with it they spiral past.
	_twist = rng.randf_range(-0.16, 0.16)

	_radius = rng.randf_range(1.0, 1.7)
	_swell = rng.randf_range(0.10, 0.30)
	_swell_w = rng.randf_range(26.0, 70.0)
	_rib_every = rng.randi_range(5, 14)
	_rib_depth = rng.randf_range(0.08, 0.22)
	var seam_n := rng.randi_range(0, 4)
	var seam_off := rng.randi() % SIDES
	for i in seam_n:
		_seams.append((seam_off + i * int(SIDES / maxi(1, seam_n))) % SIDES)
	_band_gain = rng.randf_range(0.09, 0.22)
	_band_wind = rng.randf_range(0.015, 0.075)

	_base_speed = rng.randf_range(4.0, 10.0)
	_speed = _base_speed
	lens.fov = rng.randf_range(66.0, 92.0)          # wide: the walls have to rush past
	_lit = rng.randf_range(0.85, 1.25)
	_light_th = rng.randf() * TAU
	_fog = _sch.color(_sch.opposed(_hue, 0.22), 0.55, rng.randf_range(0.05, 0.16))

	_mote_n = 0 if rng.randf() < 0.25 else rng.randi_range(30, 110)
	for i in _mote_n:
		# `at` is an ABSOLUTE arclength, not an offset from the camera, and that distinction was
		# the whole bug. Held as an offset, every mote sits the same distance ahead for ever: the
		# field travels WITH the eye and nothing ever streams past, which on the nearest objects
		# in the frame is most of what "jittery" was. It is recycled a view-length forward once
		# the camera has gone by - see _advance_track.
		_motes.append({
			"at": rng.randf() * STEP * float(AHEAD),
			"th": rng.randf() * TAU,
			"rr": rng.randf_range(0.10, 0.86),
			"sz": rng.randf_range(0.012, 0.045),
			"hue": fposmod(_hue + rng.randf_range(-0.12, 0.12), 1.0),
		})

	_seed_track()

	return {
		"track": track,
		"profile": _prof_name,
		"mood": _sch.name,
		"speed": _base_speed,
		"radius": _radius,
		"fov": lens.fov,
		"twist": _twist,
		"ribs": _rib_every,
		"seams": seam_n,
		"motes": _mote_n,
	}


func _pick(rng: RandomNumberGenerator, band: Array) -> float:
	return rng.randf_range(float(band[0]), float(band[1]))


## One curvature function: two sinusoids in arclength. Their wavelengths are deliberately NOT
## in a simple ratio - a track whose turns share a period comes back to the same shape every
## few seconds, and on a tunnel that reads as driving in circles.
func _roll_curve(rng: RandomNumberGenerator, amp: Array, wave: Array) -> PackedFloat32Array:
	var w0 := _pick(rng, wave)
	var out := PackedFloat32Array()
	out.append(_pick(rng, amp))
	out.append(TAU / w0)
	out.append(rng.randf() * TAU)
	out.append(_pick(rng, amp) * rng.randf_range(0.3, 0.7))
	out.append(TAU / (w0 * rng.randf_range(0.37, 0.61)))
	out.append(rng.randf() * TAU)
	return out


func _curve(k: PackedFloat32Array, s: float) -> float:
	return k[0] * sin(k[1] * s + k[2]) + k[3] * sin(k[4] * s + k[5])


# ---------------------------------------------------------------------------------
# The track
# ---------------------------------------------------------------------------------

func _seed_track() -> void:
	_pos.clear()
	_bas.clear()
	_s0 = 0.0
	# THE CAMERA STARTS WHERE IT WILL LIVE, `BEHIND` stations into the buffer, and not at zero.
	# _advance_track holds it at that offset for the rest of the scene, so starting anywhere else
	# means the opening seconds run in a part of the buffer the interpolation cannot serve: with
	# the cubic needing a station either side, the eye is PINNED at station one until the track
	# has caught up, and then lurches into motion. It measured as a single acceleration spike
	# sixty times the mean - worse, on that measure, than the polyline it replaced.
	_s = float(BEHIND) * STEP
	_pos.append(Vector3.ZERO)
	_bas.append(Basis.IDENTITY)
	while _pos.size() < AHEAD + BEHIND + 2:
		_extend()


## Push one more station, by turning the carried frame and stepping along its forward axis.
## Godot's convention is that -Z is forward, so the step is `-basis.z`; yaw turns about the
## frame's own up and pitch about its own right, which is what makes the two compose into
## corkscrews and loops instead of into a wobble in a fixed world plane.
func _extend() -> void:
	var b: Basis = _bas[_bas.size() - 1]
	var p: Vector3 = _pos[_pos.size() - 1]
	var s := _s0 + float(_pos.size() - 1) * STEP
	b = b.rotated(b.y, _curve(_k_yaw, s) * STEP)
	b = b.rotated(b.x, _curve(_k_pitch, s) * STEP)
	b = b.rotated(-b.z, _curve(_k_roll, s) * STEP)
	# Re-orthonormalised every step. Three successive rotations of a stored basis accumulate
	# skew, and over the thousands of stations one song's worth of track is, the frame quietly
	# stops being orthogonal - which shows up as the tube shearing into an ellipse.
	b = b.orthonormalized()
	_pos.append(p + (-b.z) * STEP)
	_bas.append(b)


## Drop stations the camera has passed and generate the same number ahead, so the buffer holds
## the same window of track whatever distance has been covered.
func _advance_track() -> void:
	while _s - _s0 > float(BEHIND) * STEP and _pos.size() > 2:
		_pos.remove_at(0)
		_bas.remove_at(0)
		_s0 += STEP
	while _pos.size() < AHEAD + BEHIND + 2:
		_extend()
	# Motes the camera has gone past come back a view-length ahead, so a fixed handful of them
	# serves an endless track.
	var span := STEP * float(AHEAD)
	for m in _motes:
		if float(m["at"]) < _s - STEP:
			m["at"] = float(m["at"]) + span


## The frame at an arbitrary arclength. Returns [position, basis].
##
## CUBIC, AND THAT IS THE WHOLE FIX FOR THE JITTER. Stations are joined by STRAIGHT segments -
## `_extend` steps along the frame's forward axis - so interpolating between them linearly puts
## the eye on a POLYLINE. Its position is continuous, which is why this looked correct in a still
## frame; its VELOCITY is not. At every station the direction of travel changes abruptly by the
## curvature times the step, and at eight units a second over a step of 0.55 that is fifteen
## direction changes a second. Reported as "super jittery... a sequence of discrete jumps at each
## step", which is exactly what it is.
##
## A Catmull-Rom through the four surrounding stations is C1: the velocity is continuous, so
## there is no per-station kink left to see. The orientation gets the same treatment on the
## quaternion, because a lerp of two bases has the identical defect one derivative up. The
## polyline was only ever Euler's approximation of the curve the curvature functions describe,
## so this is nearer the real track as well as smoother.
func _frame_at(s: float) -> Array:
	var fi := (s - _s0) / STEP
	# One station of margin either side, for the cubic's four points.
	var i := clampi(int(floor(fi)), 1, _pos.size() - 3)
	var t := clampf(fi - float(i), 0.0, 1.0)
	var p: Vector3 = (_pos[i] as Vector3).cubic_interpolate(
		_pos[i + 1] as Vector3, _pos[i - 1] as Vector3, _pos[i + 2] as Vector3, t)
	var q := Quaternion(_bas[i] as Basis).spherical_cubic_interpolate(
		Quaternion(_bas[i + 1] as Basis), Quaternion(_bas[i - 1] as Basis),
		Quaternion(_bas[i + 2] as Basis), t)
	return [p, Basis(q)]


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	# Almost nothing. A tunnel is only a tunnel while the vanishing point holds still, and the
	# 2D view drift that suits a field scene reads here as the camera coming loose.
	drift_view(f, 0.004, 0.006)
	update_layers(f, delta)
	_ch = chroma_hue()
	var e := clampf(f.energy, 0.0, 1.0)
	_energy = lerpf(_energy, e, 1.0 - exp(-2.5 * delta))
	_glow = maxf(_glow * exp(-3.6 * delta), clampf(f.beat, 0.0, 1.0))

	# SPEED IS THE FEELING, so it is what the music gets. Bounded well above zero: a tunnel that
	# stops is a corridor, and the whole read of the scene is that it cannot stop.
	_speed = _base_speed * (0.55 + 0.95 * _energy)
	_s += _speed * delta
	_advance_track()

	var fr := _frame_at(_s)
	var p: Vector3 = fr[0]
	var b: Basis = fr[1]
	# The eye sits at the station and looks a few units up the track rather than straight along
	# the tangent. Aiming at a point AHEAD is what makes a bend read as a bend being taken -
	# the vanishing point leads into the turn the way a driver's eyes do - where a pure tangent
	# keeps the turn permanently at the edge of frame.
	var la := _frame_at(_s + STEP * LOOK_AHEAD)
	lens.eye = p
	lens.look = la[0]
	lens.up = b.y
	queue_redraw()

	var job := TunnelJob.new()
	job.f = f
	job.s = _s
	job.s0 = _s0
	job.step = STEP
	job.sides = SIDES
	job.n = mini(_pos.size(), AHEAD + BEHIND + 2)
	for i in job.n:
		var q: Vector3 = _pos[i]
		var bb: Basis = _bas[i]
		job.px.append(q.x)
		job.py.append(q.y)
		job.pz.append(q.z)
		job.rx.append(bb.x.x)
		job.ry.append(bb.x.y)
		job.rz.append(bb.x.z)
		job.ux.append(bb.y.x)
		job.uy.append(bb.y.y)
		job.uz.append(bb.y.z)
	job.lens = Lens3D.new()
	job.lens.eye = lens.eye
	job.lens.look = lens.look
	job.lens.up = lens.up
	job.lens.fov = lens.fov
	job.u = unit()
	job.radius = _radius
	job.a1 = _a1
	job.n1 = _n1
	job.o1 = _o1
	job.a2 = _a2
	job.n2 = _n2
	job.o2 = _o2
	job.twist = _twist
	job.swell = _swell
	job.swell_w = _swell_w
	job.rib_every = _rib_every
	job.rib_depth = _rib_depth
	job.seams = _seams
	job.band_gain = _band_gain
	job.band_wind = _band_wind
	# The wall's hue travels along the track, so the ride passes through colour zones instead of
	# being one tinted pipe; the tonal centre pulls the whole thing toward the music's key.
	job.hue = GhostScene.blend_hue(_hue, _ch.x, clampf(_ch.y, 0.0, 1.0) * 0.35)
	job.hue_travel = _hue_travel
	job.accent = _sch.accent
	job.sat = _sch.sat
	job.lit = _lit * (0.8 + 0.45 * _energy)
	job.glow = _glow
	job.fog = _fog
	job.light_th = _light_th
	# COPIED, not handed over. `_motes` keeps being recycled by _advance_track while the worker
	# is reading, and a job that shares a mutable array with the sim is the one thing
	# [FrameForge]'s contract forbids.
	for m in _motes:
		job.m_at.append(float(m["at"]))
		job.m_th.append(float(m["th"]))
		job.m_rr.append(float(m["rr"]))
		job.m_sz.append(float(m["sz"]))
		job.m_hue.append(float(m["hue"]))
	_forge.kick(job.run, {}, self, job)


func _draw() -> void:
	begin_draw()
	# The fog colour behind everything: the far end of the tube fades into it, and the ring of
	# frame outside the near wall on a hard bend is it too.
	var x := size.x * 0.5 * 1.25 / maxf(0.001, view.zoom_actual())
	var y := size.y * 0.5 * 1.25 / maxf(0.001, view.zoom_actual())
	draw_colored_polygon(PackedVector2Array([
		Vector2(-x, -y), Vector2(x, -y), Vector2(x, y), Vector2(-x, y)]), _fog)
	_forge.submit(self)


# ---------------------------------------------------------------------------------
# The worker job: ring the track, cull, shade, sort, batch
# ---------------------------------------------------------------------------------

## One frame of tunnel, built off the main thread ([FrameForge]'s job form: a fresh object per
## kick, so nothing the worker reads can be mutated underneath it). It reads only its own
## members - never the scene node - so a Director cut mid-build is harmless.
class TunnelJob:
	extends RefCounted

	## The eye plane the wall is clipped against. Further out than [member Lens3D.near] on
	## purpose: a vertex sitting a hundredth of a unit in front of the camera projects tens of
	## thousands of pixels off screen, and the geometry is no more visible for it.
	const NEAR := 0.12

	## The fraction of the generated track at which the fog is complete. Under 1.0 so the last
	## several rings are pure haze and there is no end to see.
	const FOG_REACH := 0.72

	var f: AudioFeatures
	var lens: Lens3D
	var u := 1.0
	var s := 0.0
	var s0 := 0.0
	var step := 0.55
	var sides := 16
	var n := 0
	# The stations, flattened: position and the two cross-section axes. Flat arrays because this
	# crosses onto a worker thread while the scene keeps generating track.
	var px := PackedFloat32Array()
	var py := PackedFloat32Array()
	var pz := PackedFloat32Array()
	var rx := PackedFloat32Array()
	var ry := PackedFloat32Array()
	var rz := PackedFloat32Array()
	var ux := PackedFloat32Array()
	var uy := PackedFloat32Array()
	var uz := PackedFloat32Array()

	var radius := 1.25
	var a1 := 0.0
	var n1 := 3.0
	var o1 := 0.0
	var a2 := 0.0
	var n2 := 7.0
	var o2 := 0.0
	var twist := 0.0
	var swell := 0.18
	var swell_w := 40.0
	var rib_every := 9
	var rib_depth := 0.16
	var seams := PackedInt32Array()
	var band_gain := 0.16
	var band_wind := 0.05
	var hue := 0.0
	var hue_travel := 0.004
	var accent := 0.5
	var sat := 0.6
	var lit := 1.0
	var glow := 0.0
	var fog := Color.BLACK
	var light_th := 1.2
	var m_at := PackedFloat32Array()
	var m_th := PackedFloat32Array()
	var m_rr := PackedFloat32Array()
	var m_sz := PackedFloat32Array()
	var m_hue := PackedFloat32Array()

	func run(_snapshot: Dictionary) -> Array:
		var tb := TriBatch.new()
		if n < 3 or sides < 3:
			return tb.take_chunks()
		lens.prepare()
		var reach := step * float(n - 1)
		var faces: Array = []

		# --- THE RING VERTICES, projected AND SHADED once each. Adjacent quads share them, so
		# doing either per quad would repeat the work four times over - and shading per quad is
		# also what made the first cut read as a stack of flat plates rather than as a tube.
		# Colour belongs to the vertex here (Gouraud): the wall is a smooth surface and its
		# shading has to be continuous across a quad edge or the facets ARE the picture.
		var vx := PackedFloat32Array()
		var vy := PackedFloat32Array()
		var vd := PackedFloat32Array()          # camera depth
		var vc := PackedColorArray()
		var wx := PackedFloat32Array()          # ...and the world position, for the cull
		var wy := PackedFloat32Array()
		var wz := PackedFloat32Array()
		var cnt := n * sides
		vx.resize(cnt)
		vy.resize(cnt)
		vd.resize(cnt)
		vc.resize(cnt)
		wx.resize(cnt)
		wy.resize(cnt)
		wz.resize(cnt)
		var rad := PackedFloat32Array()
		rad.resize(n)
		for i in n:
			var si := s0 + float(i) * step
			# The radius: the seeded swell (chambers opening and necking down) in arclength, so
			# it belongs to the TUNNEL and not to the camera.
			var rr := radius * (1.0 + swell * sin(TAU * si / maxf(4.0, swell_w)))
			# THE RIBS BELONG TO THE TUNNEL, not to the window on it. `i % rib_every` counts
			# from whichever station happens to be at the near end of the buffer, and that end
			# moves: `_advance_track` retires one station every time the camera covers a step,
			# which at this speed is ten times a second. So every rib in shot stepped BACKWARD
			# by a station, ten times a second, while the walls and the fog - which are keyed
			# to arclength - carried on forward. Reported exactly that way: "it almost feels
			# like the camera is moving forward while the geometry is moving backward, and then
			# it jumps ahead to correct itself... at least 5 times per second."
			#
			# `si / step` is the ABSOLUTE station number (s0 advances by exactly one step per
			# retirement, so the quotient stays a whole number), which is fixed in the world.
			var rib := rib_every > 0 and (int(round(si / step)) % rib_every) == 0
			if rib:
				rr *= 1.0 - rib_depth
			rad[i] = rr
			var base := i * sides
			# THE HUE WALKS BETWEEN THE SCHEME'S TWO COLOURS AND COMES BACK, rather than
			# accumulating. `hue + travel * s` was the first cut and arclength grows without
			# bound: fourteen seconds in, the hue had turned two thirds of the way round the
			# wheel and the tunnel was a colour the scheme never contains. Bounded, it does what
			# it was for - the ride passes through colour zones - and stays on palette.
			var hue_i := GhostScene.blend_hue(hue, accent,
				0.5 - 0.5 * cos(TAU * si * hue_travel))
			for k in sides:
				var th := TAU * float(k) / float(sides) + twist * si
				# THE SPECTRUM, WOUND ALONG THE TRACK. Mapping the bands straight onto the angle
				# makes a bar chart that happens to be bent into a circle; adding the arclength
				# winds it into a helix, so the camera flies THROUGH the music.
				var band: float = f.sample(fposmod(th / TAU + si * band_wind, 1.0))
				var pr := rr * _prof(th) * (1.0 + band_gain * band)
				var cx := cos(th)
				var cy := sin(th)
				var qx := px[i] + rx[i] * cx * pr + ux[i] * cy * pr
				var qy := py[i] + ry[i] * cx * pr + uy[i] * cy * pr
				var qz := pz[i] + rz[i] * cx * pr + uz[i] * cy * pr
				var pj := lens.project(Vector3(qx, qy, qz))
				var vi := base + k
				vx[vi] = pj.x * u
				vy[vi] = pj.y * u
				vd[vi] = pj.z
				wx[vi] = qx
				wy[vi] = qy
				wz[vi] = qz
				# THE LIGHT IS FIXED IN THE TUBE'S OWN CROSS-SECTION, not in the world and not
				# on the camera. A headlight is the obvious choice and it is wrong twice over:
				# the inward normal of a tube is very nearly perpendicular to the view axis, so
				# a headlight shades the whole wall identically and the tube comes out flat -
				# and a light fixed in the WORLD sweeps around the wall as the track banks,
				# which reads as the tunnel rotating rather than as the vehicle rolling. A lit
				# strip running along the tunnel does neither: it rolls WITH the tube, which is
				# what finally makes a barrel roll legible.
				# WRAPPED, not clamped. `max(0, cos)` puts a hard terminator halfway round the
				# tube and everything past it at the ambient floor - which on a wide lens is
				# half the frame, dead. Remapping to 0..1 keeps a full gradient the whole way
				# round: a lit side, a shadow side, and no line between them.
				var lam := 0.5 + 0.5 * cos(th - light_th)
				# A fine banding along the track, so the near wall - which at this field of view
				# is most of the frame - has some surface to it rather than being a smooth wash.
				var panel := 0.90 + 0.10 * sin(si * 2.7)
				var shade := (0.20 + 0.80 * pow(lam, 1.35)) * panel
				var sa := sat
				var vv := shade * lit
				var h := hue_i
				if rib:
					h = GhostScene.blend_hue(hue_i, accent, 0.65)
					vv = vv * 0.9 + 0.22 + 0.30 * glow
					sa = clampf(sat * 1.1, 0.0, 1.0)
				elif seams.has(k):
					vv = vv * 0.85 + 0.30
					sa = clampf(sat * 0.55, 0.0, 1.0)
				vc[vi] = _fogged(Color.from_hsv(h, sa, clampf(vv, 0.0, 1.35)), si - s, reach)

		# --- The wall, one quad per (station, side), with its four vertex colours.
		for i in n - 1:
			for k in sides:
				var k2 := (k + 1) % sides
				var i0 := i * sides + k
				var i1 := i * sides + k2
				var i2 := (i + 1) * sides + k2
				var i3 := (i + 1) * sides + k
				var c0 := Vector3(wx[i0], wy[i0], wz[i0])
				var c2 := Vector3(wx[i2], wy[i2], wz[i2])
				var nrm := (Vector3(wx[i1], wy[i1], wz[i1]) - c0).cross(c2 - c0)
				if nrm.length_squared() < 1e-12:
					continue
				# Back-face cull. Inside a tube the visible surface is the one whose normal
				# points back at the eye; the other half of every ring is the far wall's
				# outside, which is hidden and costs a quad to draw.
				if nrm.dot(lens.eye - (c0 + c2) * 0.5) <= 0.0:
					continue
				var behind := int(vd[i0] <= NEAR) + int(vd[i1] <= NEAR) \
					+ int(vd[i2] <= NEAR) + int(vd[i3] <= NEAR)
				if behind == 4:
					continue
				if behind == 0:
					faces.append({"d": (vd[i0] + vd[i2]) * 0.5,
						"p": PackedVector2Array([
							Vector2(vx[i0], vy[i0]), Vector2(vx[i1], vy[i1]),
							Vector2(vx[i2], vy[i2]), Vector2(vx[i3], vy[i3])]),
						"c": PackedColorArray([vc[i0], vc[i1], vc[i2], vc[i3]])})
					continue
				# STRADDLING THE EYE PLANE, and this case is the whole reason the clip exists.
				# Dropping such a quad outright was the first cut and it punches BLACK WEDGES
				# into the frame - not slivers at the very edge, because at a wide field of view
				# the ring nearest the camera is enormous on screen, so one lost quad of it is
				# a hole a tenth of the frame across. Seen in the first render, three of them.
				var cl := _clip_near(
					[Vector3(wx[i0], wy[i0], wz[i0]), Vector3(wx[i1], wy[i1], wz[i1]),
						c2, Vector3(wx[i3], wy[i3], wz[i3])],
					[vd[i0], vd[i1], vd[i2], vd[i3]],
					[vc[i0], vc[i1], vc[i2], vc[i3]])
				var cp: Array = cl[0]
				if cp.size() < 3:
					continue
				var sp := PackedVector2Array()
				var sc := PackedColorArray()
				var dsum := 0.0
				for w in cp.size():
					var pj3 := lens.project(cp[w])
					sp.append(Vector2(pj3.x * u, pj3.y * u))
					sc.append((cl[1] as Array)[w])
					dsum += pj3.z
				faces.append({"d": dsum / float(cp.size()), "p": sp, "c": sc})

		# --- Motes: specks hanging in the tube, so the air between the camera and the wall is
		# not empty. They are what makes the SPEED legible up close - the wall's own detail is
		# all far away and slides slowly, and a fast tunnel with nothing near the lens reads
		# oddly serene.
		for k in m_at.size():
			var si2: float = m_at[k]
			# INTERPOLATED between stations, never snapped to one. `int((si - s0) / step)` was the
			# first cut and it teleports every mote a whole station at a time - a fifth of a
			# second of travel, on the objects nearest the lens, fifteen times a second.
			var fidx := (si2 - s0) / step
			var idx := int(floor(fidx))
			if idx < 0 or idx >= n - 1:
				continue
			var ft := clampf(fidx - float(idx), 0.0, 1.0)
			var rr2: float = lerpf(rad[idx], rad[idx + 1], ft) * m_rr[k]
			var th2: float = m_th[k] + twist * si2
			var cx2 := cos(th2)
			var cy2 := sin(th2)
			var pj2 := lens.project(Vector3(
				lerpf(px[idx], px[idx + 1], ft)
					+ lerpf(rx[idx], rx[idx + 1], ft) * cx2 * rr2
					+ lerpf(ux[idx], ux[idx + 1], ft) * cy2 * rr2,
				lerpf(py[idx], py[idx + 1], ft)
					+ lerpf(ry[idx], ry[idx + 1], ft) * cx2 * rr2
					+ lerpf(uy[idx], uy[idx + 1], ft) * cy2 * rr2,
				lerpf(pz[idx], pz[idx + 1], ft)
					+ lerpf(rz[idx], rz[idx + 1], ft) * cx2 * rr2
					+ lerpf(uz[idx], uz[idx + 1], ft) * cy2 * rr2))
			if pj2.z <= 0.35:
				continue                        # too close to be a speck; it would be a plate
			# Clamped, and it has to be: a screen size of `world / depth` is unbounded as the
			# depth goes to zero, and an unclamped mote arriving at the lens is a flat pasted
			# square filling a tenth of the frame. Seen, in the first render.
			var sz := clampf(m_sz[k] * u / pj2.z, 1.0, u * 0.008)
			var near_fade := clampf((pj2.z - 0.35) * 1.6, 0.0, 1.0)
			var c := Vector2(pj2.x * u, pj2.y * u)
			var mc := Color.from_hsv(m_hue[k], clampf(sat * 0.7, 0.0, 1.0),
				clampf(0.75 + 0.5 * glow, 0.0, 1.0), near_fade)
			var fc := _fogged(mc, si2 - s, reach)
			fc.a = mc.a
			faces.append({"d": pj2.z,
				"p": PackedVector2Array([
					c + Vector2(-sz, -sz), c + Vector2(sz, -sz),
					c + Vector2(sz, sz), c + Vector2(-sz, sz)]),
				"c": PackedColorArray([fc, fc, fc, fc])})

		# Far first. A tube seen from inside mostly does not occlude itself - each ring's quads
		# are at different angles from the eye - but a track that loops brings far track back
		# into the near field, and then it does. The sort is what makes that case correct.
		faces.sort_custom(func(a, b): return float(a["d"]) > float(b["d"]))
		for fc2 in faces:
			# Emitted as a FAN rather than as a quad, because the near clip can hand back a
			# triangle or a pentagon as easily as a quad.
			var fp: PackedVector2Array = fc2["p"]
			var fcol: PackedColorArray = fc2["c"]
			for w in range(1, fp.size() - 1):
				tb.tri_colored(fp[0], fp[w], fp[w + 1], fcol[0], fcol[w], fcol[w + 1])
		return tb.take_chunks()


	## Sutherland-Hodgman against the eye plane: keep the part of a polygon in front of the
	## camera, cutting each crossing edge at the plane and interpolating its colour there.
	## Returns [points, colours].
	func _clip_near(pts: Array, dep: Array, cols: Array) -> Array:
		var op: Array = []
		var oc: Array = []
		var m := pts.size()
		for e in m:
			var b := (e + 1) % m
			var da: float = dep[e]
			var db: float = dep[b]
			if da > NEAR:
				op.append(pts[e])
				oc.append(cols[e])
			if (da > NEAR) != (db > NEAR):
				var t := clampf((NEAR - da) / (db - da), 0.0, 1.0)
				op.append((pts[e] as Vector3).lerp(pts[b] as Vector3, t))
				oc.append((cols[e] as Color).lerp(cols[b] as Color, t))
		return [op, oc]

	## The cross-section, as a two-term Fourier series in the angle. One roll of the amplitudes
	## gives a circle, a lobed star, a flattened duct or a fluted column - which is the whole
	## vocabulary of tube shapes this scene has, out of four numbers.
	func _prof(th: float) -> float:
		return maxf(0.35, 1.0 + a1 * cos(n1 * th + o1) + a2 * cos(n2 * th + o2))

	## Fade toward the fog by DISTANCE ALONG THE TRACK, not by distance from the camera.
	##
	## Camera depth is the obvious choice and it is wrong on exactly the tracks this scene exists
	## for. Where the tunnel bends hard, the far end of the generated track is only a few units
	## from the eye in a straight line while being thirty units along the tube - so it fogs as
	## though it were close, and the last ring shows as a faceted blob hanging in mid-air with
	## nothing behind it. Rendered, seen, twice. Arclength has no such failure: the end of the
	## track is always the end of the track, so it is always fully haze and the tube has no
	## visible end however it is bent.
	##
	## Smoothstep rather than an exponential, because this has to REACH one: an exponential is
	## asymptotic, and the four percent it leaves behind at the far ring is that same blob at
	## lower contrast.
	func _fogged(c: Color, ahead: float, reach: float) -> Color:
		return c.lerp(fog, smoothstep(0.04, FOG_REACH, ahead / maxf(0.01, reach)))
