extends GhostScene

## Clockwork - meshing gears under forced restraint, dramatic by physics not by clipart.
##
## Gears go cheesy the instant they become decoration: brass cogs, googly spinning, a
## rainbow steampunk sticker. This scene avoids all of that by being a real mechanism in
## the dark. Every gear is a luminous cog edge over a near-black body; the only colour is
## one cold metal tint and the white specular that sweeps the lit teeth as the wheel
## turns. What sells it is the MOTION, and the motion is true:
##
##   meshing   - two gears mesh only if they share a module (tooth size); a child's tooth
##               count is m = 2R, so a bigger wheel has more teeth. Meshed wheels
##               counter-rotate with angular speed inversely proportional to teeth
##               (omega_b = -omega_a * N_a/N_b), and their phases interlock tip-into-gap
##               and STAY interlocked for all time (a rolling constraint, derived once).
##   in unison - a group can also be a chorus: equal wheels, unmeshed, all turning at the
##               same speed and direction - a hypnotic wall, not a meshed train.
##   ticking   - a group can run an escapement: instead of turning smoothly it advances
##               one tooth per tick, snapping on a slight underdamped spring (the clock
##               recoil), locked to the beat and a fallback cadence.
##   async     - several groups at once, each its own speed / direction / mode - and a
##               vast slow wheel bigger than the frame arcing behind everything for depth.
##
## A scene's seed picks a mode (one big mechanism, scattered trains, a clock, a chorus)
## and then samples every constant, so it is never the same machine twice. Audio drives
## the turn rate, the tick, the glow, and the travelling specular highlight.

const MODES := ["orrery", "trains", "clockwork", "chorus", "split", "split"]   # split weighted up

# A small set of restrained metal tints (hue, saturation). No rainbow: one cold metal,
# desaturated, so the picture reads as machined material lit in the dark.
const METALS := [
	[0.58, 0.18],   # steel blue
	[0.09, 0.22],   # brass / bronze
	[0.04, 0.20],   # copper
	[0.10, 0.05],   # bone / nickel
	[0.62, 0.10],   # gunmetal
]

var _f: AudioFeatures = AudioFeatures.new()
var _rng := RandomNumberGenerator.new()
var _gears: Array = []        # each: {group, pos, R, teeth, phase, omega, depth, geom...}
var _groups: Array = []       # each: motion driver shared by its gears (see _new_group)
var _hue := 0.58
var _sat := 0.18
var _glow := 0.0
var _beat_pulse := 0.0
var _beat_prev := 0.0
var _light_ang := 0.0         # world angle of the key light; the specular sweep tracks it
var _light_drift := 0.0
var _light_sigma := 0.8       # angular width of the lit arc of teeth


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	_rng.seed = rng.randi()
	framing = "subject"                       # a mechanism is a subject: allow the slow push-in
	var metal: Array = METALS[rng.randi() % METALS.size()]
	_hue = fposmod(float(metal[0]) + rng.randf_range(-0.02, 0.02), 1.0)
	_sat = float(metal[1])
	_light_ang = rng.randf_range(-PI, PI)
	_light_drift = rng.randf_range(-0.06, 0.06)
	_light_sigma = rng.randf_range(0.5, 1.05)

	var mode: String = MODES[rng.randi() % MODES.size()]
	match mode:
		"orrery":
			_build_orrery(rng)
		"trains":
			var n := rng.randi_range(2, 3)
			for t in n:
				_build_train(rng, "tick" if rng.randf() < 0.45 else "smooth")   # some trains click
		"clockwork":
			_build_train(rng, "smooth")          # a smooth going-train ...
			_build_train(rng, "tick")            # ... beside an escapement that ticks
			if rng.randf() < 0.5:
				_build_train(rng, "tick")
		"chorus":
			_build_chorus(rng)
		"split":
			_build_split(rng)

	# Some free, unmeshed gears - teeth need not always connect. (Skip for split: it is already
	# a deliberate composition.)
	if mode != "split" and rng.randf() < 0.55:
		_build_free(rng, rng.randi_range(1, 2))
	# A gear or two behind the others, counter-rotating, for depth.
	_build_bg(rng, 1 if mode == "split" else rng.randi_range(1, 2))
	# A vast, dim, almost-still wheel arcing behind it all, sometimes - extra depth and gravity.
	if rng.randf() < 0.35:
		_build_backdrop(rng)

	# Paint far wheels first so near ones occlude them.
	_gears.sort_custom(func(a, b): return float(a.depth) < float(b.depth))
	return {"mode": mode}


# --- group + gear construction ----------------------------------------------

# A motion driver shared by every gear assigned to it. "smooth" integrates an angle
# continuously; "tick" advances one driver-tooth per tick and settles on a spring.
func _new_group(kind: String, rng: RandomNumberGenerator, depth: float) -> int:
	_groups.append({
		"kind": kind,
		"s": 0.0, "vel": 0.0, "s_goal": 0.0,         # shared spin scalar (+ spring state for tick)
		"speed": rng.randf_range(0.12, 0.55),        # smooth: radians of s per second
		"dir": 1.0 if rng.randf() < 0.5 else -1.0,
		"tooth_step": 0.2,                           # tick: one driver tooth = TAU / N_driver
		"tick_t": 0.0, "tick_period": rng.randf_range(0.55, 1.30),
		"beat_sync": true,                           # tick on the beat too (false -> own cadence, for sequences)
		"stiffness": rng.randf_range(150.0, 320.0),  # tick spring: high = a crisp snap
		"damping": rng.randf_range(14.0, 24.0),      # under critical -> a little recoil overshoot
		"depth": depth,
		"sway": Vector2(rng.randf_range(-1.0, 1.0), rng.randf_range(-1.0, 1.0)).normalized()
			* rng.randf_range(0.0, 0.03),            # per-group parallax sway amplitude
		"sway_off": Vector2.ZERO,
	})
	return _groups.size() - 1


# Build one gear's local geometry (unit pitch radius = 1) and append it. The tooth
# count fixes the module; geometry is sampled per wheel so spokes and hub vary.
func _add_gear(group: int, pos: Vector2, R: float, teeth: int, phase: float,
		omega: float, depth: float) -> Dictionary:
	# Tooth LENGTH varies per wheel: the addendum is the module baseline scaled by a sampled
	# factor, so some wheels wear short stubby teeth and others long ones (not all the same).
	var add: float = clampf(2.2 / float(teeth), 0.05, 0.20) * _rng.randf_range(0.6, 1.6)
	var ded := add * _rng.randf_range(1.0, 1.4)              # dedendum (root depth) varies too
	var tip_r := 1.0 + add
	var root_r := 1.0 - ded
	var p := TAU / float(teeth)
	# Tooth WIDTH varies per wheel, at BOTH ends: tip half-width and the wider root half-width
	# (the flank angle), so teeth range from slim and pointed to broad and blocky across gears.
	var tw := _rng.randf_range(0.08, 0.26)                    # tip half-width
	var bw := _rng.randf_range(0.30, 0.46)                    # root (base) half-width - always > tw
	var style := "flat" if _rng.randf() < 0.4 else "wire"     # some solid flat-colour, some skeletal
	var pts := PackedVector2Array()
	for k in teeth:
		var a0 := float(k) * p
		# A trapezoidal tooth: root, up the flank to a flat tip, down the far flank, root.
		var angs := [a0 - bw * p, a0 - tw * p, a0 + tw * p, a0 + bw * p]
		var rads := [root_r, tip_r, tip_r, root_r]
		for j in 4:
			pts.append(Vector2(cos(angs[j]), sin(angs[j])) * rads[j])
	pts.append(pts[0])                                        # close the rim

	var spoke_n: int = [3, 4, 5, 6][_rng.randi() % 4]
	var spokes := []
	var s0 := _rng.randf_range(-PI, PI)
	for i in spoke_n:
		spokes.append(s0 + TAU * float(i) / float(spoke_n))

	# Rust: some wheels carry corroded patches - soft orange-brown blobs in their alpha, pinned
	# to the wheel (polar a, d) so they turn with it, mottling the metal as a worn texture.
	var rust := []
	if _rng.randf() < 0.45:
		for i in _rng.randi_range(4, 9):
			rust.append({"a": _rng.randf_range(-PI, PI), "d": _rng.randf_range(0.10, root_r * 0.9),
				"r": root_r * _rng.randf_range(0.12, 0.36), "al": _rng.randf_range(0.18, 0.45)})

	var g := {
		"group": group, "pos": pos, "R": R, "teeth": teeth, "style": style,
		"phase": phase, "omega": omega, "depth": depth, "rust": rust,
		"teeth_local": pts, "tip_r": tip_r, "root_r": root_r,
		"hub_r": root_r * _rng.randf_range(0.18, 0.30),
		"bore_r": root_r * _rng.randf_range(0.06, 0.12),
		"body_r": root_r * 0.99, "spokes": spokes,
	}
	_gears.append(g)
	return g


# The meshing solution: given a parent gear and the direction `alpha` from parent centre
# to the child centre, return the child's [phase, omega] so the two interlock tip-into-gap
# and remain meshed for all time. Counter-rotation and the 1/N speed ratio fall out; the
# phase is solved so the contact-line tooth coordinates stay complementary (sum = 1/2),
# which - because d(sum)/ds = 0 - holds forever, not just at the first frame.
func _mesh(parent: Dictionary, alpha: float, child_teeth: int) -> Array:
	var np := float(parent.teeth)
	var nc := float(child_teeth)
	var omega_c: float = -float(parent.omega) * np / nc
	var parent_coord := (alpha - float(parent.phase)) * np / TAU
	var target := 0.5 - parent_coord                          # desired child coord at contact
	var phase_c := (alpha + PI) - target * TAU / nc
	return [phase_c, omega_c]


# Sun-and-planets: one large central wheel with several satellites meshed around it, all
# coupled into a single smooth mechanism that turns in concert.
func _build_orrery(rng: RandomNumberGenerator) -> void:
	var depth := rng.randf_range(0.45, 0.9)
	var grp := _new_group("smooth", rng, depth)
	var module := rng.randf_range(0.018, 0.030)
	var rc := rng.randf_range(0.28, 0.42)
	var nc := maxi(9, roundi(2.0 * rc / module))
	var centre := Vector2(rng.randf_range(-0.12, 0.12), rng.randf_range(-0.12, 0.12))
	var driver := _add_gear(grp, centre, rc, nc, rng.randf_range(-PI, PI), 1.0, depth)
	_groups[grp].tooth_step = TAU / float(nc)
	var sat := rng.randi_range(3, 6)
	for i in sat:
		var rs := rng.randf_range(0.10, 0.19)
		var ns := maxi(7, roundi(2.0 * rs / module))
		var a := TAU * float(i) / float(sat) + rng.randf_range(-0.18, 0.18)
		var pos := centre + Vector2(cos(a), sin(a)) * (rc + rs)
		var m := _mesh(driver, a, ns)
		_add_gear(grp, pos, rs, ns, m[0], m[1], clampf(depth + rng.randf_range(-0.04, 0.10), 0.0, 1.0))


# A short meshed chain placed somewhere in frame: gear drives gear drives gear. `kind`
# makes the whole chain run smoothly or tick like an escapement.
func _build_train(rng: RandomNumberGenerator, kind: String) -> void:
	var depth := rng.randf_range(0.30, 0.95)
	var grp := _new_group(kind, rng, depth)
	var module := rng.randf_range(0.020, 0.032)
	var r0 := rng.randf_range(0.13, 0.24)
	var n0 := maxi(8, roundi(2.0 * r0 / module))
	_groups[grp].tooth_step = TAU / float(n0)
	var pos := Vector2(rng.randf_range(-0.55, 0.55), rng.randf_range(-0.45, 0.45))
	var prev := _add_gear(grp, pos, r0, n0, rng.randf_range(-PI, PI), 1.0, depth)
	var chain := rng.randi_range(1, 3)
	for c in chain:
		var rn := rng.randf_range(0.09, 0.20)
		var nn := maxi(7, roundi(2.0 * rn / module))
		var a := rng.randf_range(-PI, PI)
		var npos: Vector2 = prev.pos + Vector2(cos(a), sin(a)) * (float(prev.R) + rn)
		var m := _mesh(prev, a, nn)
		prev = _add_gear(grp, npos, rn, nn, m[0], m[1], clampf(depth + rng.randf_range(-0.06, 0.06), 0.0, 1.0))


# A chorus: equal, unmeshed wheels on a loose grid, ALL turning at one speed and one
# direction - in unison. Either smooth (a slow synchronized field) or ticking together
# (a wall of clocks striking as one).
func _build_chorus(rng: RandomNumberGenerator) -> void:
	var depth := rng.randf_range(0.45, 0.9)
	var ticking := rng.randf() < 0.65          # lean toward click-turn over smooth
	var module := rng.randf_range(0.022, 0.032)
	var r := rng.randf_range(0.11, 0.17)
	var n := maxi(8, roundi(2.0 * r / module))
	var dir := 1.0 if rng.randf() < 0.5 else -1.0
	var cols := rng.randi_range(3, 5)
	var rows := rng.randi_range(2, 3)
	var gx := rng.randf_range(0.30, 0.42)
	var gy := gx
	var x0 := -gx * float(cols - 1) / 2.0
	var y0 := -gy * float(rows - 1) / 2.0
	var count := cols * rows
	var period := rng.randf_range(0.5, 0.9)
	# One SHARED smooth group (unison) unless ticking, in which case each wheel gets its OWN
	# escapement with a staggered tick phase and no beat-lock, so a wave of clicks travels across
	# the grid in sequence - a row of clocks ticking one after another, not all at once.
	var shared := -1
	if not ticking:
		shared = _new_group("smooth", rng, depth)
		_groups[shared].dir = dir
		_groups[shared].tooth_step = TAU / float(n)
	var idx := 0
	for rr in rows:
		for cc in cols:
			var pos := Vector2(x0 + cc * gx, y0 + rr * gy) \
				+ Vector2(rng.randf_range(-0.015, 0.015), rng.randf_range(-0.015, 0.015))
			var grp := shared
			if ticking:
				grp = _new_group("tick", rng, depth)
				_groups[grp].dir = dir
				_groups[grp].tooth_step = TAU / float(n)
				_groups[grp].tick_period = period
				_groups[grp].tick_t = period * float(idx) / float(count)   # stagger -> sequence
				_groups[grp].beat_sync = false
			_add_gear(grp, pos, r, n, rng.randf_range(-PI, PI), 1.0,
				clampf(depth + rng.randf_range(-0.05, 0.05), 0.0, 1.0))
			idx += 1


# Split composition: 2-3 LARGE wheels stacked on one side, and a loose CLUSTER of ~20 small
# wheels on the other - all with variance (size, tooth width, flat/wire, speed, direction, and
# the small cluster ticking asynchronously). The asymmetry reads as a real, busy mechanism.
func _build_split(rng: RandomNumberGenerator) -> void:
	var big_left := rng.randf() < 0.5
	var big_x := -0.55 if big_left else 0.55
	var cl_x := 0.5 if big_left else -0.5
	# The few big wheels, stacked down one side.
	var bmod := rng.randf_range(0.022, 0.034)
	var by := -0.32
	for i in rng.randi_range(2, 3):
		var depth := rng.randf_range(0.5, 0.95)
		var grp := _new_group("tick" if rng.randf() < 0.4 else "smooth", rng, depth)
		_groups[grp].dir = 1.0 if rng.randf() < 0.5 else -1.0
		var r := rng.randf_range(0.22, 0.40)
		var teeth := maxi(12, roundi(2.0 * r / bmod))
		_groups[grp].tooth_step = TAU / float(teeth)
		_add_gear(grp, Vector2(big_x + rng.randf_range(-0.12, 0.12), by), r, teeth,
			rng.randf_range(-PI, PI), 1.0, depth)
		by += r * 1.5
	# The cluster of many small wheels on the other side.
	var cmod := rng.randf_range(0.018, 0.030)
	for i in rng.randi_range(16, 22):
		var depth := rng.randf_range(0.35, 0.95)
		var grp := _new_group("tick" if rng.randf() < 0.5 else "smooth", rng, depth)
		_groups[grp].dir = 1.0 if rng.randf() < 0.5 else -1.0
		_groups[grp].speed = rng.randf_range(0.15, 0.6)
		if String(_groups[grp].kind) == "tick":
			_groups[grp].tick_period = rng.randf_range(0.4, 1.0)
			_groups[grp].beat_sync = false                       # the cluster clicks asynchronously
		var r := rng.randf_range(0.05, 0.13)
		var teeth := maxi(7, roundi(2.0 * r / cmod))
		_groups[grp].tooth_step = TAU / float(teeth)
		_add_gear(grp, Vector2(cl_x + rng.randf_range(-0.35, 0.35), rng.randf_range(-0.5, 0.5)),
			r, teeth, rng.randf_range(-PI, PI), 1.0, depth)


# One enormous, dim, nearly-still wheel behind everything - only an arc of its rim ever
# crosses the frame. Pure depth and weight.
func _build_backdrop(rng: RandomNumberGenerator) -> void:
	var depth := rng.randf_range(0.05, 0.18)
	var grp := _new_group("smooth", rng, depth)
	_groups[grp].speed = rng.randf_range(0.03, 0.10)         # ponderous
	var module := rng.randf_range(0.030, 0.050)
	var r := rng.randf_range(0.75, 1.25)                     # larger than the frame
	var n := maxi(24, roundi(2.0 * r / module))
	_groups[grp].tooth_step = TAU / float(n)
	var pos := Vector2(rng.randf_range(-0.5, 0.5), rng.randf_range(-0.5, 0.5))
	_add_gear(grp, pos, r, n, rng.randf_range(-PI, PI), 1.0, depth)


# Free gears: unmeshed wheels that just spin on their own - the teeth need not always connect.
# Each is its own group with its own speed and DIRECTION, scattered (often overlapping the
# meshed mechanism), some smooth and some ticking.
func _build_free(rng: RandomNumberGenerator, n: int) -> void:
	for i in n:
		var depth := rng.randf_range(0.35, 0.95)
		var grp := _new_group("tick" if rng.randf() < 0.45 else "smooth", rng, depth)
		_groups[grp].dir = 1.0 if rng.randf() < 0.5 else -1.0
		var module := rng.randf_range(0.020, 0.034)
		var r := rng.randf_range(0.10, 0.26)
		var teeth := maxi(8, roundi(2.0 * r / module))
		_groups[grp].tooth_step = TAU / float(teeth)
		var pos := Vector2(rng.randf_range(-0.6, 0.6), rng.randf_range(-0.5, 0.5))
		_add_gear(grp, pos, r, teeth, rng.randf_range(-PI, PI), 1.0, depth)


# Background wheels: large, dim, slow gears placed to OVERLAP the mechanism from behind (low
# depth -> drawn first + hazy), each COUNTER-rotating at its own rate, so gears turn behind
# gears and the picture gains real depth.
func _build_bg(rng: RandomNumberGenerator, n: int) -> void:
	for i in n:
		var depth := rng.randf_range(0.04, 0.22)
		var grp := _new_group("tick" if rng.randf() < 0.3 else "smooth", rng, depth)
		_groups[grp].speed = rng.randf_range(0.05, 0.18)        # ponderous
		_groups[grp].dir = 1.0 if rng.randf() < 0.5 else -1.0   # counter-rotates vs the foreground
		var module := rng.randf_range(0.026, 0.045)
		var r := rng.randf_range(0.30, 0.70)
		var teeth := maxi(16, roundi(2.0 * r / module))
		_groups[grp].tooth_step = TAU / float(teeth)
		var pos := Vector2(rng.randf_range(-0.45, 0.45), rng.randf_range(-0.4, 0.4))   # overlap centre
		_add_gear(grp, pos, r, teeth, rng.randf_range(-PI, PI), 1.0, depth)


# --- update ------------------------------------------------------------------

func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.02, 0.025)
	_glow = Nonlinear.flare(_glow, clampf(0.30 * f.energy + 0.70 * f.beat, 0.0, 1.0), delta, 9.0, 1.6)
	var beat_edge: bool = f.beat > 0.55 and _beat_prev <= 0.55
	_beat_prev = f.beat
	_beat_pulse = maxf(_beat_pulse - delta * 4.0, 0.0)
	if beat_edge:
		_beat_pulse = 1.0
	# The key light drifts and shivers with the treble, so the specular never sits still.
	_light_ang = wrapf(_light_ang + delta * (_light_drift + 0.4 * f.treble * sin(_life * 21.0)), -PI, PI)

	var drive := 0.30 + 1.0 * f.energy + 0.6 * _glow
	for grp in _groups:
		grp.sway_off = grp.sway * sin(_life * 0.25) * (1.0 - float(grp.depth))
		if grp.kind == "smooth":
			grp.s += float(grp.dir) * float(grp.speed) * delta * drive
		else:
			# Escapement: advance one driver-tooth per tick (beat-locked, with a fallback
			# cadence), then chase the goal on an underdamped spring for the clock recoil.
			grp.tick_t += delta
			if (bool(grp.beat_sync) and beat_edge) or grp.tick_t >= float(grp.tick_period):
				grp.tick_t = 0.0
				grp.s_goal += float(grp.dir) * float(grp.tooth_step)
			var accel: float = float(grp.stiffness) * (float(grp.s_goal) - float(grp.s)) \
				- float(grp.damping) * float(grp.vel)
			grp.vel += accel * delta
			grp.s += float(grp.vel) * delta
	queue_redraw()


# --- draw --------------------------------------------------------------------

func _draw() -> void:
	begin_draw()
	var u := unit()
	for g in _gears:
		_draw_gear(g, u)


func _draw_gear(g: Dictionary, u: float) -> void:
	var grp: Dictionary = _groups[int(g.group)]
	var theta: float = float(g.phase) + float(g.omega) * float(grp.s)
	var sc: float = float(g.R) * u
	var centre: Vector2 = (Vector2(g.pos) + Vector2(grp.sway_off)) * u
	var co := cos(theta)
	var si := sin(theta)
	# Rotation+scale+translation in one transform; applies to the whole tooth ring at once.
	var xf := Transform2D(Vector2(co, si) * sc, Vector2(-si, co) * sc, centre)

	var depth: float = float(g.depth)
	var db := 0.35 + 0.65 * depth                         # nearer wheels read brighter ...
	var af := 0.5 + 0.5 * depth                           # ... and more opaque (far = hazy)
	var rw := maxf(1.5, 0.010 * sc)

	var world: PackedVector2Array = xf * PackedVector2Array(g.teeth_local)
	if String(g.style) == "flat":
		# Solid flat-colour cog: the whole tooth ring filled, with a dark rim edge for relief.
		draw_colored_polygon(world, Color.from_hsv(_hue, _sat, clampf(0.16 * db + 0.26 + 0.30 * _glow, 0.0, 1.0), 0.95 * af))
		draw_polyline(world, Color.from_hsv(_hue, _sat, 0.05, 0.6 * af), maxf(1.0, rw * 0.7), true)
	else:
		# Skeletal wire wheel: a near-black body, spokes + bolts, and a bright luminous tooth rim.
		draw_circle(centre, float(g.body_r) * sc, Color.from_hsv(_hue, _sat * 0.55, 0.05 * db + 0.015, 0.6 * af))
		var spoke_c := Color.from_hsv(_hue, _sat * 0.85, clampf(0.22 * db + 0.20 + 0.30 * _glow, 0.0, 1.0), 0.85 * af)
		var sw := maxf(1.5, 0.014 * sc)
		for sa in g.spokes:
			var a := theta + float(sa)
			var d := Vector2(cos(a), sin(a))
			draw_line(centre + d * float(g.hub_r) * sc, centre + d * float(g.body_r) * 0.95 * sc, spoke_c, sw, true)
			draw_circle(centre + d * float(g.body_r) * 0.80 * sc, sw * 1.1, spoke_c)
		draw_polyline(world, Color.from_hsv(_hue, _sat, clampf(0.26 * db + 0.28 + 0.35 * _glow, 0.0, 1.0), 0.92 * af), rw, true)

	# Rust: soft orange-brown patches mottling the metal in the alpha, turning with the wheel -
	# a worn, corroded texture over the body (under the hub).
	for sp in g.rust:
		var ra := theta + float(sp.a)
		var rp := centre + Vector2(cos(ra), sin(ra)) * float(sp.d) * sc
		Layer.soft_blob(self, rp, float(sp.r) * sc,
			Color.from_hsv(fposmod(0.05 + _hue * 0.1, 1.0), 0.6, 0.30 * db, float(sp.al) * af), 5)

	# Hub and bore.
	draw_circle(centre, float(g.hub_r) * sc, Color.from_hsv(_hue, _sat, clampf(0.30 * db + 0.18 + 0.30 * _glow, 0.0, 1.0), 0.95 * af))
	draw_circle(centre, float(g.bore_r) * sc, Color(0.015, 0.02, 0.03, af))

	# Hub glow, flaring on the beat.
	var gv := clampf(0.12 + 0.7 * _glow, 0.0, 1.0) * af
	Layer.glow(self, centre, float(g.hub_r) * sc * (1.3 + 1.4 * _glow),
		Color.from_hsv(fposmod(_hue + 0.04, 1.0), _sat * 0.6, 1.0, 0.5 * gv), 4)
