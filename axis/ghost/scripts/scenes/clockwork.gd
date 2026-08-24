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

# METAL. This used to be five hardcoded tints, which meant five machines forever.
# The restraint that keeps gears from going steampunk-clipart is not the SHORT LIST
# though - it is the low saturation. So the list becomes any mood a worked metal
# plausibly takes (blued steel, nickel, brass, copper, verdigris, anodised), and the
# restraint becomes a CAP: whatever mood is drawn, its saturation is scaled down and
# clamped, so the wheels always read as machined material lit in the dark.
const METAL_MOODS := ["ash", "bone", "brass", "sodium", "ember", "glacier", "abyss",
	"teal", "verdant", "violet"]
const SAT_CAP := 0.34

# FINISH. Every gear rolled flat-or-wire independently at a fixed 40%, so every
# machine was the same mongrel mix. A shop builds to one spec: a skeletal frame of
# spoked wheels, solid plate cogs, or a machine assembled from both.
const FINISHES := [
	{"name": "skeletal", "flat": 0.04},
	{"name": "mixed", "flat": 0.40},
	{"name": "plate", "flat": 0.88},
]

## HOW MUCH OF THE RIM THE SPOKES TAKE UP, sampled per machine and jittered per wheel. This is
## the knob the wheel vocabulary was missing entirely: the spoke was a `draw_line` at a FIXED
## 1.4% of the wheel's radius, so every spoked wheel in every machine wore the same hairline.
## Measured over 60 machines and 1038 wheels before this existed, spoke width over radius was
## the constant 0.014 - not a narrow range, a single value - which is what "the gears always
## have a very similar kind of look: long spokes, with many of them" was.
##
## Expressed as a FRACTION OF THE CIRCUMFERENCE rather than as a width, because that is the
## quantity that has to stay sane: whatever the spoke count, the spokes together take up this
## much of the rim and the gaps take up the rest, so three fat spokes and eight thin ones are
## both reachable and neither can ever overlap itself.
const FILL_MIN := 0.12
const FILL_MAX := 0.78

## WHAT THE MACHINE HAS CORRODED INTO. Wear used to be one colour and one shape: the mark hue
## was `0.05 + hue * 0.1` (orange-brown, always), the saturation was the constant 0.60 and the
## value the constant 0.30, and every mark was a soft round blob. Reported as "rust is always
## brown, and blotchy; it always looks exactly the same", and two of its three colour channels
## were literally constants, so it was.
##
## Metals do not all fail the same way, and the ones here are the ones you can name on sight:
## iron goes orange, copper and bronze go green, brass darkens to olive, a hot machine sooths
## over black, a neglected one pits, and a machine in service wears BRIGHT where it is handled.
## `mark` picks the shape as well as the colour - a blotch, a radial streak, a scatter of pits,
## or a burnished arc - because a patina that is always the same shape reads as a texture that
## was pasted on rather than as something that happened to the metal.
const WEARS := [
	{"name": "rust", "h": [0.015, 0.075], "s": [0.45, 0.85], "v": [0.18, 0.44], "mark": "blotch"},
	{"name": "verdigris", "h": [0.33, 0.45], "s": [0.25, 0.60], "v": [0.20, 0.46], "mark": "blotch"},
	{"name": "soot", "h": [0.02, 0.13], "s": [0.04, 0.22], "v": [0.015, 0.10], "mark": "blotch"},
	{"name": "tarnish", "h": [0.07, 0.15], "s": [0.12, 0.42], "v": [0.07, 0.20], "mark": "streak"},
	{"name": "pitting", "h": [0.01, 0.07], "s": [0.30, 0.70], "v": [0.08, 0.26], "mark": "pits"},
	{"name": "burnish", "h": [0.05, 0.16], "s": [0.02, 0.16], "v": [0.35, 0.62], "mark": "arc"},
]

var _f: AudioFeatures = AudioFeatures.new()
var _rng := RandomNumberGenerator.new()
var _gears: Array = []        # each: {group, pos, R, teeth, phase, omega, depth, geom...}
var _groups: Array = []       # each: motion driver shared by its gears (see _new_group)
var _hue := 0.58
var _sat := 0.18
var _glow_hue := 0.08        # the scheme's accent: the hub light, a colour the metal is not
var _glow_sat := 0.10
var _flat_p := 0.4           # this machine's finish: chance any one wheel is a solid plate
var _spoke_pool: Array = [3, 4, 5, 6]   # its spoke vocabulary (usually one repeated count)
var _fill := 0.30            # what fraction of the rim's circumference the spokes occupy
var _hole_p := 0.0           # chance a solid plate is drilled with lightening holes
var _wear_kind: Dictionary = WEARS[0]   # what this machine has corroded INTO - see WEARS
var _mod_mul := 1.0          # tooth-size multiplier: fine teeth or chunky ones, machine-wide
var _wear := 0.45            # how corroded this machine is - 0 for a pristine one
var _glow := 0.0
var _beat_pulse := 0.0
var _beat_prev := 0.0
var _light_ang := 0.0         # world angle of the key light; the specular sweep tracks it
var _light_drift := 0.0
var _light_sigma := 0.8       # angular width of the lit arc of teeth


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	_rng.seed = rng.randi()
	framing = "subject"                       # a mechanism is a subject: allow the slow push-in
	var sch := Scheme.among(METAL_MOODS, rng)
	_hue = sch.hue
	_sat = clampf(sch.sat * rng.randf_range(0.30, 0.60), 0.04, SAT_CAP)
	# The hub light is the scheme's accent, unclamped-ish: the one warm/cold spark in
	# an otherwise grey mechanism, and harmonious with the metal by construction.
	_glow_hue = sch.accent
	_glow_sat = clampf(sch.sat * 0.55, 0.03, 0.55)
	var finish: Dictionary = FINISHES[rng.randi() % FINISHES.size()]
	_flat_p = float(finish["flat"])
	# Most machines are built to ONE spoke pattern, as a single shop would; a minority
	# are assembled from mixed parts.
	# TWO IS A COUNT. A wheel cast with a pair of opposed arms is a real thing and it was not
	# reachable - the pool started at three - which is half of "fewer".
	var spokes_all := [2, 3, 3, 4, 4, 5, 6, 8]
	_spoke_pool = spokes_all if rng.randf() < 0.35 \
		else [spokes_all[rng.randi() % spokes_all.size()]]
	# ...and how heavy the arms are, machine-wide, because a shop casts to one weight.
	_fill = rng.randf_range(FILL_MIN, FILL_MAX)
	# A plate cog with holes drilled in it is a spoked wheel by another name, and it is the
	# only interior structure a solid cog has ever had here - before this they were blank discs.
	_hole_p = 0.0 if rng.randf() < 0.35 else rng.randf_range(0.35, 0.9)
	_wear_kind = WEARS[rng.randi() % WEARS.size()]
	# Tooth size, machine-wide: a fine-toothed instrument or a chunky mill drive. Every
	# module sample below scales by this, so meshing stays exact - only the grain changes.
	_mod_mul = rng.randf_range(0.7, 1.6)
	# Wear: a quarter of machines come out of the crate pristine, the rest corroded to
	# varying degrees.
	_wear = 0.0 if rng.randf() < 0.25 else rng.randf_range(0.20, 0.85)
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
		_build_free(rng, rng.randi_range(1, 3))
	# A gear or two behind the others, counter-rotating, for depth.
	_build_bg(rng, 1 if mode == "split" else rng.randi_range(1, 3))
	# A vast, dim, almost-still wheel arcing behind it all, sometimes - extra depth and gravity.
	if rng.randf() < 0.40:
		_build_backdrop(rng)

	# Paint far wheels first so near ones occlude them.
	_gears.sort_custom(func(a, b): return float(a.depth) < float(b.depth))
	return {"mode": mode, "mood": sch.name, "finish": String(finish["name"]),
		"tooth_scale": _mod_mul, "wear": _wear, "gears": _gears.size(),
		# What the wheels are SHAPED like and what they have corroded into - the two rolls the
		# console could not show before, and the two the scene was reported as never varying.
		"patina": String(_wear_kind["name"]), "arms": _spoke_pool, "arm_weight": _fill,
		"drilled": _hole_p > 0.0}


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
	var style := "flat" if _rng.randf() < _flat_p else "wire"  # per the machine's finish
	var pts := PackedVector2Array()
	for k in teeth:
		var a0 := float(k) * p
		# A trapezoidal tooth: root, up the flank to a flat tip, down the far flank, root.
		var angs := [a0 - bw * p, a0 - tw * p, a0 + tw * p, a0 + bw * p]
		var rads := [root_r, tip_r, tip_r, root_r]
		for j in 4:
			pts.append(Vector2(cos(angs[j]), sin(angs[j])) * rads[j])
	pts.append(pts[0])                                        # close the rim

	var spoke_n: int = int(_spoke_pool[_rng.randi() % _spoke_pool.size()])
	var spokes := []
	var s0 := _rng.randf_range(-PI, PI)
	for i in spoke_n:
		spokes.append(s0 + TAU * float(i) / float(spoke_n))

	# THE WHEEL'S PROPORTIONS, and all three of them were missing. A spoke ran from the hub to
	# very nearly the tooth root at a fixed hairline width, so every spoked wheel was long,
	# thin and many-armed. It now has a hub to start at, a RIM BAND to stop against, and a real
	# width - so a wheel can be three fat arms between a heavy hub and a heavy rim, or eight
	# hairlines, or anything between.
	#
	# `rim_in` is where the spokes end and the rim begins; a low one is a thick rim, which
	# shortens the spokes from the outside as a bigger hub shortens them from the inside.
	var rim_in := root_r * _rng.randf_range(0.55, 0.94)
	var hub_r := root_r * _rng.randf_range(0.16, 0.42)
	# Half-width in wheel units: the fraction of the ROOM BETWEEN ARMS that the arm fills,
	# jittered per wheel off the machine's figure. Half the gap between two adjacent arms at
	# the rim is `rim_in * sin(PI / n)`, so scaling by that is self-limiting - at any count the
	# arms can be hairlines or nearly touching and can never overlap, and the same number means
	# the same visual weight whether the wheel has two arms or eight.
	#
	# Arc length was the first parameterisation and it does not survive a wide arm: a straight
	# bar of half-width w subtends 2*asin(w/r), not 2w/r, so a "78% of the circumference" wheel
	# with two arms came out with a half-width of 2.03 - an arm twice as wide as the wheel it
	# was on. Measured, before this line was rewritten.
	var fill := clampf(_fill * _rng.randf_range(0.75, 1.3), FILL_MIN, FILL_MAX)
	var spoke_w := rim_in * sin(PI / float(spoke_n)) * fill
	# A drilled plate: one hole per gap, sized to the gap it sits in. This is what gives a
	# solid cog any interior at all.
	var holes := _rng.randf() < _hole_p

	# WEAR: patches of whatever this machine has corroded into (see WEARS), pinned to the wheel
	# in polar (a, d) so they turn with it. Every mark rolls its OWN colour inside the finish's
	# ranges - the old code fixed saturation at 0.60 and value at 0.30 for every mark on every
	# wheel of every machine, which is most of why it always looked the same.
	var rust := []
	if _rng.randf() < _wear:
		var kind := String(_wear_kind["mark"])
		var n_marks := int(round(_rng.randf_range(3.0, 7.0) + _wear * 6.0))
		if kind == "pits":
			n_marks *= 3                       # pitting is many small marks, not a few big ones
		for i in n_marks:
			var big: float = 0.10 if kind == "pits" else 0.34
			rust.append({
				"a": _rng.randf_range(-PI, PI),
				"d": _rng.randf_range(hub_r * 0.6, root_r * 0.95),
				"r": root_r * _rng.randf_range(big * 0.35, big),
				"al": _rng.randf_range(0.14, 0.50),
				# the mark's own colour, inside the finish's ranges
				"h": _rng.randf_range(float(_wear_kind["h"][0]), float(_wear_kind["h"][1])),
				"s": _rng.randf_range(float(_wear_kind["s"][0]), float(_wear_kind["s"][1])),
				"v": _rng.randf_range(float(_wear_kind["v"][0]), float(_wear_kind["v"][1])),
				# how far a streak runs, and which way an arc bends - unused by the other shapes
				"len": _rng.randf_range(0.10, 0.34) * root_r,
				"span": _rng.randf_range(0.35, 1.5) * (1.0 if _rng.randf() < 0.5 else -1.0),
			})

	var g := {
		"group": group, "pos": pos, "R": R, "teeth": teeth, "style": style,
		"phase": phase, "omega": omega, "depth": depth, "rust": rust,
		"teeth_local": pts, "tip_r": tip_r, "root_r": root_r,
		"hub_r": hub_r,
		"bore_r": hub_r * _rng.randf_range(0.24, 0.45),
		"body_r": root_r * 0.99, "spokes": spokes,
		"rim_in": rim_in, "spoke_w": spoke_w, "holes": holes,
		"wear_mark": String(_wear_kind["mark"]),
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
	var module := rng.randf_range(0.018, 0.030) * _mod_mul
	var rc := rng.randf_range(0.24, 0.48)
	var nc := maxi(9, roundi(2.0 * rc / module))
	var centre := Vector2(rng.randf_range(-0.12, 0.12), rng.randf_range(-0.12, 0.12))
	var driver := _add_gear(grp, centre, rc, nc, rng.randf_range(-PI, PI), 1.0, depth)
	_groups[grp].tooth_step = TAU / float(nc)
	var sat := rng.randi_range(3, 8)
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
	var module := rng.randf_range(0.020, 0.032) * _mod_mul
	var r0 := rng.randf_range(0.10, 0.28)
	var n0 := maxi(8, roundi(2.0 * r0 / module))
	_groups[grp].tooth_step = TAU / float(n0)
	var pos := Vector2(rng.randf_range(-0.55, 0.55), rng.randf_range(-0.45, 0.45))
	var prev := _add_gear(grp, pos, r0, n0, rng.randf_range(-PI, PI), 1.0, depth)
	var chain := rng.randi_range(1, 5)
	for c in chain:
		var rn := rng.randf_range(0.07, 0.22)
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
	var module := rng.randf_range(0.022, 0.032) * _mod_mul
	var dir := 1.0 if rng.randf() < 0.5 else -1.0
	# A chorus is a WALL, so the grid is sized first and the wheels are then fitted to
	# it: a wide sparse rank of small clocks and a tight block of big ones are both
	# choruses, but a fixed spacing could only ever draw one of them.
	var cols := rng.randi_range(2, 6)
	var rows := rng.randi_range(2, 4)
	var gx := minf(rng.randf_range(0.30, 0.50), 1.75 / maxf(1.0, float(cols - 1)))
	var gy := minf(gx, 1.25 / maxf(1.0, float(rows - 1)))
	var r := minf(rng.randf_range(0.09, 0.22), minf(gx, gy) * 0.46)
	var n := maxi(8, roundi(2.0 * r / module))
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
	var bmod := rng.randf_range(0.022, 0.034) * _mod_mul
	var by := -0.32
	for i in rng.randi_range(2, 4):
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
	var cmod := rng.randf_range(0.018, 0.030) * _mod_mul
	for i in rng.randi_range(10, 26):
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
	var module := rng.randf_range(0.030, 0.050) * _mod_mul
	var r := rng.randf_range(0.70, 1.55)                     # larger than the frame
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
		var module := rng.randf_range(0.020, 0.034) * _mod_mul
		var r := rng.randf_range(0.08, 0.30)
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
		var module := rng.randf_range(0.026, 0.045) * _mod_mul
		var r := rng.randf_range(0.26, 0.85)
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
	var rim_in: float = float(g.rim_in)
	var hub_r: float = float(g.hub_r)
	if String(g.style) == "flat":
		# Solid flat-colour cog: the whole tooth ring filled, with a dark rim edge for relief.
		draw_colored_polygon(world, Color.from_hsv(_hue, _sat, clampf(0.16 * db + 0.26 + 0.30 * _glow, 0.0, 1.0), 0.95 * af))
		draw_polyline(world, Color.from_hsv(_hue, _sat, 0.05, 0.6 * af), maxf(1.0, rw * 0.7), true)
		# ...and, if this shop drills them, lightening holes: one in each gap between the arms
		# the casting would have had. A blank disc is the one thing a real cog never is.
		if bool(g.holes):
			var hole_c := Color.from_hsv(_hue, _sat * 0.7, 0.045 * db + 0.012, 0.85 * af)
			var mid_r := (hub_r + rim_in) * 0.5
			var gap := TAU / float((g.spokes as Array).size())
			# Sized to the gap it sits in AND to the band it spans, so it can never eat the
			# hub or the rim however few arms the wheel has.
			var hr := minf(mid_r * sin(gap * 0.5) * 0.72, (rim_in - hub_r) * 0.42)
			if hr > 0.01:
				for sa in g.spokes:
					var ha := theta + float(sa) + gap * 0.5
					draw_circle(centre + Vector2(cos(ha), sin(ha)) * mid_r * sc, hr * sc, hole_c)
	else:
		# Skeletal wire wheel: a dark body, a rim band, the arms, and a bright tooth rim.
		draw_circle(centre, float(g.body_r) * sc, Color.from_hsv(_hue, _sat * 0.55, 0.05 * db + 0.015, 0.6 * af))
		var spoke_c := Color.from_hsv(_hue, _sat * 0.85, clampf(0.22 * db + 0.20 + 0.30 * _glow, 0.0, 1.0), 0.85 * af)
		# THE RIM AS A BAND, not as a line. It is what the arms stop against, and without it a
		# wheel with short heavy arms reads as a hub with stubs rather than as a cast wheel.
		var band: float = (float(g.body_r) - rim_in) * sc
		if band > 1.0:
			draw_arc(centre, (rim_in + float(g.body_r)) * 0.5 * sc, 0.0, TAU, 48, spoke_c, band, true)
		# THE ARMS, as filled bars rather than as lines, which is the whole of "fewer, shorter,
		# wider". A line has one width for every wheel; a bar has the width the casting was
		# given, and it is flared where it meets the hub the way a fillet is.
		var half: float = maxf(float(g.spoke_w) * sc, 1.0)
		for sa in g.spokes:
			var a := theta + float(sa)
			var d := Vector2(cos(a), sin(a))
			var q := Vector2(-d.y, d.x)
			var r0 := hub_r * 0.85
			# A fillet where the arm meets the hub, and a MODEST one: flaring by a fixed
			# fraction of the arm's own width turns a broad cast arm into a petal, because the
			# flare grows with the thing it is supposed to be a detail on.
			var flare := 1.0 + 0.9 * minf(1.0, 3.0 / maxf(1.0, half))
			draw_colored_polygon(PackedVector2Array([
				centre + d * r0 * sc - q * half * flare,
				centre + d * r0 * sc + q * half * flare,
				centre + d * rim_in * sc + q * half,
				centre + d * rim_in * sc - q * half]), spoke_c)
			# A BOLT IS A BOLT, whatever the arm is. Sized to the wheel, not to the arm, and
			# only where there is arm to put it on - scaled to the arm it became a round pad
			# the width of the arm, and every wide wheel came out looking like a propeller.
			var boss := clampf(0.022 * sc, 1.2, half * 0.7)
			draw_circle(centre + d * (rim_in * sc - boss * 1.6), boss, spoke_c)
		draw_polyline(world, Color.from_hsv(_hue, _sat, clampf(0.26 * db + 0.28 + 0.35 * _glow, 0.0, 1.0), 0.92 * af), rw, true)

	# WEAR, in whatever this machine corroded into and in that finish's own shape - see WEARS.
	# Pinned to the wheel in polar (a, d) so it turns with the metal, and every mark carries the
	# colour it rolled for itself rather than one the whole scene shares.
	var mark := String(g.wear_mark)
	for sp in g.rust:
		var ra := theta + float(sp.a)
		var dir := Vector2(cos(ra), sin(ra))
		var rp := centre + dir * float(sp.d) * sc
		var wc := Color.from_hsv(float(sp.h), float(sp.s), float(sp.v) * (0.55 + 0.45 * db),
			float(sp.al) * af)
		match mark:
			"streak":
				# A run of corrosion following the metal outward, not a dot.
				var steps := 5
				var step := dir * float(sp.len) * sc / float(steps - 1)
				for k in steps:
					Layer.puff(self, rp + step * float(k), float(sp.r) * 0.55 * sc,
						Color(wc.r, wc.g, wc.b, wc.a * 0.55))
			"pits":
				Layer.puff(self, rp, float(sp.r) * sc, wc)
			"arc":
				# Burnished where the wheel is handled: a bright band worn along its travel.
				var aw := maxf(float(sp.r) * 0.8 * sc, 1.5)
				draw_arc(centre, float(sp.d) * sc, ra, ra + float(sp.span), 20, wc, aw, true)
			_:
				Layer.soft_blob(self, rp, float(sp.r) * sc, wc, 5)

	# Hub and bore.
	draw_circle(centre, float(g.hub_r) * sc, Color.from_hsv(_hue, _sat, clampf(0.30 * db + 0.18 + 0.30 * _glow, 0.0, 1.0), 0.95 * af))
	draw_circle(centre, float(g.bore_r) * sc, Color(0.015, 0.02, 0.03, af))

	# Hub glow, flaring on the beat.
	var gv := clampf(0.12 + 0.7 * _glow, 0.0, 1.0) * af
	Layer.glow(self, centre, float(g.hub_r) * sc * (1.3 + 1.4 * _glow),
		Color.from_hsv(_glow_hue, _glow_sat, 1.0, 0.5 * gv), 4)
