extends RefCounted
class_name Actions

## Actions - the verb registry for data-driven scenes (see scenes/stage.gd).
##
## The registry sibling of [Primitives] / [Layer] / [Shots], for *choreography*: each
## verb is a small class applied to named actors over a time window by a [Track]
## span. What used to be imperative code gated on hardcoded fraction constants inside
## the-point scenes (crystallize, tremble, burst, phase-lock, specialize, ...) is one
## verb each here, called with data.
##
## Lifecycle: `begin(stage, actors)` once when the window is entered, `apply(stage,
## actors, k, f, dt)` every frame with the EASED window fraction k (a point span gets
## a single k=1 apply), `finish(stage, actors)` once when it closes. Most verbs LATCH
## on finish (a specialized prism stays specialized) - undoing is another span.
##
## Verbs receive their args RAW (ranges intact) plus a seeded rng: values meant to be
## fixed per instance are sampled in begin() ("cattle, not pets"); per-event values
## (a gaze fixation's distance) are re-drawn as they happen. Same seed, same show.

const REGISTRY := {
	"look": Look, "blink": Blink, "arrive": Arrive, "sprout": Sprout, "split": Split,
	"crystallize": Crystallize, "tremble": Tremble, "burst": Burst, "lock": Lock,
	"desync": Desync, "hold_still": HoldStill, "sway": Sway, "counterflow": Counterflow,
	"specialize": Specialize,
	"gather": Gather, "fly": Fly, "helix_split": HelixSplit, "lane_jump": LaneJump,
	"set": SetParam, "ramp": Ramp, "pulse": Pulse, "flash": Flash,
}


static func make(key: String, args: Dictionary, rng: RandomNumberGenerator) -> Action:
	if not REGISTRY.has(key):
		push_warning("ghost: unknown action '%s' (have: %s)" % [key, ", ".join(REGISTRY.keys())])
		return null
	var a: Action = REGISTRY[key].new()
	a.args = args
	a.rng = rng
	return a


class Action extends RefCounted:
	var args := {}
	var rng: RandomNumberGenerator

	func begin(_stage, _actors: Array) -> void:
		pass

	func apply(_stage, _actors: Array, _k: float, _f: AudioFeatures, _dt: float) -> void:
		pass

	func finish(_stage, _actors: Array) -> void:
		pass

	# Sample an arg once (a [lo, hi] range collapses to one draw). Call from begin()
	# and store - repeated calls would re-roll.
	func num(key: String, fallback: float) -> float:
		if not args.has(key):
			return fallback
		return float(Storyboard.sample(args[key], rng))


## look - aim eyes at a wandering 3D focus point. ALL targets share one focus, so two
## eyes verge (toe in near, run parallel far) for free; a rare stray sends one eye
## wandering. The depth-tier table comes from the args (see [Cast.FocusSampler]).
class Look extends Action:
	var _sampler: Cast.FocusSampler

	func begin(_stage, _actors: Array) -> void:
		_sampler = Cast.FocusSampler.new(args, rng)

	func apply(_stage, actors: Array, _k: float, _f: AudioFeatures, dt: float) -> void:
		_sampler.advance(dt, actors.size())
		for i in actors.size():
			var a = actors[i]
			a.state["focus"] = _sampler.focus
			a.state["focus_age"] = 0.0
			if _sampler.stray_of == i:
				a.state["div"] = _sampler.stray_off
			else:
				a.state.erase("div")


## blink - close and reopen the eyelids once. args: duration (s).
class Blink extends Action:
	func apply(_stage, actors: Array, _k: float, _f: AudioFeatures, _dt: float) -> void:
		var dur := num("duration", 0.32)
		for a in actors:
			if a.body is EyeBody:
				(a.body as EyeBody).blink(dur)


## arrive - the actor FLIES IN with velocity from offscreen, finds its anchor, and
## lands with a small damped wobble. The path curves slightly (a swoop, not a rail)
## and the approach decelerates hard into the landing. Give the span `ease: linear` -
## the verb shapes its own motion. args: from ([x,y,z] origin, default a sampled
## offscreen point), wobble (landing amplitude, world units).
class Arrive extends Action:
	var _origin := Vector3.ZERO
	var _slot := Vector3.ZERO
	var _flight := Vector3.RIGHT
	var _side := Vector3.UP
	var _wob := 0.06
	var _freq := 13.0
	var _arc := 0.3

	func begin(_stage, actors: Array) -> void:
		var a = actors[0]
		_slot = a.pos
		if args.has("from"):
			_origin = Cast._vec3(args["from"], a.home)
		else:
			# A sampled offscreen origin: out past the frame edge, biased upward a
			# little so the entrance reads as a swoop down onto the anchor.
			var ang := rng.randf_range(0.25, 0.85) * (1.0 if rng.randf() < 0.5 else -1.0)
			var dir2 := Vector2(sin(ang), absf(cos(ang)) * 0.6 + 0.2).normalized()
			_origin = _slot + Vector3(dir2.x, dir2.y, 0.0) * rng.randf_range(2.6, 3.4)
		_wob = num("wobble", rng.randf_range(0.05, 0.09))
		_freq = rng.randf_range(11.0, 15.0)
		_arc = rng.randf_range(0.2, 0.42) * (1.0 if rng.randf() < 0.5 else -1.0)
		_flight = (_slot - _origin).normalized()
		_side = Vector3(-_flight.y, _flight.x, 0.0)
		a.fade = 1.0

	func apply(_stage, actors: Array, k: float, _f: AudioFeatures, _dt: float) -> void:
		var a = actors[0]
		var approach := 1.0 - pow(1.0 - k, 3.0)             # fast in, decelerating hard
		var curve: float = _arc * sin(PI * minf(k * 1.15, 1.0)) * (1.0 - k)
		var land := 0.0
		if k > 0.55:                                        # the damped landing wobble
			var kl := (k - 0.55) / 0.45
			land = _wob * sin(_freq * kl) * exp(-4.5 * kl) * (1.0 - kl * 0.3)
		a.pos = _origin.lerp(_slot, approach) + _side * curve + _flight * land
		a.home = _slot

	func finish(_stage, actors: Array) -> void:
		actors[0].pos = _slot
		actors[0].home = _slot


## sprout - a STANDING ROSE beside the anchor: the stem grows up from below the
## frame (rooted in the void's floor - never from the camera), rising almost
## vertical beside the eye's anchor, cresting above it, and drooping its head over
## so the tip hangs downward; the eye swells at the hanging tip like fruit, facing
## the ground. As it hangs, the bloom SHEDS a petal or two; then the eye DETACHES -
## the release is what whips the freed stem back, and it retreats down along itself
## while the dropped eye settles onto its anchor and slowly rights itself to face
## forward. The vine and petals are drawn by [Cast.EyeActor] from `state.vine`.
## Give the span `ease: linear`. args: side (-1/1 stem side, default sampled).
class Sprout extends Action:
	var _slot := Vector3.ZERO
	var _R := 0.34
	var _base := Vector3.ZERO         # the stem's root, beyond the frame's lower side corner
	var _grow_k := 0.42               # phase boundaries + character, sampled per instance
	var _fruit_k := 0.62
	var _drop_k := 0.74
	var _hang := Vector3.ZERO         # where the eye's centre hangs (a touch above the slot)
	var _sag := 0.05                  # how far the fruit's weight bends the tip down
	var _sway_f := 1.6
	var _whip_f := 15.0
	var _whip_a := 0.16
	var _seed := 0
	var _hue := 0.3
	var _petals: Array = []           # {spawn: k, at: Vector3, ph, size} - shed as it wilts
	var _detached := false
	var _t := 0.0

	func begin(_stage, actors: Array) -> void:
		var a = actors[0]
		_slot = a.pos
		_R = a.scale
		var side := signf(num("side", -1.0 if rng.randf() < 0.5 else 1.0))
		_base = _slot + Vector3(side * rng.randf_range(0.34, 0.52), rng.randf_range(-1.25, -1.05), 0.0)
		_grow_k = rng.randf_range(0.38, 0.46)
		_fruit_k = _grow_k + rng.randf_range(0.16, 0.22)
		_drop_k = _fruit_k + rng.randf_range(0.09, 0.14)
		_hang = _slot + Vector3(0, rng.randf_range(0.08, 0.14), 0)
		_sag = rng.randf_range(0.04, 0.07)
		_sway_f = rng.randf_range(1.2, 2.0)
		_whip_f = rng.randf_range(12.0, 18.0)
		_whip_a = rng.randf_range(0.12, 0.20)
		_seed = rng.randi()
		_hue = rng.randf_range(0.24, 0.36)          # deep plant green, sampled
		# The shed petals: one or two, let go while the bloom hangs (before the drop).
		for i in rng.randi_range(1, 2):
			_petals.append({"spawn": rng.randf_range(_fruit_k + 0.02, _drop_k + 0.03),
				"ph": rng.randf() * TAU, "size": rng.randf_range(0.030, 0.045), "at": Vector3.ZERO})
		a.fade = 0.0
		if a.body is EyeBody:
			(a.body as EyeBody).droop = 1.1

	func apply(stage, actors: Array, k: float, _f: AudioFeatures, dt: float) -> void:
		var a = actors[0]
		_t += dt
		var kg := clampf(k / _grow_k, 0.0, 1.0)                                 # growth
		var kf := clampf((k - _grow_k) / (_fruit_k - _grow_k), 0.0, 1.0)        # fruiting
		var kd := clampf((k - _drop_k) / maxf(0.05, 1.0 - _drop_k), 0.0, 1.0)   # drop + retreat
		# The fruit's weight bends the hanging tip lower; a light pendulum sway.
		var sag := _sag * smoothstep(0.0, 1.0, kf)
		var sway := 0.022 * sin(_t * TAU * _sway_f) * kf * (1.0 - kd)
		var tip := _hang + Vector3(sway, _R * 1.02 - sag, 0.0)
		# The standing rose: a near-vertical rise beside the anchor (a gentle lean),
		# cresting above it, the head drooping over onto the hanging tip.
		var c1 := Vector3(_base.x * 1.08, lerpf(_base.y, tip.y, 0.72), 0.0)
		var c2 := Vector3(tip.x, tip.y + 0.5, 0.0)
		var grown := 1.0 - pow(1.0 - kg, 2.6)               # quick, decelerating growth
		if not _detached and k >= _drop_k:
			_detached = true
			stage.spatter(tip, 3, rng, Color.from_hsv(_hue, 0.4, 0.55))
		if _detached:
			# Freed of the weight, the vine WHIPS back and retreats along itself.
			var wob := exp(-4.0 * kd * 3.0) * sin(_whip_f * kd * 3.0)
			c2 += Vector3(-signf(_base.x - _slot.x) * _whip_a * wob, _whip_a * 0.7 * absf(wob), 0.0)
			c1.x += _whip_a * 0.4 * wob * signf(_base.x - _slot.x)
			grown = 1.0 - smoothstep(0.0, 1.0, minf(kd * 1.6, 1.0))
		# The shed petals flutter down from wherever the tip was when they let go.
		var pets: Array = []
		for p in _petals:
			var sp := float(p.spawn)
			if k < sp:
				continue
			if (p.at as Vector3) == Vector3.ZERO:
				p.at = tip
			var fall := clampf((k - sp) / 0.34, 0.0, 1.0)
			if fall >= 1.0:
				continue
			var ph := float(p.ph)
			pets.append({
				"p": (p.at as Vector3) + Vector3(sin(fall * 7.0 + ph) * 0.07 * (1.0 - 0.4 * fall),
					-pow(fall, 1.6) * 0.55, 0.0),
				"ang": ph + fall * 5.0, "size": float(p.size),
				"a": 1.0 - smoothstep(0.55, 1.0, fall)})
		a.state["vine"] = {"base": _base, "c1": c1, "c2": c2, "tip": tip,
			"g": grown, "seed": _seed, "hue": _hue, "r": _R, "petals": pets}
		# The eye: swells at the hanging tip, drops free, settles, rights itself.
		if k < _drop_k:
			a.fade = smoothstep(0.0, 1.0, minf(kf * 1.6, 1.0))
			a.scale = _R * pow(maxf(kf, 0.0001), 0.45)      # fruit swell
			a.pos = tip - Vector3(0, a.scale * 1.02, 0.0)   # hanging from the tip
			if a.body is EyeBody:
				(a.body as EyeBody).droop = 1.1
		else:
			a.scale = _R
			var fall := 1.0 - exp(-5.0 * kd) * cos(13.0 * kd)   # small drop, damped bounce
			a.pos = Vector3(_slot.x, lerpf(_hang.y - sag, _slot.y, fall), _slot.z)
			if a.body is EyeBody:
				(a.body as EyeBody).droop = 1.1 * (1.0 - smoothstep(0.0, 1.0, kd))
		a.home = _slot
		# The verb owns the gaze: neutral under the droop while it hangs and wakes.
		a.state["focus"] = a.pos + Vector3(0, 0, 9.0)
		a.state["focus_age"] = 0.0

	func finish(_stage, actors: Array) -> void:
		var a = actors[0]
		a.state.erase("vine")
		a.pos = _slot
		a.home = _slot
		a.scale = _R
		a.fade = 1.0
		if a.body is EyeBody:
			(a.body as EyeBody).droop = 0.0


## split - MITOSIS: one eye divides into two. The twin buds off the source as a blank
## wet ball (no iris yet), connected by a stretching tissue membrane (drawn by
## [Cast.EyeActor] from `state.mit`); volume is conserved, so the parent visibly
## deflates as the bud swells (one R eye -> two ~0.79R twins). The pull is stick-slip
## (the tissue resists in jerks) up to the SNAP, where the membrane tears: both eyes
## recoil on a damped spring the rest of the way to their slots, droplets spatter,
## and only then does the newborn's iris surface - followed by its first blink.
## args: into (actor id), slots ([[x,y,z],[x,y,z]]).
class Split extends Action:
	var _from_pos := Vector3.ZERO
	var _R := 0.3                     # the parent's whole volume, as a radius
	var _slots: Array = []
	var _tense_k := 0.30              # phase boundaries + dynamics, sampled per instance
	var _snap_k := 0.75
	var _pre_frac := 0.55             # separation reached at the snap (the membrane holds the rest)
	var _rec_freq := 22.0
	var _rec_damp := 6.0
	var _drops := 4
	var _seed := 0
	var _snapped := false

	func begin(stage, actors: Array) -> void:
		var src = actors[0]
		var into = stage.actor(String(args.get("into", "")))
		_from_pos = src.pos
		_R = src.scale
		_slots = args.get("slots", [])
		_tense_k = rng.randf_range(0.26, 0.36)
		_snap_k = rng.randf_range(0.70, 0.80)
		# The membrane must STRETCH on screen: separation at the snap has to well
		# exceed the sum of the two radii (~0.48 of the slot distance), or the bridge
		# stays hidden inside the overlapping spheres and tears unseen.
		_pre_frac = rng.randf_range(0.78, 0.88)
		_rec_freq = rng.randf_range(17.0, 26.0)
		_rec_damp = rng.randf_range(5.0, 7.5)
		_drops = rng.randi_range(3, 6)
		_seed = rng.randi()
		if into != null:
			into.hue = src.hue
			if into.body is EyeBody and src.body is EyeBody:
				(into.body as EyeBody).hue = (src.body as EyeBody).hue
				(into.body as EyeBody).gaze = (src.body as EyeBody).gaze
				(into.body as EyeBody).target = (src.body as EyeBody).target
				(into.body as EyeBody).iris_fade = 0.0    # born featureless
			into.pos = src.pos
			into.scale = 0.0
			into.fade = 0.0

	func apply(stage, actors: Array, k: float, _f: AudioFeatures, _dt: float) -> void:
		var src = actors[0]
		var into = stage.actor(String(args.get("into", "")))
		if into == null or _slots.size() < 2:
			return
		var s0: Vector3 = Cast._vec3(_slots[0], src.home)
		var s1: Vector3 = Cast._vec3(_slots[1], into.home)
		var kt := clampf(k / _tense_k, 0.0, 1.0)                              # tension builds
		var kp := clampf((k - _tense_k) / (_snap_k - _tense_k), 0.0, 1.0)     # the pull
		var ks := clampf((k - _snap_k) / maxf(0.05, 1.0 - _snap_k), 0.0, 1.0) # snap + settle
		# Volume conservation: the bud inflates to half the parent's volume by the snap.
		var v := 0.5 * pow(clampf(k / _snap_k, 0.0, 1.0), 1.3)
		src.scale = _R * pow(1.0 - v, 1.0 / 3.0)
		into.scale = _R * pow(maxf(v, 0.0001), 1.0 / 3.0)
		# Separation: a stick-slip creep to _pre_frac (the tissue gives way in jerks),
		# then the snap's damped spring carries the rest - with an overshoot recoil.
		var sep: float
		if k < _snap_k:
			var jerk := pow(absf(sin(kp * 37.0 + float(_seed % 7)) * sin(kp * 53.0 + 1.7)), 3.0)
			sep = _pre_frac * (pow(kp, 1.15) + 0.06 * jerk * kp)
		else:
			var spring := 1.0 - exp(-_rec_damp * ks) * cos(_rec_freq * ks)
			sep = _pre_frac + (1.0 - _pre_frac) * spring
		# The parent LEANS toward the bud while the tissue drags on it, then recoils.
		var lean := 0.05 * kp * (1.0 - kp) * 4.0 * (1.0 if k < _snap_k else 0.0)
		var toward := (s1 - s0).normalized()
		src.pos = _from_pos.lerp(s0, sep) + toward * (lean * _R)
		into.pos = _from_pos.lerp(s1, sep)
		into.pos.z += 0.02 * (1.0 - ks)          # a hair closer, so the bud draws over its parent
		src.home = src.pos
		into.home = into.pos
		into.fade = smoothstep(0.0, 1.0, clampf(k / (_tense_k * 0.8), 0.0, 1.0))
		# Strain: the tremble channel gives the jiggle AND gathers light around the eye.
		if k < _snap_k:
			src.state["tremble"] = clampf(0.30 * kt + 0.35 * kp, 0.0, 0.6)
		else:
			src.state["tremble"] = maxf(0.0, 0.45 * (1.0 - ks * 3.0))
		# The division owns the gaze: the parent STARES dead ahead under the strain
		# (no saccading while it labors - one thing happening, not three), and the
		# newborn holds the same stare once it tears free. The next entry's look verb
		# takes over naturally when this stops feeding (focus_age expiry).
		src.state["focus"] = src.pos + Vector3(0, 0, 9.0)
		src.state["focus_age"] = 0.0
		if _snapped:
			into.state["focus"] = into.pos + Vector3(0, 0, 9.0)
			into.state["focus_age"] = 0.0
		# The membrane, for the actor's draw pass. `waist` starves through the pull.
		src.state["mit"] = {"to": String(args.get("into", "")), "seed": _seed,
			"waist": 1.0 - kp, "snap": -1.0 if not _snapped else float(src.state["mit"].get("snap", -1.0)) + _dt}
		if k >= _snap_k and not _snapped:
			_snapped = true
			src.state["mit"]["snap"] = 0.0
			stage.spatter(_from_pos.lerp(s0.lerp(s1, 0.5), sep), _drops, rng)
		# The aftermath: the newborn's iris surfaces once it has torn free.
		if _snapped and into.body is EyeBody:
			(into.body as EyeBody).iris_fade = smoothstep(0.0, 1.0, clampf((ks - 0.25) / 0.6, 0.0, 1.0))

	func finish(stage, actors: Array) -> void:
		var src = actors[0]
		var into = stage.actor(String(args.get("into", "")))
		src.state.erase("mit")
		src.state.erase("tremble")
		if _slots.size() >= 2:
			src.pos = Cast._vec3(_slots[0], src.home)
			src.home = src.pos
		var half := _R * pow(0.5, 1.0 / 3.0)
		src.scale = half
		if into != null:
			if _slots.size() >= 2:
				into.pos = Cast._vec3(_slots[1], into.home)
				into.home = into.pos
			into.scale = half
			into.fade = 1.0
			if into.body is EyeBody:
				(into.body as EyeBody).iris_fade = 1.0


## crystallize - the eye FREEZES OVER into the prism (a staged physical transition,
## not a crossfade). Phase 1, stilling: the gaze locks to one point and crystal
## edges creep across the ball ([Cast.EyeActor] draws the cage from `state.crys`)
## while the pupil constricts to a pinpoint and the iris hue chills toward the
## crystal's. Phase 2, faceting: the cage tightens and glints, the iris features
## die away (a blank frozen ball), and the prism's wireframe materializes aligned
## over it. Phase 3, collapse: the ball is sucked into the crystal's core - cold
## shards, an icy flash, and the prism solidifies alive. Give the span `ease:
## linear` - the verb shapes its own phases. args: into (actor id).
class Crystallize extends Action:
	var _facet_k := 0.4               # phase boundaries + character, sampled per instance
	var _fall_k := 0.8
	var _seed := 0
	var _stare := Vector3.ZERO
	var _base_scale := 0.27
	var _base_hue := 0.3
	var _flashed := false

	func begin(stage, actors: Array) -> void:
		var src = actors[0]
		_facet_k = rng.randf_range(0.34, 0.44)
		_fall_k = rng.randf_range(0.76, 0.84)
		_seed = rng.randi()
		_base_scale = src.scale
		# The stare it dies with: one fixed world point, sampled once - the FIRST sign
		# of the freeze is that the eye stops moving.
		_stare = src.pos + Vector3(rng.randf_range(-0.4, 0.4), rng.randf_range(-0.25, 0.25), 7.0)
		if src.body is EyeBody:
			_base_hue = (src.body as EyeBody).hue
		var into = stage.actor(String(args.get("into", "")))
		if into != null:
			into.fade = 0.0

	func apply(stage, actors: Array, k: float, _f: AudioFeatures, _dt: float) -> void:
		var src = actors[0]
		var into = stage.actor(String(args.get("into", "")))
		if into == null:
			return
		var kf := clampf((k - _facet_k) / maxf(0.05, _fall_k - _facet_k), 0.0, 1.0)  # faceting
		var kc := clampf((k - _fall_k) / maxf(0.05, 1.0 - _fall_k), 0.0, 1.0)        # collapse
		var chill := clampf(into.hue, 0.0, 1.0)
		# Stilling: the locked stare, the dying pupil, the chilling iris.
		src.state["focus"] = _stare
		src.state["focus_age"] = 0.0
		if src.body is EyeBody:
			var eye := src.body as EyeBody
			eye.dilate_bias = -0.30 * smoothstep(0.0, 1.0, minf(k / _facet_k, 1.0))
			eye.hue = lerpf(_base_hue, chill, smoothstep(0.0, 1.0, k))
			eye.iris_fade = 1.0 - smoothstep(0.0, 1.0, kf)         # features die in phase 2
		# The cage, drawn by the actor: growth through phases 1-2, gone with the ball.
		src.state["crys"] = {"k": k, "facet": kf, "seed": _seed, "hue": chill}
		# Collapse: the ball is sucked into the crystal's core; the prism solidifies.
		src.scale = _base_scale * (1.0 - smoothstep(0.0, 1.0, kc))
		src.fade = 1.0 if kc < 1.0 else 0.0
		into.fade = smoothstep(0.0, 1.0, kf * 0.55 + kc * 0.45)
		into.drive_gain = 0.25 + 0.75 * (kf * 0.4 + kc * 0.6)
		if not _flashed and kc > 0.0:
			_flashed = true
			var icy := Color.from_hsv(chill, 0.35, 1.0)
			stage.flash(into, icy)
			stage.spatter(src.pos, rng.randi_range(4, 7), rng, icy)

	func finish(stage, actors: Array) -> void:
		var src = actors[0]
		src.fade = 0.0
		src.scale = _base_scale
		src.state.erase("crys")
		if src.body is EyeBody:
			(src.body as EyeBody).iris_fade = 1.0
			(src.body as EyeBody).dilate_bias = 0.0
		var into = stage.actor(String(args.get("into", "")))
		if into != null:
			into.fade = 1.0
			into.drive_gain = 1.0


## tremble - the riser: the eye vibrates harder as k rises, light building around it
## (the drawing lives in [Cast.EyeActor]). Latches at its final level - the burst
## that replaces the eye is what ends it.
class Tremble extends Action:
	func apply(_stage, actors: Array, k: float, f: AudioFeatures, _dt: float) -> void:
		var drive := clampf(f.energy * 0.85 + f.beat * 0.6, 0.0, 1.0)
		for a in actors:
			a.state["tremble"] = clampf(k * (0.55 + 0.6 * drive), 0.0, 1.0)


## burst - the target bursts into being in a flash (the DROP): revealed at full fade,
## optionally replacing another actor (the eye it erupts from). Usually a point span
## gated on a beat: `{at: 0, on: beat, by: 0.8, action: burst, ...}`.
class Burst extends Action:
	func apply(stage, actors: Array, _k: float, _f: AudioFeatures, _dt: float) -> void:
		for a in actors:
			a.fade = 1.0
			stage.flash(a, Color(1.0, 0.5, 0.4))
		var rep = stage.actor(String(args.get("replaces", "")))
		if rep != null:
			rep.fade = 0.0
			rep.state.erase("tremble")


## lock - phase-lock the target's pose to another prism (they turn as one), with a
## brief bright tie between them at the snap. Latches until a desync. args: to
## (actor id), snap (window fraction where the snap lands, default 0.05).
class Lock extends Action:
	var _snapped := false
	var _snap_k := 0.05

	func begin(_stage, _actors: Array) -> void:
		_snap_k = num("snap", 0.05)

	func apply(stage, actors: Array, k: float, _f: AudioFeatures, _dt: float) -> void:
		if k < _snap_k and not _snapped:
			return
		for a in actors:
			if not _snapped:
				var to = stage.actor(String(args.get("to", "")))
				if to != null:
					stage.tie(a, to)
			a.state["lock_to"] = String(args.get("to", ""))
		_snapped = true


## desync - break a phase-lock: the target resumes its own rotation.
class Desync extends Action:
	func apply(_stage, actors: Array, _k: float, _f: AudioFeatures, _dt: float) -> void:
		for a in actors:
			a.state.erase("lock_to")


## hold_still - damp the target's own spin toward rest (it keeps breathing). Latches.
class HoldStill extends Action:
	func apply(_stage, actors: Array, k: float, _f: AudioFeatures, _dt: float) -> void:
		for a in actors:
			a.state["still"] = k


## sway - slip the anchor and drift weightlessly around it, pushing and pulling (the
## float). Each target wanders on its own seeded phases; the audio leans into it.
## args: amp (world units, default [0.07, 0.10] - sampled per target).
class Sway extends Action:
	var _t := 0.0
	var _ph: Array = []               # per-target {px, py, fx, fy, amp}

	func begin(_stage, actors: Array) -> void:
		for i in actors.size():
			_ph.append({"px": rng.randf() * TAU, "py": rng.randf() * TAU,
				"fx": rng.randf_range(0.55, 0.95), "fy": rng.randf_range(0.6, 1.0),
				"amp": num("amp", rng.randf_range(0.07, 0.10))})

	func apply(_stage, actors: Array, k: float, f: AudioFeatures, dt: float) -> void:
		_t += dt
		var drive := clampf(f.energy * 0.85 + f.beat * 0.6, 0.0, 1.0)
		var follow := 1.0 - exp(-3.0 * dt)
		for i in actors.size():
			var a = actors[i]
			var p: Dictionary = _ph[i]
			var w := Vector2(sin(_t * float(p.fx) + float(p.px)), cos(_t * float(p.fy) + float(p.py))) \
				* (float(p.amp) * k * (0.5 + drive))
			a.pos = a.pos.lerp(a.home + Vector3(w.x, w.y, 0.0), follow)


## counterflow - the pair take OPPOSITE HIGHWAYS: each merges onto its own lane and
## cruises against the other (they pass mid-scene), rolling about its travel axis
## like a drill boring down its road; in the final stretch each pulls a LOOP off its
## lane - the arc that carries them into the ouroboros. Cruise speed rides the
## actor's live time_scale, so a specialized small/fast prism zips while the large/
## slow one glides. Give the span `ease: linear`. args: lane (half-gap between the
## highways), speed (world/s), loop_k (where the exit loop begins, 0..1).
class Counterflow extends Action:
	var _lane := 0.18
	var _speed := 0.35
	var _loop_k := 0.7
	var _loop_r := 0.3
	var _roll := 2.8
	var _from: Array = []
	var _dir: Array = []              # +1 / -1 travel direction per target
	var _trav: Array = []             # accumulated cruise distance per target
	var _loop_at: Array = []          # position where each target left its lane

	func begin(_stage, actors: Array) -> void:
		_lane = num("lane", rng.randf_range(0.14, 0.22))
		_speed = num("speed", rng.randf_range(0.28, 0.40))
		_loop_k = num("loop_k", rng.randf_range(0.66, 0.74))
		_loop_r = rng.randf_range(0.19, 0.26)
		_roll = rng.randf_range(2.2, 3.4)
		for i in actors.size():
			var a = actors[i]
			_from.append(a.pos)
			_trav.append(0.0)
			_loop_at.append(Vector3.ZERO)
			# Travel TOWARD the other side of the frame, so the two pass each other.
			_dir.append(-signf(a.pos.x) if absf(a.pos.x) > 0.01 else (1.0 if i == 0 else -1.0))

	func apply(_stage, actors: Array, k: float, _f: AudioFeatures, dt: float) -> void:
		# The CURRENT: the pair start adrift and the flow builds slowly, easing them
		# onto their lanes and up to cruise - caught, not launched.
		var current := smoothstep(0.0, 0.38, k)
		for i in actors.size():
			var a = actors[i]
			var d := float(_dir[i])
			var up := 1.0 if i == 0 else -1.0          # which side its lane and loop live on
			var lane_y := _lane * up
			var merge := smoothstep(0.0, 0.42, k)
			if k < _loop_k:
				_trav[i] = float(_trav[i]) + dt * _speed * current * (0.4 + 0.6 * a.time_scale)
				var p := Vector3(float((_from[i] as Vector3).x) + d * float(_trav[i]),
					lerpf(float((_from[i] as Vector3).y), lane_y, merge), 0.0)
				a.pos = p
				_loop_at[i] = p
			else:
				# The exit loop: a circular arc off the lane, curling up (blue) or down
				# (red) - and as it turns, the same current draws it into a SPIRAL that
				# sinks toward the centre-depth, so the cut into the ouroboros catches
				# both converging on the point the swarm then blossoms out of.
				var kl := smoothstep(0.0, 1.0, (k - _loop_k) / maxf(0.05, 1.0 - _loop_k))
				var c := (_loop_at[i] as Vector3) + Vector3(0, up * _loop_r, 0)
				var ph := kl * 3.7                     # ~210 degrees of the circle
				var sink := smoothstep(0.45, 1.0, kl)
				var on_loop := c + Vector3(sin(ph) * d, -cos(ph) * up, 0.0) * _loop_r
				a.pos = on_loop.lerp(Vector3(0.0, 0.0, -0.85), sink * 0.8)
			a.home = a.pos
			# Roll about the travel axis - a drill boring along its highway; opposite
			# directions, opposite rolls. Builds with the current, eases off in the loop.
			if a.body is PrismBody:
				var want := Vector3(d * _roll * current * (0.4 + 0.6 * a.time_scale), 0.12 * d, 0.0)
				if k >= _loop_k:
					want *= 0.4
				var pb := a.body as PrismBody
				pb._vel = pb._vel.lerp(want, 1.0 - exp(-3.0 * dt))


## specialize - diverge in character: scale eases to size x, the body's whole tempo
## to `tempo` x, and a scale pulse latches ({amp, rate}) - bias vs variance made
## physical. args: size (default 1.0), tempo (default 1.0), pulse {amp, rate}.
class Specialize extends Action:
	var _base: Array = []
	var _size := 1.0
	var _tempo := 1.0

	func begin(_stage, actors: Array) -> void:
		_size = num("size", 1.0)
		_tempo = num("tempo", 1.0)
		for a in actors:
			_base.append(a.scale)

	func apply(_stage, actors: Array, k: float, _f: AudioFeatures, _dt: float) -> void:
		var pulse: Dictionary = args.get("pulse", {}) if typeof(args.get("pulse")) == TYPE_DICTIONARY else {}
		for i in actors.size():
			var a = actors[i]
			a.scale = lerpf(float(_base[i]), float(_base[i]) * _size, k)
			a.time_scale = lerpf(1.0, _tempo, k)
			if not pulse.is_empty():
				a.state["pulse"] = {"amp": float(pulse.get("amp", 0.1)) * k,
					"rate": float(pulse.get("rate", 1.2))}


## gather - the swarm's members ease in one at a time and form up (k plays the
## staggered joins through; see [Cast.SwarmActor]).
class Gather extends Action:
	func apply(_stage, actors: Array, k: float, _f: AudioFeatures, _dt: float) -> void:
		for a in actors:
			a.state["gather"] = k


## fly - the formation flies forward along its track, the stage camera following a
## touch slower so it pulls ahead and recedes (the ONE camera move in the brief).
## args: speed (x, default 1.0), follow (camera fraction of the travel, default 0.72).
class Fly extends Action:
	var _base_z := 0.0
	var _speed := 1.0
	var _follow := 0.72

	func begin(stage, _actors: Array) -> void:
		_base_z = stage.lens.eye.z
		_speed = num("speed", 1.0)
		_follow = num("follow", 0.72)

	func apply(stage, actors: Array, k: float, f: AudioFeatures, dt: float) -> void:
		var drive := clampf(f.energy * 0.8 + f.beat * 0.6, 0.0, 1.0)
		for a in actors:
			var travel := float(a.state.get("travel", 0.0)) + dt * k * (1.2 + 0.8 * drive) * _speed
			a.state["travel"] = travel
			stage.lens.eye.z = _base_z + travel * _follow
			stage.lens.look.z = stage.lens.eye.z + 6.0


## helix_split - the single track opens into a double helix: the spread widens and
## the counter-winding red strand fades in (see [Cast.SwarmActor]).
class HelixSplit extends Action:
	func apply(_stage, actors: Array, k: float, _f: AudioFeatures, _dt: float) -> void:
		for a in actors:
			a.state["split_k"] = k


## lane_jump - the swarm banks to its side and leaps across to the other strand,
## then holds there (k rides the bank's arc through the crossing).
class LaneJump extends Action:
	func apply(_stage, actors: Array, k: float, _f: AudioFeatures, _dt: float) -> void:
		for a in actors:
			a.state["lane"] = k


## set - set one actor parameter instantly. args: param, value.
class SetParam extends Action:
	func apply(_stage, actors: Array, _k: float, _f: AudioFeatures, _dt: float) -> void:
		var v := num("value", 0.0)
		for a in actors:
			a.set_param(String(args.get("param", "")), v)


## ramp - ease one actor parameter from its current value (or `from`) to `to`.
class Ramp extends Action:
	var _from: Array = []
	var _to := 0.0

	func begin(_stage, actors: Array) -> void:
		_to = num("to", 0.0)
		for a in actors:
			_from.append(num("from", a.get_param(String(args.get("param", "")))))

	func apply(_stage, actors: Array, k: float, _f: AudioFeatures, _dt: float) -> void:
		for i in actors.size():
			actors[i].set_param(String(args.get("param", "")), lerpf(float(_from[i]), _to, k))


## pulse - a scale breathing on the target while the span runs ({amp, rate});
## latches unless args.latch is false.
class Pulse extends Action:
	var _amp := 0.1
	var _rate := 1.2

	func begin(_stage, _actors: Array) -> void:
		_amp = num("amp", 0.1)
		_rate = num("rate", 1.2)

	func apply(_stage, actors: Array, k: float, _f: AudioFeatures, _dt: float) -> void:
		for a in actors:
			a.state["pulse"] = {"amp": _amp * k, "rate": _rate}

	func finish(_stage, actors: Array) -> void:
		if not bool(args.get("latch", true)):
			for a in actors:
				a.state.erase("pulse")


## flash - a bright burst overlay on the target (a stinger accent).
class Flash extends Action:
	func apply(stage, actors: Array, _k: float, _f: AudioFeatures, _dt: float) -> void:
		var h := num("hue", -1.0)
		var col := Color(1, 1, 1) if h < 0.0 else Color.from_hsv(h, 0.45, 1.0)
		for a in actors:
			stage.flash(a, col)
