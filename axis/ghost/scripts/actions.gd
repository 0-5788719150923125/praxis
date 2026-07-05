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
	"look": Look, "blink": Blink, "split": Split, "crystallize": Crystallize,
	"tremble": Tremble, "burst": Burst, "lock": Lock, "desync": Desync,
	"hold_still": HoldStill, "sway": Sway, "specialize": Specialize,
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
	var _src_blinked := false
	var _born_blinked := false

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
		# The membrane, for the actor's draw pass. `waist` starves through the pull.
		src.state["mit"] = {"to": String(args.get("into", "")), "seed": _seed,
			"waist": 1.0 - kp, "snap": -1.0 if not _snapped else float(src.state["mit"].get("snap", -1.0)) + _dt}
		if k >= _snap_k and not _snapped:
			_snapped = true
			src.state["mit"]["snap"] = 0.0
			stage.spatter(_from_pos.lerp(s0.lerp(s1, 0.5), sep), _drops, rng)
		# The aftermath: the parent blinks off the strain; the newborn's iris surfaces,
		# then it blinks awake.
		if _snapped and into.body is EyeBody:
			(into.body as EyeBody).iris_fade = smoothstep(0.0, 1.0, clampf((ks - 0.25) / 0.6, 0.0, 1.0))
			if not _src_blinked and ks > 0.15 and src.body is EyeBody:
				_src_blinked = true
				(src.body as EyeBody).blink(0.34)
			if not _born_blinked and ks > 0.85:
				_born_blinked = true
				(into.body as EyeBody).blink(0.42)

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


## crystallize - the target dissolves while `into` (a prism on the same slot) forms
## in its place, with a form-flash at the crossover. args: into (actor id).
class Crystallize extends Action:
	var _flashed := false

	func apply(stage, actors: Array, k: float, _f: AudioFeatures, _dt: float) -> void:
		var src = actors[0]
		var into = stage.actor(String(args.get("into", "")))
		src.fade = clampf(1.0 - k, 0.0, 1.0)
		if into != null:
			into.fade = smoothstep(0.0, 1.0, k)
			into.drive_gain = 0.4 + 0.6 * k        # it comes to life as it finishes forming
			if not _flashed and k >= 0.5:
				_flashed = true
				stage.flash(into, Color(0.75, 0.86, 1.0))

	func finish(stage, actors: Array) -> void:
		actors[0].fade = 0.0
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
