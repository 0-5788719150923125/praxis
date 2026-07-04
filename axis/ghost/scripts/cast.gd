extends RefCounted
class_name Cast

## Cast - the actor registry for data-driven scenes (see scenes/stage.gd).
##
## The registry sibling of [Primitives] (forces), [Layer] (visuals) and [Shots]
## (framing), for *performers*: an **actor** wraps a live body ([EyeBody],
## [PrismBody], ...) with the state a storyboard's verbs steer - slot position,
## scale, hue, visibility, tempo - so choreography lives in the data (the track's
## action spans) instead of bespoke scene code. Every numeric config value accepts a
## `[lo, hi]` range, sampled once per instance ("cattle, not pets").
##
## An actor is a RefCounted, so a stage hands its LIVE actors to the next stage
## through the morph payload and a body literally continues across a cut.
##
## Verbs communicate with actors through `state` (a plain Dictionary): e.g.
## `state.tremble` (0..1 riser vibration), `state.lock_to` (pose-lock source id),
## `state.still` (spin damping), `state.pulse` ({amp, rate} scale breathing),
## `state.focus` (a world point the eye looks at). Kind classes read what they
## understand; unknown keys are inert - a misspelled verb can't crash a show.

const REGISTRY := {
	"eye": "eye",
	"prism": "prism",
	"swarm": "swarm",
}


## Build an actor from a storyboard cast entry. `raw_cfg` is the un-sampled entry
## (ranges intact); each instance draws its own values from [param rng].
static func make(kind: String, rng: RandomNumberGenerator, raw_cfg: Dictionary) -> Actor:
	if not REGISTRY.has(kind):
		push_warning("ghost: unknown actor kind '%s' (have: %s)" % [kind, ", ".join(REGISTRY.keys())])
		return null
	var cfg: Dictionary = Storyboard.sample(raw_cfg, rng)
	var a: Actor
	match kind:
		"eye":
			a = EyeActor.new()
		"prism":
			a = PrismActor.new()
		"swarm":
			a = SwarmActor.new()
	a.id = String(cfg.get("id", kind))
	a.kind = kind
	a.spec = cfg
	a.pos = _vec3(cfg.get("at"), Vector3.ZERO)
	a.home = a.pos
	a.fade = 0.0 if bool(cfg.get("hidden", false)) else 1.0
	a.setup(int(cfg.get("seed", rng.randi())), cfg, rng)
	return a


static func _vec3(v: Variant, fallback: Vector3) -> Vector3:
	if typeof(v) == TYPE_ARRAY and (v as Array).size() == 3:
		var arr: Array = v
		return Vector3(float(arr[0]), float(arr[1]), float(arr[2]))
	return fallback


## The base performer. Subclasses own a body and know how to advance and draw it;
## everything a verb steers lives here.
class Actor extends RefCounted:
	var id := ""
	var kind := ""
	var spec := {}                    # the sampled config this instance was built from
	var body: RefCounted = null       # the live EyeBody / PrismBody / ...
	var pos := Vector3.ZERO           # current world slot
	var home := Vector3.ZERO          # anchor (sway strains pos away from home)
	var scale := 0.3                  # world size (an eye's radius, a prism's footprint)
	var hue := 0.6
	var fade := 1.0                   # 0 hidden .. 1 shown (verbs reveal/dissolve via this)
	var time_scale := 1.0             # tempo lever (specialize): scales the body's clock
	var drive_gain := 1.0             # how hard the audio drives the body
	var state := {}                   # verb scratch - see the class doc
	var _t := 0.0                     # local clock (jitter phases, pulse breathing)

	func setup(_seed: int, _cfg: Dictionary, _rng: RandomNumberGenerator) -> void:
		pass

	func update(_f: AudioFeatures, _dt: float, _stage) -> void:
		pass

	## Depth-sortable draw items: [{d: camera depth, call: Callable}]. The stage
	## collects every actor's items, sorts far-first, and runs the callables.
	func draw_items(_stage, _lens: Lens3D, _u: float) -> Array:
		return []

	## The drawn size: `scale` breathing with a latched pulse ({amp, rate}), if any.
	func draw_scale() -> float:
		var p: Variant = state.get("pulse")
		if typeof(p) == TYPE_DICTIONARY:
			return scale * (1.0 + float(p.get("amp", 0.0)) * sin(_t * float(p.get("rate", 1.0))))
		return scale

	## Generic parameter access for the set/ramp verbs. Unknown names land in
	## `state` so kind classes can expose their own levers.
	func set_param(name: String, v: float) -> void:
		match name:
			"scale": scale = v
			"fade": fade = v
			"hue": hue = v
			"time_scale": time_scale = v
			"drive": drive_gain = v
			_: state[name] = v

	func get_param(name: String) -> float:
		match name:
			"scale": return scale
			"fade": return fade
			"hue": return hue
			"time_scale": return time_scale
			"drive": return drive_gain
			_: return float(state.get(name, 0.0))


## A human eye ([EyeBody]) on a slot. Gaze is driven by a look verb feeding
## `state.focus` (shared focus = real vergence); when no verb has fed it for a
## moment it falls back to the body's own centre-biased self-saccades.
class EyeActor extends Actor:
	func setup(seed_value: int, cfg: Dictionary, rng: RandomNumberGenerator) -> void:
		scale = float(cfg.get("radius", cfg.get("scale", rng.randf_range(0.24, 0.34))))
		hue = float(cfg.get("hue", rng.randf_range(0.05, 0.6)))
		body = EyeBody.new(seed_value, hue)

	func set_param(name: String, v: float) -> void:
		match name:
			"dilate": (body as EyeBody).dilate_bias = v
			"lid": (body as EyeBody).lid = v
			_: super.set_param(name, v)

	func get_param(name: String) -> float:
		match name:
			"dilate": return (body as EyeBody).dilate_bias
			"lid": return (body as EyeBody).lid
			_: return super.get_param(name)

	func update(f: AudioFeatures, dt: float, _stage) -> void:
		_t += dt
		var eye := body as EyeBody
		var drive := clampf((f.energy * 0.7 + f.beat * 0.4) * drive_gain, 0.0, 1.0)
		var tr := float(state.get("tremble", 0.0))
		if tr > 0.0:
			drive = clampf(drive + tr * 0.5, 0.0, 1.0)   # it widens as it strains
		if state.has("focus"):
			eye.autonomous = false
			eye.look_at_point(pos, state["focus"])
			if state.has("div"):
				eye.target += state["div"]               # the rare one-eye divergence
			state["focus_age"] = float(state.get("focus_age", 0.0)) + dt
			if float(state["focus_age"]) > 0.8:          # the look verb went quiet
				state.erase("focus")
		else:
			eye.autonomous = true
		eye.update(dt * time_scale, drive)

	func draw_items(stage, lens: Lens3D, u: float) -> Array:
		if fade <= 0.003:
			return []
		var p := pos
		var tr := float(state.get("tremble", 0.0))
		var items := []
		if tr > 0.001:
			# High-frequency vibration (the riser): jitter the world slot, and gather
			# a soft light around the eye as it builds. (From eye_prism, now a verb.)
			var jx := (sin(_t * 61.0) + 0.6 * sin(_t * 97.0 + 1.3)) * 0.5
			var jy := (sin(_t * 71.0 + 0.7) + 0.6 * sin(_t * 113.0)) * 0.5
			p += Vector3(jx, jy, 0.0) * (0.05 * tr)
			var pj := lens.project(p)
			if pj.z > lens.near:
				var light_p := p
				items.append({"d": lens.depth(p) + 0.01, "call": func() -> void:
					_draw_riser_light(stage, lens, u, light_p, tr)})
		var eye := body as EyeBody
		var at := p
		var r := draw_scale()
		var a := fade
		items.append({"d": lens.depth(p), "call": func() -> void:
			eye.draw(stage, lens, u, at, r, a)})
		return items

	# A soft radial glow gathering on the eye as it trembles - "light building around it".
	func _draw_riser_light(stage, lens: Lens3D, u: float, p: Vector3, k: float) -> void:
		var pj := lens.project(p)
		if pj.z <= lens.near:
			return
		var c := Vector2(pj.x, pj.y) * u
		var base := scale * lens._focal / maxf(0.1, pj.z) * u
		var rings := 6
		for i in rings:
			var fr := 1.0 - float(i) / float(rings - 1)
			var rr := base * (1.2 + 3.0 * fr) * (0.7 + 0.5 * k)
			stage.draw_circle(c, rr, Color(0.7, 0.82, 1.0, clampf(0.05 * k * (1.0 - 0.85 * fr), 0.0, 0.3)))


## A living wireframe prism ([PrismBody]) on a slot, drawn through the projected-slot
## bridge (world slot -> screen centre + perspective scale) so it lines up exactly
## with 3D bodies sharing the lens.
class PrismActor extends Actor:
	var _spin := 1.0                  # eases toward 0 while state.still holds the body

	func setup(seed_value: int, cfg: Dictionary, rng: RandomNumberGenerator) -> void:
		scale = float(cfg.get("scale", rng.randf_range(0.24, 0.30)))
		hue = float(cfg.get("hue", 0.6))
		body = PrismBody.new(seed_value)

	func update(f: AudioFeatures, dt: float, _stage) -> void:
		_t += dt
		var prism := body as PrismBody
		var drive := clampf((f.energy * 0.85 + f.beat * 0.6) * drive_gain, 0.0, 1.0)
		prism.update(dt * time_scale, drive)
		# hold_still: damp the body's own spin toward rest (it keeps breathing).
		var still := float(state.get("still", 0.0))
		var want := 1.0 - clampf(still, 0.0, 1.0)
		_spin = lerpf(_spin, want, 1.0 - exp(-4.0 * dt))
		if _spin < 0.999:
			prism._vel *= (0.02 + 0.98 * _spin)

	func draw_items(stage, lens: Lens3D, u: float) -> Array:
		if fade <= 0.003:
			return []
		var pj := lens.project(pos)
		if pj.z <= lens.near:
			return []
		var prism := body as PrismBody
		var center := Vector2(pj.x, pj.y) * u
		var px := draw_scale() * lens._focal / maxf(0.1, pj.z) * u * 1.15
		var h := hue
		var a := fade
		return [{"d": pj.z, "call": func() -> void:
			prism.draw(stage, center, px, h, a)}]


## A formation of prisms streaming along an invisible track into the void - a GROUP
## actor (many bodies, one performer), because the swarm's coordinated strand math is
## one thing, not N independent choreographies. Ported from prism_swarm.gd. Verbs
## steer it through state: `gather` (members ease in one at a time), `travel` (flown
## distance - the fly verb), `split_k` (the track opens into a double helix; the red
## strand fades in on the OTHER winding), `lane` (the bank-and-jump to the far
## strand). config: count / count_red / spacing / head / size / helix / r_min /
## r_max / hue (blue strand) / hue_red / bank ("left" / "right"; by seed otherwise) /
## lead (an actor id whose live prism becomes member 0 across a carry).
class SwarmActor extends Actor:
	var _blue: Array = []             # PrismBody per blue member (0 = the lead)
	var _red: Array = []
	var _entry: Array = []            # per blue member: fly-in origin (unit fractions)
	var _bank := 1.0
	var _spacing := 0.45
	var _head := 3.6
	var _size := 0.42
	var _helix := 1.15
	var _r_min := 0.30
	var _r_max := 0.95
	var _hue_red := 0.0
	var _stag := 0.1                  # gather stagger per member (fraction of the gather)

	func setup(seed_value: int, cfg: Dictionary, rng: RandomNumberGenerator) -> void:
		var seeded := RandomNumberGenerator.new()
		seeded.seed = seed_value
		hue = float(cfg.get("hue", 0.6))
		_hue_red = float(cfg.get("hue_red", 0.0))
		_spacing = float(cfg.get("spacing", 0.45))
		_head = float(cfg.get("head", 3.6))
		_size = float(cfg.get("size", 0.42))
		_helix = float(cfg.get("helix", 1.15))
		_r_min = float(cfg.get("r_min", 0.30))
		_r_max = float(cfg.get("r_max", 0.95))
		match String(cfg.get("bank", "")):
			"left":
				_bank = -1.0
			"right":
				_bank = 1.0
			_:
				_bank = 1.0 if seeded.randf() < 0.5 else -1.0
		var count := maxi(1, int(cfg.get("count", 7)))
		var count_red := maxi(1, int(cfg.get("count_red", count)))
		_stag = 0.7 / float(count)
		for i in count:
			_blue.append(PrismBody.new(seeded.randi()))
			if i == 0:
				_entry.append(Vector2(0.15, 0.0))     # replaced by the carried lead's slot
			else:
				var side := 1.0 if (i % 2 == 0) else -1.0
				_entry.append(Vector2(side * seeded.randf_range(0.7, 1.1),
					seeded.randf_range(-0.5, 0.5)))
		for i in count_red:
			_red.append(PrismBody.new(seeded.randi()))
		state["gather"] = 1.0             # formed unless a gather span plays it in

	## Continue a carried prism as the formation's lead: its live body becomes member
	## 0 and its last screen slot the fly-in origin (called from stage.begin_morph).
	func adopt_lead(old: Actor, entry_frac: Vector2) -> void:
		if old.body is PrismBody and not _blue.is_empty():
			_blue[0] = old.body
			_entry[0] = entry_frac

	func update(f: AudioFeatures, dt: float, _stage) -> void:
		_t += dt
		var drive := clampf((f.energy * 0.8 + f.beat * 0.6) * drive_gain, 0.0, 1.0)
		for b in _blue:
			(b as PrismBody).update(dt * time_scale, drive)
		if float(state.get("split_k", 0.0)) > 0.01:       # reds live once the helix opens
			for r in _red:
				(r as PrismBody).update(dt * time_scale, drive)

	# A point on one strand of the double helix at track distance d. strand_dir sets
	# the angular offset AND the twist direction, so the strands counter-wind and weave.
	func _strand(d: float, strand_dir: float, r: float) -> Vector3:
		var ang := d * _helix * strand_dir + PI * (0.5 - 0.5 * strand_dir)
		return Vector3(cos(ang) * r, sin(ang) * r * 0.72, d)

	func draw_items(stage, lens: Lens3D, u: float) -> Array:
		if fade <= 0.003:
			return []
		var items := []
		var split_k := float(state.get("split_k", 0.0))
		var r := lerpf(_r_min, _r_max, split_k)
		var travel := float(state.get("travel", 0.0))
		var gather := float(state.get("gather", 1.0))
		var lane := float(state.get("lane", 0.0))
		var arc := sin(lane * PI) * _bank * r * 0.5       # the bank/overshoot of the leap
		for i in _blue.size():
			var d := _head - i * _spacing + travel
			if d <= 0.05:
				continue
			var on := clampf((gather - i * _stag) / 0.25, 0.0, 1.0)
			if on <= 0.001:
				continue
			var world := _strand(d, 1.0, r).lerp(_strand(d, -1.0, r), lane) + Vector3(arc, 0.0, 0.0)
			var pj := lens.project(world)
			if pj.z <= lens.near:
				continue
			var center := Vector2(pj.x, pj.y) * u
			var alpha := clampf(1.0 - (pj.z - 1.0) / 12.0, 0.12, 1.0) * fade
			if on < 1.0:                                  # easing in from its entry point
				center = (_entry[i] as Vector2 * u).lerp(center, smoothstep(0.0, 1.0, on))
				alpha *= on
			var b: PrismBody = _blue[i]
			var c := center
			var px := _size * lens._focal / pj.z * u
			var a := alpha
			items.append({"d": pj.z, "call": func() -> void:
				b.draw(stage, c, px, hue, a)})
		if split_k > 0.01:
			for i in _red.size():
				var d := _head - i * _spacing + travel
				if d <= 0.05:
					continue
				var world := _strand(d, -1.0, r)          # counter-winding, weaving past blue
				var pj := lens.project(world)
				if pj.z <= lens.near:
					continue
				var b: PrismBody = _red[i]
				var c := Vector2(pj.x, pj.y) * u
				var px := _size * lens._focal / pj.z * u
				var a := clampf(1.0 - (pj.z - 1.0) / 12.0, 0.12, 1.0) * fade * split_k
				items.append({"d": pj.z, "call": func() -> void:
					b.draw(stage, c, px, _hue_red, a)})
		return items


## The deduplicated focus-tier gaze sampler (it was baked, differently tuned, into
## eye.gd, two_eyes.gd AND eye_prism.gd). A data table of depth tiers - probability,
## distance range, dwell range - plus a lateral excursion range and a rare one-eye
## stray. Ranges are re-drawn per refixation from the OWNING verb's seeded rng, so a
## look is deterministic per show yet never twice the same fixation.
class FocusSampler extends RefCounted:
	const DEFAULT_TIERS := [
		{"p": 0.32, "dist": [2.2, 4.0], "dwell": [3.0, 5.5]},     # near - linger, eyes verge
		{"p": 0.36, "dist": [5.0, 11.0], "dwell": [2.0, 3.6]},    # mid
		{"p": 0.32, "dist": [22.0, 80.0], "dwell": [3.5, 7.0]},   # far - linger, near-parallel
	]

	var tiers: Array = DEFAULT_TIERS
	var lateral: Variant = [0.12, 0.45]
	var stray := 0.0                  # chance per refixation that ONE eye wanders
	var focus := Vector3(0, 0, 6.0)
	var stray_of := -1                # which target strays this fixation (-1 = none)
	var stray_off := Vector2.ZERO
	var _dwell := 0.0
	var _rng: RandomNumberGenerator

	func _init(cfg: Dictionary, rng: RandomNumberGenerator) -> void:
		_rng = rng
		if typeof(cfg.get("tiers")) == TYPE_ARRAY:
			tiers = cfg["tiers"]
		lateral = cfg.get("lateral", lateral)
		stray = float(cfg.get("stray", 0.0))

	## Advance the dwell; on expiry pick a new tier / focus / (maybe) stray. Returns
	## true when a refixation happened.
	func advance(dt: float, n_targets: int) -> bool:
		_dwell -= dt
		if _dwell > 0.0:
			return false
		var roll := _rng.randf()
		var acc := 0.0
		var tier: Dictionary = tiers.back()
		for t in tiers:
			acc += float((t as Dictionary).get("p", 1.0 / maxi(1, tiers.size())))
			if roll <= acc:
				tier = t
				break
		var d := _range(tier.get("dist", [5.0, 11.0]))
		_dwell = _range(tier.get("dwell", [2.0, 3.6]))
		var sacc := EyeBody.saccade_target(_rng)
		var lat := _range(lateral)
		focus = Vector3(sacc.x * lat, sacc.y * lat * 0.7, d)
		stray_of = -1
		if n_targets > 1 and _rng.randf() < stray:
			stray_of = _rng.randi() % n_targets
			stray_off = Vector2(_rng.randf_range(-1, 1), _rng.randf_range(-1, 1)).normalized() \
				* _rng.randf_range(0.06, 0.16)
		return true

	# A `[lo, hi]` range re-drawn per call, or a plain number as-is.
	func _range(v: Variant) -> float:
		if typeof(v) == TYPE_ARRAY and (v as Array).size() == 2:
			return _rng.randf_range(float(v[0]), float(v[1]))
		return float(v)
