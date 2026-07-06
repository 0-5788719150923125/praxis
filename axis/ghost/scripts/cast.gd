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
	# Live-dial modulation overlay (see [Dial]): set fresh by the stage every frame,
	# NEVER accumulated into the actor's own state - neutral values mean the dial is
	# quiet and everything below costs nothing.
	var mod_scale := 1.0              # multiplies the drawn size
	var mod_hue := 0.0                # added to the drawn hue (prisms/swarm; eyes hold their iris)
	var mod_time := 1.0               # multiplies the body's tempo
	var mod_drive := 1.0              # multiplies the audio drive
	var mod_off := Vector2.ZERO       # world-space x/y slot displacement

	func setup(_seed: int, _cfg: Dictionary, _rng: RandomNumberGenerator) -> void:
		pass

	func update(_f: AudioFeatures, _dt: float, _stage) -> void:
		pass

	## Depth-sortable draw items: [{d: camera depth, call: Callable}]. The stage
	## collects every actor's items, sorts far-first, and runs the callables.
	func draw_items(_stage, _lens: Lens3D, _u: float) -> Array:
		return []

	## The drawn size: `scale` breathing with a latched pulse ({amp, rate}), if any,
	## times the live-dial overlay.
	func draw_scale() -> float:
		var s := scale * mod_scale
		var p: Variant = state.get("pulse")
		if typeof(p) == TYPE_DICTIONARY:
			return s * (1.0 + float(p.get("amp", 0.0)) * sin(_t * float(p.get("rate", 1.0))))
		return s

	## The drawn slot: the choreographed position plus the live-dial displacement.
	func draw_pos() -> Vector3:
		return pos + Vector3(mod_off.x, mod_off.y, 0.0)

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
			"iris": (body as EyeBody).iris_fade = v
			_: super.set_param(name, v)

	func get_param(name: String) -> float:
		match name:
			"dilate": return (body as EyeBody).dilate_bias
			"lid": return (body as EyeBody).lid
			"iris": return (body as EyeBody).iris_fade
			_: return super.get_param(name)

	func update(f: AudioFeatures, dt: float, _stage) -> void:
		_t += dt
		var eye := body as EyeBody
		var drive := clampf((f.energy * 0.7 + f.beat * 0.4) * drive_gain * mod_drive, 0.0, 1.0)
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
		eye.update(dt * time_scale * mod_time, drive)

	func draw_items(stage, lens: Lens3D, u: float) -> Array:
		# The sprout VINE draws even while the eye itself is still unfruited (fade 0):
		# the plant precedes the fruit.
		var pre := []
		var vine: Variant = state.get("vine")
		if typeof(vine) == TYPE_DICTIONARY:
			var vd: Dictionary = vine
			pre.append({"d": lens.depth(pos) + 0.10, "call": func() -> void:
				_draw_vine(stage, lens, u, vd)})
		if fade <= 0.003:
			return pre
		var p := draw_pos()
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
		items.append_array(pre)
		var eye := body as EyeBody
		var at := p
		var r := draw_scale()
		var a := fade
		items.append({"d": lens.depth(p), "call": func() -> void:
			eye.draw(stage, lens, u, at, r, a)})
		# The mitosis MEMBRANE: while the split verb stashes `state.mit`, a stretchy
		# tissue bridge connects this eye to its budding twin - drawn just behind both
		# eyeballs so the spheres cap its ends and only the strained middle shows.
		var mit: Variant = state.get("mit")
		if typeof(mit) == TYPE_DICTIONARY:
			var other = stage.actor(String((mit as Dictionary).get("to", "")))
			if other != null:
				var m: Dictionary = mit
				items.append({"d": lens.depth(p) + 0.08, "call": func() -> void:
					_draw_membrane(stage, lens, u, m, other)})
		# The crystallization CAGE: while the crystallize verb stashes `state.crys`,
		# frost-like crystal edges creep across the ball - drawn just in front of it.
		var crys: Variant = state.get("crys")
		if typeof(crys) == TYPE_DICTIONARY and fade > 0.003:
			var cr: Dictionary = crys
			items.append({"d": lens.depth(p) - 0.05, "call": func() -> void:
				_draw_crystal_cage(stage, lens, u, cr)})
		return items

	# Frost taking the eyeball: seeded chord edges that each GROW outward from their
	# midpoint in a staggered wave (creeping ice), brightening and pulling tight as
	# the faceting sets in, with vertex glints where edges complete and a cold rim.
	func _draw_crystal_cage(stage, lens: Lens3D, u: float, cr: Dictionary) -> void:
		var pj := lens.project(draw_pos())
		if pj.z <= lens.near:
			return
		var c := Vector2(pj.x, pj.y) * u
		var r := draw_scale() * lens._focal / maxf(0.1, pj.z) * u
		if r < 2.0:
			return
		var k := clampf(float(cr.get("k", 0.0)), 0.0, 1.0)
		var facet := clampf(float(cr.get("facet", 0.0)), 0.0, 1.0)
		var seed: int = int(cr.get("seed", 0))
		var hue := clampf(float(cr.get("hue", 0.6)), 0.0, 1.0)
		var ice := Color.from_hsv(hue, 0.30, 1.0)
		var tight := 1.0 - 0.12 * facet                    # the cage pulls in as it sets
		var lw: float = maxf(1.2, r * 0.022)
		var edges := 16
		for j in edges:
			var h1 := float(hash([seed, j, 1]) % 10000) / 10000.0
			var h2 := float(hash([seed, j, 2]) % 10000) / 10000.0
			var h3 := float(hash([seed, j, 3]) % 10000) / 10000.0
			var h4 := float(hash([seed, j, 4]) % 10000) / 10000.0
			var a1 := h1 * TAU
			var a2 := a1 + 0.5 + h2 * 1.1                  # chord span
			var f1 := (0.55 + 0.45 * h3) * tight
			var f2 := (0.55 + 0.45 * h4) * tight
			var pa := c + Vector2(cos(a1), sin(a1)) * (r * f1)
			var pb := c + Vector2(cos(a2), sin(a2)) * (r * f2)
			# Staggered growth: each edge extends from its midpoint outward in its window.
			var start := 0.7 * float(hash([seed, j, 5]) % 1000) / 1000.0
			var g := clampf((k - start) / 0.25, 0.0, 1.0)
			if g <= 0.0:
				continue
			var qa := pa.lerp(pb, 0.5 - 0.5 * g)
			var qb := pa.lerp(pb, 0.5 + 0.5 * g)
			var a := (0.30 + 0.45 * facet) * fade
			stage.draw_line(qa, qb, Color(ice.r, ice.g, ice.b, a), lw, true)
			if g >= 1.0 and facet > 0.05:                  # completed edges glint at the joints
				var ga := (0.35 + 0.4 * facet) * fade
				stage.draw_circle(pa, lw * 1.1, Color(1, 1, 1, ga))
				stage.draw_circle(pb, lw * 1.1, Color(1, 1, 1, ga))
		# A cold rim as the freeze sets in.
		if facet > 0.02:
			var rim := 30
			var pts := PackedVector2Array()
			for i in rim + 1:
				var th := TAU * float(i) / float(rim)
				pts.append(c + Vector2(cos(th), sin(th)) * (r * 1.015))
			stage.draw_polyline(pts, Color(ice.r, ice.g, ice.b, 0.28 * facet * fade), lw * 0.8, true)

	# The stretchy connective tissue of the split: a metaball-style neck between the
	# two projected eyeballs, drooping under its own weight, veined and wet-lit, that
	# thins to slimy strands and - after the snap - leaves retracting nubs and a pair
	# of dangling remnants. All geometry + layered fills; shapes derive from `m.seed`
	# so the same show tears the same way.
	func _draw_membrane(stage, lens: Lens3D, u: float, m: Dictionary, other) -> void:
		var p1 := lens.project(draw_pos())
		var p2 := lens.project(other.draw_pos())
		if p1.z <= lens.near or p2.z <= lens.near:
			return
		var c1 := Vector2(p1.x, p1.y) * u
		var c2 := Vector2(p2.x, p2.y) * u
		var r1 := draw_scale() * lens._focal / maxf(0.1, p1.z) * u
		var r2: float = float(other.draw_scale()) * lens._focal / maxf(0.1, p2.z) * u
		var d := c2 - c1
		var L := d.length()
		if L < 1.0 or r2 < 1.0:
			return
		var dir := d / L
		var n := Vector2(-dir.y, dir.x)
		if n.y > 0.0:
			n = -n                                     # n points screen-up; droop is -n
		var waist := clampf(float(m.get("waist", 1.0)), 0.0, 1.0)
		var snap_t := float(m.get("snap", -1.0))
		var seed: int = int(m.get("seed", 0))
		var alpha := clampf(minf(fade, float(other.fade)), 0.0, 1.0)
		if snap_t >= 0.0:
			_draw_membrane_torn(stage, c1, c2, dir, n, r1, r2, snap_t, seed, alpha)
			return
		# --- the intact bridge ---
		var w1 := r1 * 0.66
		var w2 := r2 * 0.74
		var wm := minf(r1, r2) * 0.78 * pow(waist, 1.35)   # the waist starves as the bud pulls
		var droop := L * 0.085 * (1.0 - waist) * clampf(L / maxf(r1, 1.0) - 1.2, 0.0, 1.0)
		var segs := 24
		var mid := PackedVector2Array()
		var wid := PackedFloat32Array()
		for i in segs + 1:
			var t := float(i) / float(segs)
			var c := c1.lerp(c2, t)
			c -= n * (sin(PI * t) * droop)             # the tissue sags under its own weight
			mid.append(c)
			wid.append((1.0 - t) * (1.0 - t) * w1 + 2.0 * (1.0 - t) * t * wm + t * t * w2)
		# Layered flesh: shadowed outer sheet, brighter core, a wet specular streak.
		stage.draw_colored_polygon(_membrane_poly(mid, wid, n, 1.0), Color(0.50, 0.42, 0.43, 0.92 * alpha))
		stage.draw_colored_polygon(_membrane_poly(mid, wid, n, 0.58), Color(0.72, 0.64, 0.64, 0.9 * alpha))
		var streak := PackedVector2Array()
		for i in segs + 1:
			streak.append(mid[i] + n * (wid[i] * 0.28))
		stage.draw_polyline(streak, Color(1.0, 0.98, 0.97, 0.20 * alpha), maxf(1.2, wm * 0.26), true)
		# Irritated veins, angrier as the waist starves.
		var vein_a := (0.20 + 0.5 * (1.0 - waist)) * alpha
		for v in 2:
			var ph := float(hash([seed, "vein", v]) % 628) * 0.01
			var off := (0.30 + 0.22 * float(v)) * (1.0 if v == 0 else -1.0)
			var vp := PackedVector2Array()
			for i in segs + 1:
				var t := float(i) / float(segs)
				var wob := sin(t * 9.0 + ph) * 0.14
				vp.append(mid[i] + n * (wid[i] * (off + wob)))
			stage.draw_polyline(vp, Color(0.62, 0.12, 0.12, vein_a), maxf(1.4, minf(r1, r2) * 0.030), true)
		# Near the tear: the sheet has already given way to a few slimy strands.
		if waist < 0.42:
			var sag_amp := L * 0.10
			for s in 3:
				var hh := float(hash([seed, "strand", s]) % 1000) / 1000.0
				var y0 := (hh - 0.5) * 1.2
				var a0 := c1 + dir * (r1 * 0.85) + n * (w1 * y0 * 0.5)
				var a1 := c2 - dir * (r2 * 0.85) + n * (w2 * y0 * 0.5)
				var strand := PackedVector2Array()
				for i in 13:
					var t := float(i) / 12.0
					var c := a0.lerp(a1, t)
					c -= n * (sin(PI * t) * sag_amp * (0.6 + hh))
					strand.append(c)
				var sa := clampf((0.42 - waist) / 0.42, 0.0, 1.0) * 0.8 * alpha
				stage.draw_polyline(strand, Color(0.80, 0.69, 0.69, sa), maxf(1.4, minf(r1, r2) * 0.045), true)

	# After the snap: the neck is gone - each eye keeps a small retracting NUB where it
	# tore, and a slack remnant strand swings from each side before wicking away.
	func _draw_membrane_torn(stage, c1: Vector2, c2: Vector2, dir: Vector2, n: Vector2,
			r1: float, r2: float, snap_t: float, seed: int, alpha: float) -> void:
		var g := clampf(1.0 - snap_t / 0.75, 0.0, 1.0)
		if g <= 0.0:
			return
		for side in 2:
			var c := c1 if side == 0 else c2
			var r := r1 if side == 0 else r2
			var toward := dir if side == 0 else -dir
			# The nub: a teardrop bump easing back into the ball.
			var base := c + toward * (r * 0.94)
			var nubs := 4
			for i in nubs:
				var t := float(i) / float(nubs - 1)
				var rr := r * 0.16 * g * (1.0 - t * 0.75)
				if rr < 0.6:
					continue
				var pcen := base + toward * (r * 0.16 * g * t) - n * (rr * 0.35 * t)
				stage.draw_circle(pcen, rr, Color(0.66, 0.56, 0.57, 0.8 * g * alpha))
			stage.draw_circle(base + toward * (r * 0.05), r * 0.10 * g, Color(0.95, 0.9, 0.9, 0.25 * g * alpha))
			# The dangling remnant: a slack filament swinging from the tear point.
			var hh := float(hash([seed, "rem", side]) % 1000) / 1000.0
			var sway := sin(snap_t * (7.0 + 4.0 * hh) + hh * 6.0) * 0.35
			var len := r * (0.55 + 0.3 * hh) * g
			var rem := PackedVector2Array()
			for i in 9:
				var t := float(i) / 8.0
				rem.append(base + toward * (len * t * (0.4 + sway * t)) - n * (len * t * t * (1.1 - absf(sway))))
			stage.draw_polyline(rem, Color(0.75, 0.63, 0.63, 0.55 * g * alpha), maxf(1.0, r * 0.045 * g), true)

	# The sprout's plant-arm: a tapered ribbon along a cubic crook (base -> c1 -> c2 ->
	# hanging tip), revealed to fraction `g` of its length - growth extends it, the
	# post-detach retreat slides it back down. A dark fleshy-green stem with a brighter
	# spine, a wet edge highlight, and a couple of seeded leaf nubs.
	func _draw_vine(stage, lens: Lens3D, u: float, v: Dictionary) -> void:
		# Shed petals first - they keep falling even as the stem retreats away.
		for pt in v.get("petals", []):
			var pp := lens.project(pt.p as Vector3)
			if pp.z <= lens.near:
				continue
			var pc := Vector2(pp.x, pp.y) * u
			var ps: float = float(pt.size) * lens._focal / maxf(0.1, pp.z) * u
			var ca := cos(float(pt.ang))
			var sa := sin(float(pt.ang))
			var oval := PackedVector2Array()
			for i in 10:
				var th := TAU * float(i) / 10.0
				var lx := cos(th) * ps
				var ly := sin(th) * ps * 0.52
				oval.append(pc + Vector2(lx * ca - ly * sa, lx * sa + ly * ca))
			stage.draw_colored_polygon(oval, Color.from_hsv(0.965, 0.38, 0.60, float(pt.a)))
		var g := clampf(float(v.get("g", 0.0)), 0.0, 1.0)
		if g <= 0.015:
			return
		var b: Vector3 = v.base
		var c1: Vector3 = v.c1
		var c2: Vector3 = v.c2
		var tip: Vector3 = v.tip
		var seed: int = int(v.get("seed", 0))
		var hue := float(v.get("hue", 0.3))
		var r := float(v.get("r", 0.3))
		var segs := 24
		var mid := PackedVector2Array()
		var wid := PackedFloat32Array()
		for i in segs + 1:
			var tc := g * float(i) / float(segs)           # position along the FULL curve
			var p3 := _bez(b, c1, c2, tip, tc)
			var pj := lens.project(p3)
			if pj.z <= lens.near:
				return
			mid.append(Vector2(pj.x, pj.y) * u)
			wid.append(r * lerpf(0.30, 0.11, tc) * lens._focal / maxf(0.1, pj.z) * u)
		var nrm := PackedVector2Array()
		for i in segs + 1:
			var tan := mid[mini(i + 1, segs)] - mid[maxi(i - 1, 0)]
			if tan.length() < 0.001:
				tan = Vector2(0, -1)
			nrm.append(Vector2(-tan.y, tan.x).normalized())
		# The stem body: thick line segments with joint-fill discs - NO polygons, so
		# the tight crook (where ribbon quads twist into bowties) can never fail to
		# triangulate. The width tapers stepwise along the 24 segments.
		var flesh := Color.from_hsv(hue, 0.50, 0.27)
		for i in segs:
			if mid[i].distance_to(mid[i + 1]) < 0.5:
				continue                       # a just-sprouted stem is pixels long - skip
			stage.draw_line(mid[i], mid[i + 1], flesh, wid[i] + wid[i + 1], true)
			stage.draw_circle(mid[i + 1], (wid[i] + wid[i + 1]) * 0.5, flesh)
		stage.draw_circle(mid[0], wid[0], flesh)
		stage.draw_polyline(mid, Color.from_hsv(hue, 0.40, 0.46),
			maxf(1.2, wid[segs / 2] * 0.85), true)
		var edge := PackedVector2Array()
		for i in segs + 1:
			edge.append(mid[i] + nrm[i] * (wid[i] * 0.55))
		stage.draw_polyline(edge, Color(0.72, 0.88, 0.70, 0.28), maxf(1.0, wid[0] * 0.22), true)
		# Leaf nubs at seeded positions along the revealed stem.
		for lf in 2:
			var ht := 0.25 + 0.35 * float(hash([seed, "leaf", lf]) % 1000) / 1000.0
			if g < ht + 0.05:
				continue
			var i := clampi(int(ht / g * float(segs)), 1, segs - 1)
			if wid[i] < 1.2:
				continue                       # too small to shape - a degenerate triangle
			var side := 1.0 if lf == 0 else -1.0
			var along := (mid[i + 1] - mid[i - 1]).normalized()
			var apex := mid[i] + nrm[i] * (side * wid[i] * 3.2) + along * (wid[i] * 1.4)
			stage.draw_colored_polygon(PackedVector2Array([
				mid[i] - along * wid[i], mid[i] + along * wid[i], apex]),
				Color.from_hsv(hue + 0.02, 0.48, 0.38))

	func _bez(b: Vector3, c1: Vector3, c2: Vector3, tip: Vector3, t: float) -> Vector3:
		var s := 1.0 - t
		return b * (s * s * s) + c1 * (3.0 * s * s * t) + c2 * (3.0 * s * t * t) + tip * (t * t * t)

	# The closed outline of the bridge sheet at a width fraction (top edge out, bottom back).
	func _membrane_poly(mid: PackedVector2Array, wid: PackedFloat32Array, n: Vector2, frac: float) -> PackedVector2Array:
		var pts := PackedVector2Array()
		for i in mid.size():
			pts.append(mid[i] + n * (wid[i] * frac))
		for i in range(mid.size() - 1, -1, -1):
			pts.append(mid[i] - n * (wid[i] * frac))
		return pts

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
		var drive := clampf((f.energy * 0.85 + f.beat * 0.6) * drive_gain * mod_drive, 0.0, 1.0)
		prism.update(dt * time_scale * mod_time, drive)
		# hold_still: damp the body's own spin toward rest (it keeps breathing).
		var still := float(state.get("still", 0.0))
		var want := 1.0 - clampf(still, 0.0, 1.0)
		_spin = lerpf(_spin, want, 1.0 - exp(-4.0 * dt))
		if _spin < 0.999:
			prism._vel *= (0.02 + 0.98 * _spin)

	func draw_items(stage, lens: Lens3D, u: float) -> Array:
		if fade <= 0.003:
			return []
		var pj := lens.project(draw_pos())
		if pj.z <= lens.near:
			return []
		var prism := body as PrismBody
		var center := Vector2(pj.x, pj.y) * u
		var px := draw_scale() * lens._focal / maxf(0.1, pj.z) * u * 1.15
		var h := fposmod(hue + mod_hue, 1.0)
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

	## Continue a carried prism as the RED strand's lead: its live body becomes red
	## member 0, dormant until the helix opens - the ouroboros closing its loop (the
	## red that drove off down the opposite highway comes back as the counter-strand).
	func adopt_lead_red(old: Actor) -> void:
		if old.body is PrismBody and not _red.is_empty():
			_red[0] = old.body

	func update(f: AudioFeatures, dt: float, _stage) -> void:
		_t += dt
		var drive := clampf((f.energy * 0.8 + f.beat * 0.6) * drive_gain * mod_drive, 0.0, 1.0)
		for b in _blue:
			(b as PrismBody).update(dt * time_scale * mod_time, drive)
		if float(state.get("split_k", 0.0)) > 0.01:       # reds live once the helix opens
			for r in _red:
				(r as PrismBody).update(dt * time_scale * mod_time, drive)

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
			var px := _size * lens._focal / pj.z * u * mod_scale
			var h := fposmod(hue + mod_hue, 1.0)
			var a := alpha
			items.append({"d": pj.z, "call": func() -> void:
				b.draw(stage, c, px, h, a)})
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
				var px := _size * lens._focal / pj.z * u * mod_scale
				var h := fposmod(_hue_red + mod_hue, 1.0)
				var a := clampf(1.0 - (pj.z - 1.0) / 12.0, 0.12, 1.0) * fade * split_k
				items.append({"d": pj.z, "call": func() -> void:
					b.draw(stage, c, px, h, a)})
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
