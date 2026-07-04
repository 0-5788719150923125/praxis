extends Scene3D

## Stage - the data-driven scene: its entire content comes from the storyboard entry.
##
## Where every other scene is code that rolls a look from a seed, a stage is a
## RENDERER for a description: the entry's `cast:` names actors from the [Cast]
## registry (with sampled-range params), its `track:` schedules [Actions] verbs over
## them (see [Track]), and `camera:` places the [Lens3D]. Nothing here knows about
## eyes or prisms specifically - new performers and verbs extend the registries, and
## scene behavior is authored in the data.
##
##   - scene: stage
##     hold: 4
##     camera: {eye: [0, 0, 4.0], look: [0, 0, 0], fov: 48}
##     cast:
##       - {id: left, kind: eye, at: [0, 0, 0], radius: [0.30, 0.40]}
##     track:
##       nominal: 4
##       spans:
##         - {at: 0.5, action: blink, target: left}
##
## Continuity: a stage declares `morph_out = morph_in = "stage"`, so when two stage
## entries run back to back the Director plays a morph and [method begin_morph]
## adopts the previous entry's LIVE actors by matching id - a body continues across
## the cut (new slots come from the new entry; identity, pose and verb latches ride
## along). Set `carry: false` on an entry to open with a clean cut instead.

var _actors := {}                  # id -> Cast.Actor (insertion-ordered)
var _track: Track = null
var _nominal := 8.0
var _t := 0.0
var _flashes: Array = []           # transient overlays: {actor, k, col}
var _ties: Array = []              # snap ties: {a, b, k}
# The elastic timeline clock (entry `elastic`, 0 = off): the keyframe clock breathes
# with the music - an energetic passage runs it up to (1 + elastic)x, a quiet one
# down to (1 - elastic)x. The signal is ENDOGENOUS and zero-mean by construction
# (fast energy EMA vs slow baseline EMA), so the timeline stretches and contracts
# WITH the song but stays anchored to the authored length overall - and with no
# audio the two EMAs agree, the rate is exactly 1, and timing is unchanged.
var _elastic := 0.0
var _e_fast := 0.0
var _e_slow := 0.0


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "plane"
	morph_out = "stage"
	morph_in = "" if not bool(spec.get("carry", true)) else "stage"

	_elastic = clampf(float(spec.get("elastic", 0.0)), 0.0, 0.9)
	var cam: Dictionary = spec.get("camera") if typeof(spec.get("camera")) == TYPE_DICTIONARY else {}
	lens.eye = Storyboard.sample_vec3(cam.get("eye"), rng, Vector3(0, 0, 4.0))
	lens.look = Storyboard.sample_vec3(cam.get("look"), rng, Vector3.ZERO)
	lens.fov = float(Storyboard.sample(cam.get("fov", 48.0), rng))

	for item in spec.get("cast", []):
		if typeof(item) != TYPE_DICTIONARY:
			continue
		var a := Cast.make(String(item.get("kind", "")), rng, item)
		if a != null:
			_actors[a.id] = a
	if typeof(spec.get("track")) == TYPE_DICTIONARY:
		_track = Track.new(spec["track"], rng)
		_nominal = _track.nominal
	return {}


## The live actors, for the next stage entry to adopt.
func morph_payload() -> Dictionary:
	return {"actors": _actors}


## Adopt the previous stage's live actors by matching id: the LIVE actor continues
## wholesale - body, identity, verb latches, group members (a swarm's formation and
## travel) - only its SLOT and spec come from this entry's own cast, so a board can
## re-stage a carried actor without resetting what it has become.
func begin_morph(from: GhostScene) -> void:
	var p := from.morph_payload()
	if typeof(p.get("actors")) != TYPE_DICTIONARY:
		return
	var carried: Dictionary = p["actors"]
	for id in _actors.keys():
		if not carried.has(id):
			continue
		var fresh: Cast.Actor = _actors[id]
		var old: Cast.Actor = carried[id]
		if old.kind != fresh.kind:
			continue
		old.pos = fresh.pos
		old.home = fresh.home
		old.spec = fresh.spec
		_actors[id] = old
	# A swarm may continue a carried prism as its lead (cfg `lead: <actor id>`): the
	# live body becomes member 0 and its LAST SCREEN SLOT (projected through the old
	# stage's lens - the cameras differ) becomes the fly-in origin.
	for a in _actors.values():
		if a is Cast.SwarmActor and carried.has(String(a.spec.get("lead", ""))):
			var old: Cast.Actor = carried[String(a.spec["lead"])]
			var frac := Vector2(0.15, 0.0)
			if from.has_method("project_actor"):
				var pr: Dictionary = from.project_actor(old)
				if not pr.is_empty():
					frac = (pr.center as Vector2) / maxf(1.0, from.unit())
			(a as Cast.SwarmActor).adopt_lead(old, frac)
	_t = 0.0


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	drift_view(f, 0.004, 0.008)                 # nearly static; behavior gates it anyway
	# The keyframe clock only runs on screen: the Director pre-warms scenes with a few
	# update() calls before they enter the tree, and those must not consume early spans
	# (a blink half a second in would fire during the warm-up). Ambient body life below
	# still advances, so the first shown frame is settled.
	if is_inside_tree():
		_t += delta * _clock_rate(f, delta)
		if _track != null:
			_track.advance(_t, phase_span(_nominal), self, f, delta)
	# Live-dial overlay: sample the performance bus per actor (index-diverse, so the
	# cast modulates as a group without lockstep) and stamp fresh neutral-off values
	# every frame - the dial perturbs the reading of the choreography, never the
	# choreography itself.
	var ai := 0
	for a in _actors.values():
		a.mod_scale = 1.0 + 0.22 * Director.dial_value("scale", ai)
		a.mod_hue = 0.10 * Director.dial_value("hue", ai)
		a.mod_time = clampf(1.0 + 0.5 * Director.dial_value("tempo", ai), 0.35, 2.0)
		a.mod_drive = clampf(1.0 + 0.65 * Director.dial_value("drive", ai), 0.0, 2.0)
		a.mod_off = Vector2(Director.dial_value("off_x", ai), Director.dial_value("off_y", ai)) * 0.05
		ai += 1
		a.update(f, delta, self)
	# Pose locks resolve after every body has advanced, so a locked pair is exact.
	for a in _actors.values():
		var to_id := String(a.state.get("lock_to", ""))
		if to_id != "" and _actors.has(to_id):
			var to: Cast.Actor = _actors[to_id]
			if a.body is PrismBody and to.body is PrismBody:
				(a.body as PrismBody).lock_pose_to(to.body as PrismBody)
	for fl in _flashes:
		fl.k -= delta * 2.4
	_flashes = _flashes.filter(func(x): return x.k > 0.001)
	for tf in _ties:
		tf.k -= delta * 2.6
	_ties = _ties.filter(func(x): return x.k > 0.001)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	lens.prepare()
	var u := unit()
	var items := []
	for a in _actors.values():
		items.append_array(a.draw_items(self, lens, u))
	items.sort_custom(func(x, y): return x.d > y.d)      # far first
	for it in items:
		(it.call as Callable).call()
	# Transient overlays ride on top: form/burst flashes and the phase-lock snap tie.
	for fl in _flashes:
		var pr := project_actor(fl.actor)
		if pr.is_empty():
			continue
		var col: Color = fl.col
		draw_circle(pr.center, pr.px * (0.6 + 1.4 * fl.k), Color(col.r, col.g, col.b, 0.5 * fl.k))
		draw_circle(pr.center, pr.px * (0.3 + 0.65 * fl.k), Color(1, 1, 1, 0.6 * fl.k))
	for tf in _ties:
		var pa := project_actor(tf.a)
		var pb := project_actor(tf.b)
		if pa.is_empty() or pb.is_empty():
			continue
		draw_line(pa.center, pb.center, Color(1, 1, 1, 0.5 * tf.k), 2.0, true)
		draw_circle((pa.center + pb.center) * 0.5, u * 0.02 * tf.k, Color(1, 1, 1, 0.7 * tf.k))


# The breathing rate of the keyframe clock (see _elastic above). Ambient body life
# always runs on the raw delta - only WHEN events land stretches, never how things move.
func _clock_rate(f: AudioFeatures, delta: float) -> float:
	if _elastic <= 0.0:
		return 1.0
	_e_fast = lerpf(_e_fast, f.energy, 1.0 - exp(-delta / 0.8))
	_e_slow = lerpf(_e_slow, f.energy, 1.0 - exp(-delta / 8.0))
	return clampf(1.0 + _elastic * tanh((_e_fast - _e_slow) * 4.0), 1.0 - _elastic, 1.0 + _elastic)


## Look up an actor by id (verbs use this to reach their counterparts). Null if absent.
func actor(id: String) -> Cast.Actor:
	return _actors.get(id)


## All actors, in cast order (a span with target "all").
func actors() -> Array:
	return _actors.values()


## A one-shot bright burst overlay on an actor (crystallize's form-flash, the drop).
func flash(a: Cast.Actor, col: Color) -> void:
	_flashes.append({"actor": a, "k": 1.0, "col": col})


## A brief bright tie between two actors (the phase-lock snap).
func tie(a: Cast.Actor, b: Cast.Actor) -> void:
	_ties.append({"a": a, "b": b, "k": 1.0})


## Project an actor's slot to screen: {center: px Vector2, px: perspective size, z}.
## Empty if behind the camera. (The same slot->screen bridge the bodies draw with.)
func project_actor(a: Cast.Actor) -> Dictionary:
	var pj := lens.project(a.pos)
	if pj.z <= lens.near:
		return {}
	return {"center": Vector2(pj.x, pj.y) * unit(),
		"px": a.draw_scale() * lens._focal / maxf(0.1, pj.z) * unit() * 1.15, "z": pj.z}
