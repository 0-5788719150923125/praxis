extends Scene3D

## Prism swarm - the swarm forms, flies the track, splits into a helix, and jumps (the-point,
## scenes 12-15).
##
##   12 gather - more BLUE prisms fly in one at a time beside the first, a sparse swarm of 6-7,
##               forming up into a tight unison formation streaming along one track.
##   13 fly    - the formation flies FORWARD in unison along an invisible track into the void.
##   14 split  - the single track splits into TWO entangled tracks twisting like a double helix,
##               one strand all BLUE, the other all RED, weaving in OPPOSING directions.
##   15 jump   - the swarm commits to ONE side (banks left or right by seed), then JUMPS across
##               to the other track - a quick lane-change leap - and holds, leaving room for a card.
##
## The only scene where the camera travels (the brief allows motion from the swarm on): it flies
## forward a touch slower than the swarm, so the formation pulls ahead and recedes into the void.
## Arrives by morph from `two_prisms` (`morph_in = "prisms"`): the live blue prism leads the swarm.

var _f: AudioFeatures = AudioFeatures.new()
var _rng := RandomNumberGenerator.new()
var _blue: Array = []          # blue swarm PrismBodies (index 0 = the lead, handed over)
var _red: Array = []           # red strand PrismBodies (appear at the split)
var _entry: Array = []         # per blue member: fly-in origin (2D unit-fraction)
var _t := 0.0
var _frac := 0.0               # fraction of the hold elapsed (drives the keyframe phases)
var _travel := 0.0             # distance the swarm has flown along the track
var _lane := 0.0               # blue swarm's strand: 0 = its own, 1 = jumped to the other
var _bank := 1.0               # which way it banks before the jump (+1 / -1 by seed)
var _lead_entry := Vector2(0.15, 0.0)   # where the lead was handed over (for a smooth entry)

const N_BLUE := 7
const N_RED := 7
const SPACING := 0.45          # along-track gap between consecutive members
const D_HEAD := 3.6            # track distance of the lead at t=0 (all members start in front)
const HELIX_W := 1.15          # twist (radians) per unit of track depth
const R_MIN := 0.30            # the single track's gentle spread before the split
const R_MAX := 0.95            # helix radius when fully split
const SIZE := 0.42             # base prism size (world; perspective scales it on screen)
const STAGGER_F := 0.037       # gather join interval, as a fraction of the hold
const HUE_BLUE := 0.6
const HUE_RED := 0.0
# Phase thresholds as FRACTIONS of the hold, so gather / fly / split / jump all land whatever the
# scene length; NOMINAL paces it in auto mode (no fixed hold).
const FL_FLY := 0.26           # formation is up; begin flying forward
const FL_SPLIT := 0.46         # the track opens into a double helix; red strand fades in
const FL_JUMP := 0.74          # the lane-change leap
const NOMINAL := 5.0


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "scene3d"
	framing = "plane"
	morph_in = "prisms"
	_rng.seed = rng.randi()
	_bank = 1.0 if rng.randf() < 0.5 else -1.0
	for i in N_BLUE:
		_blue.append(PrismBody.new(rng.randi()))
		# The lead flies in from where it was handed over; the rest sweep in from the frame edges.
		if i == 0:
			_entry.append(_lead_entry)
		else:
			var side := 1.0 if (i % 2 == 0) else -1.0
			_entry.append(Vector2(side * rng.randf_range(0.7, 1.1), rng.randf_range(-0.5, 0.5)))
	for i in N_RED:
		_red.append(PrismBody.new(rng.randi()))
	lens.eye = Vector3(0, 0, -2.6)
	lens.look = Vector3(0, 0, 5.0)
	lens.fov = 56.0
	_t = 0.0
	return {}


# Arrived from two_prisms: the live blue prism leads the swarm, easing in from its last position.
func begin_morph(from: GhostScene) -> void:
	var p := from.morph_payload()
	if p.is_empty():
		return
	if p.has("blue") and not _blue.is_empty():
		_blue[0] = p["blue"]
	_lead_entry = p.get("blue_pos", _lead_entry)
	if not _entry.is_empty():
		_entry[0] = _lead_entry
	_t = 0.0
	_travel = 0.0
	_lane = 0.0


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	last_f = f
	tick(f, delta)
	_t += delta
	# Keyframes (gather/fly/split/jump) run on the fraction of the hold elapsed, so they land whatever
	# the scene length; the prisms fly and spin on the raw delta, so the tempo never speeds the motion.
	_frac = clampf(_t / phase_span(NOMINAL), 0.0, 1.0)
	var drive := clampf(f.energy * 0.8 + f.beat * 0.6, 0.0, 1.0)

	# Fly forward: the swarm accelerates once formed; the camera follows a touch slower so the
	# formation pulls ahead and recedes into the void (room opens up for the end card).
	var rate := smoothstep(FL_FLY, FL_FLY + 0.24, _frac) * (1.2 + 0.8 * drive)
	_travel += delta * rate
	lens.eye.z = -2.6 + _travel * 0.72
	lens.look.z = lens.eye.z + 6.0

	# The track opens into a double helix; the lane-change leap crosses the swarm to the far strand.
	var lane_t := smoothstep(FL_JUMP, FL_JUMP + 0.14, _frac)
	_lane = lerpf(_lane, lane_t, 1.0 - exp(-9.0 * delta))

	for b in _blue:
		b.update(delta, drive)
	for r in _red:
		r.update(delta, drive)
	queue_redraw()


# The track's radius: a gentle spread before the split (one coherent stream), opening out to the
# full double-helix radius as it splits.
func _radius() -> float:
	return lerpf(R_MIN, R_MAX, smoothstep(FL_SPLIT, FL_SPLIT + 0.2, _frac))


# A point on one strand of the double helix at track distance d. strand_dir sets the angular
# offset AND the twist direction, so the two strands wind in opposing directions and weave.
func _strand(d: float, strand_dir: float, r: float) -> Vector3:
	var ang := d * HELIX_W * strand_dir + PI * (0.5 - 0.5 * strand_dir)
	return Vector3(cos(ang) * r, sin(ang) * r * 0.72, d)


func _draw() -> void:
	begin_draw()
	lens.prepare()
	var u := unit()
	var r := _radius()
	var items := []            # {d, center, size, hue, alpha} - depth-sorted before drawing

	# Blue swarm: a stream of prisms nose-to-tail along the track. Before the split they run a
	# straight line into the void; after it they follow the blue strand, weaving. On the jump the
	# whole stream banks to one side and leaps to the red strand (its angular offset crosses to it).
	var blue_dir := 1.0
	for i in _blue.size():
		var d: float = D_HEAD - i * SPACING + _travel
		if d <= 0.05:
			continue
		# Strand angle, with the lane-change: the blue stream banks (a lateral lean) then crosses
		# from its strand to the other one over the jump.
		var base := _strand(d, blue_dir, r)
		var other := _strand(d, -blue_dir, r)
		var arc := sin(_lane * PI) * _bank * r * 0.5     # the bank/overshoot of the leap
		var world: Vector3 = base.lerp(other, _lane) + Vector3(arc, 0.0, 0.0)
		# Gather: member i eases in from its entry point, joining one at a time.
		var on := clampf((_frac - i * STAGGER_F) / 0.12, 0.0, 1.0)
		var it := _project_item(world, u)
		if it.is_empty():
			continue
		if on < 1.0:
			var ec: Vector2 = _entry[i] * u
			it.center = ec.lerp(it.center, smoothstep(0.0, 1.0, on))
			it.alpha *= on
		it.hue = HUE_BLUE
		it.body = _blue[i]
		items.append(it)

	# Red strand: fades in with the split, running the OTHER way (opposing direction) so the two
	# strands weave past each other. All red.
	var red_fade := smoothstep(FL_SPLIT, FL_SPLIT + 0.18, _frac)
	if red_fade > 0.01:
		for i in _red.size():
			var d: float = D_HEAD - i * SPACING + _travel
			if d <= 0.05:
				continue
			var world := _strand(d, -1.0, r)   # the other strand: counter-winding, weaving past blue
			var it := _project_item(world, u)
			if it.is_empty():
				continue
			it.alpha *= red_fade
			it.hue = HUE_RED
			it.body = _red[i]
			items.append(it)

	items.sort_custom(func(a, c): return a.d > c.d)    # far first
	for it in items:
		it.body.draw(self, it.center, it.size, it.hue, clampf(it.alpha, 0.0, 1.0))


# Project a world point on the track to a draw item (screen centre, perspective size, depth fade),
# or {} if it is behind the camera.
func _project_item(world: Vector3, u: float) -> Dictionary:
	var pj := lens.project(world)
	if pj.z <= lens.near:
		return {}
	var center := Vector2(pj.x, pj.y) * u
	var size := SIZE * lens._focal / pj.z * u
	# Fade with distance: near prisms are solid, far ones dissolve into the void.
	var alpha := clampf(1.0 - (pj.z - 1.0) / 12.0, 0.12, 1.0)
	return {"d": pj.z, "center": center, "size": size, "hue": 0.6, "alpha": alpha, "body": null}
