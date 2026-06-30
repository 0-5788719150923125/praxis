extends Scene3D

## Shatter glass - a real pane of glass, shattering in true 3D.
##
## A flat pane stands in space and is seen at a three-quarter angle through the
## [Lens3D] camera (not pinned flat to the screen). On a beat it fractures into
## irregular angular shards - [Geo.fracture] cracks that radiate from an impact, not
## pizza slices - and the shards burst off the plane and **tumble through space**,
## each spiralling on its own 3D axis, before easing back together (loop) or
## drifting to rest (oneshot). Real depth: a shard in front occludes one behind, and
## the glass catches the light by how each shard faces the camera. The old version
## was flat shards sliding in the 2D plane; this one is dimensional.

var _f: AudioFeatures = AudioFeatures.new()
var _rng := RandomNumberGenerator.new()
var _shards: Array = []
var _hue := 0.0
var _oneshot := false
var _fired := false
var _moved := false
var _beat_prev := 0.0
var _angle := 0.5
var _stress := 0.0           # accumulated harmonic stress; releases as a fracture at threshold
var _stress_thresh := 4.0    # sampled release point (re-rolled per fracture, so cracks are uneven)
var _max_shards := 60        # stop cracking once the pane is this finely divided
var _cracks: Array = []      # transient bright fracture flashes ({a, b world endpoints, age})

const PANE := 1.25       # pane half-size, world units


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "plane"
	_rng.seed = rng.randi()
	_hue = rng.randf()
	_oneshot = rng.randf() < 0.4
	lifecycle = "oneshot" if _oneshot else "loop"
	lens.fov = rng.randf_range(40.0, 52.0)
	_angle = rng.randf_range(0.35, 0.7)

	# The intact pane (a disc or a rectangle), fractured into shards.
	var base := PackedVector2Array()
	if rng.randf() < 0.5:
		var n := rng.randi_range(12, 18)
		for i in n:
			var a := TAU * float(i) / float(n)
			base.append(Vector2(cos(a), sin(a)) * PANE)
	else:
		var hw := PANE * rng.randf_range(0.85, 1.1)
		var hh := PANE * rng.randf_range(0.6, 1.0)
		base = PackedVector2Array([Vector2(-hw, -hh), Vector2(hw, -hh), Vector2(hw, hh), Vector2(-hw, hh)])
	var impact := Vector2(rng.randf_range(-1, 1), rng.randf_range(-1, 1)) * PANE * 0.4
	var shards := Geo.fracture(base, rng.randi_range(18, 34), impact, PANE * 0.12, rng)
	for poly: PackedVector2Array in shards:
		var cen := Geo.centroid(poly)
		var local := PackedVector2Array()
		for v in poly:
			local.append(v - cen)
		# Each shard resonates on its own spectral CHANNEL: centre shards on the low end
		# (bass), rim shards on the high end (treble), with a little jitter so it is not a
		# clean ring. Its channel level drives transparency in _draw, so quiet shards turn
		# glassy and reveal the ones behind them.
		var chan := clampf(cen.length() / (PANE * 1.05) + rng.randf_range(-0.1, 0.1), 0.0, 1.0)
		_shards.append({
			"poly": local,
			"home": Vector3(cen.x, cen.y, 0.0),
			"pos": Vector3(cen.x, cen.y, 0.0),
			"basis": Basis.IDENTITY,
			"vel": Vector3.ZERO,
			"spin": Vector3.ZERO,
			"hue": fposmod(_hue + 0.22 * cen.length() / PANE + 0.12 * chan, 1.0),
			"noise": Vector3(rng.randf_range(-1, 1), rng.randf_range(-1, 1), rng.randf_range(-1, 1)),
			"chan": chan,
			"resonance": rng.randf_range(0.6, 1.0),   # how strongly this shard answers its band
			"lev": 0.0,                               # smoothed channel level (its resonance)
		})
	# Cracks are driven by the music, not a clock: harmonic stress accumulates and releases as
	# a connected fracture at a sampled threshold (see update / _fracture_run).
	_stress_thresh = rng.randf_range(2.6, 5.5)
	_max_shards = mini(72, _shards.size() * 3)
	return {}


func finished() -> bool:
	return _oneshot and _moved and _settled()


func _settled() -> bool:
	for s in _shards:
		if s.vel.length() > 0.04:
			return false
	return true


# Burst every shard off the plane: outward in-plane + toward the viewer + a random
# kick, with a hard random spin so they tumble and spiral rather than slide flat.
func _burst(strength: float) -> void:
	for s in _shards:
		var radial: Vector3 = s.home
		var dir: Vector3 = radial.normalized() if radial.length() > 0.01 else Vector3(0, 0, 1)
		s.vel += (dir + s.noise * 0.7 + Vector3(0, 0, _rng.randf_range(0.4, 1.1))) * strength
		s.spin += s.noise * _rng.randf_range(2.5, 6.0)


# A connected fracture RUN: a crack starts in a stressed shard and propagates from region to
# region along a continuing, meandering line - splitting each shard it crosses - so a tiny
# fracture turns a stretch of the pane into shards, the way real glass cracks. The line jags
# non-linearly (occasional sharp kinks), so the resulting pattern is irregular and alive.
func _fracture_run() -> void:
	var start := _pick_stressed()
	if start < 0:
		return
	var s0: Dictionary = _shards[start]
	var p := Vector2(s0.home.x, s0.home.y)             # pane-space crack head (the seed of the fracture)
	var dir := Vector2.from_angle(_rng.randf() * TAU)
	# Usually a short fracture, sometimes a long run that races across the pane (non-linear).
	var steps := 2 + int(pow(_rng.randf(), 1.4) * 4.0)
	for _i in steps:
		if _shards.size() >= _max_shards:
			break
		var idx := _shard_at(p)
		if idx < 0:
			break                                      # ran off the pane (into a gap / the rim)
		_split_shard_at(idx, p, dir)
		p += dir * _rng.randf_range(0.10, 0.26)        # advance into the next region
		var turn := _rng.randf_range(-0.35, 0.35)
		if _rng.randf() < 0.30:
			turn += _rng.randf_range(-1.0, 1.0)        # a kink: the crack jags and the pattern branches
		dir = dir.rotated(turn)


# Pick a shard to start a fracture in: large, and weighted toward the ones currently
# resonating (the most stressed glass). Returns its index, or -1 if none qualifies.
func _pick_stressed() -> int:
	var best := -1
	var best_score := 0.0
	for i in _shards.size():
		var s: Dictionary = _shards[i]
		var ar: float = Geo.area(s.poly)
		if ar < 0.025:                                  # don't seed in a sliver
			continue
		var score: float = ar * (0.4 + float(s.lev)) * _rng.randf()
		if score > best_score:
			best_score = score
			best = i
	return best


# The shard whose resting (pane-space) outline contains point `p`, or -1. The fracture run
# walks the pane through these to find the next region to split.
func _shard_at(p: Vector2) -> int:
	for i in _shards.size():
		var s: Dictionary = _shards[i]
		var poly := PackedVector2Array()
		var hx: float = s.home.x
		var hy: float = s.home.y
		for v: Vector2 in s.poly:
			poly.append(Vector2(hx + v.x, hy + v.y))
		if Geometry2D.is_point_in_polygon(p, poly):
			return i
	return -1


# Split shard `idx` along the line through pane-space point `pane_pt` running in direction
# `dir`. Children inherit the parent's motion, get a gentle opening kick, and leave a
# PERMANENT thin gap so the crack never heals; the fresh break also flashes briefly.
func _split_shard_at(idx: int, pane_pt: Vector2, dir: Vector2) -> bool:
	var s: Dictionary = _shards[idx]
	var local_pt: Vector2 = pane_pt - Vector2(s.home.x, s.home.y)   # crack head in the shard's local frame
	var nrm := Vector2(-dir.y, dir.x)                  # normal to the fracture line
	if nrm.length() < 0.5:
		return false
	nrm = nrm.normalized()
	var pieces := Geo.split(s.poly, local_pt, nrm)
	var a: PackedVector2Array = pieces[0]
	var b: PackedVector2Array = pieces[1]
	if a.size() < 3 or b.size() < 3 or Geo.area(a) < 0.008 or Geo.area(b) < 0.008:
		return false
	_shards.remove_at(idx)
	var wnrm: Vector3 = (s.basis as Basis) * Vector3(nrm.x, nrm.y, 0.0)
	var sep := _rng.randf_range(0.015, 0.05)           # gentle: a crack opening, not a burst
	var hgap: float = _rng.randf_range(0.008, 0.020)
	var hsep := Vector3(nrm.x, nrm.y, 0.0) * hgap
	_shards.append(_child_from(s, a, wnrm * sep, hsep))
	_shards.append(_child_from(s, b, -wnrm * sep, -hsep))
	# Flash this segment of the running crack, centred at the crack head and along the line.
	var wpt: Vector3 = (s.basis as Basis) * Vector3(local_pt.x, local_pt.y, 0.0) + Vector3(s.pos)
	var lined: Vector3 = (s.basis as Basis) * Vector3(dir.x, dir.y, 0.0)
	var halflen := sqrt(maxf(Geo.area(s.poly), 0.001)) * 0.7
	_cracks.append({"a": wpt + lined * halflen, "b": wpt - lined * halflen, "age": 0.0})
	return true


# Build a child shard from one piece of a split: re-centre the piece on its own centroid and
# place it so it sits EXACTLY where it currently is on the parent (no jump), with a matching
# resting home in the pane, then inherit motion plus the small separating kick.
func _child_from(parent: Dictionary, piece: PackedVector2Array, kick: Vector3, home_off := Vector3.ZERO) -> Dictionary:
	var c := Geo.centroid(piece)                       # piece centroid, in the parent's local frame
	var local := PackedVector2Array()
	for v in piece:
		local.append(v - c)
	var basis: Basis = parent.basis
	var pos: Vector3 = basis * Vector3(c.x, c.y, 0.0) + Vector3(parent.pos)
	var home: Vector3 = Vector3(parent.home) + Vector3(c.x, c.y, 0.0) + home_off   # permanent crack gap
	return {
		"poly": local, "home": home, "pos": pos, "basis": basis,
		"vel": Vector3(parent.vel) + kick, "spin": Vector3(parent.spin),
		"hue": fposmod(float(parent.hue) + _rng.randf_range(-0.02, 0.02), 1.0),
		"noise": Vector3(_rng.randf_range(-1, 1), _rng.randf_range(-1, 1), _rng.randf_range(-1, 1)),
		"chan": clampf(float(parent.chan) + _rng.randf_range(-0.05, 0.05), 0.0, 1.0),
		"resonance": float(parent.resonance), "lev": float(parent.lev),
	}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	# A three-quarter view that drifts slowly, so the pane reads as a plane in space.
	var ang := _angle + 0.18 * sin(_life * 0.12)
	lens.eye = Vector3(sin(ang) * 1.7, 0.7 + 0.2 * sin(_life * 0.1), cos(ang) * 3.6)
	lens.look = Vector3.ZERO

	var beat_edge: bool = f.beat > 0.55 and _beat_prev <= 0.55
	_beat_prev = f.beat
	if _oneshot:
		if not _fired and (beat_edge or _life > 1.2):
			_burst(1.0)
			_fired = true
	elif beat_edge:
		_burst(0.5)

	# Cracks answer the HARMONICS, not a clock. Spectral change (flux), energy, and beats feed
	# a "stress" that builds through a non-linear curve; when it crosses a sampled threshold it
	# releases as a connected fracture and re-rolls the threshold. So cracks are sparse, uneven,
	# and tied to what the music is doing - quiet stretches barely crack, busy ones split more.
	var harmonic: float = clampf(0.45 * f.energy + 3.0 * f.flux + 0.4 * f.beat, 0.0, 1.4)
	_stress += delta * (0.05 + pow(harmonic, 1.7))     # 0.05 = a faint baseline so silence still creeps
	if _stress >= _stress_thresh and _shards.size() < _max_shards:
		_stress = 0.0
		_stress_thresh = _rng.randf_range(2.6, 5.5)
		_fracture_run()

	# Age out the fracture flashes.
	for cr in _cracks:
		cr.age += delta
	_cracks = _cracks.filter(func(cr): return float(cr.age) < 0.6)

	for s in _shards:
		# Follow this shard's spectral channel with a resonant ease (quick to swell, slower
		# to settle), so it lights and fades with its own band rather than the whole pane.
		var target: float = f.sample(float(s.chan)) * float(s.resonance)
		var rate: float = 9.0 if target > s.lev else 3.5
		s.lev = lerpf(float(s.lev), target, 1.0 - exp(-rate * delta))
		s.pos += s.vel * delta
		# Re-orthonormalize: repeated incremental rotation accumulates float drift,
		# which would otherwise make slerp/quaternion conversion below fail.
		s.basis = (s.basis * Basis.from_euler(s.spin * delta)).orthonormalized()
		s.vel *= maxf(0.0, 1.0 - 1.1 * delta)        # drag
		s.spin *= maxf(0.0, 1.0 - 0.4 * delta)
		if not _oneshot:                              # pull back home and re-knit
			s.vel += (s.home - s.pos) * 2.6 * delta
			if (s.pos - s.home).length() < 0.02:
				s.basis = s.basis.slerp(Basis.IDENTITY, 1.0 - exp(-3.0 * delta))
	if not _settled():
		_moved = true
	queue_redraw()


func _draw() -> void:
	begin_draw()
	lens.prepare()
	var u := unit()
	# Depth-sort shards back-to-front so nearer ones occlude farther ones.
	var order := range(_shards.size())
	order.sort_custom(func(a, b): return lens.depth(_shards[a].pos) > lens.depth(_shards[b].pos))
	for idx in order:
		var s: Dictionary = _shards[idx]
		var poly := PackedVector2Array()
		var ok := true
		for v: Vector2 in s.poly:
			var world: Vector3 = s.basis * Vector3(v.x, v.y, 0.0) + s.pos
			var pr := lens.project(world)
			if pr.z <= lens.near:
				ok = false
				break
			poly.append(Vector2(pr.x, pr.y) * u)
		if not ok or poly.size() < 3 or Geo.area(poly) < 1.0:   # skip edge-on (degenerate) shards
			continue
		# Glassy shading: how squarely the shard faces the camera sets its glint.
		var nrm: Vector3 = (s.basis * Vector3(0, 0, 1)).normalized()
		var face := absf(nrm.dot((s.pos - lens.eye).normalized()))
		var scatter := clampf((s.pos - s.home).length() * 1.4, 0.0, 1.0)
		var res: float = clampf(float(s.lev), 0.0, 1.0)
		# Transparency tracks the shard's channel: when its band is quiet it goes glassy and
		# see-through (the depth-sorted shards behind it show), when it resonates it solidifies
		# and brightens. Drawn back-to-front (above), so the alpha layers compose correctly.
		var val := clampf(0.30 + 0.45 * res + 0.28 * face, 0.1, 1.0)
		var alpha := clampf(0.24 + 0.5 * res + 0.16 * face, 0.2, 0.92)
		draw_colored_polygon(poly, Color.from_hsv(s.hue, 0.4, val, alpha))
		var edge := poly.duplicate()
		edge.append(poly[0])
		draw_polyline(edge, Color.from_hsv(s.hue, 0.15, 1.0, clampf(0.18 + 0.4 * res + 0.4 * scatter, 0.0, 1.0)), 1.0, true)

	# Fracture flashes: a bright streak along each freshly formed crack, fading as it ages -
	# the visible "new break" the eye catches before it settles into the permanent seam.
	for cr in _cracks:
		var pa := lens.project(Vector3(cr.a))
		var pb := lens.project(Vector3(cr.b))
		if pa.z <= lens.near or pb.z <= lens.near:
			continue
		var k := 1.0 - clampf(float(cr.age) / 0.6, 0.0, 1.0)
		draw_line(Vector2(pa.x, pa.y) * u, Vector2(pb.x, pb.y) * u,
			Color(1.0, 1.0, 1.0, 0.85 * k), maxf(1.0, 2.0 * k), true)
