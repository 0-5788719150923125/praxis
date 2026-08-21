extends Scene3D

## Neural field - a layered network in real depth, lit only where the harmony routes.
##
## Three to six flat SHEETS of neurons hang one behind the other in genuine perspective,
## seen from a slow three-quarter yaw so the sheets overlap and the depth reads at once.
## Between consecutive sheets runs a sparse fan of straight edges, and EVERY edge is always
## there - the geometry never changes, not once, for the whole life of the scene. What
## changes is only which edges are LIT, and an edge lights only when BOTH of its endpoints
## are firing at the same time. That single rule is the whole scene: you are not watching a
## uniform twitch, you are watching a shifting subset of ROUTES through a slab.
##
## This is the corrected version of the constant, meaningless twitching a network
## visualization usually degenerates into - hairline edges destroyed and re-randomised off a
## wall clock, every node reacting to loudness at once. Here nothing is destroyed, nothing is
## re-randomised, and loudness alone lights nothing.
##
## WHAT THE SEED DECIDES. The mood ([Scheme], every hue in the frame from one family, each
## sheet taking `hue_at(l, layers)` so depth reads as a colour gradient too). How many sheets,
## and their TOPOLOGY - `slab` keeps every sheet the same size, `funnel` narrows toward the
## output, `fan_out` widens, `bottleneck` pinches in the middle and opens again - so the
## silhouette of the network itself is sampled, not just its colours. Rows and columns, the
## positional jitter that keeps the grid from reading as graph paper, the fan-out degree and
## how far a neuron may reach sideways for its targets, the exponent that shapes the weight
## distribution (high exponent = most weights near zero = a darker, more selective graph),
## the fraction of INHIBITORY weights (drawn in the opposed hue), the sparsity/attack/decay
## of each sheet's [Activation], the lens distance / pitch / yaw sway, layer spacing, edge
## width in pixels, how many sub-segments each edge splits into, disc facet count, and the
## darkness floor that unlit edges rest at.
##
## AUDIO. The input sheet is read across its width: neuron at column fraction cx takes
## `f.sample(cx)`, so bass enters one side of the sheet and treble the other. Every neuron
## also carries a PITCH CLASS assigned at build, and mixes in `Spectrum.harmonic_signature()`
## for that class - so which neurons are ELIGIBLE to fire is decided by the harmony rather
## than by loudness, and a chord change re-routes the slab without the level moving at all.
## (The signature array is EMPTY until the analyzer is up, hence the `.size() > pc` guard.)
## One [Activation] per sheet, driven by the sheet's aggregate, supplies the slow seeded GATE
## - who is on this passage - while a faster per-neuron envelope supplies the actual firing.
## Sheet L+1's drive is the mean of sheet L's levels times a sampled gain, so activity really
## propagates instead of every sheet reacting to the same frame independently. On a beat
## onset a bright PACKET leaves the input sheet and walks one pre-rolled route forward, one
## layer per `beat_period * hops`, lighting each neuron it lands on, and you can follow it
## through the fan until it exits the far side. A `movement` spike eases the lens toward a new
## seeded three-quarter angle, so the framing turns over on section changes. `chroma_hue()`
## pulls every hue in the frame toward the tonal centre by `0.35 * strength`.
##
## NOTHING HERE MULTIPLIES A RADIUS BY ENERGY. The discs are a constant world size and shrink
## only with distance; sound moves colour, brightness and which routes exist.
##
## MEASUREMENT AND THE COSTS THAT SHAPED THE CODE.
##
##   Edges are SPLIT into sub-segments before sorting. A painter's sort over whole edges makes
##   each edge win or lose against the discs as a block, which is wrong for any edge that
##   passes between two sheets - the exact failure measured at 20-45% of quads for the old
##   water sheet (terrain.gd's collect_surface doc). Three to five segments per edge lets the
##   sort interleave an edge with the neurons it passes.
##
##   The item budget is enforced, not hoped for. Neurons per sheet are capped so no seed can
##   ask for more than [constant MAX_PER_SHEET], and the fan-out degree is then clamped so
##   `edges * segments` never exceeds [constant MAX_SEGMENTS]. Left unbudgeted, 324 neurons
##   with k = 5 and 4 segments is ~6.8k sortable items, and at the ~2.2 us/item anchor
##   measured for terrain that is a whole frame gone on emission alone.
##
##   [Scene3D]'s own `add_body` / `render_world` are deliberately NOT used: render_world
##   sort_customs its item list (a GDScript lambda per comparison) and each body flushes the
##   batch, so a few hundred neurons would be a few hundred draw calls plus a few hundred
##   thousand lambda calls. Everything here is emitted into one [TriBatch] inside a
##   [FrameForge] job object instead - one or two draw calls, sorted natively by
##   `TriBatch.painter_sort`, and built entirely off the main thread.
##
##   Dark edges are emitted with TriBatch's cheap hard-edged line (2 triangles) and only lit
##   ones pay for the feathered form (6). The dark ones are the overwhelming majority and are
##   near-black by design, so there is nothing there for antialiasing to smooth.
##
## The propagation and the packets run on a [SimClock], not on `update()`: the Director
## sub-steps a scene up to 15 times in a frame, pre-warms it 12 times before its first frame,
## and an [Echo] re-localize can fast-forward hundreds of calls. A network that advanced once
## per call would arrive on screen already saturated.

## Neurons a single sheet may hold. Rows and columns are each sampled in [4,9], which would
## allow 81 per sheet and 486 across six - and every neuron costs a projection, an envelope
## step and a sorted disc every frame. 64 keeps the worst case near 380 while leaving the
## sampled ranges themselves untouched (the column count is simply drawn from what is left
## after the row count).
const MAX_PER_SHEET := 64

## Ceiling on sortable edge sub-segments per frame. The fan-out degree is clamped down until
## `edges * segments` fits, so the item count is bounded whatever the seed asks for.
const MAX_SEGMENTS := 2600

## Beat level a rising edge must cross to count as an onset.
const BEAT_ON := 0.55

## How far every hue is pulled toward the live tonal centre, scaled by how tonal the moment is.
const CHROMA_PULL := 0.35

## Below this the material counts as SILENCE and the drive reference stops learning, so a held
## pause can never divide a tiny number by a tinier one and light the slab up. Same value and
## same role as [member Director.silence_floor], which is already the "nothing is playing" line
## the cut logic trusts.
const SILENCE_FLOOR := 0.03

## Where the material's OWN AVERAGE lands on the 0..1 drive dial. Half, so an ordinary passage
## sits mid-scale and a passage that leans in has somewhere to go. This is a normalisation
## convention, not a look knob - the look knobs are all sampled.
const DRIVE_MID := 0.5

# How each topology scales a sheet's row/column counts along the slab.
const TOPOLOGIES := ["slab", "funnel", "bottleneck", "fan_out"]

var _f: AudioFeatures = AudioFeatures.new()
var _sig := PackedFloat32Array()      # 12 chroma channels + coarse shape, read live
var _ch := Vector2.ZERO               # (hue, strength) of the music's tonality
var _sch: Scheme = null
var _sim := SimClock.new(45.0, 3)
var _forge := FrameForge.new()

# --- the graph (immutable after build) ---
var _pos := PackedVector3Array()      # neuron world positions, normalized to unit radius
var _layer_of := PackedInt32Array()
var _cx := PackedFloat32Array()       # column fraction 0..1, for f.sample()
var _pc := PackedInt32Array()         # pitch class 0..11
var _hjit := PackedFloat32Array()     # per-neuron hue jitter
var _start := PackedInt32Array()      # first neuron index of each sheet
var _count := PackedInt32Array()      # neurons in each sheet
var _esrc := PackedInt32Array()
var _edst := PackedInt32Array()
var _ew := PackedFloat32Array()       # signed weight; sign = excitatory / inhibitory
var _ehue := PackedFloat32Array()     # per-edge base hue, precomputed
var _eoff := PackedInt32Array()       # first outgoing edge of each neuron
var _ecnt := PackedInt32Array()
var _routes: Array = []               # pre-rolled packet paths (PackedInt32Array of neurons)

# --- the state ---
var _level := PackedFloat32Array()    # per-neuron firing envelope 0..1
var _acts: Array = []                 # one Activation per sheet (the route gate)
var _gain := PackedFloat32Array()     # sheet-to-sheet transmission (see _step)
var _packets: Array = []
var _glow := 0.0
var _prev_beat := 0.0
var _last_launch := -10.0
var _last_reframe := 0.0
var _route_i := 0
var _yaw_i := 0

# The material's own recent level, which every drive here is measured against (see _step).
var _e_ref := 0.0                     # EMA of f.energy
var _b_ref := 0.0                     # EMA of the loudest band this frame
var _ref_seeded := false

# --- sampled definition kept as typed members (hot in the per-frame path) ---
var _layers := 4
var _rows := 6
var _cols := 6
var _segs := 4
var _disc_r := 0.03
var _edge_w := 1.4
var _sides := 8
var _dark := 0.12
var _band_mix := 0.8
var _chroma_mix := 0.8
var _gate_floor := 0.12
var _w_att := 0.3
var _w_dec := 0.05
var _packets_on := true
var _burst := 2
var _hops := 1
var _move_thresh := 0.45
var _reframe_gap := 24.0
var _dist := 1.7
var _push := 0.0
var _pitch := 0.1
var _yaw_center := 0.65
var _yaw_goal := 0.65
var _yaw_amp := 0.08
var _yaw_rate := 0.02
var _yaw_phase := 0.0
var _yaw_ease := 0.1
var _yaw_bag := PackedFloat32Array()
var _ref_tau := 25.0
var _a_floor := 0.20
var _bleach := 0.85
var _ghost_sat := 0.45
var _edge_gamma := 0.5
var _node_gamma := 0.6


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	framing = "plane"                       # the yaw comes from the lens, not the shot
	_sch = Scheme.pick(rng)                 # a graph is abstract: no mood is wrong for it
	_layers = rng.randi_range(3, 6)
	_rows = rng.randi_range(4, 9)
	# Draw the column count from what the per-sheet budget leaves, so the sampled range is
	# honoured wherever it can be and the item count is still bounded.
	_cols = rng.randi_range(4, clampi(MAX_PER_SHEET / _rows, 4, 9))
	var topo := String(TOPOLOGIES[rng.randi() % TOPOLOGIES.size()])
	var jitter := rng.randf_range(0.0, 0.35)
	var spacing := rng.randf_range(0.5, 1.2)
	var wexp := rng.randf_range(0.6, 2.2)
	var inhib := rng.randf_range(0.15, 0.40)
	var reach := rng.randi_range(1, 3)      # how far sideways a neuron may look for a target
	var k_want := rng.randi_range(2, 5)
	_segs = rng.randi_range(3, 5)
	_disc_r = rng.randf_range(0.07, 0.14)   # as a fraction of a cell, scaled below
	_edge_w = rng.randf_range(0.8, 2.2)
	_sides = rng.randi_range(5, 12)
	_dark = rng.randf_range(0.05, 0.25)
	_band_mix = rng.randf_range(0.5, 1.0)
	_chroma_mix = rng.randf_range(0.5, 1.1)
	_gate_floor = rng.randf_range(0.05, 0.20)
	_packets_on = rng.randf() < 0.75
	_burst = rng.randi_range(1, 4)
	_hops = rng.randi_range(1, 3)
	_move_thresh = rng.randf_range(0.35, 0.60)
	# Seconds the framing must hold before a section change may re-aim it at all. Was 5-9, which
	# on narration (where `movement` spikes at nearly every sentence) meant the camera re-aimed at
	# almost every opportunity - see the yaw block below for what that did.
	_reframe_gap = rng.randf_range(18.0, 32.0)
	# How long the drive reference takes to forget, in seconds. Short = the slab tracks each
	# sentence's own loudness; long = it tracks the passage.
	_ref_tau = rng.randf_range(15.0, 40.0)
	# The lit/unlit mapping (see NeuralJob.run). `bleach` is how completely a firing route burns
	# out to white, `ghost_sat` how much colour the unlit graph keeps, `a_floor` how present the
	# unlit graph is at all, and the gammas how sharply firing translates into light.
	# The gammas are ABOVE one, which is the opposite of the 0.75 that shipped - and that is the
	# tell that the old exponent was compensating for a signal that was not there. With the drive
	# starved, only a gamma below one lifted anything off the floor at all, and it lifted the
	# whole graph together. With the drive restored, levels reach 1 on their own and the exponent
	# can do its actual job: hold the unlit majority down so a genuinely hot pair stands out.
	_a_floor = rng.randf_range(0.14, 0.26)
	_bleach = rng.randf_range(0.75, 0.95)
	_ghost_sat = rng.randf_range(0.35, 0.60)
	_edge_gamma = rng.randf_range(1.30, 2.10)
	_node_gamma = rng.randf_range(1.50, 2.40)
	var sparsity := rng.randf_range(0.25, 0.70)
	var attack := rng.randf_range(4.0, 9.0)
	var decay := rng.randf_range(0.8, 1.8)
	# The firing envelope is deliberately CRISPER than the Activation gate underneath it: the
	# gate is the slow question "is this neuron on this passage", the envelope is the fast one
	# "is it firing right now". Both weights are constants because SimClock's dt is constant.
	_w_att = 1.0 - exp(-attack * 1.5 * _sim.dt)
	_w_dec = 1.0 - exp(-decay * _sim.dt)

	# --- place the sheets ---
	var hw := 1.0
	var hy := hw * float(_rows) / float(_cols)      # square-ish cells whatever the grid
	var cell := 2.0 * hw / float(_cols)
	var n_total := 0
	for l in _layers:
		var s := _topo_scale(topo, l)
		var cl := maxi(2, int(round(float(_cols) * s)))
		var rl := maxi(2, int(round(float(_rows) * s)))
		var z := (float(_layers - 1) * 0.5 - float(l)) * spacing   # sheet 0 nearest the camera
		_start.append(n_total)
		_count.append(rl * cl)
		for r in rl:
			for c in cl:
				var fx := (float(c) + 0.5) / float(cl)
				var fy := (float(r) + 0.5) / float(rl)
				var px := (fx * 2.0 - 1.0) * hw + rng.randf_range(-jitter, jitter) * cell
				var py := (fy * 2.0 - 1.0) * hy + rng.randf_range(-jitter, jitter) * cell
				_pos.append(Vector3(px, py, z))
				_layer_of.append(l)
				_cx.append(fx)
				# Pitch class runs across the sheet (so the harmonic routing has a spatial
				# shape) with a one-step seeded nudge, so the classes are not perfect stripes.
				var pc := (int(fx * 12.0) + rng.randi_range(-1, 1)) % 12
				_pc.append(pc if pc >= 0 else pc + 12)
				_hjit.append(rng.randf_range(-_sch.spread, _sch.spread) * 0.5)
				n_total += 1
	_level.resize(n_total)
	for i in n_total:
		_level[i] = 0.0

	# Normalize the whole slab to a unit bounding radius. Without this the lens distance means
	# something different for every seed - a six-sheet slab at 1.2 spacing is three times the
	# depth of a three-sheet slab at 0.5 - and the same sampled distance would frame one and
	# sit inside the other.
	var radius := 0.001
	for i in n_total:
		radius = maxf(radius, _pos[i].length())
	var fit := 1.0 / radius
	for i in n_total:
		_pos[i] = _pos[i] * fit
	_disc_r = _disc_r * cell * fit

	# --- the fan ---
	var n_src := n_total - int(_count[_layers - 1])
	var k := clampi(int(MAX_SEGMENTS / maxi(1, n_src * _segs)), 2, k_want)
	_eoff.resize(n_total)
	_ecnt.resize(n_total)
	for l in _layers:
		var s0 := int(_start[l])
		var n0 := int(_count[l])
		if l == _layers - 1:
			for j in n0:
				_eoff[s0 + j] = _esrc.size()
				_ecnt[s0 + j] = 0
			break
		var s1 := int(_start[l + 1])
		var n1 := int(_count[l + 1])
		var hue_a := _sch.hue_at(l, _layers)
		var hue_b := _sch.hue_at(l + 1, _layers)
		var opp := _sch.opposed(hue_a)
		var dh := fposmod(hue_b - hue_a + 0.5, 1.0) - 0.5      # shortest way round
		for j in n0:
			var i := s0 + j
			_eoff[i] = _esrc.size()
			var picked: Array = []
			for _t in k:
				# Targets are drawn from a NEIGHBOURHOOD of the source's own position in the
				# next sheet, so the graph reads as a fan of local connections rather than as
				# spaghetti - which is what makes a route legible to the eye at all.
				var tgt := _near_target(rng, i, s1, n1, reach, picked)
				if tgt < 0:
					continue
				picked.append(tgt)
				_esrc.append(i)
				_edst.append(tgt)
				var w := pow(rng.randf(), wexp)
				var neg := rng.randf() < inhib
				_ew.append((-w) if neg else w)
				# An edge takes the colour of the gap it crosses: halfway between its two
				# sheets' hues, or the opposed hue when it is inhibitory.
				_ehue.append(opp if neg else fposmod(hue_a + dh * 0.5, 1.0))
			_ecnt[i] = _esrc.size() - int(_eoff[i])

	# --- per-sheet activation + propagation transmission ---
	# TRANSMISSION, not amplification. The old 1.1-1.9 gain existed to fight the collapse of the
	# whole-sheet MEAN: a sparse sheet's mean is dominated by the neurons that are off, so each hop
	# multiplied the drive by roughly the sparsity and the far sheets went dead (measured on this
	# project's own narration bake: sheet levels 0.33 / 0.17 / 0.026 / 0.007 / 0.0002 / 0.000 -
	# the sixth sheet was EXACTLY zero, every frame, every seed). _step now propagates what
	# actually FIRED instead of the mean, which does not collapse, so this becomes a per-sheet
	# transmission a shade either side of unity and depth attenuates gently rather than geometrically.
	for l in _layers:
		_acts.append(Activation.new(int(_count[l]), rng, sparsity, attack, decay))
		_gain.append(rng.randf_range(0.72, 0.92))

	# --- pre-rolled packet routes and lens angles ---
	# Both are drawn HERE, never at the moment of the audio event that uses them: a live
	# analyzer and an offline bake do not produce identical feature streams, so an rng draw on
	# a beat would desynchronise the export from the preview.
	var nroutes := rng.randi_range(10, 24)
	for _r in nroutes:
		var chain := PackedInt32Array()
		var cur := int(_start[0]) + rng.randi() % int(_count[0])
		chain.append(cur)
		for _l in _layers - 1:
			if int(_ecnt[cur]) <= 0:
				break
			var e := int(_eoff[cur]) + rng.randi() % int(_ecnt[cur])
			cur = int(_edst[e])
			chain.append(cur)
		if chain.size() >= 2:
			_routes.append(chain)
	# THE CAMERA IS ESSENTIALLY FIXED. The side of the slab is chosen ONCE, here, and never
	# changes: the old bag drew an independent sign per entry, so two consecutive framings could
	# be +0.95 and -0.95 - a 109-degree swing, eased at 0.7/s (a peak of 77 DEGREES PER SECOND),
	# re-triggered every 5-9 seconds, and passing straight through yaw = 0 where the sheets stack
	# on top of each other and the depth the whole scene exists to show collapses. It read exactly
	# as reported: the array spinning wildly back and forth. Same side, narrow band, slow ease -
	# the framing now drifts at about a degree per second and never crosses zero.
	var side := 1.0 if rng.randf() < 0.5 else -1.0
	for _b in 8:
		_yaw_bag.append(side * rng.randf_range(0.55, 0.85))
	_yaw_center = _yaw_bag[0]
	_yaw_goal = _yaw_center
	_yaw_amp = rng.randf_range(0.04, 0.10)
	_yaw_rate = rng.randf_range(0.012, 0.035)
	_yaw_phase = rng.randf() * TAU
	# Seconds-ish to close a re-aim. At the widest gap the bag allows (0.30 rad) this is a peak of
	# ~1.7 deg/s; the breath on top adds at most ~1.3 deg/s.
	_yaw_ease = rng.randf_range(0.07, 0.14)
	_yaw_i = 1
	# CLOSE. At the old 2.2-4.0 the whole slab sat in the middle of the frame as a small object -
	# on a unit-radius slab at distance 3 with a 50 degree lens, a neuron disc projects to a ~14 px
	# radius with ~135 px between neighbours, which is a diagram seen from across the room. Halving
	# the distance doubles both, so the near sheet overruns the frame and the far ones recede: you
	# are inside the fan rather than looking at the whole graph. The near sheet still clears the
	# 0.05 near plane by a wide margin (worst case ~0.7 world units in front of the eye).
	_dist = rng.randf_range(1.35, 2.10)
	# A slow cinematic push, in world units over the scene's whole life - at most ~0.01/s.
	_push = rng.randf_range(0.08, 0.26)
	_pitch = rng.randf_range(-0.24, 0.24)
	lens.fov = rng.randf_range(48.0, 62.0)

	# A faint wash behind the slab, on the mood: a network in a pure black void reads as a
	# diagram rather than as a place.
	add_layer("bed", rng, {
		"hue": _sch.accent,
		"sat": clampf(_sch.sat * rng.randf_range(0.4, 0.8), 0.05, 1.0),
		"val": rng.randf_range(0.04, 0.11),
		"pools": rng.randi_range(1, 3),
	})
	return {
		"mood": _sch.name, "layers": _layers, "rows": _rows, "cols": _cols,
		"topology": topo, "neurons": n_total, "edges": _esrc.size(), "fan_out": k,
		"reach": reach, "jitter": jitter, "spacing": spacing, "weight_exp": wexp,
		"inhibitory": inhib, "segments": _segs, "sparsity": sparsity,
		"attack": attack, "decay": decay, "dark_floor": _dark, "disc_sides": _sides,
		"edge_width": _edge_w, "packets": _packets_on, "burst": _burst, "hops": _hops,
		"dist": _dist, "pitch": _pitch, "yaw_rate": _yaw_rate, "routes": _routes.size(),
	}


# How a topology scales sheet `l`'s row/column counts. Both axes take the scale, so a funnel
# narrows in area rather than only in width.
func _topo_scale(topo: String, l: int) -> float:
	var t := float(l) / float(maxi(1, _layers - 1))
	match topo:
		"funnel":
			return lerpf(1.0, 0.55, t)
		"fan_out":
			return lerpf(0.55, 1.0, t)
		"bottleneck":
			return lerpf(1.0, 0.5, 1.0 - absf(t * 2.0 - 1.0))
		_:
			return 1.0


# One more target for `src` in the next sheet, near it in the sheet plane and not already taken.
# The next sheet's grid may be a different SHAPE (the topologies resize them), so "near" is
# measured in world x/y rather than by row/column arithmetic that would not survive a funnel.
#
# `reach` is expressed as how many seeded candidates get compared: best-of-8 lands almost on the
# source's own axis, best-of-4 wanders. Sampling candidates and keeping the closest UNTAKEN one
# is what stops a five-way fan from collapsing onto the same two neighbours, which is what a
# plain independent draw per edge does. Returns -1 when the sheet has nothing left to give.
func _near_target(rng: RandomNumberGenerator, src: int, s1: int, n1: int, reach: int,
		taken: Array) -> int:
	if taken.size() >= n1:
		return -1
	var p: Vector3 = _pos[src]
	var best := -1
	var bd := 1e20
	for _t in maxi(3, 10 - reach * 2):
		var c := s1 + rng.randi() % n1
		if taken.has(c):
			continue
		var q: Vector3 = _pos[c]
		var d := (q.x - p.x) * (q.x - p.x) + (q.y - p.y) * (q.y - p.y)
		if d < bd:
			bd = d
			best = c
	return best


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.01, 0.02)
	update_layers(f, delta)
	_sig = Spectrum.harmonic_signature()
	_ch = chroma_hue()
	_glow = Nonlinear.flare(_glow, clampf(0.35 * f.energy + 0.7 * f.beat, 0.0, 1.0), delta, 9.0, 1.6)

	# A beat ONSET (a rising edge, not the level) launches packets. The Director legitimately
	# calls update() several times with the same frame, and storing the level afterwards makes
	# that idempotent rather than a burst.
	if _packets_on and f.beat > BEAT_ON and _prev_beat <= BEAT_ON and _life - _last_launch > 0.12:
		_last_launch = _life
		_launch(f)
	_prev_beat = f.beat

	# A section change re-frames the slab: ease the yaw centre toward the next PRE-ROLLED
	# three-quarter angle. Rate-limited, because a busy passage would otherwise re-aim every
	# few frames and the camera would read as nervous rather than as deliberate.
	if f.movement > _move_thresh and _life - _last_reframe > _reframe_gap and _yaw_bag.size() > 0:
		_last_reframe = _life
		_yaw_goal = _yaw_bag[_yaw_i % _yaw_bag.size()]
		_yaw_i += 1
	_yaw_center = lerpf(_yaw_center, _yaw_goal, 1.0 - exp(-_yaw_ease * delta))
	_yaw_phase += delta * _yaw_rate * TAU
	# The yaw BREATHES inside a narrow three-quarter band on ONE side of the slab (the side is
	# fixed at build). Head-on the sheets stack exactly on top of each other and the depth
	# vanishes; near a quarter turn they collapse to lines. Neither is the picture, and the band
	# is now narrow enough that neither is ever approached.
	var yaw := _yaw_center + _yaw_amp * sin(_yaw_phase)
	# The push-in is on the scene's OWN clock, not the audio: a cinematic move should be one
	# unhurried gesture across the whole shot, not something that restarts on every loud sentence.
	var dist := _dist - _push * smoothstep(0.0, 1.0, clampf(_life / 40.0, 0.0, 1.0))
	lens.orbit(Vector3.ZERO, maxf(dist, 0.9), yaw, _pitch + 0.02 * mod.value("tilt"))

	for _i in _sim.ticks(delta):
		_step(_sim.dt)

	var job := NeuralJob.new()
	job.u = unit()
	job.reveal = view.reveal
	job.fov = lens.fov
	job.lens = Lens3D.new()
	job.lens.eye = lens.eye
	job.lens.look = lens.look
	job.lens.up = lens.up
	job.lens.fov = lens.fov
	job.lens.near = lens.near
	job.pos = _pos                       # immutable after build - safe by reference
	job.layer_of = _layer_of
	job.hjit = _hjit
	job.esrc = _esrc
	job.edst = _edst
	job.ew = _ew
	job.ehue = _ehue
	job.level = _level.duplicate()       # main mutates this every sim tick
	job.hues = PackedFloat32Array()
	for l in _layers:
		job.hues.append(_sch.hue_at(l, _layers))
	job.ch_h = _ch.x
	job.ch_s = _ch.y
	job.sat = _sch.sat
	job.val = _sch.val
	job.dark = _dark
	job.a_floor = _a_floor
	job.bleach = _bleach
	job.ghost_sat = _ghost_sat
	job.edge_gamma = _edge_gamma
	job.node_gamma = _node_gamma
	job.segs = _segs
	job.sides = _sides
	job.disc_r = _disc_r
	job.edge_w = _edge_w
	job.glow = _glow
	job.packets = _packet_marks()
	_forge.kick(job.run, {}, self, job)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	draw_layers()
	_forge.submit(self)


# One fixed generation: the sheets settle, then every live packet steps forward. Called only
# from SimClock.ticks(), never once per update - see the class doc.
func _step(dt: float) -> void:
	var f := _f
	# THE DRIVE IS CONTENT-RELATIVE, NOT ABSOLUTE. `f.energy` documents itself as "roughly 0..1"
	# and for a mastered track it is; for SPEECH it is not. Measured over this project's own
	# narration bake (686 s, 20581 frames): median 0.047, p90 0.072, max 0.108 - against the ~0.4
	# a mastered track sits at even at rest. Fed in raw it left every Activation threshold
	# unreachable, so `min(level_a, level_b)` peaked at 0.045 across the whole take and the
	# brightest edge in eleven minutes reached 20/255 on black. That is the reported "impossible
	# to see them", and no amount of remapping the colour can recover a signal that is not there.
	#
	# Director._lean() already learned this exact lesson one layer up ("a ratio has no genre baked
	# into it: it fires on the passage where THIS material leans in, whatever this material
	# happens to be"), and its comment records the same measurement for the cut logic. This is
	# that lesson in the scene: the moment is judged against the material's own recent level.
	var bmax := 0.0
	for v in f.bands:
		bmax = maxf(bmax, v)
	if not _ref_seeded:
		_ref_seeded = true
		_e_ref = f.energy
		_b_ref = bmax
	elif f.energy > SILENCE_FLOOR:
		# Silence does not teach the reference anything - it would only drag it down until the
		# next syllable read as a climax.
		var k := 1.0 - exp(-dt / _ref_tau)
		_e_ref = lerpf(_e_ref, f.energy, k)
		_b_ref = lerpf(_b_ref, bmax, k)
	var drive := clampf(f.energy / maxf(_e_ref, SILENCE_FLOOR) * DRIVE_MID, 0.0, 1.0)
	var bgain := 1.0 / maxf(_b_ref, SILENCE_FLOOR)

	var prev_fire := 0.0
	var agg_in := 0.0                     # sheet 0's aggregate, the yardstick for every sheet after
	var fire_in := 0.0                    # ...and how hard sheet 0 actually fired under it
	for l in _layers:
		var s := int(_start[l])
		var n := int(_count[l])
		var act: Activation = _acts[l]
		# The sheet's AGGREGATE drives the gate: which neurons are eligible this passage.
		#
		# SELF-NORMALISED, so that transmission alone decides how far activity reaches. Each
		# Activation is a soft THRESHOLD, not a scale: once a sheet's aggregate falls under the
		# thresholds its gates collapse to `gate_floor` and everything past it is dead - a cliff,
		# not a slope. Handing sheet L+1 the raw firing measure walked straight off that cliff
		# (measured levels by sheet: 0.34 / 0.17 / 0.02 / 0.005 / 0.001 / 0.000). Expressing it
		# as a FRACTION of what sheet 0 managed under its own aggregate keeps every sheet in the
		# same working range, and `_gain` - now a shade under unity - is what makes depth fade.
		var agg: float
		if l == 0:
			agg = clampf(1.8 * drive + 0.6 * f.beat, 0.0, 1.0)
			agg_in = agg
		else:
			agg = clampf(agg_in * (prev_fire / maxf(fire_in, 1e-3)) * float(_gain[l]), 0.0, 1.0)
		act.update(agg, dt)
		var sum := 0.0
		var sum_sq := 0.0
		for j in n:
			var i := s + j
			var raw: float
			if l == 0:
				raw = clampf(f.sample(_cx[i]) * bgain, 0.0, 1.0) * _band_mix
			else:
				raw = agg
			# The harmony decides WHO, the level decides HOW MUCH. On the input sheet the
			# chroma channel adds to the band drive; deeper in it scales what arrived, so a
			# chord change re-routes the slab without the loudness moving at all.
			var pc := int(_pc[i])
			if _sig.size() > pc:
				if l == 0:
					raw += _sig[pc] * _chroma_mix
				else:
					raw *= 0.40 + 1.1 * _sig[pc] * _chroma_mix
			var gate := _gate_floor + (1.0 - _gate_floor) * act.level(j)
			var target := clampf(raw * gate, 0.0, 1.0)
			var cur := _level[i]
			_level[i] = cur + (target - cur) * (_w_att if target > cur else _w_dec)
			sum += _level[i]
			sum_sq += _level[i] * _level[i]
		# WHAT FIRED, not what is there. The level-weighted mean level (sum of squares over sum)
		# is the plain mean when the sheet is uniform and the MAX when a single neuron carries it,
		# which is the statistic this propagation always wanted. A sparse sheet's plain mean is
		# mostly its dark majority, so passing THAT on multiplied the drive by roughly the sparsity
		# at every hop and the far sheets died (the measured collapse is in build_params, at the
		# transmission ranges). One extra multiply-add per neuron, no sort.
		prev_fire = sum_sq / maxf(sum, 1e-6)
		if l == 0:
			fire_in = prev_fire

	var pi := _packets.size() - 1
	while pi >= 0:
		var pk: Dictionary = _packets[pi]
		var route: PackedInt32Array = pk["route"]
		var before := int(float(pk["p"]))
		var np: float = float(pk["p"]) + dt / maxf(0.05, float(pk["transit"]))
		pk["p"] = np
		var after := int(np)
		if after > before and after < route.size():
			_level[route[after]] = 1.0        # the packet ARRIVES: that neuron fires outright
		if np >= float(route.size() - 1):
			_packets.remove_at(pi)
		pi -= 1


# Send this onset's packets down the next pre-rolled routes. The route is taken by a counter,
# never by an rng draw, so the sequence is identical live and in an export.
func _launch(f: AudioFeatures) -> void:
	if _routes.is_empty():
		return
	var transit := clampf(f.beat_period, 0.2, 1.6) * float(_hops)
	for _b in _burst:
		if _packets.size() >= 20:
			_packets.remove_at(0)
		var route: PackedInt32Array = _routes[_route_i % _routes.size()]
		_route_i += 1
		_packets.append({"route": route, "p": 0.0, "transit": transit})
		_level[route[0]] = 1.0


# Where each live packet is right now, in world space - resolved on the main thread so the
# worker never has to walk the route arrays.
func _packet_marks() -> Array:
	var out: Array = []
	for pk in _packets:
		var p: Dictionary = pk
		var route: PackedInt32Array = p["route"]
		var pr: float = clampf(float(p["p"]), 0.0, float(route.size() - 1))
		var li := mini(int(pr), route.size() - 2)
		var t := pr - float(li)
		var a: Vector3 = _pos[route[li]]
		var b: Vector3 = _pos[route[li + 1]]
		out.append(a.lerp(b, t))
	return out


## The whole frame, built off the main thread (the [FrameForge] job object form): project every
## neuron once, colour and SPLIT every edge, sort the lot together, and cut it into batch runs.
## Reads only its own members - never the scene node - so a Director cut mid-build is harmless.
class NeuralJob:
	extends RefCounted

	# Restated here rather than read from the outer script: a GDScript inner class does not
	# inherit the outer script's constants (spires' SpireJob restates its own for the same
	# reason), and the job must be readable as the self-contained thing it is.
	const CHROMA_PULL := 0.35

	var lens: Lens3D
	var u := 1.0
	var fov := 50.0
	var reveal := 1.0
	var pos := PackedVector3Array()
	var layer_of := PackedInt32Array()
	var hjit := PackedFloat32Array()
	var esrc := PackedInt32Array()
	var edst := PackedInt32Array()
	var ew := PackedFloat32Array()
	var ehue := PackedFloat32Array()
	var level := PackedFloat32Array()
	var hues := PackedFloat32Array()
	var packets: Array = []
	var ch_h := 0.0
	var ch_s := 0.0
	var sat := 0.6
	var val := 0.9
	var dark := 0.12
	var a_floor := 0.20
	var bleach := 0.85
	var ghost_sat := 0.45
	var edge_gamma := 1.7
	var node_gamma := 1.9
	var segs := 4
	var sides := 8
	var disc_r := 0.03
	var edge_w := 1.4
	var glow := 0.0

	func run(_s: Dictionary) -> Array:
		var out: Array = []
		if reveal < 0.02:
			return out
		lens.prepare()
		var n := pos.size()
		var sp := PackedVector2Array()
		sp.resize(n)
		var dp := PackedFloat32Array()
		dp.resize(n)
		for i in n:
			var pr := lens.project(pos[i])
			sp[i] = Vector2(pr.x, pr.y) * u
			dp[i] = pr.z
		var near := lens.near
		var focal := 1.0 / tan(deg_to_rad(fov) * 0.5)
		var items: Array = []
		var inv := 1.0 / float(segs)

		# EDGES. Every one is emitted, every frame - the geometry is immutable and only the
		# colour moves. `min(level_a, level_b) * |weight|` is the discipline of the scene: an
		# edge is bright only when BOTH ends are hot, so what lights up is a route, not a mood.
		# A FIRING ROUTE RUNS TO WHITE. The background is black, so the only way a lit edge reads
		# as lit is to leave the mood's colour behind on its way up: the hue carries the middle of
		# the range and burns out at the top. `lit_val` is therefore ~1, not the mood's own value.
		var lit_val := lerpf(val, 1.0, 0.72 + 0.24 * glow)
		for e in esrc.size():
			var a := int(esrc[e])
			var b := int(edst[e])
			var da := dp[a]
			var db := dp[b]
			if da <= near or db <= near:
				continue
			# ONE pow, and the weight rides in the EXPONENT rather than scaling the intensity. A
			# strong edge lights on a modest pair, a weak one needs both ends genuinely hot - the
			# same "a route, not a mood" discipline, but it can now reach the top of the range
			# instead of being multiplied away from it before the curve is ever applied.
			var sh := pow(minf(level[a], level[b]), edge_gamma * (1.6 - 0.8 * minf(absf(ew[e]), 1.0)))
			var v := dark + maxf(0.0, lit_val - dark) * sh
			var h := _tonal(ehue[e])
			# Colour in the middle, white at the top: hold most of the mood's saturation until the
			# edge is genuinely lit, then bleach it out.
			var csat := sat * (ghost_sat + (1.0 - ghost_sat) * minf(sh * 3.0, 1.0)) * (1.0 - bleach * sh)
			var col := Color.from_hsv(h, clampf(csat, 0.0, 1.0),
				clampf(v, 0.0, 1.0), reveal * clampf(a_floor + (1.0 - a_floor) * sh, 0.0, 1.0))
			# Split the edge so the painter's sort can INTERLEAVE it with the discs it passes.
			# An edge crossing a whole sheet gap otherwise wins or loses against every neuron
			# along it as a single block, which is exactly wrong for a scene about routes.
			var pa := sp[a]
			var pb := sp[b]
			var soft := sh > 0.30
			for s in segs:
				var t0 := float(s) * inv
				var t1 := float(s + 1) * inv
				var it := {
					"d": lerpf(da, db, (t0 + t1) * 0.5),
					"a": pa.lerp(pb, t0), "b": pa.lerp(pb, t1),
					"col": col, "w": edge_w}
				if soft:
					it["soft"] = true
				items.append(it)

		# NEURONS. A constant world radius, so on screen a disc shrinks with distance and with
		# nothing else. Sound never touches this number.
		var cs := PackedFloat32Array()
		var sn := PackedFloat32Array()
		for k in sides:
			var ang := TAU * float(k) / float(sides)
			cs.append(cos(ang))
			sn.append(sin(ang))
		var node_floor := clampf(dark + 0.06, 0.08, 0.34)
		for i in n:
			var d := dp[i]
			if d <= near:
				continue
			var fire := level[i]
			var r := disc_r * focal / d * u
			if r < 0.35:
				continue
			var c := sp[i]
			var poly := PackedVector2Array()
			poly.resize(sides)
			for k in sides:
				poly[k] = Vector2(c.x + cs[k] * r, c.y + sn[k] * r)
			var h := _tonal(hues[layer_of[i]] + hjit[i])
			var shn := pow(fire, node_gamma)
			var v := node_floor + maxf(0.0, 1.0 - node_floor) * shn
			var nsat := sat * (ghost_sat + (1.0 - ghost_sat) * minf(shn * 3.0, 1.0)) * (1.0 - bleach * shn)
			items.append({"d": d, "poly": poly,
				"flat": Color.from_hsv(h, clampf(nsat, 0.0, 1.0),
					clampf(v, 0.0, 1.0), reveal)})

		# PACKETS - the one thing in the frame allowed to be white.
		for p in packets:
			var wp: Vector3 = p
			var pr2 := lens.project(wp)
			if pr2.z <= near:
				continue
			var r2 := disc_r * 1.7 * focal / pr2.z * u
			var c2 := Vector2(pr2.x, pr2.y) * u
			var poly2 := PackedVector2Array()
			poly2.resize(sides)
			for k in sides:
				poly2[k] = Vector2(c2.x + cs[k] * r2, c2.y + sn[k] * r2)
			items.append({"d": pr2.z, "poly": poly2,
				"flat": Color.from_hsv(_tonal(hues[hues.size() - 1]), sat * 0.25, 1.0, reveal)})

		items = TriBatch.painter_sort(items)
		var tb := TriBatch.new()
		for it2 in items:
			var d2: Dictionary = it2
			if d2.has("poly"):
				tb.poly(d2["poly"], d2["flat"])
			else:
				# The dark majority take the cheap 2-triangle line; only lit edges pay for the
				# feathered 6-triangle form, since a near-black hairline has no edge to smooth.
				tb.line(d2["a"], d2["b"], d2["col"], float(d2["w"]), d2.has("soft"))
		return tb.take_chunks()

	# Pull a hue toward the music's tonal centre, the shortest way round the wheel, scaled by
	# how tonal the moment actually is (0 when the spectrum is noise, so nothing shifts).
	func _tonal(h: float) -> float:
		var d := ch_h - h
		return fposmod(h + (d - round(d)) * CHROMA_PULL * ch_s, 1.0)
