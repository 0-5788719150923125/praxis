extends GhostScene

## Murmuration - thousands of birds deciding together, against a bright sky.
##
## A wide, nearly empty dusk sky - pale at the horizon, deepening upward - with one flock
## in it. The birds wheel as a single body: the flock draws out into a ribbon, folds back
## through itself, thickens into a dark sheet where the wings come broadside to the lens
## and thins to a haze where they roll edge-on. Waves of that darkening travel across the
## whole flock in half a second. A hawk cuts through the middle on a beat, the flock opens
## around it and closes behind, and a pale wave of alarm spreads outward from the strike
## by neighbour contact alone - faster than any bird in it is flying.
##
## WHY IT IS NEW. Nothing in the catalogue flocks. `prism_swarm` is a scripted formation on
## a fixed double-helix track, and [Primitives]' eight forces are field-like: every agent
## reads the same wind and the same attractor, and none of them has ever had an opinion
## about the agent beside it. This is the first subject whose shape is decided by consensus
## rather than by a seeded parameter, and the first crowd - the alarm wave is the point of
## it, a signal propagating through neighbours at a speed that has nothing to do with the
## audio that triggered it, so what you watch outlives its own trigger.
##
## WHAT THE SEED DECIDES. How many birds and how far they can see; the k-nearest cap and
## the scan cap that bound the neighbour loop; the fixed separation weight and the two
## endpoints the cohesion/alignment pair walks between; speed range and the acceleration
## ceiling that is really a turn-rate ceiling; the roost's position, its pull and how far
## it may wander; whether there is a hawk at all (about seven seeds in ten), and if so its
## cadence, its speed and four sampled approaches - straight through the middle, a stoop
## from above, a lateral cross, or a climb from below; the sky's mood, how high the horizon
## sits, how deep the zenith goes, whether there is cloud and whether there is a distant
## treeline; the bird silhouette - body length, wing span, how far the wings beat and the
## RANGE of wingbeat rates, so the flock's flicker is asynchronous rather than one strobe;
## the depth of the lens volume the flock occupies; and the roll wave's amplitude, its
## wavelength and how long it takes to cross.
##
## AUDIO, WHICH IS THE THREE WEIGHTS. Separation never moves - personal space is not a
## musical parameter. Cohesion is driven by f.bass, so low sustained energy pulls the flock
## into a dense ball; alignment is driven by f.high + f.treble, so bright detail makes it
## stream out into ribbons. f.movement and f.flux together walk the weight point along a
## seeded segment between the two sampled endpoints, and the weights are RATE-LIMITED on
## the way, which is what gives sharp folds on an agitated passage and long lazy arcs on a
## calm one. f.beat on an armed latch launches the hawk; after that the strike is on its
## own clock and the audio has no further say in it. A slow vertical thermal in the air
## comes from f.bass. Only the sky reads chroma_hue() and f.energy - the birds' brightness
## comes from their BANKING ANGLE, never from the music, because a flock that flashed with
## the beat would stop being an animal.
##
## MEASURED REALITY. f.energy is a mean over 64 bands and rarely passes 0.5; f.flux lives
## around 0.01-0.05. So the drives here are written for those ranges - alignment reads
## (f.high + f.treble) * 1.7 clamped, and the walk speed is movement + 14 * flux, not the
## nominal unit quantities a 0..1 reading would suggest.
##
## COST, AND AN HONEST COUNT. The solver and the batching both live in a [FrameForge] job;
## the scene never touches a bird. A real murmuration is tens of thousands of starlings and
## the temptation is to write that number down, but it is not a number GDScript reaches:
## the working ceiling on a ghost worker is around half a million interpreted operations
## per tick (falling_sand's 23k-cell sweep at 25 Hz is the reference point), and a bird
## costs roughly three hundred of them once the eight-cell block query, the scan cap and
## the query stride are all in. That puts the honest range at [900, 2400], which is what is
## sampled - a flock, drawn as animals with wings that beat, rather than nine thousand dots
## that never render. The sim runs on a [SimClock] at a
## sampled 22-30 Hz and the renderer interpolates between the last two solver states, so
## the motion is smooth at display rate while the physics stays frame-rate independent -
## which is not optional here, since the Director sub-steps update() up to fifteen times in
## one frame, pre-warms every scene twelve times before it is drawn, and an Echo re-localize
## can fast-forward hundreds of calls. Any of the three would have the flock in a corner
## before the first frame anyone sees.

## Ceiling on solver ticks handed to one build. [SimClock] caps a single update() call;
## this caps what several capped calls may ACCUMULATE between two drawn frames.
const MAX_PENDING := 5

## The hawk's approaches. Each is a start point (in units of the flock box's half-extents),
## a direction, and a speed multiplier - a stoop comes in steep and fast from above, a
## cross is slow and lateral and only shaves the edge of the flock.
const APPROACHES := ["through", "stoop", "cross", "climb"]

var _flock: Boids
var _clock: SimClock
var _forge := FrameForge.new()
var _job: FlockJob
var _flow: Flow2D
var _tb := TriBatch.new()

# --- flock steering state (the audio mapping's own memory) -------------------------
var _sep := 1.5
var _coh_a := 0.6
var _coh_b := 1.5
var _ali_a := 1.4
var _ali_b := 0.5
var _walk := 0.0
var _walk_dir := 1.0
var _walk_gain := 0.5
var _walk_cap := 0.9
var _turn_cap := 1.1
var _w_coh := 0.9
var _w_ali := 1.0

# --- roost ------------------------------------------------------------------------
var _roost_home := Vector3.ZERO
var _roost_reach := Vector3(0.3, 0.14, 0.22)
var _roost_pull := 0.45
var _thermal := 0.16

# --- hawk -------------------------------------------------------------------------
var _hawk_present := true
var _hawk_cad := 9.0
var _hawk_arm := 0.0
var _hawk_left := 0.0
var _hawk_i := 0
var _hawk_speed := 0.6
var _hawk_pos := Vector3.ZERO
var _hawk_vel := Vector3.ZERO
var _passes: Array = []
var _beat_prev := 0.0

# --- sky --------------------------------------------------------------------------
var _sky_hue := 0.58
var _sky_hue_top := 0.62
var _horizon := 0.52
var _zenith_sat := 0.34
var _zenith_val := 0.62
var _horizon_sat := 0.10
var _horizon_val := 0.90
var _bands := 16
var _glow := 0.0
var _ch := Vector2.ZERO
var _trees := PackedFloat32Array()
var _tree_h := 0.06
var _tree_val := 0.16
var _ground_val := 0.55
var _ground_sat := 0.20

# --- camera -----------------------------------------------------------------------
var _fov := 46.0
## How far past the visible edge the flock's walls sit. The birds have to be turning where the
## camera cannot see them, not at the border of the frame.
const WALL_OUT := 1.18

var _dist := 1.9
var _eye_lift := 0.10

var _ext := Vector3(0.78, 0.44, 0.60)
var _since_tick := 0.0


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "particles"
	framing = "field"

	var sch := Scheme.among(["glacier", "dawn", "rose", "ash", "bone", "brass", "teal"], rng)
	_sky_hue = sch.hue
	_sky_hue_top = sch.opposed(sch.hue, 0.06)
	# The horizon sits LOW, always: the subject is the sky and the flock in it, and a
	# horizon near the middle turns the picture into a landscape with birds over it.
	_horizon = rng.randf_range(0.60, 0.93)
	_horizon_val = rng.randf_range(0.76, 0.97)
	_horizon_sat = rng.randf_range(0.04, 0.16)
	# Sampled as a FRACTION of the horizon rather than freely, so the sky deepens upward
	# for every seed. Drawn independently the two ranges overlap and about a third of
	# seeds come out brighter at the zenith, which reads as an overcast ceiling.
	_zenith_val = _horizon_val * rng.randf_range(0.48, 0.90)
	_zenith_sat = rng.randf_range(0.18, 0.48)
	_bands = rng.randi_range(12, 22)
	# The land below the horizon: never black, because at dusk it is lit by the same sky
	# it sits under, but always darker than the air it meets - which is the only thing that
	# makes the horizon read as an edge rather than as a colour change.
	_ground_val = _horizon_val * rng.randf_range(0.35, 0.78)
	_ground_sat = rng.randf_range(0.08, 0.34)

	# The lens volume. A shallow box makes a flat sheet of birds and a deep one lets the
	# flock fold through itself in depth, which is where the density read comes from.
	_ext = Vector3(rng.randf_range(0.66, 0.92), rng.randf_range(0.34, 0.50),
		rng.randf_range(0.34, 0.78))
	_fov = rng.randf_range(38.0, 58.0)
	_dist = rng.randf_range(1.55, 2.35)
	_eye_lift = rng.randf_range(-0.04, 0.20)

	var count := rng.randi_range(900, 2400)
	_flock = Boids.new()
	_flock.ext = _ext
	# THE WALLS HAVE TO BE ABLE TO MOVE OUT. `_ext` is seeded, and a seeded box has no idea where
	# the camera is: reported as "I can watch as birds bounce off of invisible boundaries". The
	# box is widened from the live view every frame (see update), so the grid has to be laid for
	# the widest it can reach - computed from THIS instance's own lens rather than a blind
	# multiplier, so the extra cells are only the ones that can actually be needed.
	#
	# The worst case is a field scene's camera at its limit on an ultrawide frame: a `pull_back`
	# shot bottoms out near 0.98 zoom, drift_view takes another 0.08 off it and pans about 0.17,
	# and 21:9 puts the horizontal half-extent near 1.75 in unit fractions. It is the one bound
	# here that has to be assumed rather than measured, because a scene's `size` is still zero
	# while build_params runs - the Director does not add it to the tree until afterwards.
	var m := _wall_scale()
	_flock.ext_max = Vector3(maxf(_ext.x, 1.75 * m), maxf(_ext.y, 0.80 * m), _ext.z)
	_flock.radius = rng.randf_range(0.028, 0.044)
	_flock.sep_radius = _flock.radius * rng.randf_range(0.30, 0.50)
	_flock.k_cap = rng.randi_range(6, 14)
	# The scan cap must stay comfortably above k_cap or a dense knot starves: a bird that
	# spends its whole budget on the first cell never reaches the seven neighbours it is
	# allowed to have.
	_flock.scan_cap = _flock.k_cap + rng.randi_range(8, 16)
	_flock.query_stride = 2 if count > 1600 else rng.randi_range(1, 2)
	_flock.speed_min = rng.randf_range(0.10, 0.16)
	_flock.speed_max = _flock.speed_min + rng.randf_range(0.10, 0.24)
	_flock.accel_max = rng.randf_range(0.9, 2.2)
	_flock.alarm_tau = rng.randf_range(0.7, 1.6)
	_flock.alarm_transfer = rng.randf_range(0.80, 0.93)
	_flock.alarm_speed = rng.randf_range(0.25, 0.6)
	_flock.bank_gain = rng.randf_range(1.6, 3.4)
	_flock.bank_rate = rng.randf_range(4.0, 9.0)
	_flock.flap_lo = rng.randf_range(5.5, 8.0)
	_flock.flap_hi = _flock.flap_lo + rng.randf_range(2.5, 6.0)

	# The travelling roll wave. Its speed is sampled as a CROSSING TIME rather than as a
	# frequency, because what the eye reads is "the darkening got from one side of the
	# flock to the other in about half a second" - so the wavelength and the box width fix
	# the temporal frequency rather than the other way round.
	var wave_len := rng.randf_range(0.22, 0.55)
	var cross := rng.randf_range(0.35, 0.9)
	_flock.wave_k = TAU / wave_len
	_flock.wave_w = _flock.wave_k * (2.0 * _ext.x / cross)
	_flock.wave_amp = rng.randf_range(0.25, 0.75)
	_flock.wave_phase = rng.randf() * TAU
	var wa := rng.randf() * TAU
	_flock.wave_dir = Vector3(cos(wa), rng.randf_range(-0.35, 0.35), sin(wa)).normalized()

	# The weight point. Separation is fixed; the pair that walks is cohesion against
	# alignment, and the two endpoints are a BALL (all cohesion) and a RIBBON (all
	# alignment), sampled with enough slack that no two seeds walk the same segment.
	_sep = rng.randf_range(1.1, 2.1)
	_coh_a = rng.randf_range(0.9, 1.8)
	_ali_a = rng.randf_range(0.25, 0.7)
	_coh_b = rng.randf_range(0.15, 0.55)
	_ali_b = rng.randf_range(1.0, 2.0)
	_walk_gain = rng.randf_range(0.30, 0.85)
	_walk_cap = rng.randf_range(0.5, 1.3)
	_turn_cap = rng.randf_range(0.6, 1.8)
	_walk = rng.randf()
	_walk_dir = 1.0 if rng.randf() < 0.5 else -1.0
	_w_coh = lerpf(_coh_a, _coh_b, _walk)
	_w_ali = lerpf(_ali_a, _ali_b, _walk)

	_roost_home = Vector3(rng.randf_range(-0.16, 0.16), rng.randf_range(-0.14, 0.06),
		rng.randf_range(-0.14, 0.14))
	_roost_reach = Vector3(_ext.x * rng.randf_range(0.20, 0.46),
		_ext.y * rng.randf_range(0.14, 0.38), _ext.z * rng.randf_range(0.16, 0.40))
	_roost_pull = rng.randf_range(0.22, 0.70)
	_thermal = rng.randf_range(0.08, 0.30)
	_flow = Flow2D.new(rng.randi(), rng.randf_range(1.2, 2.6), rng.randf_range(0.02, 0.07))

	# --- the hawk ---------------------------------------------------------------------
	_hawk_present = rng.randf() < 0.7
	_hawk_cad = rng.randf_range(6.0, 16.0)
	_hawk_speed = rng.randf_range(0.45, 0.95)
	_flock.hawk_radius = rng.randf_range(0.07, 0.15)
	_flock.hawk_push = rng.randf_range(4.5, 10.0)
	var styles: Array = []
	for i in 5:
		var st := String(APPROACHES[rng.randi() % APPROACHES.size()])
		var side := 1.0 if rng.randf() < 0.5 else -1.0
		var start := Vector3.ZERO
		var dir := Vector3.ZERO
		var mult := 1.0
		match st:
			"stoop":
				start = Vector3(rng.randf_range(-0.5, 0.5) * _ext.x, -_ext.y * 1.6,
					side * _ext.z * 1.5)
				dir = Vector3(rng.randf_range(-0.3, 0.3), 1.0, -side * 1.3)
				mult = rng.randf_range(1.2, 1.7)
			"cross":
				start = Vector3(-side * _ext.x * 1.7, rng.randf_range(-0.4, 0.4) * _ext.y,
					rng.randf_range(-0.5, 0.5) * _ext.z)
				dir = Vector3(side * 1.0, rng.randf_range(-0.15, 0.15), rng.randf_range(-0.4, 0.4))
				mult = rng.randf_range(0.7, 1.0)
			"climb":
				start = Vector3(rng.randf_range(-0.5, 0.5) * _ext.x, _ext.y * 1.6,
					side * _ext.z * 1.4)
				dir = Vector3(rng.randf_range(-0.3, 0.3), -1.0, -side * 1.1)
				mult = rng.randf_range(0.9, 1.3)
			_:
				start = Vector3(-side * _ext.x * 1.6, rng.randf_range(-0.3, 0.3) * _ext.y,
					_ext.z * 1.5)
				dir = Vector3(side * 1.1, rng.randf_range(-0.2, 0.2), -1.2)
				mult = rng.randf_range(1.0, 1.4)
		_passes.append({"style": st, "start": start, "dir": dir.normalized(), "mult": mult})
		styles.append(st)

	# --- sky furniture -----------------------------------------------------------------
	var treeline := rng.randf() < 0.45
	if treeline:
		var nt := rng.randi_range(90, 190)
		_tree_h = rng.randf_range(0.025, 0.075)
		_tree_val = rng.randf_range(0.06, 0.24)
		_trees.resize(nt)
		# A ragged canopy, not a comb: two octaves of seeded bumps plus the odd tall one.
		var ph0 := rng.randf() * TAU
		var ph1 := rng.randf() * TAU
		var f0 := rng.randf_range(3.0, 7.0)
		var f1 := rng.randf_range(11.0, 23.0)
		for i in nt:
			var u := float(i) / float(nt)
			var h := 0.55 + 0.28 * sin(u * TAU * f0 + ph0) + 0.17 * sin(u * TAU * f1 + ph1)
			if rng.randf() < 0.06:
				h += rng.randf_range(0.25, 0.6)
			_trees[i] = clampf(h * rng.randf_range(0.82, 1.18), 0.12, 1.6)
	var clouded := rng.randf() < 0.6
	if clouded:
		add_layer("clouds", rng, {"z": "back", "count": rng.randi_range(3, 7),
			"hue": _sky_hue, "sat": rng.randf_range(0.03, 0.12),
			"val": rng.randf_range(0.85, 1.0), "alpha": rng.randf_range(0.05, 0.13)})
	var rayed := rng.randf() < 0.3
	if rayed:
		add_layer("rays", rng, {"z": "back", "count": rng.randi_range(3, 6),
			"hue": sch.accent, "alpha": rng.randf_range(0.03, 0.08)})

	# --- the silhouette -----------------------------------------------------------------
	var body := rng.randf_range(0.007, 0.014)
	var span := body * rng.randf_range(1.5, 2.6)
	var flap_amp := rng.randf_range(0.5, 1.1)

	_clock = SimClock.new(rng.randf_range(22.0, 30.0), 2)
	_flock.build(count, rng)

	_job = FlockJob.new()
	_job.flock = _flock
	_job.dt = _clock.dt
	_job.body_len = body
	_job.wing_span = span
	_job.flap_amp = flap_amp
	_job.bird_hue = fposmod(_sky_hue + rng.randf_range(-0.06, 0.06), 1.0)
	_job.bird_sat = rng.randf_range(0.10, 0.40)
	_job.dark_val = rng.randf_range(0.04, 0.16)
	_job.pale_val = _job.dark_val + rng.randf_range(0.18, 0.42)
	_job.alarm_hue = fposmod(sch.accent + rng.randf_range(-0.05, 0.05), 1.0)
	_job.alarm_sat = rng.randf_range(0.05, 0.35)
	_job.alarm_val = rng.randf_range(0.80, 1.0)
	_job.alpha_lo = rng.randf_range(0.20, 0.42)
	_job.alpha_hi = rng.randf_range(0.75, 1.0)
	_job.haze_near = _dist - _ext.z * 0.9
	_job.haze_far = _dist + _ext.z * 1.5
	_job.haze_lift = rng.randf_range(0.45, 0.85)
	_job.fov = _fov
	_job.setup()

	return {
		"birds": count, "mood": sch.name, "radius": _flock.radius,
		"sep_radius": _flock.sep_radius, "k_cap": _flock.k_cap,
		"scan_cap": _flock.scan_cap, "stride": _flock.query_stride,
		"sim_rate": _clock.rate, "speed": Vector2(_flock.speed_min, _flock.speed_max),
		"accel_max": _flock.accel_max, "box": _ext,
		"w_sep": _sep, "coh_span": Vector2(_coh_a, _coh_b),
		"ali_span": Vector2(_ali_a, _ali_b), "turn_cap": _turn_cap,
		"roost_pull": _roost_pull, "thermal": _thermal,
		"hawk": _hawk_present, "hawk_cadence": _hawk_cad, "hawk_speed": _hawk_speed,
		"approaches": styles, "wave_amp": _flock.wave_amp, "wave_cross": cross,
		"wave_len": wave_len, "body": body, "wing_span": span, "flap_amp": flap_amp,
		"flap_rate": Vector2(_flock.flap_lo, _flock.flap_hi),
		"horizon": _horizon, "horizon_val": _horizon_val, "zenith_val": _zenith_val,
		"ground_val": _ground_val, "clouds": clouded, "rays": rayed, "treeline": treeline,
		"fov": _fov, "dist": _dist,
	}


func update(f: AudioFeatures, delta: float) -> void:
	tick(f, delta)
	# A field scene, and a very wide one: the camera barely moves, because the flock is
	# already doing all the moving there is.
	drift_view(f, 0.012, 0.02)
	update_layers(f, delta)
	_ch = chroma_hue()
	_glow = Nonlinear.flare(_glow, clampf(f.energy * 1.8 + 0.4 * f.beat, 0.0, 1.0),
		delta, 5.0, 0.9)

	# --- the weight walk ----------------------------------------------------------------
	# The point moves along its segment at a speed set by how much the music is CHANGING,
	# not by how loud it is, and turns round at the ends rather than being re-drawn - an
	# rng draw on an audio-conditioned event would put the export on a different path from
	# the preview for the rest of the song.
	var speed := clampf(_walk_gain * (f.movement + 14.0 * f.flux), 0.0, _walk_cap)
	_walk += speed * _walk_dir * delta
	if _walk >= 1.0:
		_walk = 1.0
		_walk_dir = -1.0
	elif _walk <= 0.0:
		_walk = 0.0
		_walk_dir = 1.0
	var coh_t := lerpf(_coh_a, _coh_b, _walk) * (0.45 + 1.25 * clampf(f.bass * 1.5, 0.0, 1.0))
	var ali_t := lerpf(_ali_a, _ali_b, _walk) * (0.40 + 1.30 * clampf((f.high + f.treble) * 1.7, 0.0, 1.0))
	# The rate limit is the whole character of the motion: without it the flock snaps
	# between ball and ribbon on every transient and reads as a switch being thrown.
	_w_coh = move_toward(_w_coh, coh_t, _turn_cap * delta)
	_w_ali = move_toward(_w_ali, ali_t, _turn_cap * delta)
	# Everything the audio steers is written onto the JOB, never onto the solver: the worker
	# may be mid-tick, and a weight changing under a half-finished generation is exactly the
	# kind of race that shows up once an hour and never reproduces. run() copies them across
	# before it steps anything.
	_job.w_sep = _sep
	_job.w_coh = _w_coh
	_job.w_ali = _w_ali
	_job.w_roost = _roost_pull
	# THE BOX FOLLOWS THE CAMERA, and only ever outward: shrinking it would put birds through a
	# wall that moved onto them. Through the job like every other steered value, because the
	# solver may be mid-tick on a worker.
	var need := view_half() * _wall_scale()
	_ext.x = clampf(maxf(_ext.x, need.x), 0.0, _flock.ext_max.x)
	_ext.y = clampf(maxf(_ext.y, need.y), 0.0, _flock.ext_max.y)
	_job.box_ext = _ext
	# Negative y is up here, so a rising thermal is a negative bias.
	_job.thermal = -_thermal * (0.25 + 1.4 * clampf(f.bass * 1.4, 0.0, 1.0))

	# --- the roost drifts on the wind ----------------------------------------------------
	# One Flow2D sample a frame moves the attractor the whole flock hangs off, which reads
	# as the flock riding a wind without any bird having to sample a field.
	_flow.advance(delta)
	var w := _flow.at(Vector2(_roost_home.x, _roost_home.y) + Vector2(0.3, 0.0))
	var w2 := _flow.at(Vector2(_roost_home.z, _roost_home.y) - Vector2(0.25, 0.1))
	_job.roost = _roost_home + Vector3(
		w.x * _roost_reach.x, w.y * _roost_reach.y, w2.x * _roost_reach.z)

	# --- the hawk -------------------------------------------------------------------------
	# The strike is the ONLY audio event in the scene, and it is one-shot: the beat launches
	# it and then has no further say. Everything after - the hole opening, the alarm running
	# outward, the flock closing behind - is the solver's own business, which is what makes
	# the wave outlive the transient that caused it.
	_hawk_arm += delta
	var beat_up: bool = f.beat > 0.55 and _beat_prev <= 0.55
	_beat_prev = f.beat
	if _hawk_left > 0.0:
		_hawk_left -= delta
		_hawk_pos += _hawk_vel * delta
		_job.hawk = _hawk_pos
		_job.hawk_vel = _hawk_vel
		_job.hawk_on = _hawk_left > 0.0
	elif _hawk_present and beat_up and _hawk_arm >= _hawk_cad and not _passes.is_empty():
		# WHICH pass is an index step, never a draw. Same reason as the walk above.
		var pass_d: Dictionary = _passes[_hawk_i % _passes.size()]
		_hawk_i += 1
		_hawk_arm = 0.0
		var st: Vector3 = pass_d["start"]
		var dr: Vector3 = pass_d["dir"]
		var mult: float = pass_d["mult"]
		_hawk_pos = st
		_hawk_vel = dr * (_hawk_speed * mult)
		var travel := st.length() * 2.2
		_hawk_left = travel / maxf(0.05, _hawk_vel.length())
		_job.hawk = _hawk_pos
		_job.hawk_vel = _hawk_vel
		_job.hawk_on = true

	# --- camera ----------------------------------------------------------------------------
	# Held nearly still, with a slow lateral breath; the flock supplies the motion, and a
	# camera that also swings makes the picture unreadable.
	_job.eye = Vector3(0.10 * mod.value("panx"), _eye_lift + 0.05 * mod.value("pany"), _dist)
	_job.look = Vector3(0.04 * mod.value("look"), 0.02 * mod.value("tiltc"), 0.0)
	_job.fov = _fov

	# --- clock --------------------------------------------------------------------------
	var n := _clock.ticks(delta)
	if n > 0:
		_since_tick = 0.0
		_job.pending = mini(_job.pending + n, MAX_PENDING)
	else:
		_since_tick += delta
	# A one-tick lag, interpolated: at a 25 Hz solver that is 40 ms behind, which nobody
	# sees, and it is the difference between smooth flight and a 25 fps flicker.
	_job.blend = clampf(_since_tick / _clock.dt, 0.0, 1.0)
	_job.unit_px = unit()
	_forge.kick(_job.run, {}, self, _job)
	queue_redraw()


## How far out a wall has to sit, per unit of visible half-extent, for the birds to turn OFF
## SCREEN. A bird at the box edge sits at the nearest face of the box, so its projected offset is
## `focal * ext / (dist - ext.z)`; inverting that gives the box extent a given screen extent
## demands, and the margin puts the turn safely outside the frame rather than on its edge.
func _wall_scale() -> float:
	var focal := 1.0 / tan(deg_to_rad(clampf(_fov, 10.0, 170.0)) * 0.5)
	return WALL_OUT * maxf(0.25, _dist - _ext.z) / focal


func _draw() -> void:
	begin_draw()
	# What the camera can see, not the viewport times a constant - dividing by the zoom covers a
	# pull-back and nothing else. See [method SceneView.visible_half].
	var hpx := view_half_px()
	var hx := hpx.x
	var hy := hpx.y
	var tint := _ch.y * 0.45

	# The ground is the pale air at the horizon, flat and full-bleed. `bed` cannot be this
	# - it vignettes, and its brightest possible pixel is a mid tone - and a dusk sky that
	# darkens at its corners reads as a spotlight on paper rather than as sky.
	paint_ground(_sky_hue, _horizon_sat, clampf(_horizon_val + 0.06 * _glow, 0.0, 1.0),
		tint, _ch.x)

	# The vertical ramp: pale at the horizon, deepening upward. Painted as a small stack of
	# opaque strips rather than as a gradient texture, because a dozen quads through
	# [TriBatch] is one draw call and needs no resource.
	var yh := hy * (2.0 * _horizon - 1.0)
	var top := -hy
	var span := yh - top
	if span > 1.0:
		var dh := fposmod(_sky_hue_top - _sky_hue + 0.5, 1.0) - 0.5
		for i in _bands:
			var t0 := float(i) / float(_bands)
			var t1 := float(i + 1) / float(_bands)
			var y0 := top + span * t0
			var y1 := top + span * t1
			var c0 := _sky_at(1.0 - t0, dh, tint)
			var c1 := _sky_at(1.0 - t1, dh, tint)
			_tb.quad_colored(
				PackedVector2Array([Vector2(-hx, y0), Vector2(hx, y0),
					Vector2(hx, y1), Vector2(-hx, y1)]),
				PackedColorArray([c0, c0, c1, c1]))
	# The land, as a short ramp off the horizon. Four strips is enough because it occupies
	# a tenth of the frame at most and is mostly hidden behind the treeline when there is
	# one; its whole job is to stop the bottom of the picture reading as more sky.
	var gspan := hy - yh
	if gspan > 1.0:
		for i in 4:
			var g0 := float(i) / 4.0
			var g1 := float(i + 1) / 4.0
			var ga := _ground_color(g0)
			var gb := _ground_color(g1)
			_tb.quad_colored(
				PackedVector2Array([Vector2(-hx, yh + gspan * g0), Vector2(hx, yh + gspan * g0),
					Vector2(hx, yh + gspan * g1), Vector2(-hx, yh + gspan * g1)]),
				PackedColorArray([ga, ga, gb, gb]))
	_tb.flush(self)

	draw_layers("back")

	# A distant treeline, when the seed drew one: the flock reads as being over LAND rather
	# than in a void, and the birds crossing it are the only depth cue that costs nothing.
	if not _trees.is_empty():
		var nt := _trees.size()
		var cw := hx * 2.0 / float(nt)
		var base := clampf(_tree_val + 0.10 * _glow, 0.0, 1.0)
		var col := Color.from_hsv(fposmod(_sky_hue + 0.03, 1.0),
			clampf(_horizon_sat + 0.18, 0.0, 1.0), base, 0.92)
		for i in nt:
			var x0 := -hx + cw * float(i)
			var th := _trees[i] * _tree_h * hy * 2.0
			_tb.quad(Vector2(x0, yh - th), Vector2(x0 + cw * 1.02, yh - th),
				Vector2(x0 + cw * 1.02, hy), Vector2(x0, hy), col)
		_tb.flush(self)

	# The flock: simulated AND batched off-thread, submitted here in microseconds.
	_forge.submit(self)
	draw_layers("front")


## The sky colour at height [param t] (0 at the horizon, 1 at the zenith), dragged toward
## the music's tonal centre by [param tint]. Saturation climbs and value falls with height,
## which is what makes a dusk sky read as depth rather than as a two-colour gradient.
func _sky_at(t: float, dh: float, tint: float) -> Color:
	var e := t * t * (3.0 - 2.0 * t)                 # smoothstep: the band edges vanish
	var h := fposmod(_sky_hue + dh * e, 1.0)
	if tint > 0.001:
		var d := _ch.x - h
		d = d - round(d)
		h = fposmod(h + d * tint * 0.5, 1.0)
	return Color.from_hsv(h,
		clampf(lerpf(_horizon_sat, _zenith_sat, e), 0.0, 1.0),
		clampf(lerpf(_horizon_val, _zenith_val, e) + 0.05 * _glow * (1.0 - e), 0.0, 1.0))


## The land at [param t] below the horizon (0 at the horizon, 1 at the frame's bottom edge).
func _ground_color(t: float) -> Color:
	return Color.from_hsv(fposmod(_sky_hue + 0.02, 1.0),
		clampf(lerpf(_horizon_sat, _ground_sat, t), 0.0, 1.0),
		clampf(lerpf(_horizon_val * 0.94, _ground_val, t) + 0.04 * _glow, 0.0, 1.0))


## The off-thread job: it owns the [Boids] solver, advances it on the pending [SimClock]
## ticks, and batches every bird into one triangle buffer - all on the worker, so the
## scene never touches a bird and a Director cut mid-build is harmless.
##
## The triangle layout is FIXED - three triangles, nine vertices, per bird, always, with a
## culled bird written as degenerate zero-alpha geometry rather than skipped. Two things
## follow. The index buffer is built ONCE at setup and never touched again, since it is
## just 0..9n-1 and the triangle count never changes. And the vertex and colour buffers
## are written BY INDEX rather than appended to, so a bird costs no GDScript method calls
## at all - where [TriBatch]'s tri() would be three calls and eighteen appends each, which
## at two thousand birds a frame is the whole budget.
class FlockJob:
	extends RefCounted

	var flock: Boids
	var lens := Lens3D.new()
	var eye := Vector3(0, 0, 2.0)
	var look := Vector3.ZERO
	var fov := 46.0
	var pending := 0
	var dt := 1.0 / 25.0
	var blend := 0.0
	var unit_px := 720.0

	# Steering, staged here by the scene and copied onto the solver at the top of run().
	var w_sep := 1.5
	var box_ext := Vector3.ZERO
	var w_coh := 0.9
	var w_ali := 1.0
	var w_roost := 0.45
	var roost := Vector3.ZERO
	var thermal := 0.0
	var hawk_on := false
	var hawk := Vector3.ZERO
	var hawk_vel := Vector3.ZERO

	var body_len := 0.010
	var wing_span := 0.020
	var flap_amp := 0.8
	var bird_hue := 0.58
	var bird_sat := 0.20
	var dark_val := 0.09
	var pale_val := 0.34
	var alarm_hue := 0.10
	var alarm_sat := 0.20
	var alarm_val := 0.95
	var alpha_lo := 0.30
	var alpha_hi := 0.90
	var haze_near := 1.0
	var haze_far := 3.0
	var haze_lift := 0.65

	var _idx := PackedInt32Array()
	var _verts := 0
	var _sx := PackedFloat32Array()
	var _sy := PackedFloat32Array()
	var _hx := PackedFloat32Array()
	var _hy := PackedFloat32Array()
	var _dep := PackedFloat32Array()
	var _keys := PackedInt64Array()

	## Allocate the scratch once. The index buffer is the only thing that survives between
	## builds: it is shared into every packet and never written after this, so the main
	## thread submitting an old packet can never see a torn one. The vertex and colour
	## buffers are minted fresh per build for the same reason - the previous packet is
	## still in flight and must not be rewritten under it, and a resize of a few hundred
	## kilobytes is a memset, far below the cost of the geometry that fills it.
	func setup() -> void:
		var n := flock.n
		# One extra slot past the flock: the hawk, drawn last so it is never buried by the
		# birds it is scattering.
		_verts = (n + 1) * 9
		_idx.resize(_verts)
		for i in _verts:
			_idx[i] = i
		_sx.resize(n)
		_sy.resize(n)
		_hx.resize(n)
		_hy.resize(n)
		_dep.resize(n)
		_keys.resize(n)

	func run(_s: Dictionary) -> Array:
		var n := pending
		pending -= n          # consume; a concurrent increment losing a race costs one tick
		flock.w_sep = w_sep
		if box_ext.x > 0.0:
			flock.ext = box_ext
		flock.w_coh = w_coh
		flock.w_ali = w_ali
		flock.w_roost = w_roost
		flock.roost = roost
		flock.thermal = thermal
		flock.hawk_on = hawk_on
		var hp := hawk
		for k in n:
			# The hawk flies on SIM time inside the loop, so a build that runs two ticks
			# does not have it teleport - the hole it opens is a continuous furrow.
			flock.hawk = hp
			flock.step(dt)
			hp += hawk_vel * dt

		lens.eye = eye
		lens.look = look
		lens.fov = fov
		lens.prepare()
		var vf := (look - eye).normalized()
		var focal := 1.0 / tan(deg_to_rad(fov) * 0.5)
		var u := unit_px
		var nb := flock.n
		var bl := blend
		# Local aliases for the hot loops. A packed array reached through `flock.px[i]` is
		# a property lookup and a Variant round trip per access, and there are a dozen per
		# bird per pass; aliasing costs one refcount each and nothing else, since neither
		# pass ever writes to them.
		var fqx := flock.qx
		var fqy := flock.qy
		var fqz := flock.qz
		var fpx := flock.px
		var fpy := flock.py
		var fpz := flock.pz
		var fvx := flock.vx
		var fvy := flock.vy
		var fvz := flock.vz
		var fbank := flock.bank
		var fflap := flock.flap
		var falarm := flock.alarm

		# --- pass 1: project the body axis, and key each bird by depth -------------------
		for i in nb:
			var wx := lerpf(fqx[i], fpx[i], bl)
			var wy := lerpf(fqy[i], fpy[i], bl)
			var wz := lerpf(fqz[i], fpz[i], bl)
			var vxi := fvx[i]
			var vyi := fvy[i]
			var vzi := fvz[i]
			var sp := sqrt(vxi * vxi + vyi * vyi + vzi * vzi)
			if sp < 1e-9:
				sp = 1e-9
			var c := lens.project(Vector3(wx, wy, wz))
			_sx[i] = c.x
			_sy[i] = c.y
			_dep[i] = c.z
			var h := lens.project(Vector3(wx + vxi / sp * body_len,
				wy + vyi / sp * body_len, wz + vzi / sp * body_len))
			_hx[i] = h.x
			_hy[i] = h.y
			# Painter key, packed natively: quantized depth in the high bits, index in the
			# low 22, so the whole ordering is one sort of a PackedInt64Array. A list of
			# Dictionaries through sort_custom would be thousands of allocations and
			# hundreds of thousands of interpreted comparisons every frame.
			var q := int(clampf(c.z, -100.0, 100.0) * 16384.0)
			_keys[i] = (-q << 22) | i

		_keys.sort()

		var pts := PackedVector2Array()
		var cols := PackedColorArray()
		pts.resize(_verts)
		cols.resize(_verts)

		var near := lens.near
		var d_span := maxf(0.001, haze_far - haze_near)

		# --- pass 2: three triangles per bird, far to near -------------------------------
		for rank in nb:
			var i := int(_keys[rank] & 0x3FFFFF)
			var o := rank * 9
			var depth := _dep[i]
			var cx := _sx[i] * u
			var cy := _sy[i] * u
			if depth <= near or absf(_sx[i]) > 1.8 or absf(_sy[i]) > 1.8:
				var z := Vector2(cx, cy)
				var clear := Color(0, 0, 0, 0)
				for k in 9:
					pts[o + k] = z
					cols[o + k] = clear
				continue

			var scale := focal / depth * u
			var dx := _hx[i] * u - cx
			var dy := _hy[i] * u - cy
			var dl := sqrt(dx * dx + dy * dy)
			var ux := 1.0
			var uy := 0.0
			if dl > 1e-6:
				ux = dx / dl
				uy = dy / dl
			var body := maxf(dl, 1.4)
			var px_ := -uy
			var py_ := ux

			# How much of the bird's heading points down the view axis. It sets the
			# foreshortening of the circle the wingtips sweep, and with the roll angle it
			# decides how much wing area the lens actually sees.
			var vxi2 := fvx[i]
			var vyi2 := fvy[i]
			var vzi2 := fvz[i]
			var spd := sqrt(vxi2 * vxi2 + vyi2 * vyi2 + vzi2 * vzi2)
			if spd < 1e-9:
				spd = 1e-9
			var along := absf((vxi2 * vf.x + vyi2 * vf.y + vzi2 * vf.z) / spd)

			var roll := fbank[i]
			var fl := flap_amp * sin(fflap[i])
			var tl := roll + fl
			var tr := roll + PI - fl
			var span := wing_span * scale
			var cl := cos(tl)
			var sl := sin(tl)
			var cr := cos(tr)
			var sr := sin(tr)
			# The wingtips sweep a circle in the plane normal to the heading; projected,
			# that circle is an ellipse with a full-length axis across the screen heading
			# and an axis along it foreshortened by exactly `along`. The length of the
			# projected span IS how much wing the lens sees - which is where the dark
			# sheets and the pale hazes come from, and it is geometry, not audio.
			var lx := cx + px_ * (span * cl) + ux * (span * along * sl)
			var ly := cy + py_ * (span * cl) + uy * (span * along * sl)
			var rx := cx + px_ * (span * cr) + ux * (span * along * sr)
			var ry := cy + py_ * (span * cr) + uy * (span * along * sr)

			var cb := cos(roll)
			var sb := sin(roll)
			var seen := sqrt(cb * cb + along * along * sb * sb)
			var haze := clampf((depth - haze_near) / d_span, 0.0, 1.0)
			var al := falarm[i]
			var v := lerpf(pale_val, dark_val, seen)
			v = lerpf(v, 1.0, haze * haze_lift)
			var sat := bird_sat
			var hue := bird_hue
			var a := lerpf(alpha_lo, alpha_hi, seen) * (1.0 - 0.72 * haze)
			if al > 0.01:
				var d := alarm_hue - hue
				d = d - round(d)
				hue = fposmod(hue + d * al, 1.0)
				v = lerpf(v, alarm_val, al)
				sat = lerpf(sat, alarm_sat, al)
				a = clampf(a + 0.30 * al, 0.0, 1.0)
			var col := Color.from_hsv(fposmod(hue, 1.0), clampf(sat, 0.0, 1.0),
				clampf(v, 0.0, 1.0), clampf(a, 0.0, 1.0))

			var hd := Vector2(cx + ux * body * 0.62, cy + uy * body * 0.62)
			var tw := maxf(body * 0.16, 0.6)
			var t1 := Vector2(cx - ux * body * 0.38 + px_ * tw, cy - uy * body * 0.38 + py_ * tw)
			var t2 := Vector2(cx - ux * body * 0.38 - px_ * tw, cy - uy * body * 0.38 - py_ * tw)
			var ra := Vector2(cx + ux * body * 0.16, cy + uy * body * 0.16)
			var rb := Vector2(cx - ux * body * 0.16, cy - uy * body * 0.16)

			pts[o] = hd
			pts[o + 1] = t1
			pts[o + 2] = t2
			pts[o + 3] = ra
			pts[o + 4] = rb
			pts[o + 5] = Vector2(lx, ly)
			pts[o + 6] = ra
			pts[o + 7] = rb
			pts[o + 8] = Vector2(rx, ry)
			for k in 9:
				cols[o + k] = col

		# --- the hawk, in the reserved last slot ------------------------------------------
		# Same three triangles as a bird, at two and a half times the span and with the
		# wings HELD rather than beating - a hunting bird does not flap, and that alone
		# separates it from the thousand animals it is flying through.
		var ho := nb * 9
		var drawn := false
		if hawk_on:
			var hc := lens.project(hawk)
			if hc.z > near and absf(hc.x) < 1.8 and absf(hc.y) < 1.8:
				var hv := hawk_vel
				if hv.length_squared() < 1e-12:
					hv = Vector3(0, 0, -1)
				hv = hv.normalized()
				var hh := lens.project(hawk + hv * body_len * 3.0)
				var hcx := hc.x * u
				var hcy := hc.y * u
				var hdx := hh.x * u - hcx
				var hdy := hh.y * u - hcy
				var hdl := sqrt(hdx * hdx + hdy * hdy)
				var hux := 1.0
				var huy := 0.0
				if hdl > 1e-6:
					hux = hdx / hdl
					huy = hdy / hdl
				var hb := maxf(hdl, 3.0)
				var hpx := -huy
				var hpy := hux
				var hspan := wing_span * 2.4 * (focal / hc.z * u)
				var swept := hb * 0.32
				var hcol := Color.from_hsv(fposmod(bird_hue, 1.0),
					clampf(bird_sat * 1.3, 0.0, 1.0), clampf(dark_val * 0.55, 0.0, 1.0), 0.96)
				pts[ho] = Vector2(hcx + hux * hb * 0.68, hcy + huy * hb * 0.68)
				pts[ho + 1] = Vector2(hcx - hux * hb * 0.42 + hpx * hb * 0.20,
					hcy - huy * hb * 0.42 + hpy * hb * 0.20)
				pts[ho + 2] = Vector2(hcx - hux * hb * 0.42 - hpx * hb * 0.20,
					hcy - huy * hb * 0.42 - hpy * hb * 0.20)
				pts[ho + 3] = Vector2(hcx + hux * hb * 0.20, hcy + huy * hb * 0.20)
				pts[ho + 4] = Vector2(hcx - hux * hb * 0.20, hcy - huy * hb * 0.20)
				pts[ho + 5] = Vector2(hcx + hpx * hspan - hux * swept,
					hcy + hpy * hspan - huy * swept)
				pts[ho + 6] = Vector2(hcx + hux * hb * 0.20, hcy + huy * hb * 0.20)
				pts[ho + 7] = Vector2(hcx - hux * hb * 0.20, hcy - huy * hb * 0.20)
				pts[ho + 8] = Vector2(hcx - hpx * hspan - hux * swept,
					hcy - hpy * hspan - huy * swept)
				for k2 in 9:
					cols[ho + k2] = hcol
				drawn = true
		if not drawn:
			for k3 in 9:
				pts[ho + k3] = Vector2.ZERO
				cols[ho + k3] = Color(0, 0, 0, 0)

		return [{"pts": pts, "cols": cols, "idx": _idx}]
