extends GhostScene

## Filaments - the procedural-growth primitive, showcased.
##
## One scene, three lives chosen by seed, all the same [Filament] mechanism on a
## curl-noise [Flow2D] field - proof that the primitive composes:
##   lightning - arcs that strike on the beat: a jagged forked path floods in fast,
##               blazes, then fades, and re-strikes on the next hit (slow-motion bolts).
##   neural    - tendrils that creep from scattered seeds and slowly regrow, a living
##               network coiling through the frame.
##   thread    - long smooth threads flowing across on the flow, barely branching.
## Audio drives the strikes / growth surge and the brightness; nonlinearity (the
## flow's meander, the spike-shaped drive, the asymmetric flare) is what animates it.

const MODES := {
	"lightning": {"variant": "lightning", "count_lo": 4, "count_hi": 7, "grow": 1.6,
		"fade": 1.1, "strike": true, "hue": 0.60, "sat": 0.35, "w_lo": 3.0, "w_hi": 5.0,
		"len_lo": 0.55, "len_hi": 0.85, "evolve": 0.10, "jitter": 0.0, "cluster": 0.55},
	"neural": {"variant": "tendril", "count_lo": 7, "count_hi": 12, "grow": 0.5,
		"fade": 0.0, "strike": false, "hue": 0.75, "sat": 0.7, "w_lo": 3.0, "w_hi": 6.0,
		"len_lo": 0.35, "len_hi": 0.55, "evolve": 0.05, "jitter": 0.012, "cluster": 0.55},
	"thread": {"variant": "thread", "count_lo": 5, "count_hi": 9, "grow": 0.42,
		"fade": 0.0, "strike": false, "hue": 0.50, "sat": 0.6, "w_lo": 2.0, "w_hi": 4.0,
		"len_lo": 0.6, "len_hi": 0.95, "evolve": 0.08, "jitter": 0.006, "cluster": 0.4},
}

var _f: AudioFeatures = AudioFeatures.new()
var _rng := RandomNumberGenerator.new()
var _flow: Flow2D
var _cfg: Dictionary
var _mode := "lightning"
var _fils: Array = []
var _hue := 0.0
var _glow := 0.0
var _spark := 0.0            # a rare SPONTANEOUS activation pulse - keeps the scene alive in dead air
var _beat_prev := 0.0
var _loud := 0.0             # lightning: smoothed loudness (energy) - the audio-liveness gate
var _flux_prev := 0.0        # lightning: smoothed spectral flux, for onset (transient) detection
var _life_alpha := 1.0       # set per-filament before its draw, read by _color_for
var _ever_alive := false     # lightning: has the audio-liveness gate ever opened yet
var _strike_acc := 0.0       # lightning: time since the last strike
var _strike_period := 1.6    # lightning: fallback re-strike cadence (sampled), so it
                             # strikes even with no detectable beats - never pure black
var _converge := Vector2(0.0, 0.2)   # lightning: the point the spread bolts strike toward
var _strike_mode := false    # true for the lightning variant (per-depth fade + channels)
var _cur_fade := 1.0         # the bolt being drawn: its life (set before draw_growing)
var _cur_maxd := 1           # the bolt being drawn: its deepest branch level
var _cur_persist := false    # the bolt being drawn: is a persistent channel


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	_rng.seed = rng.randi()
	var keys := MODES.keys()
	_mode = keys[rng.randi() % keys.size()]
	_cfg = MODES[_mode]
	_strike_mode = bool(_cfg.strike)
	# Lightning converges on a point: the spread bolts all strike toward it.
	_converge = Vector2(rng.randf_range(-0.35, 0.35), rng.randf_range(0.0, 0.45))
	_hue = fposmod(float(_cfg.hue) + rng.randf_range(-0.08, 0.08), 1.0)
	_flow = Flow2D.new(rng.randi(), rng.randf_range(2.0, 3.5), float(_cfg.evolve))
	_strike_period = rng.randf_range(1.1, 2.0)
	var count := rng.randi_range(int(_cfg.count_lo), int(_cfg.count_hi))
	for i in count:
		var fil := {"fil": null, "grown": 0.0, "life": 0.0, "mature": 0.0,
			"origin": Vector2.ZERO, "heading": 0.0, "active": false,
			"state": "grow", "timer": 0.0, "rate": 1.0, "hold": 2.0, "mode": "fade", "retract_to": 0.0,
			"delay": 0.0,         # lightning: ignition stagger, counted down before the bolt grows
			"persist": false, "rest_floor": 0.0,   # lightning: a persistent channel + its rest glow
			"far": false, "reach": 1.0}   # spawns from well outside the frame, with a longer reach
		_seed_path(fil, i, count)
		_regrow(fil)
		if not bool(_cfg.strike):
			# Continuous modes run a staggered, rate-varied lifecycle (see _update_continuous).
			fil.life = 1.0
			fil.grown = rng.randf_range(0.0, 1.0)   # start anywhere in the lifecycle (async)
			fil.rate = _gauss_rate()
			fil.hold = rng.randf_range(1.6, 4.5)
			fil.mode = _roll_mode()
		_fils.append(fil)
	# Mark some bolts as persistent CHANNELS: their base never fully fades and the next
	# strike re-ionises the SAME path (real lightning re-uses its route). The rest are
	# transient forks that come and go. Guarantee at least one channel. NO opening strike is
	# seeded any more: lightning scatters from real audio energy (see the loudness gate in
	# _update_strikes), so a silent intro / long fade-in stays dark instead of flashing out
	# of dead air.
	if bool(_cfg.strike):
		var any_persist := false
		for fil in _fils:
			if _rng.randf() < 0.45:
				fil.persist = true
				fil.rest_floor = _rng.randf_range(0.10, 0.16)
				any_persist = true
		if not any_persist and not _fils.is_empty():
			_fils[0].persist = true
			_fils[0].rest_floor = _rng.randf_range(0.10, 0.16)
	return {}


# Where a path starts and which way it heads, by mode.
func _seed_path(fil: Dictionary, i: int, count: int) -> void:
	match _mode:
		"lightning":
			# Spawn in several places around the upper arc, then CONVERGE: each bolt heads
			# toward a shared strike point, so they fan in instead of falling in parallel.
			fil.origin = Vector2(_rng.randf_range(-0.92, 0.92), _rng.randf_range(-0.72, -0.30))
			# Some bolts come from the DISTANCE: pushed well outside the frame along their own
			# line to the strike point, with a longer reach, so they streak in already full
			# rather than sprouting from just past the edge - the world extends past the view.
			if _rng.randf() < 0.45:
				var dir: Vector2 = (Vector2(fil.origin) - _converge).normalized()
				if dir.length() < 0.5:
					dir = Vector2(0, -1)
				fil.origin = _converge + dir * _rng.randf_range(1.4, 2.7)
				fil.far = true
				fil.reach = _rng.randf_range(1.8, 3.0)
			fil.heading = (_converge - fil.origin).angle() + _rng.randf_range(-0.22, 0.22)
		"thread":
			# Lanes spread across the full height so the threads fill the frame.
			fil.origin = Vector2(_rng.randf_range(-0.7, -0.45), _rng.randf_range(-0.62, 0.62))
			if _rng.randf() < 0.45:                                # some flow in from far off-left
				fil.origin.x = _rng.randf_range(-2.4, -1.4)
				fil.far = true
				fil.reach = _rng.randf_range(1.8, 3.0)
			fil.heading = _rng.randf_range(-0.3, 0.3)              # rightward flow
		_:  # neural - scattered seeds spread across the frame, any heading
			fil.origin = Vector2(_rng.randf_range(-0.62, 0.62), _rng.randf_range(-0.6, 0.6))
			fil.heading = TAU * float(i) / float(count) + _rng.randf_range(-0.5, 0.5)
			if _rng.randf() < 0.35:                                # some creep in from outside
				var ang := _rng.randf() * TAU
				fil.origin = Vector2(cos(ang), sin(ang)) * _rng.randf_range(1.2, 2.2)
				fil.far = true
				fil.reach = _rng.randf_range(1.6, 2.6)
				fil.heading = (-fil.origin).angle() + _rng.randf_range(-0.7, 0.7)   # head inward


func _regrow(fil: Dictionary) -> void:
	# A distant filament reaches further (a longer path with more segments) so it spans the
	# gap from off-screen into the frame and arrives already developed.
	var reach: float = float(fil.get("reach", 1.0))
	var steps := int(round(_rng.randi_range(10, 18) * clampf(reach, 1.0, 3.0)))
	var length := _rng.randf_range(float(_cfg.len_lo), float(_cfg.len_hi)) * reach
	var width := _rng.randf_range(float(_cfg.w_lo), float(_cfg.w_hi))
	fil.heading += _rng.randf_range(-0.4, 0.4)
	fil.fil = Filament.grow(String(_cfg.variant), fil.origin, fil.heading, length, width,
		steps, _flow, _rng, float(_cfg.get("cluster", 0.0)))
	fil.grown = 0.0
	fil.mature = 0.0


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.03, 0.05)
	_flow.advance(delta)
	# Spontaneous activation: a RARE nonlinear self-ignition (independent of the audio), so the scene
	# never freezes into dead space when the song fades out - there is always the odd flicker of life.
	# Deliberately low-chance and impulsive (not the constant fallback that used to strobe the intro).
	_spark = maxf(0.0, _spark - delta * 1.3)
	var spont := false
	if _rng.randf() < 0.22 * delta:
		_spark = _rng.randf_range(0.55, 1.0)
		spont = true
	_glow = Nonlinear.flare(_glow, clampf(0.3 * f.energy + 0.7 * f.beat + 0.6 * _spark, 0.0, 1.0), delta, 9.0, 1.6)
	var beat_edge: bool = f.beat > 0.55 and _beat_prev <= 0.55
	_beat_prev = f.beat

	var grow := float(_cfg.grow)
	if bool(_cfg.strike):
		_update_strikes(f, delta, grow, beat_edge, spont)
	else:
		_update_continuous(f, delta, grow, spont)
	queue_redraw()


# Lightning: bolts strike on beats and on harmonic transients, flood in, blaze, then fade
# out. They strike ONLY when the audio is actually sounding - a smoothed-loudness gate keeps
# the scene dark through a silent intro / fade-in, so bolts scatter from real spectral energy
# rather than out of dead air. While the audio is alive, a fallback cadence still flickers
# the bolts and a strike is forced if none are lit, so loud passages are never black.
func _update_strikes(f: AudioFeatures, delta: float, grow: float, beat_edge: bool, spont: bool) -> void:
	var drive := 0.5 + 1.5 * Nonlinear.apply("spike", clampf(0.6 * f.energy + f.beat + _spark, 0.0, 1.0), 2.0)
	# Audio-liveness gate. `energy` is the overall loudness proxy (the per-band levels are a
	# normalised spectral shape, so they read high even in near-silence and can't gate). Smooth
	# it so a fade-in ramps in gradually, and treat sub-threshold loudness as dead air.
	_loud = lerpf(_loud, f.energy, 1.0 - exp(-5.0 * delta))
	var alive := _loud > 0.13
	if alive:
		_ever_alive = true
	# A spectral-flux onset (a harmonic transient) scatters a fresh strike while alive.
	var onset := alive and f.flux > 0.02 and f.flux > _flux_prev * 1.5
	_flux_prev = lerpf(_flux_prev, f.flux, 0.35)
	_strike_acc += delta
	var any_active := false
	for fil in _fils:
		if fil.active:
			any_active = true
	# Beats fire whenever the beat detector triggers (it only fires on real audio anyway). The
	# fallback cadence and the never-black backstop are gated on `alive`, so silence stays dark.
	# `spont` = the rare spontaneous activation: strike even in dead air, so a silent OUTRO still
	# flickers - but only once the song has actually been alive at least once, so an intro's
	# fade-in never scatters a bolt out of dead air before there is any real sound to react to.
	var trigger := beat_edge or onset or (spont and _ever_alive) \
		or (alive and (_strike_acc >= _strike_period or not any_active))
	if trigger:
		_strike_acc = 0.0
		_strike_some((alive or spont) and not any_active)
	for fil in _fils:
		if not fil.active:
			continue
		# Honor the per-bolt ignition stagger: a struck bolt holds dark until its delay
		# elapses, so the cluster lights as a connected cascade, not one global flash.
		if fil.delay > 0.0:
			fil.delay = maxf(0.0, fil.delay - delta)
			continue
		# Per-bolt rate breaks the lockstep advance (the "single global oscillation").
		fil.grown = minf(1.0, fil.grown + delta * grow * drive * float(fil.rate))
		if fil.grown >= 1.0:
			# Fade once grown: a channel settles to its rest glow and stays (the ionised path
			# lingers); a transient bolt fades to nothing and goes idle. Branches die tip-first
			# in the draw (see _seg_alpha), so the system never blinks out all at once.
			var floor_v: float = float(fil.rest_floor) if fil.persist else 0.0
			fil.life = maxf(floor_v, fil.life - delta * float(_cfg.fade))
			if not fil.persist and fil.life <= 0.02:
				fil.active = false


# Strike a random subset of idle bolts. If `force`, guarantee at least one strikes even
# when the random draw lights none (the never-pure-black backstop). The struck bolts are
# then staggered into a connected cascade rather than all igniting on the same frame.
func _strike_some(force: bool) -> void:
	var struck: Array = []
	for fil in _fils:
		if _strikeable(fil) and _rng.randf() < 0.6:
			_ignite(fil)
			struck.append(fil)
	if force and struck.is_empty():
		for fil in _fils:
			if _strikeable(fil):
				_ignite(fil)
				struck.append(fil)
				break
	_stagger_ignition(struck)


# A bolt can be struck when it is idle, or when it is a channel that has flared and settled
# back to its rest glow (ready to re-ionise) - but not mid-flash.
func _strikeable(fil: Dictionary) -> bool:
	if not fil.active:
		return true
	return fil.persist and fil.grown >= 1.0 and fil.life <= float(fil.rest_floor) + 0.04


# Arm a single bolt for a strike: full life and its own growth rate. A persistent channel
# mostly re-ionises its EXISTING path in place (re-brighten, no dark blink, no re-flood) -
# real lightning re-uses its route - and only occasionally re-routes. A transient bolt
# always grows a fresh forked path. The ignition delay is set afterward by _stagger_ignition.
func _ignite(fil: Dictionary) -> void:
	fil.active = true
	fil.life = 1.0
	fil.rate = _gauss_rate()
	fil.delay = 0.0
	if fil.persist and fil.fil != null and _rng.randf() < 0.82:
		fil.grown = 1.0                  # re-flare the established channel in place
	else:
		_regrow(fil)                     # fresh path (grown reset to 0 -> floods from the root)


# Light a cluster of struck bolts as a CONNECTED CASCADE, not one global flash. They
# ignite in sequence sweeping outward from the shared convergence point (so the cascade
# reads as one propagating strike, "connected"), and the gaps between ignitions are
# nonlinear - an eased power curve, randomly accelerating or settling per strike - so it
# never ticks like a metronome. The nearest bolt fires immediately (delay 0), so a strike
# is always visible at once; the rest fan in over a short window.
func _stagger_ignition(struck: Array) -> void:
	# Channels re-flash immediately (no stagger) so they never blink; only fresh transient
	# bolts cascade in.
	var seq: Array = []
	for fil in struck:
		if fil.persist:
			fil.delay = 0.0
		else:
			seq.append(fil)
	if seq.size() <= 1:
		return
	seq.sort_custom(func(a, b):
		return (Vector2(a.origin) - _converge).length_squared() \
			< (Vector2(b.origin) - _converge).length_squared())
	var window := _rng.randf_range(0.10, 0.26)   # total spread of the cascade (seconds)
	var gamma := _rng.randf_range(0.55, 2.2)     # <1 the cascade accelerates, >1 it settles
	var n := seq.size()
	for k in n:
		var u := float(k) / float(n - 1)         # 0 (nearest) .. 1 (farthest)
		seq[k].delay = window * pow(u, gamma) + _rng.randf_range(0.0, 0.02)


# Neural / thread: each tendril runs an independent, staggered lifecycle - grow slowly
# to full, hold, then retire gracefully (FADE out, or REWIND its front back inward) and
# regrow on a fresh path. Never a clear-and-pop; always something growing. A steady
# creep surged by energy through a spike curve.
func _update_continuous(f: AudioFeatures, delta: float, grow: float, spont: bool) -> void:
	# The spark feeds the drive, so a spontaneous activation visibly surges the growth in dead air.
	var drive := 0.5 + 1.1 * Nonlinear.apply("spike", clampf(0.7 * f.energy + f.beat + _spark, 0.0, 1.0), 2.0)
	# On a fresh spark, kick a random RESTING tendril back into growth so something new sprouts.
	if spont and not _fils.is_empty():
		var fil: Dictionary = _fils[_rng.randi() % _fils.size()]
		if String(fil.state) == "hold" or String(fil.state) == "fade":
			fil.state = "grow"
	for fil in _fils:
		match fil.state:
			"grow":
				fil.grown = minf(1.0, fil.grown + delta * grow * 0.5 * float(fil.rate) * drive)
				if fil.grown >= 1.0:
					fil.state = "hold"
					fil.timer = fil.hold
			"hold":
				fil.timer -= delta
				if fil.timer <= 0.0:
					fil.state = fil.mode
					if fil.state == "rewind":
						fil.retract_to = _roll_retract()
			"fade":
				fil.life = maxf(0.0, fil.life - delta * 0.55)
				if fil.life <= 0.0:
					_recycle(fil)
			"rewind":
				var floor_v: float = fil.retract_to
				fil.grown = maxf(floor_v, fil.grown - delta * 0.4 * (0.6 + 0.5 * drive))
				if fil.grown <= floor_v + 0.005:
					if floor_v < 0.08:
						_recycle(fil)               # fully retracted -> new path
					else:
						# Partial retract: regrow the SAME tendril back up, re-rolling only
						# the next hold/retire so it keeps varying.
						fil.state = "grow"
						fil.hold = _rng.randf_range(1.6, 4.5)
						fil.mode = _roll_mode()


# Regrow a continuous tendril on a fresh path and re-roll its lifecycle constants, so
# each life differs (path, rate, hold, retire mode).
func _recycle(fil: Dictionary) -> void:
	_regrow(fil)
	fil.life = 1.0
	fil.state = "grow"
	fil.rate = _gauss_rate()
	fil.hold = _rng.randf_range(1.6, 4.5)
	fil.mode = "rewind" if _rng.randf() < 0.30 else "fade"


# A per-filament growth-speed multiplier drawn from a ~normal distribution (sum of three
# uniforms), with a wide spread - so the strands grow at a real variety of speeds (some
# slow crawlers, some fast shoots, most middling) rather than all at roughly one rate.
# How far a rewinding strand pulls its growth front back. Almost always a SHALLOW dieback
# (the tip recedes a little, then grows out again on the same tendril); a deep pull-back
# that nearly erases the strand is rare - a strand should partly die, not vanish.
func _roll_retract() -> float:
	if _rng.randf() < 0.10:
		return _rng.randf_range(0.0, 0.30)    # rare: deep retract (and < 0.08 = full restart)
	return _rng.randf_range(0.55, 0.88)        # usual: a small partial dieback


# Retire mode for one growth cycle: mostly a partial REWIND (dieback + regrow), only
# occasionally a full FADE-out + fresh path - so strands rarely disappear completely.
func _roll_mode() -> String:
	return "fade" if _rng.randf() < 0.12 else "rewind"


func _gauss_rate() -> float:
	var g := (_rng.randf() + _rng.randf() + _rng.randf()) / 3.0
	return 0.3 + 1.7 * g


func _draw() -> void:
	begin_draw()
	var u := unit()
	for fil in _fils:
		if fil.fil == null or fil.life <= 0.0:
			continue
		if fil.delay > 0.0:
			continue                      # struck but not yet ignited - stays dark until its turn
		# The bolt being drawn: its fade level, depth, and whether it is a persistent channel
		# (read by _seg_alpha to fade branches tip-first and keep the base channel alive).
		_cur_fade = float(fil.life)
		_cur_maxd = maxi(1, (fil.fil as Filament).max_depth())
		_cur_persist = bool(fil.persist)
		# Nonlinear lifecycle: the fade eases (smoothstep) rather than ramping linearly,
		# and the growth front advances on an eased curve - slow-fast-slow - so sprouting
		# and dying read as living motion, not a uniform slider.
		_life_alpha = smoothstep(0.0, 1.0, float(fil.life))
		var eased_grown: float = Nonlinear.apply("smoothstep", clampf(float(fil.grown), 0.0, 1.0))
		# A glowing bud where the strand sprouts - a complementary node, alive with energy.
		var bud_v: float = clampf(0.25 + 0.7 * _f.energy + 0.5 * _glow, 0.0, 1.0) * _life_alpha
		var bud := Color.from_hsv(fposmod(_hue + 0.5, 1.0), 0.3, 1.0, 0.5 * bud_v)
		var bsz: float = u * (0.006 + 0.010 * bud_v)
		Layer.glow(self, Vector2(fil.origin) * u, bsz * 3.0, bud, 4)
		var tip := Color.from_hsv(fposmod(_hue + 0.5, 1.0), 0.2, 1.0, 0.9 * _life_alpha)
		var jitter := float(_cfg.jitter) * (0.6 + 0.6 * _f.energy)
		fil.fil.draw_growing(self, u, eased_grown, _color_for, tip, jitter, _life)


func _color_for(depth: int, along := 0.0) -> Color:
	var h := fposmod(_hue + 0.04 * float(depth), 1.0)
	var sat: float = float(_cfg.sat) * (0.6 + 0.4 * float(depth == 0))
	var band := 0.08 * sin(along * 30.0 + float(depth) * 1.5)        # a faint grain along the strand
	var v := clampf(0.45 + 0.4 * _f.energy + 0.45 * _glow + band, 0.1, 1.0)
	return Color.from_hsv(h, sat, v, 0.92 * _seg_alpha(depth))


# Per-segment alpha for lightning: branches die from the TIPS inward (a deeper segment
# starts fading while the bolt's life is still high) so the structure doesn't blink out all
# at once, and a persistent channel keeps a faint floor on its base (low depth) so the
# ionised path lingers and the next strike re-uses it. Continuous modes keep the old
# uniform alpha (the whole strand fades together, which is right for roots/threads).
func _seg_alpha(depth: int) -> float:
	if not _strike_mode:
		return _life_alpha
	var dn := float(depth) / float(_cur_maxd)            # 0 trunk .. 1 tip
	var thr := dn * 0.72                                  # tips begin fading at higher life
	var a := smoothstep(0.0, 1.0, clampf((_cur_fade - thr) / 0.28, 0.0, 1.0))
	if _cur_persist and dn <= 0.34:
		a = maxf(a, 0.5 * (1.0 - dn / 0.34))             # the base channel never goes fully dark
	return a
