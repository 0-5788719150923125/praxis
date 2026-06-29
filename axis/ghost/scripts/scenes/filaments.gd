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
		"len_lo": 0.55, "len_hi": 0.85, "evolve": 0.10, "jitter": 0.0},
	"neural": {"variant": "tendril", "count_lo": 7, "count_hi": 12, "grow": 0.5,
		"fade": 0.0, "strike": false, "hue": 0.75, "sat": 0.7, "w_lo": 3.0, "w_hi": 6.0,
		"len_lo": 0.35, "len_hi": 0.55, "evolve": 0.05, "jitter": 0.012},
	"thread": {"variant": "thread", "count_lo": 5, "count_hi": 9, "grow": 0.42,
		"fade": 0.0, "strike": false, "hue": 0.50, "sat": 0.6, "w_lo": 2.0, "w_hi": 4.0,
		"len_lo": 0.6, "len_hi": 0.95, "evolve": 0.08, "jitter": 0.006},
}

var _f: AudioFeatures = AudioFeatures.new()
var _rng := RandomNumberGenerator.new()
var _flow: Flow2D
var _cfg: Dictionary
var _mode := "lightning"
var _fils: Array = []
var _hue := 0.0
var _glow := 0.0
var _beat_prev := 0.0
var _life_alpha := 1.0       # set per-filament before its draw, read by _color_for
var _strike_acc := 0.0       # lightning: time since the last strike
var _strike_period := 1.6    # lightning: fallback re-strike cadence (sampled), so it
                             # strikes even with no detectable beats - never pure black


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	_rng.seed = rng.randi()
	var keys := MODES.keys()
	_mode = keys[rng.randi() % keys.size()]
	_cfg = MODES[_mode]
	_hue = fposmod(float(_cfg.hue) + rng.randf_range(-0.08, 0.08), 1.0)
	_flow = Flow2D.new(rng.randi(), rng.randf_range(2.0, 3.5), float(_cfg.evolve))
	_strike_period = rng.randf_range(1.1, 2.0)
	var count := rng.randi_range(int(_cfg.count_lo), int(_cfg.count_hi))
	for i in count:
		var fil := {"fil": null, "grown": 0.0, "life": 0.0, "mature": 0.0,
			"origin": Vector2.ZERO, "heading": 0.0, "active": false,
			"state": "grow", "timer": 0.0, "rate": 1.0, "hold": 2.0, "mode": "fade", "retract_to": 0.0}
		_seed_path(fil, i, count)
		_regrow(fil)
		if not bool(_cfg.strike):
			# Continuous modes run a staggered, rate-varied lifecycle (see _update_continuous).
			fil.life = 1.0
			fil.grown = rng.randf_range(0.0, 1.0)   # start anywhere in the lifecycle (async)
			fil.rate = _gauss_rate()
			fil.hold = rng.randf_range(1.6, 4.5)
			fil.mode = "rewind" if rng.randf() < 0.30 else "fade"
		_fils.append(fil)
	# Lightning: seed an opening strike (mid-flash) so the very first frame already has
	# bolts on screen, rather than waiting in the dark for the first beat.
	if bool(_cfg.strike):
		for j in mini(2, _fils.size()):
			var fil: Dictionary = _fils[j]
			fil.active = true
			fil.life = 1.0
			fil.grown = rng.randf_range(0.5, 1.0)
	return {}


# Where a path starts and which way it heads, by mode.
func _seed_path(fil: Dictionary, i: int, count: int) -> void:
	match _mode:
		"lightning":
			# Strike across the whole width / top, not one patch.
			fil.origin = Vector2(_rng.randf_range(-0.78, 0.78), _rng.randf_range(-0.62, -0.42))
			fil.heading = PI * 0.5 + _rng.randf_range(-0.4, 0.4)   # downward strike
		"thread":
			# Lanes spread across the full height so the threads fill the frame.
			fil.origin = Vector2(_rng.randf_range(-0.7, -0.45), _rng.randf_range(-0.62, 0.62))
			fil.heading = _rng.randf_range(-0.3, 0.3)              # rightward flow
		_:  # neural - scattered seeds spread across the frame, any heading
			fil.origin = Vector2(_rng.randf_range(-0.62, 0.62), _rng.randf_range(-0.6, 0.6))
			fil.heading = TAU * float(i) / float(count) + _rng.randf_range(-0.5, 0.5)


func _regrow(fil: Dictionary) -> void:
	var steps := _rng.randi_range(10, 18)
	var length := _rng.randf_range(float(_cfg.len_lo), float(_cfg.len_hi))
	var width := _rng.randf_range(float(_cfg.w_lo), float(_cfg.w_hi))
	fil.heading += _rng.randf_range(-0.4, 0.4)
	fil.fil = Filament.grow(String(_cfg.variant), fil.origin, fil.heading, length, width,
		steps, _flow, _rng)
	fil.grown = 0.0
	fil.mature = 0.0


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	tick(f, delta)
	drift_view(f, 0.03, 0.05)
	_flow.advance(delta)
	_glow = Nonlinear.flare(_glow, clampf(0.3 * f.energy + 0.7 * f.beat, 0.0, 1.0), delta, 9.0, 1.6)
	var beat_edge: bool = f.beat > 0.55 and _beat_prev <= 0.55
	_beat_prev = f.beat

	var grow := float(_cfg.grow)
	if bool(_cfg.strike):
		_update_strikes(f, delta, grow, beat_edge)
	else:
		_update_continuous(f, delta, grow)
	queue_redraw()


# Lightning: bolts strike on beats, flood in, blaze, then fade out. They also strike on
# a fallback cadence (so quiet/no-audio stretches still flicker), and a strike is forced
# whenever none are alive - so the scene is never pure black.
func _update_strikes(f: AudioFeatures, delta: float, grow: float, beat_edge: bool) -> void:
	var drive := 0.5 + 1.5 * Nonlinear.apply("spike", clampf(0.6 * f.energy + f.beat, 0.0, 1.0), 2.0)
	_strike_acc += delta
	var any_active := false
	for fil in _fils:
		if fil.active:
			any_active = true
	if beat_edge or _strike_acc >= _strike_period or not any_active:
		_strike_acc = 0.0
		_strike_some(not any_active)
	for fil in _fils:
		if fil.active:
			fil.grown = minf(1.0, fil.grown + delta * grow * drive)
			if fil.grown >= 1.0:
				fil.life = maxf(0.0, fil.life - delta * float(_cfg.fade))
				if fil.life <= 0.02:
					fil.active = false


# Strike a random subset of idle bolts. If `force`, guarantee at least one strikes even
# when the random draw lights none (the never-pure-black backstop).
func _strike_some(force: bool) -> void:
	var struck := false
	for fil in _fils:
		if not fil.active and _rng.randf() < 0.6:
			_regrow(fil)
			fil.active = true
			fil.life = 1.0
			struck = true
	if force and not struck:
		for fil in _fils:
			if not fil.active:
				_regrow(fil)
				fil.active = true
				fil.life = 1.0
				break


# Neural / thread: each tendril runs an independent, staggered lifecycle - grow slowly
# to full, hold, then retire gracefully (FADE out, or REWIND its front back inward) and
# regrow on a fresh path. Never a clear-and-pop; always something growing. A steady
# creep surged by energy through a spike curve.
func _update_continuous(f: AudioFeatures, delta: float, grow: float) -> void:
	var drive := 0.5 + 1.1 * Nonlinear.apply("spike", clampf(0.7 * f.energy + f.beat, 0.0, 1.0), 2.0)
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
						# How far it pulls back varies: usually a PARTIAL retract (the front
						# recedes part way, then grows out again on the same tendril); only a
						# near-zero target fully deletes and starts a fresh path.
						fil.retract_to = _rng.randf_range(0.0, 0.65)
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
						fil.mode = "rewind" if _rng.randf() < 0.30 else "fade"


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
func _gauss_rate() -> float:
	var g := (_rng.randf() + _rng.randf() + _rng.randf()) / 3.0
	return 0.3 + 1.7 * g


func _draw() -> void:
	begin_draw()
	var u := unit()
	for fil in _fils:
		if fil.fil == null or fil.life <= 0.0:
			continue
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


func _color_for(depth: int) -> Color:
	var h := fposmod(_hue + 0.04 * float(depth), 1.0)
	var sat: float = float(_cfg.sat) * (0.6 + 0.4 * float(depth == 0))
	var v := clampf(0.45 + 0.4 * _f.energy + 0.45 * _glow, 0.1, 1.0)
	return Color.from_hsv(h, sat, v, 0.92 * _life_alpha)
