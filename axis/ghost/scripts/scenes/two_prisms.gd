extends GhostScene

## Two prisms - the pair, from the drop through specialization (the-point, scenes 6-11).
##
## The trembling eye BURSTS into the glowing RED [PrismBody] on the beat drop; now two prisms
## face forward, the BLUE (continued from `eye_prism`) and the RED. They run a timeline:
##   6  burst      - the red forms in a flash where the eye was; blue keeps living.
##   7  phase-lock - the two SNAP into sync, red locked to blue's pose, rotating as one.
##   8  scan       - locked, they scan the void in unison.
##   9  desync     - the blue breaks sync and looks around on its own; the red holds steady.
##   10 unlock     - both slip their anchors and SWAY, weightless, straining and drifting.
##   11 specialize - the blue SWELLS and pulses slowly; the red SHRINKS and quickens. They
##                   grow visibly distinct (bias vs. variance made physical).
##
## Arrives by morph from `eye_prism` (`morph_in = "eye2prism"`): reuses the live blue prism and
## bursts the red at the eye's slot. Hands off to `prism_swarm` (`morph_out = "prisms"`).

var _f: AudioFeatures = AudioFeatures.new()
var _blue: PrismBody
var _red: PrismBody
var _bpos := Vector2(0.15, 0.0)     # blue centre, unit-fractions from screen centre
var _rpos := Vector2(-0.15, 0.0)    # red centre
var _banch := Vector2(0.15, 0.0)    # anchors the pair sway around
var _ranch := Vector2(-0.15, 0.0)
var _bscale := 0.18                 # blue draw scale (unit-fraction)
var _rscale := 0.18                 # red draw scale
var _base := 0.18
var _t := 0.0
var _burst := 0.0                   # the red's form-in flash envelope
var _lock := 0.0                    # 0 free .. 1 red phase-locked to blue
var _lockflash := 0.0
var _rspin := 1.0                   # red's own spin scale (damped to 0 when it "holds steady")
const HUE_BLUE := 0.6
const HUE_RED := 0.0

# Phase thresholds as FRACTIONS of the scene's hold (beats 6-11 spread across it), so every event
# lands whatever the hold length - a shorter scene just marches through them sooner.
const FL_LOCK := 0.19      # snap into phase-lock
const FL_DESYNC := 0.50    # blue breaks sync
const FL_UNLOCK := 0.62    # slip anchors and sway
const FL_SPECIAL := 0.80   # specialize
const NOMINAL := 16.0      # design length (used to pace in auto mode, where there is no fixed hold)


func build_params(rng: RandomNumberGenerator) -> Dictionary:
	render_kind = "scene3d"        # drawn on the canvas, but a 3D-feeling body; keep the plane framing
	framing = "plane"
	morph_in = "eye2prism"
	morph_out = "prisms"
	_blue = PrismBody.new(rng.randi())
	_red = PrismBody.new(rng.randi())
	_base = rng.randf_range(0.16, 0.20)
	_bscale = _base
	_rscale = _base
	var off := rng.randf_range(0.14, 0.17)
	_banch = Vector2(off, 0.0)
	_ranch = Vector2(-off, 0.0)
	_bpos = _banch
	_rpos = _ranch
	_t = 0.0
	return {}


# Arrived from eye_prism: keep the live blue prism, and burst the red in at the eye's slot.
func begin_morph(from: GhostScene) -> void:
	var p := from.morph_payload()
	if p.is_empty():
		return
	if p.has("blue"):
		_blue = p["blue"]
	_banch = p.get("blue_slot", _banch)
	_ranch = p.get("eye_slot", _ranch)
	_bpos = _banch
	_rpos = _ranch
	_base = float(p.get("scale", _base))
	_bscale = _base
	_rscale = _base
	_t = 0.0
	_burst = 1.0                     # the drop: the red bursts into being


func morph_payload() -> Dictionary:
	return {"blue": _blue, "red": _red, "blue_pos": _bpos, "red_pos": _rpos,
		"blue_scale": _bscale, "red_scale": _rscale}


func update(f: AudioFeatures, delta: float) -> void:
	_f = f
	last_f = f
	tick(f, delta)
	drift_view(f, 0.006, 0.012)
	_t += delta
	var drive := clampf(f.energy * 0.85 + f.beat * 0.6, 0.0, 1.0)
	_burst = maxf(0.0, _burst - delta * 2.2)
	# Keyframe phases run on the fraction of the hold elapsed (so they always land); the bodies' spin
	# and pulse below run on the raw delta and never speed up with the tempo.
	var f2 := clampf(_t / phase_span(NOMINAL), 0.0, 1.0)

	# Phase-lock: ease _lock up around FL_LOCK; a snap-flash on the crossover. While locked, the red
	# mirrors the blue's pose exactly (they turn as one).
	var lock_t := smoothstep(FL_LOCK - 0.04, FL_LOCK + 0.03, f2)
	if f2 < FL_DESYNC:
		if _lock < 0.5 and lock_t >= 0.5:
			_lockflash = 1.0
		_lock = lock_t
	else:
		_lock = maxf(0.0, _lock - delta * 3.0)     # blue breaks sync at FL_DESYNC
	_lockflash = maxf(0.0, _lockflash - delta * 2.6)

	# When the blue desyncs, the red HOLDS STEADY: damp its own spin to zero (blue keeps looking).
	var want_rspin := 1.0 if f2 < FL_DESYNC else 0.0
	_rspin = lerpf(_rspin, want_rspin, 1.0 - exp(-4.0 * delta))
	_red._vel *= (0.02 + 0.98 * _rspin) if f2 >= FL_DESYNC else 1.0

	# Specialize: blue swells + slows, red shrinks + quickens. Time-scales change each body's whole
	# tempo (its core pulse and its looking-around), so they diverge in character, not just size.
	var sp := smoothstep(FL_SPECIAL, FL_SPECIAL + 0.15, f2)
	var b_speed := lerpf(1.0, 0.55, sp)
	var r_speed := lerpf(1.0, 1.8, sp)
	var b_size := lerpf(_base, _base * 1.55, sp) * (1.0 + 0.12 * sp * sin(_t * 1.2))   # slow pulse
	var r_size := lerpf(_base, _base * 0.6, sp) * (1.0 + 0.06 * sp * sin(_t * 6.5))    # quick flutter
	_bscale = lerpf(_bscale, b_size, 1.0 - exp(-5.0 * delta))
	_rscale = lerpf(_rscale, r_size, 1.0 - exp(-5.0 * delta))

	# Unlock + sway: past FL_UNLOCK the pair slips its anchors and drifts weightlessly, pushing and
	# pulling against them (a gentle spring back, with an audio-driven wander on top).
	var sway := smoothstep(FL_UNLOCK, FL_UNLOCK + 0.09, f2)
	var bw := Vector2(sin(_t * 0.7 + 0.0), cos(_t * 0.9 + 1.0)) * (0.05 * sway * (0.5 + drive))
	var rw := Vector2(sin(_t * 0.8 + 2.1), cos(_t * 0.6 + 3.0)) * (0.05 * sway * (0.5 + drive))
	var follow := 1.0 - exp(-3.0 * delta)
	_bpos = _bpos.lerp(_banch + bw, follow)
	_rpos = _rpos.lerp(_ranch + rw, follow)

	_blue.update(delta * b_speed, clampf(drive * (0.6 + 0.4 * (1.0 - _lock)), 0.0, 1.0))
	# The red drives on its own energy; while locked its pose is overwritten to the blue's.
	_red.update(delta * r_speed, clampf(drive, 0.0, 1.0))
	if _lock > 0.5:
		_red.lock_pose_to(_blue)
	queue_redraw()


func _draw() -> void:
	begin_draw()
	var u := unit()
	var bc := _bpos * u
	var rc := _rpos * u
	# The red bursts into being where the eye was: a bright flash that fades as it forms.
	if _burst > 0.001:
		draw_circle(rc, _rscale * u * (0.6 + 1.6 * _burst), Color(1.0, 0.5, 0.4, 0.5 * _burst))
		draw_circle(rc, _rscale * u * (0.3 + 0.7 * _burst), Color(1, 1, 1, 0.6 * _burst))
	_blue.draw(self, bc, _bscale * u, HUE_BLUE, 1.0)
	_red.draw(self, rc, _rscale * u, HUE_RED, clampf(1.0 - _burst * 0.5, 0.0, 1.0))
	# The phase-lock snap: a brief bright tie between the two centres as they sync.
	if _lockflash > 0.001:
		var mid := (bc + rc) * 0.5
		draw_line(bc, rc, Color(1, 1, 1, 0.5 * _lockflash), 2.0, true)
		draw_circle(mid, u * 0.02 * _lockflash, Color(1, 1, 1, 0.7 * _lockflash))
