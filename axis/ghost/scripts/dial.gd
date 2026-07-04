extends RefCounted
class_name Dial

## Dial - the first live performance control (the semi-automatic mode's first lever).
##
## Not a parameter knob: a SCULPTING instrument. Turning it injects energy into
## seeded modulation *deposits*; each deposit surges briefly (the transient you feel
## on the turn) and then decays into a smaller PERSISTENT level - a standing pattern
## the scene keeps. Deposits accumulate for the whole session: the more the dial has
## been turned, the richer the standing weave of modulations. Turning is purely
## additive over time.
##
## Structure of a turn: one revolution is divided into 5 or 6 WEDGES (per turn, by
## seed). Each (turn, wedge) has its own seeded SIGNATURE - which modulation slots it
## drives, at what frequencies, with what waveform shapes, how hard it surges, how
## fast it decays and how much of it persists. Sweeping through a revolution therefore
## pumps a succession of distinct transformations; crossing into the next revolution
## increments the turn counter and RE-ROLLS the signature vocabulary (same wedge, new
## turn, new character). What a dial does to a scene is deliberately arbitrary - but
## deterministic: signatures derive from the dial's seed (the session seed), so the
## same song and the same gesture give the same response.
##
## Consumption: [method value] sums every live deposit's waveform for a modulation
## SLOT (the scene-facing vocabulary below), bounded by tanh into (-1, 1). Callers
## pass an element index `i` for phase diversity, so a cast of actors modulates as a
## group without moving in lockstep. Scenes read slots through
## `Director.dial_value(slot, i)` and apply them to whatever they mean by them.

## The modulation vocabulary scenes can honor. Deliberately abstract - a slot is a
## CHANNEL, not a parameter; each consumer decides what "scale" or "drive" bends.
const SLOTS := ["scale", "hue", "drive", "tempo", "off_x", "off_y"]

const ENV_MAX := 1.6              # transient ceiling per deposit
const TURN_GAIN := 0.5            # injected energy per radian of turning

var seed_value := 0
var angle := 0.0                  # unbounded radians of accumulated turning
var _deposits := {}               # "turn:wedge" -> deposit dict
var _t := 0.0                     # waveform clock
var _spin := 0.0                  # smoothed |turn rate|, for UI feel


func _init(seed_v := 0) -> void:
	seed_value = seed_v


## Rotate by [param delta_angle] radians (either direction). Energy lands in the
## wedge the needle is IN, surging its transient and growing its persistent base.
func turn(delta_angle: float) -> void:
	angle += delta_angle
	var d := _deposit_at(angle)
	var inj := absf(delta_angle) * TURN_GAIN
	d["env"] = minf(float(d["env"]) + inj * float(d["surge"]), ENV_MAX)
	d["base"] = minf(float(d["base"]) + inj * float(d["persist"]), float(d["cap"]))
	_spin = minf(_spin + absf(delta_angle), 3.0)


## Advance waveforms and decay the transients. Call once per frame.
func advance(dt: float) -> void:
	_t += dt
	_spin *= exp(-3.0 * dt)
	for d in _deposits.values():
		d["env"] = float(d["env"]) * exp(-dt / float(d["decay"]))


## The summed modulation on [param slot], in (-1, 1). [param i] decorrelates
## elements of a group (a golden-angle phase offset per index).
func value(slot: String, i := 0) -> float:
	var total := 0.0
	for d in _deposits.values():
		var level := float(d["env"]) + float(d["base"])
		if level <= 0.004:
			continue
		var slots: Array = d["slots"]
		for k in slots.size():
			if String(slots[k]) != slot:
				continue
			var p: Dictionary = d["per"][k]
			total += level * float(d["amp"]) * float(p["weight"]) * _wave(p, i)
	return tanh(total)


## This turn's wedge count: 5 or 6, re-rolled per revolution.
func wedges_of(turn_i: int) -> int:
	return 5 + (_sig_hash(turn_i, -1) & 1)


func turn_count() -> int:
	return floori(angle / TAU)


## Needle position within the current revolution, 0..1.
func phase() -> float:
	return fposmod(angle, TAU) / TAU


## The wedge the needle is in (for UI ticks).
func wedge() -> int:
	return mini(int(phase() * wedges_of(turn_count())), wedges_of(turn_count()) - 1)


## Total live level across deposits (transient + standing), for UI glow. Unbounded-ish.
func glow() -> float:
	var g := 0.0
	for d in _deposits.values():
		g += float(d["env"]) + float(d["base"])
	return g


## Smoothed recent turning, for UI feel.
func spin() -> float:
	return _spin


# The deposit under the needle at absolute angle `a`, created on first touch with a
# signature seeded from (dial seed, turn, wedge) - stable for the whole session.
func _deposit_at(a: float) -> Dictionary:
	var t := floori(a / TAU)
	var w := mini(int(fposmod(a, TAU) / TAU * wedges_of(t)), wedges_of(t) - 1)
	var key := "%d:%d" % [t, w]
	if _deposits.has(key):
		return _deposits[key]
	var rng := RandomNumberGenerator.new()
	rng.seed = _sig_hash(t, w)
	# Which slots this transformation drives (1-3 of them), and each one's voice.
	var count := 1 + rng.randi() % 3
	var pool := SLOTS.duplicate()
	var slots: Array = []
	var per: Array = []
	for k in count:
		var slot: String = pool.pop_at(rng.randi() % pool.size())
		slots.append(slot)
		per.append({
			"freq": rng.randf_range(0.08, 1.4),          # Hz
			"shape": rng.randi() % 4,
			"ph": rng.randf() * TAU,
			"weight": rng.randf_range(0.5, 1.0),
		})
	var d := {
		"slots": slots, "per": per,
		"amp": rng.randf_range(0.5, 1.0),
		"surge": rng.randf_range(0.8, 1.5),              # how hard a turn kicks it
		"decay": rng.randf_range(1.5, 6.0),              # transient half-life-ish (s)
		"persist": rng.randf_range(0.12, 0.4),           # fraction that becomes standing
		"cap": rng.randf_range(0.3, 0.8),                # standing ceiling
		"env": 0.0, "base": 0.0,
	}
	_deposits[key] = d
	return d


# One deposit's waveform at element index i: four voices, all slow and organic.
func _wave(p: Dictionary, i: int) -> float:
	var ph := _t * float(p["freq"]) * TAU + float(p["ph"]) + float(i) * 2.399963  # golden angle
	match int(p["shape"]):
		1:
			var s := sin(ph)
			return s * s * s                             # soft spike
		2:
			return tanh(2.6 * sin(ph))                   # rounded square (pulse train)
		3:
			return sin(ph) * sin(ph * 0.371 + 1.7)       # two incommensurate sines (organic beat)
		_:
			return sin(ph)


# Deterministic per (dial, turn, wedge) - String.hash() is a stable algorithm.
func _sig_hash(t: int, w: int) -> int:
	return ("%d|%d|%d" % [seed_value, t, w]).hash()
