extends RefCounted
class_name ModBank

## ModBank - the organic-motion engine.
##
## The same idea as the evolutionary agents' voter pool: a scene's motion should
## not come from one clock but from a pool of slow oscillators blended into named
## continuous channels. Each channel is a seeded weighted sum of sines at
## incommensurate (rarely-aligning) frequencies, so the combined signal drifts
## for a very long time before it repeats - motion that feels alive, not looped.
##
## A scene asks for channels by name - mod.value("sway"), mod.value("breathe") -
## and a stable channel is minted on first use from the bank seed XOR the name.
## So any scene can invent any channel; same seed always yields the same drift.

## Oscillators per channel. More -> longer recurrence, smoother wander.
const OSC := 5
## Slow band, in Hz. Whole-screen motion wants tens-of-seconds periods - kept
## deliberately low so the drift reads as breathing, not spinning.
const FREQ_MIN := 0.010
const FREQ_MAX := 0.085

var _seed: int
var _t: float = 0.0
var _audio: float = 0.0
var _channels: Dictionary = {}   # name -> {f,p,w : PackedFloat32Array}


func _init(seed_value: int) -> void:
	_seed = seed_value


## Advance the clock once per frame. [param audio_energy] lets channels lean on
## the sound when a scene asks for an audio-coupled value.
func advance(delta: float, audio_energy: float) -> void:
	_t += delta
	_audio = audio_energy


# Mint (or fetch) a channel's oscillator table.
func _ensure(name: String) -> Dictionary:
	if not _channels.has(name):
		var rng := RandomNumberGenerator.new()
		rng.seed = _seed ^ hash(name)
		var f := PackedFloat32Array()
		var p := PackedFloat32Array()
		var w := PackedFloat32Array()
		for i in OSC:
			f.append(rng.randf_range(FREQ_MIN, FREQ_MAX))
			p.append(rng.randf_range(0.0, TAU))
			w.append(rng.randf_range(-1.0, 1.0))
		_channels[name] = {"f": f, "p": p, "w": w}
	return _channels[name]


## A smooth organic signal in roughly -1..1 for [param name].
func value(name: String) -> float:
	var ch := _ensure(name)
	var f: PackedFloat32Array = ch.f
	var p: PackedFloat32Array = ch.p
	var w: PackedFloat32Array = ch.w
	var s := 0.0
	var wsum := 0.0
	for i in f.size():
		s += w[i] * sin(TAU * f[i] * _t + p[i])
		wsum += absf(w[i])
	return s / maxf(0.001, wsum)


## Same channel mapped to 0..1.
func unit(name: String) -> float:
	return value(name) * 0.5 + 0.5


## An organic value pulled toward the audio - drifts on its own, but louder
## passages bias it positive. Good for "breathe harder when the track swells."
func audio_value(name: String, audio_mix: float = 0.5) -> float:
	return lerpf(value(name), _audio * 2.0 - 1.0, clampf(audio_mix, 0.0, 1.0))
