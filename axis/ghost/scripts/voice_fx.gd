extends RefCounted
class_name VoiceFX

## VoiceFX - the ambience from Synthesis, over any PCM.
##
## The procedural engine's echo bus and sympathetic resonance are built into its
## per-sample kernel, so they only ever applied to its own output. Neither is
## actually specific to formant synthesis: an echo is a delay line, and the
## resonance is a bank of narrow resonators excited by the voice. Both work on
## a buffer regardless of what made it.
##
## So this is that ambience, re-expressed as a filter over arbitrary audio. It
## runs in the generative path's push loop, sample by sample, with all state
## held here so a chunk boundary is not a discontinuity.
##
## THE RESONANCE, and the one real design difference. In Synthesis the
## sympathetic strings are tuned to the READING's anchor notes, which come from
## the seed's genome - each seed carries its own chord. A neural voice has no
## genome, so the chord has to come from somewhere. Deriving it from the audio
## keeps what made the effect good: it still answers the voice rather than
## playing over it. The pitch tracker is a cheap autocorrelation on a decimated
## signal, and the tones sit at consonant intervals above the tracked pitch, so
## the chord follows the narration's register instead of fighting it.

# --- echo (after voice.gd's bus: 170 ms, damped in the loop) -----------------
const ECHO_DELAY := 0.17
const ECHO_LP := 1400.0        # every repeat comes back darker

# --- resonance ---------------------------------------------------------------
const TONES := 4
# Just intervals above the tracked pitch: octave, fifth, octave+third, two
# octaves. Chosen over equal temperament because these ring against the voice's
# own harmonics rather than beating with them.
const RATIOS := [2.0, 3.0, 5.0, 8.0]
const RES_Q := 0.9985          # decay per sample: seconds-long ring
# Excitation per sample into each resonator. 0.004 measured a 5% peak lift -
# inaudible, which is what "the resonance slider does nothing" meant. The ring
# integrates over its whole decay, so this is a rate, not a level; 0.05 with the
# output normalised by tone count lands it clearly under the voice but present.
const RES_DRIVE := 0.05
const TRACK_HZ := 12.0         # pitch re-estimates per second
const PITCH_MIN := 70.0
const PITCH_MAX := 320.0
const PITCH_GLIDE := 0.08      # the chord follows the register, never jumps

# --- the pad: sustained ambient tones, DECOUPLED from the voice ---------------
#
# Distinct from the resonance above, and the distinction is the point. Resonance
# is EXCITED by the voice: it rings when the voice rings and dies when it stops,
# which makes it feel like the room the voice is in. A pad is its own instrument
# - long tones that keep sounding through pauses, choosing notes on a scale,
# breathing on their own timescale. Ambient music rather than reverb.
#
# What keeps it organic is that the KEY comes from the narration. A slow average
# of the tracked pitch gives the reader's register; the tonic is dropped two
# octaves so the pad sits underneath rather than competing, and it glides rather
# than jumping, so a change of register moves the music instead of breaking it.
const PAD_VOICES := 5
# Minor pentatonic, in semitones. Chosen because no two degrees are a semitone
# apart, so ANY combination of them is consonant - the scheduler cannot stack a
# clash however it overlaps them. That is what makes an unattended generative
# pad safe to leave running under speech.
const PAD_SCALE := [0, 3, 5, 7, 10]
const PAD_OCTAVES := [0, 12, 12, 24]     # weighted toward one octave up
const PAD_ATTACK := 3.5                  # seconds: tones swell, never start
const PAD_HOLD_MIN := 5.0
const PAD_HOLD_MAX := 14.0
const PAD_RELEASE := 7.0                 # long tails, so the bed never gates
const PAD_DETUNE := 0.004                # a second oscillator, slightly off
const PAD_GAP := 2.2                     # seconds between new tones entering
const PAD_TONIC_TAU := 12.0              # the key follows the reader, slowly
# 0.55 pulled the bed back hard whenever the voice was speaking - and narration
# is speech almost continuously, so it spent nearly all its time ducked, which
# matters perceptually far more than the average level suggests. 0.40 still
# steps aside for the voice without vanishing behind it.
const PAD_DUCK := 0.40

var sample_rate := 22050
var echo_wet := 0.0            # 0..1
var echo_feedback := 0.42
var resonance := 0.0           # 0..1
var presence := 1.0            # 1 = in the room, lower = further away
var pad := 0.0                 # 0..1 the ambient bed
var pad_seed := 0              # same text, same music
# Staged by sweeping against real narration (voice RMS 0.0474):
#   0.012 -> 11.8 dB below the voice   too quiet, reported as barely audible
#   0.022 ->  6.5 dB below             present without competing  <-- chosen
#   0.035 ->  2.5 dB below             competes with the reading
# That is at the slider's maximum, so half-way gives roughly -12 dB and the
# full range spans "just there" to "clearly part of the piece".
var pad_level := 0.022
var pad_duck := PAD_DUCK       # how far the bed steps back under speech

var _echo := PackedFloat32Array()
var _echo_i := 0
var _echo_lp := 0.0
var _res_re := PackedFloat32Array()
var _res_im := PackedFloat32Array()
var _res_env := PackedFloat32Array()
var _pitch := 0.0
var _track := PackedFloat32Array()
var _track_n := 0
var _lp := 0.0                 # presence: distance darkens as well as quietens
var _p_re := PackedFloat32Array()
var _p_im := PackedFloat32Array()
var _p_re2 := PackedFloat32Array()
var _p_im2 := PackedFloat32Array()
var _p_cos := PackedFloat32Array()
var _p_sin := PackedFloat32Array()
var _p_cos2 := PackedFloat32Array()
var _p_sin2 := PackedFloat32Array()
var _p_env := PackedFloat32Array()
var _p_hold := PackedFloat32Array()
var _p_state := PackedInt32Array()      # 0 idle, 1 attack, 2 hold, 3 release
var _tonic := 0.0                       # Hz, the pad's root
var _next_note := 0.0
var _speech := 0.0                      # smoothed |voice|, for the duck
var _rng := RandomNumberGenerator.new()


func setup(sr: int) -> void:
	sample_rate = sr
	_echo.resize(maxi(1, int(ECHO_DELAY * sr)))
	_echo.fill(0.0)
	_echo_i = 0
	_res_re.resize(TONES)
	_res_im.resize(TONES)
	_res_env.resize(TONES)
	for i in TONES:
		_res_re[i] = 0.0
		_res_im[i] = 0.0
		_res_env[i] = 0.0
	_track.resize(maxi(1, int(sr / TRACK_HZ)))
	_track_n = 0
	_pitch = 0.0
	for arr in [_p_re, _p_im, _p_re2, _p_im2, _p_cos, _p_sin, _p_cos2, _p_sin2,
			_p_env, _p_hold]:
		arr.resize(PAD_VOICES)
		arr.fill(0.0)
	_p_state.resize(PAD_VOICES)
	_p_state.fill(0)
	_tonic = 0.0
	_next_note = 0.5
	_speech = 0.0
	_rng.seed = pad_seed


## Teach the pad its key BEFORE processing, from a buffer that has not been heard yet.
##
## The pad is an independent instrument but a key-SLAVED one: `_tonic` is derived from
## the tracked pitch of the voice, two octaves down, and `_start_tone` refuses to
## schedule anything while it is zero. That is exactly right in the steady state and
## exactly wrong at the head of a take, because a leading silence has no pitch - the
## estimator returns 0 on any window under its energy floor - so a bed asked to play
## over an intro would sit mute through all of it and only fade up once the narration
## it was supposed to introduce had already begun.
##
## So: scan forward for the first genuinely voiced window, take its pitch, and set the
## tonic directly. One pass over at most a few seconds of audio, and it does not touch
## the resonators or the echo line, so the chain is still cold when [method process]
## starts. The glide in `process` then carries the key onward as the reader moves.
##
## The resonance deliberately gets no equivalent. It is a sympathetic bank excited only
## by the dry signal, so it has nothing to ring from until the voice arrives, and
## priming it would be inventing an excitation that was never played.
func prime_key(buf: PackedFloat32Array) -> void:
	if _track.is_empty():
		setup(sample_rate)
	var win := _track.size()
	if win <= 0 or buf.size() < win:
		return
	# SKIP THE SILENCE CHEAPLY FIRST. The buffer this is called on opens with the whole
	# intro of digital zeros, and the intro is the thing that is several seconds long -
	# so a scan that starts at sample 0 and gives up after a fixed couple of seconds
	# examines nothing but the padding and always comes back empty. (It did. The pad
	# stayed silent through the entire intro and the measurement is what caught it.)
	# A scalar amplitude test costs one compare per sample and finds the voice exactly.
	var first := -1
	for i in buf.size():
		if absf(buf[i]) > 0.004:
			first = i
			break
	if first < 0:
		return                                  # the whole take is silent
	# From there, two seconds is generous: it is four times the longest gap the
	# scheduler leaves before its first tone, and bounding it keeps a 20-minute chapter
	# off the critical path.
	var limit := mini(buf.size() - win, first + int(2.0 * float(sample_rate)))
	var pos := first
	while pos <= limit:
		var chunk := buf.slice(pos, pos + win)
		var f := _estimate_pitch(chunk)
		if f > 0.0:
			_pitch = f
			_tonic = f * 0.25
			return
		pos += win


## Process in place. `buf` is mono float samples; returns the same array so the
## copy-on-write semantics of PackedFloat32Array cannot silently drop the work.
func process(buf: PackedFloat32Array) -> PackedFloat32Array:
	if _echo.is_empty():
		setup(sample_rate)
	var n := buf.size()
	var res_step := PackedFloat32Array()
	res_step.resize(TONES)
	for i in n:
		var dry := buf[i]

		# --- pitch tracking, at TRACK_HZ rather than per sample ---
		_track[_track_n] = dry
		_track_n += 1
		if _track_n >= _track.size():
			_track_n = 0
			var f := _estimate_pitch(_track)
			if f > 0.0:
				_pitch = f if _pitch <= 0.0 else lerpf(_pitch, f, PITCH_GLIDE)

		var wet := dry

		# --- resonance: a chord excited by the voice's own energy ---
		if resonance > 0.0 and _pitch > 0.0:
			var sum := 0.0
			for k in TONES:
				var hz: float = _pitch * float(RATIOS[k])
				if hz >= sample_rate * 0.45:
					continue
				var w := TAU * hz / sample_rate
				# one-pole complex resonator: cheap, stable, and its magnitude
				# is the ring envelope for free
				var re := _res_re[k] * cos(w) - _res_im[k] * sin(w)
				var im := _res_re[k] * sin(w) + _res_im[k] * cos(w)
				re = re * RES_Q + dry * RES_DRIVE
				im *= RES_Q
				_res_re[k] = re
				_res_im[k] = im
				sum += re
			wet += (sum / float(TONES)) * resonance

		# --- the pad: its own instrument, on its own clock ---
		if pad > 0.0:
			_speech += (absf(dry) - _speech) * 0.0004      # ~1 s envelope
			if _pitch > 0.0:
				# two octaves down: a tonic in the speaking register would mask
				# the voice instead of supporting it
				var want := _pitch * 0.25
				_tonic = want if _tonic <= 0.0 else lerpf(_tonic, want,
					1.0 / (PAD_TONIC_TAU * sample_rate))
			var mix := 0.0
			for k in PAD_VOICES:
				if _p_state[k] == 0:
					continue
				# complex rotation: 4 mults, no sin() in the sample loop
				var pre := _p_re[k] * _p_cos[k] - _p_im[k] * _p_sin[k]
				var pim := _p_re[k] * _p_sin[k] + _p_im[k] * _p_cos[k]
				_p_re[k] = pre
				_p_im[k] = pim
				var pre2 := _p_re2[k] * _p_cos2[k] - _p_im2[k] * _p_sin2[k]
				var pim2 := _p_re2[k] * _p_sin2[k] + _p_im2[k] * _p_cos2[k]
				_p_re2[k] = pre2
				_p_im2[k] = pim2
				match _p_state[k]:
					1:
						_p_env[k] = minf(1.0, _p_env[k] + 1.0 / (PAD_ATTACK * sample_rate))
						if _p_env[k] >= 1.0:
							_p_state[k] = 2
					2:
						_p_hold[k] -= 1.0 / sample_rate
						if _p_hold[k] <= 0.0:
							_p_state[k] = 3
					3:
						_p_env[k] -= 1.0 / (PAD_RELEASE * sample_rate)
						if _p_env[k] <= 0.0:
							_p_env[k] = 0.0
							_p_state[k] = 0
				var e: float = _p_env[k]
				mix += (pre + pre2 * 0.6) * e * e      # squared: a gentler swell
			var duck := 1.0 - pad_duck * clampf(_speech * 12.0, 0.0, 1.0)
			# Staged by measurement, not by eye: at 0.22 the bed measured RMS
			# 0.14 against the narration's 0.047 - three times LOUDER than the
			# voice it is supposed to sit under, and peaking at 0.88. This puts
			# it well below the voice, which is where a bed belongs: audible
			# when you listen for it, never competing.
			wet += mix * pad * pad_level * duck
			_next_note -= 1.0 / sample_rate
			if _next_note <= 0.0 and _tonic > 0.0:
				_next_note = PAD_GAP * _rng.randf_range(0.7, 1.8)
				_start_tone()

		# --- echo: darker on every repeat ---
		if echo_wet > 0.0:
			var back := _echo[_echo_i]
			_echo_lp += (back - _echo_lp) * (1.0 - exp(-TAU * ECHO_LP / sample_rate))
			wet += _echo_lp * echo_wet
			_echo[_echo_i] = dry + _echo_lp * echo_feedback
			_echo_i = (_echo_i + 1) % _echo.size()

		# --- presence: distance is a filter first, a gain second ---
		if presence < 0.995:
			var cut := 900.0 * pow(2.0, 4.0 * presence)
			_lp += (1.0 - exp(-TAU * cut / sample_rate)) * (wet - _lp)
			wet = _lp * maxf(presence, 0.05)

		buf[i] = clampf(wet, -1.0, 1.0)
	return buf


## Bring one idle voice in on a scale degree. Notes are chosen, not swept: the
## pad is a sequence of sustained tones, and overlapping envelopes are what turn
## a sequence into a chord.
func _start_tone() -> void:
	var slot := -1
	for k in PAD_VOICES:
		if _p_state[k] == 0:
			slot = k
			break
	if slot < 0:
		return
	var degree: int = int(PAD_SCALE[_rng.randi() % PAD_SCALE.size()])
	var octave: int = int(PAD_OCTAVES[_rng.randi() % PAD_OCTAVES.size()])
	var hz: float = _tonic * pow(2.0, float(degree + octave) / 12.0)
	if hz <= 0.0 or hz >= sample_rate * 0.45:
		return
	var w := TAU * hz / sample_rate
	_p_cos[slot] = cos(w)
	_p_sin[slot] = sin(w)
	var w2 := TAU * hz * (1.0 + PAD_DETUNE) / sample_rate
	_p_cos2[slot] = cos(w2)
	_p_sin2[slot] = sin(w2)
	# random start phases: tones that all begin in step sum into a click
	var ph := _rng.randf() * TAU
	_p_re[slot] = cos(ph)
	_p_im[slot] = sin(ph)
	_p_re2[slot] = cos(ph * 1.7)
	_p_im2[slot] = sin(ph * 1.7)
	_p_env[slot] = 0.0
	_p_hold[slot] = _rng.randf_range(PAD_HOLD_MIN, PAD_HOLD_MAX)
	_p_state[slot] = 1


## Autocorrelation over one tracking window, decimated by 4 - the pitch of a
## voice is well under 400 Hz, so full rate buys nothing but cost.
func _estimate_pitch(win: PackedFloat32Array) -> float:
	var dec := 4
	var m := win.size() / dec
	if m < 64:
		return 0.0
	var x := PackedFloat32Array()
	x.resize(m)
	var energy := 0.0
	for i in m:
		x[i] = win[i * dec]
		energy += x[i] * x[i]
	if energy < 1e-5:
		return 0.0                     # silence: hold the last chord
	var sr := float(sample_rate) / dec
	var lo := int(sr / PITCH_MAX)
	var hi := mini(int(sr / PITCH_MIN), m - 1)
	var best := 0.0
	var best_lag := 0
	for lag in range(lo, hi):
		var acc := 0.0
		for i in range(m - lag):
			acc += x[i] * x[i + lag]
		if acc > best:
			best = acc
			best_lag = lag
	if best_lag <= 0 or best < energy * 0.3:
		return 0.0                     # unvoiced or noisy: do not retune on noise
	return sr / float(best_lag)
