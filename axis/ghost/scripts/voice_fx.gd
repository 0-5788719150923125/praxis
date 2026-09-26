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

# --- echo: a room at the bottom of the dial, a slapback at the top ------------
#
# THE SETTING CHOOSES THE CHARACTER, NOT JUST THE LEVEL. It used to choose only
# the level: one tap at a fixed 170 ms with a fixed 0.42 feedback, so measured
# across the whole dial the delay was 170 ms and the tail ran 0.85-1.36 s at
# EVERY setting - 5 to 8 discrete repeats whether the slider was at 0.05 or 0.5.
# Turning it down made the same effect quieter and never made it a smaller one.
#
# That is unusable on speech for a reason that gain cannot fix. Past roughly
# 50 ms the precedence effect stops fusing a repeat with the direct sound and
# the ear hears a separate event, so a discrete repeat stays plainly audible
# far below where a diffuse room would - detection thresholds for speech at
# this delay sit around -20 to -25 dB, and the old dial's first repeat is
# -26 dB at 0.05 and -20 dB at 0.10. It was AT the audibility threshold at the
# very bottom of its travel. Worse, 170 ms is about one syllable at a normal
# reading pace, so each repeat lands on the next word: maximally smearing.
#
# So the tap SHORTENS as the dial comes down, until it is inside the fusion
# window and reads as the room the voice is in; the feedback and the damping
# come down with it, and a second incommensurate tap keeps even the long
# setting from being a single hard image.
const ECHO_DELAY := 0.17       # the longest tap, and the delay line's size
const ECHO_NEAR := 0.028       # the shortest: inside the ~50 ms fusion window
const ECHO_TAP2 := 0.61        # second tap as a fraction of the first, deliberately
                               # not a simple ratio - a 1/2 or 1/3 tap re-lands on
                               # the first one's repeats and rebuilds a single image
const ECHO_TAP2_MIX := 0.30
const ECHO_LP := 1400.0        # every repeat comes back darker...
const ECHO_LP_NEAR := 650.0    # ...and a small dark room is darker still
const ECHO_FB_NEAR := 0.12     # one repeat and gone, rather than a five-tap tail
## The dial's taper. Above 1 so the bottom of the travel is genuinely quiet -
## the useful settings for narration all live down there, and a linear dial spent
## most of its length on levels that bury the voice. Not much above 1, though: at
## a square taper 0.05 lands at -52 dB, which is inaudible, and dead travel at the
## bottom of a dial is its own kind of broken. 1.5 puts 0.05 at -39 dB and 0.20 at
## -21 dB, both of which do something, and both of which are a ROOM at those
## settings rather than a repeat (see ECHO_NEAR).
const ECHO_TAPER := 1.5

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
## THE DIAL IS SEVERITY, NOT LEVEL. The bed plays at its full staged level (`pad_level`)
## whenever Ambience is on at all - it faded in over this much of the travel so 0 is still
## a clean off, not a step. What the rest of the dial buys is MORE HAPPENING on top of it.
const PAD_FADE_IN := 0.05

# --- no melody, on purpose ------------------------------------------------------
# Three melodic layers were built on top of the bed and each removed: a plucked string
# (twice) and a sliding electronic lead. They were in tune and on a beat, and they still
# sounded like random notes, because choosing WHICH note comes next needs a real model of
# music and a handful of rules is not one. The bed works because it never has to make that
# choice - sustained, always-consonant tones - and the bass works for the same reason: one
# note, the key's root, when the voice moves. Do not add a melody without that model.

# --- the bass: a low swell on the key's root, when the voice MOVES -----------------
# Cued by a shift in the reader's pitch away from its recent average - emphasis, a question,
# a change of register - and landing on the next bar line of a steady tempo drawn per reading,
# so swells a minute apart keep one pulse.
const BPM_MIN := 80.0                    # the bar grid's tempo, drawn once per reading
const BPM_MAX := 92.0
const BASS_ONSET := 0.3
const BASS_CHANCE := 0.7                 # a pitch move is answered this often at 1.0
const BASS_SHIFT := 2.0                  # semitones from the recent average that count as a move
const BASS_PITCH_TAU := 4.0              # seconds: what "recent average" means
const BASS_COOLDOWN := 4.0               # one move, one swell, not a swell per syllable
const BASS_ATTACK := 0.35
const BASS_DECAY := 2.4                  # exponential time constant, seconds
const BASS_LEVEL := 1.0
const BASS_ROOT_LO := 41.0               # the bass root sits in [LO, 2 LO)
# The key moves only when the SLOW register (`_tonic`, a 12 s average) has left it by this
# much: a key that followed each phrase's pitch hopped between keys phrase to phrase.
const KEY_HYSTERESIS := 3.0

var sample_rate := 22050
var echo_wet := 0.0            # 0..1
var echo_feedback := 0.42
var resonance := 0.0           # 0..1
## THE ROOM, which is not the echo above and is the reason both exist. The echo is
## discrete: taps at a measurable delay, which the ear hears as repeats of the
## voice however small the room is made. This is the diffuse half - a comb/allpass
## network whose response has no countable events in it at all - so it is the
## space the voice is in rather than a copy of the voice arriving late.
##
## Shared with Masking, which has had one on its bus for a while: [RoomFX] holds
## the dial and its meanings, and only the engine differs (see its class docs).
var room := RoomFX.new()
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
# Resolved from echo_wet whenever it moves (the slider is live), not per sample.
var _echo_k := -1.0            # the setting these were resolved for
var _echo_n1 := 0              # near tap, in samples behind the write head
var _echo_n2 := 0              # far tap
var _echo_gain := 0.0
var _echo_fb := 0.0
var _echo_a := 0.0             # damping coefficient
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
# The events draw from their OWN generator, so moving the dial changes what is added on
# top without re-rolling the bed's notes underneath it.
var _ev_rng := RandomNumberGenerator.new()
var _key_semi := -1000                  # the key's root, semitones from A440; -1000 = none yet
var _clock := 0                         # samples since setup
var _eighth := 1                        # samples per eighth note; a bar is eight
var _bass_at := -1                      # a swell waiting for its bar line
var _bass_hz := 0.0
var _pitch_slow := 0.0                  # the reader's recent pitch, for the bass cue
var _bass_cool := 0.0
var _bass_ph := 0.0
var _bass_w := 0.0
var _bass_env := 0.0
var _bass_rise := false

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
	_ev_rng.seed = pad_seed + 7919
	_key_semi = -1000
	_clock = 0
	_eighth = maxi(1, int(round(sr * 30.0 / _ev_rng.randf_range(BPM_MIN, BPM_MAX))))
	_bass_at = -1
	_pitch_slow = 0.0
	_bass_cool = 0.0
	_bass_env = 0.0
	_bass_rise = false
	room.setup(sr)


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
			_follow_key()
			return
		pos += win


## Turn the echo dial into an actual room: tap times, feedback, damping and gain.
##
## Everything moves together, which is the whole point - see the constants above
## for why a dial that moved only the gain was unusable on speech. The dial's own
## `echo_feedback` still sets the TOP of the feedback travel, so a caller that
## deliberately wants a long tail still gets one at a high setting.
##
## Cheap and idempotent: the slider is live, so this runs once per buffer and
## returns immediately unless the setting actually moved.
func _resolve_echo() -> void:
	var k := clampf(echo_wet, 0.0, 1.0)
	if is_equal_approx(k, _echo_k):
		return
	_echo_k = k
	var sz := maxi(1, _echo.size())
	# The tap shortens toward the fusion window as the dial comes down. `k` is
	# used raw here (not tapered) because this is about CHARACTER: the room should
	# already be small by the time the level is subtle.
	var d1 := lerpf(ECHO_NEAR, ECHO_DELAY, k)
	_echo_n1 = clampi(int(round(d1 * sample_rate)), 1, sz - 1)
	_echo_n2 = clampi(int(round(d1 * ECHO_TAP2 * sample_rate)), 1, sz - 1)
	_echo_fb = lerpf(ECHO_FB_NEAR, echo_feedback, k)
	_echo_a = 1.0 - exp(-TAU * lerpf(ECHO_LP_NEAR, ECHO_LP, k) / sample_rate)
	# The LEVEL is tapered, so the bottom of the dial is genuinely quiet on top of
	# being genuinely small.
	_echo_gain = pow(k, ECHO_TAPER)


## Process in place. `buf` is mono float samples; returns the same array so the
## copy-on-write semantics of PackedFloat32Array cannot silently drop the work.
func process(buf: PackedFloat32Array) -> PackedFloat32Array:
	if _echo.is_empty():
		setup(sample_rate)
	_resolve_echo()
	# Same discipline as _resolve_echo: the dial is live, so the room's settings
	# are resolved once here rather than per sample.
	room.prepare()
	var room_on := room.is_active()
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
			_follow_key()
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
			mix += _tick_events(dry)
			wet += mix * smoothstep(0.0, PAD_FADE_IN, pad) * pad_level * duck
			_next_note -= 1.0 / sample_rate
			if _next_note <= 0.0 and _key_semi > -1000:
				_next_note = PAD_GAP * _rng.randf_range(0.7, 1.8)
				_start_tone()

		# --- echo: a room down low, a slapback up high, darker on every repeat ---
		if echo_wet > 0.0:
			var sz := _echo.size()
			var back := _echo[(_echo_i + sz - _echo_n1) % sz] * (1.0 - ECHO_TAP2_MIX) \
				+ _echo[(_echo_i + sz - _echo_n2) % sz] * ECHO_TAP2_MIX
			_echo_lp += (back - _echo_lp) * _echo_a
			wet += _echo_lp * _echo_gain
			_echo[_echo_i] = dry + _echo_lp * _echo_fb
			_echo_i = (_echo_i + 1) % sz

		# --- the room: last, around all of it, as a room is ---
		# The same place Masking's bus chain puts it, and for the same reason: the
		# room has to hear the echo, the resonance and the bed, or the bed is
		# somewhere else than the voice is.
		if room_on:
			wet = room.tick(wet)

		# --- presence: distance is a filter first, a gain second ---
		if presence < 0.995:
			var cut := 900.0 * pow(2.0, 4.0 * presence)
			_lp += (1.0 - exp(-TAU * cut / sample_rate)) * (wet - _lp)
			wet = _lp * maxf(presence, 0.05)

		buf[i] = clampf(wet, -1.0, 1.0)
	return buf


## The key's root in Hz, or 0 before there is one.
func _key_hz() -> float:
	return 0.0 if _key_semi <= -1000 else 440.0 * pow(2.0, float(_key_semi) / 12.0)


## Hold the key on a real note, and move it only when the reader's register has moved more
## than KEY_HYSTERESIS off it - so it follows a change of speaker, not every inflection.
func _follow_key() -> void:
	if _tonic <= 0.0:
		return
	var s := 12.0 * log(_tonic / 440.0) / log(2.0)
	if _key_semi <= -1000 or absf(s - float(_key_semi)) > KEY_HYSTERESIS:
		_key_semi = int(round(s))


## The bass for one sample, in the bed's units (so it shares its level, its duck and its
## fade-in). Its cue is detected here too.
func _tick_events(dry: float) -> float:
	var dt := 1.0 / sample_rate
	var out := 0.0
	_clock += 1
	# --- cue: the voice's pitch moving ---
	_bass_cool -= dt
	if _pitch > 0.0:
		_pitch_slow = _pitch if _pitch_slow <= 0.0 else lerpf(_pitch_slow, _pitch, dt / BASS_PITCH_TAU)
		if _bass_cool <= 0.0 and absf(12.0 * log(_pitch / _pitch_slow) / log(2.0)) > BASS_SHIFT:
			_bass_cool = BASS_COOLDOWN
			if _key_semi > -1000 and _bass_at < 0 and _ev_rng.randf() < _chance(BASS_ONSET, BASS_CHANCE):
				_plan_bass()
	if _bass_at >= 0 and _clock >= _bass_at:
		_bass_at = -1
		_bass_w = TAU * _bass_hz / sample_rate
		_bass_rise = true
	# --- the bass ---
	if _bass_env > 0.0 or _bass_rise:
		if _bass_rise:
			_bass_env += dt / BASS_ATTACK
			if _bass_env >= 1.0:
				_bass_env = 1.0
				_bass_rise = false
		else:
			_bass_env *= exp(-dt / BASS_DECAY)
			if _bass_env < 1e-4:
				_bass_env = 0.0
		_bass_ph = fmod(_bass_ph + _bass_w, TAU)
		# The fundamental plus two harmonics: a small speaker cannot play the root itself, so
		# the harmonics are what make the note there at all.
		var v := sin(_bass_ph) + 0.45 * sin(2.0 * _bass_ph) + 0.18 * sin(3.0 * _bass_ph)
		out += v * _bass_env * _bass_env * BASS_LEVEL
	return out


## How often a cue is taken at the current dial: never below `onset`, rising (x^1.5, so
## the lower part of the range stays sparse) to `top` at 1.0.
func _chance(onset: float, top: float) -> float:
	var x := clampf((pad - onset) / (1.0 - onset), 0.0, 1.0)
	return top * pow(x, 1.5)


## A bass swell on the key's root (sometimes the fifth), waiting for the next bar line.
func _plan_bass() -> void:
	var hz := _key_hz()
	if _ev_rng.randf() < 0.25:
		hz *= 1.5
	while hz >= BASS_ROOT_LO * 2.0:
		hz *= 0.5
	while hz < BASS_ROOT_LO:
		hz *= 2.0
	_bass_hz = hz
	var bar := _eighth * 8
	_bass_at = (_clock / bar + 1) * bar


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
	var hz: float = _key_hz() * pow(2.0, float(degree + octave) / 12.0)
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
