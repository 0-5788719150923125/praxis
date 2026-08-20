extends RefCounted
class_name Voice

## Voice - the source-filter speech synthesizer (rungs 0-2 of next/voice.md).
##
## Klatt-lineage formant synthesis, all in-house: a glottal **source** (Rosenberg
## pulse wavetable + aspiration noise), a cascade of three formant resonators as
## the **filter** (per-phoneme targets from [Phonemes], EMA-smoothed across
## segments = coarticulation), and a Fujisaki-style **prosody** realization -
## declination as a slow drift, accent bumps on stressed syllables, everything
## converging through EMAs because that is the accepted model of human f0, not a
## trick. No generative AI, no recordings; deterministic per (text, spec seed).
##
## A voice is a [Voice.Spec]: a typed parameter bag sampled from ranges
## ("cattle, not pets" over the space of voices - reroll the seed, get a new
## speaker). `render(text, spec)` returns the PCM plus a word/phoneme timing map
## (alignment is known by construction - we synthesized it), which is what the
## karaoke subtitles and any future landmark labels key from. `write_wav` saves
## PCM16 the rest of ghost (Spectrum, the exporter) can play like any song.

# 44100 native (2026-07-26 fidelity pass; was 22050): doubles the audio
# bandwidth ceiling AND removes Godot's 22050->44100 playback resampling -
# the generator now runs at the device rate, so no linear-resample imaging.
# Everything downstream derives from SR; the per-BLOCK constants
# (LIMIT_RELEASE, the leveler's attack/release) were rescaled to keep their
# time constants, since a 64-sample block is now ~1.45 ms, not ~2.9.
const SR := 44100
# Glottal wavetable resolution. 64 points reconstructed a Rosenberg pulse's
# sharp closure - the event that makes essentially all of its high-frequency
# content - as a piecewise-linear corner whose position quantized to 1/64 of a
# period, so the closure edge landed differently every cycle. That is the
# "synthetic grain" the reference read described. 512 costs 4 KB.
# Melody: how many semitones the ORNAMENTS may contribute in total, on top of
# the structural reading. Ornaments are the motif tilt, the activation ring,
# microprosody and the ProsodyField wander.
const MELODY_DECOR := 2.5
# Spontaneous hesitations and filled "uh"s. A property of spontaneous speech,
# not of a person, and roughly 900 unrequested "uh"s across a 140k-word book.
# Belongs to a future spontaneous register, not to the narrator.
const DISFLUENCY := false
const PULSE_N := 512
const FRAME := 64                     # samples per parameter update (~1.45 ms)
const TWO_PI := TAU
# Fixed output gain in place of retroactive normalization (streaming cannot
# know the future peak). This number is STAGED BY MEASUREMENT and re-staged
# whenever the physics under it move, which they have three times: the 22050
# calibration, the 44.1 kHz move, and now the 2026-08-04 rebuild (zero-mean
# source, parallel frication branch, sixth cascade pole, Klatt bandwidths).
# Every staging number before that rebuild was measuring something other than
# the voice: first S-noise setting the peaks, then a DC pedestal holding
# 43-52% of the take's power. See `python build/scratchpad.py run` for the
# harness that produced the current value.
# Restaged 2026-08-08: the upper-pole cluster and the frication radiation fix
# together moved ~13 dB into 2-4 kHz, which pinned every fixture's peak at the
# lookahead limiter (-1.54 dBFS on all three). Never stage peaks over the
# ceiling and rely on saturation for loudness - that rule is already written
# into next/voice_rca.md round four, and the physics under it just moved.
const OUT_GAIN := 3.2
# The echo bus: a feedback delay line the output always passes through. Sends
# are zero except on echo-activated words, so the line is silent until a word
# is thrown into it - then it rings, decaying, through whatever follows
# (including the pauses). Fixed model-agnostic constants, per the house rule.
const ECHO_DELAY := 0.17              # seconds
const ECHO_FB := 0.45                 # feedback per repeat
# The echo line is DAMPED: a one-pole lowpass inside the loop, so every
# repeat comes back darker (physical echoes lose their top). Undamped, an
# echoed fricative-heavy word rang raw S-static through the following
# pause - measured as ~200 ms aperiodic 3-11 kHz bursts at ~-27 dB in gaps,
# landing wherever the walk's sparse echo gate fired: the "brief bursts of
# noise injected at random places" report. A ~15 dB/pass cut at 8 kHz kills
# the static by the first repeat while voiced ringing survives.
const ECHO_LP := 1400.0               # echo damping cutoff (Hz)
# The BROADCAST stage v3: a one-block LOOKAHEAD peak limiter. The v2 chain
# (AGC toward a sine target -> cosine dampener -> masking static) measured as
# the dominant artifact source (see next/voice_rca.md): speech has a crest
# factor of ~13x, not a sine's 1.4x, so the AGC pegged at max boost ~95% of
# the time and drove the dampener to ZERO one voiced sample in ten - it
# manufactured the crackle it existed to prevent. The limiter uses the one
# advantage a synthesizer has over a radio station: the next block is already
# known before the current one is emitted. Output runs one 64-sample block
# behind synthesis; each incoming block's peak sets a LINEAR gain ramp across
# the outgoing block, so no sample ever exceeds LIMIT and the gain curve has
# no corners (a corner is a click). Clean passages pass through untouched.
# Broadcast v4 peak handling: the lookahead RAMP is the peak authority (it
# steers every block's peak under LIMIT, cornerless and distortion-free);
# the soft clip is only a SAFETY NET for the tip that remains - IDENTITY
# below KNEE, rounding into CLIP_CEIL above it. The v3.5 always-on tanh
# (ceiling 0.8, peaks staged 2x over it) compressed the entire top half of
# the range on every take - that was the "bursts of noise" class.
const LIMIT := 0.85                   # the ramp steers block peaks under this
                                      # (at 0.92 the knee still shaved a 0.036
                                      # tip off the hottest pulse per block;
                                      # 0.85 keeps even that under ~0.013)
const KNEE := 0.7                     # soft clip is identity below this
const CLIP_CEIL := 0.98               # asymptote for residual overshoot
const LIMIT_RELEASE := 1.022          # gain recovery per block (~70 ms to unity)
# The syllable LEVELER, sharing the limiter's ramp: a bounded 2:1 compressor
# on the block envelope - the honest version of what the old AGC reached for.
# Speech here has a ~22 dB crest, so gain staging alone cannot land the take
# at a human-audible RMS without the ceiling chewing on every stressed vowel;
# the leveler evens vowel peaks a few dB and lifts quiet stretches a few dB,
# hard-bounded so the walk's dynamics (emphasis, swells, arousal) survive.
const COMP_TARGET := 0.42             # envelope level the leveler steers toward.
                                      # NOTE the units: fast attack + slow release
                                      # means cenv rides a vowel's NEAR-PEAK
                                      # envelope (~0.35-0.5 here), not its mean -
                                      # a "mean-|x|"-sized target quietly cut
                                      # every vowel by ~3 dB and pinned the RMS
const COMP_MIN := 0.62                # never cut more than ~4 dB
const COMP_MAX := 1.5                 # never lift more than ~3.5 dB
# The static bed is now ONLY the medium's grain: a constant faint noise floor.
# The adaptive masking bed (rise where the limiter worked) is gone - measured,
# it either idled at this floor (the limiter almost never works that hard) or,
# when a loud stretch did engage it, its slow floor ratchet (~60 s settle)
# held hiss up long after the moment had passed - the "never normalizes"
# report. The tanh ceiling's own warmth is all the cover hard moments need.
const FLOOR_MIN := 0.0015             # the permanent faint grain (~-56 dB)
# NOISE FX kill switch (2026-07-26): false silences every NON-PHONEMIC noise
# injection - the pitch-synchronous aspiration hiss (spec.breath), the
# air/static band (spec.air_gain), and the constant output grain (FLOOR_MIN).
# Fricatives, stop bursts and VOT keep their noise: that IS the consonants.
# Reported as "brief bursts of noise at random places": both voiced
# injections scale with the amplitude envelope, so the walk's emphases and
# swells pump them louder on effectively random words. The rng draws still
# happen when disabled, so flipping this flag A/Bs the SAME take with and
# without the noise. Hardcoded on purpose - flip to restore the feature.
# 2026-07-26 (final restore rung): back ON - the noise saga's culprit was
# frication routing, not these; breath/air are also part of the fidelity
# answer (the air band is most of the voice's top-octave life).
# 2026-08-08 (intelligibility program, next/voice_intelligibility.md): back OFF.
# Measured: breath + air supply 90.1% of vowel power at 2.5-4 kHz and 84.9% at
# 4-6 kHz, injected PRE-cascade so the formants shape them onto the voice. That
# is exactly the band the consonant cues live in, and phonemes.gd:80-82 states
# the design principle it destroys ("the cue is spectral CONTRAST against the
# neighbour, not level"). Fricative-vs-vowel contrast at 2.5-5 kHz: -4.05 dB
# with this noise, +5.96 dB without. The "top-octave life" it was restored for
# is real but it costs 10 dB of consonant contrast to buy. The principled end
# state is aspiration tied to the glottal open quotient and seated ~10-12 dB
# lower; until that lands, off.
# 2026-08-09: back ON, but TRIMMED. Turning it off bought +10 dB of consonant
# contrast and cost the masking that had been hiding an artifact class we now
# know is INHERENT to this method rather than a bug in ghost: the user compared
# against DECtalk, the commercial gold standard of Klatt-lineage synthesis, and
# found the same clicks and static there, quieter and less frequent. Gating
# noise sources on and off and retuning ringing resonators produces transients;
# DECtalk masked its own under an 8-bit-era noise floor. Seven separate
# mechanisms were tested and refuted here before that comparison was made.
#
# So the goal is minimize-and-mask, not eliminate. NOISE_TRIM seats breath and
# air ~12 dB under their old level: enough floor to bury the transients, far
# short of the 90% of vowel power at 2.5-4 kHz they used to supply.
const NOISE_FX := true
const NOISE_TRIM := 0.25
# RAW BYPASS (diagnostic, 2026-07-26): true = every seed plays the BASE
# synthesis and nothing else, to bisect "base synthesis vs modulations".
# Disables, at plan time: the walk's realized modifiers (pace, emphasis,
# activations, spontaneous hesitations, breath debt, motif tilt, ring,
# anchor gravity) and the ProsodyField wander - the walk still ADVANCES and
# the strike EVENTS still fire, so fishing bites keep working; only their
# audible realization is neutralized. At synth time: per-period jitter/
# shimmer/tension wander and the whole broadcast chain (leveler, lookahead
# limiter, safety clip, grain) - blocks go out under one fixed trim.
# [VoiceStream] also bypasses presence/location at push time when this is
# set. What REMAINS: phonemes, coarticulation EMAs, declination, accents,
# vowel reduction, punctuation pauses, terminal contours, VOT, and the
# trait vector itself. NOISE_FX above stays independently off. Restore the
# full instrument by flipping RAW_MODE false (and NOISE_FX true).
# 2026-07-26 (end of the bisect): back OFF - the noise was the frication
# (found and fixed via the pure_say ladders); the modulation stack returns
# with OUT_GAIN restaged for the S-clean signal. NOISE_FX stays off as the
# last restore rung.
const RAW_MODE := false
# THE RESONANCE - ambience from the voice itself. A bank of SYMPATHETIC
# STRINGS: ultra-narrow resonators tuned to the reading's ANCHOR notes (the
# same shelf the melody gravitates toward - each seed carries its own chord),
# excited by the voice's own output. When the melody lands on one of its
# notes the string blooms and RINGS for seconds, holding long sustained
# tones under and between the phrases - a drone that is not a separate
# instrument but the voice resonating in its own space. Deterministic,
# baked into the take (exports carry it; the scenes hear it). Static vars so
# tests can silence the bank; the app treats them as constants.
# 2026-08-08: default OFF. Measured word-boundary trough depth at a 40 ms gap:
# -15.5 dB with the bank, -31.8 dB without (80 ms: -16.6 vs -33.5). The strings
# hold a sustained floor through every pause, so silence never arrives and the
# listener never finds where words begin. It is a good atmosphere effect and a
# bad default for speech; the fishing session can turn it back on.
static var DRONE := false
const DRONE_STRINGS := 6              # tones, tuned to the first anchor notes
const DRONE_NEAR := 2.5               # st range of a note's proximity GLOW
const DRONE_ATTACK := 0.3             # s for a landed note to swell in
const DRONE_RELEASE := 4.0            # s for a left note to fade (the long hold)
const DRONE_LEVEL := 0.0025           # per-tone level into the mix (pre OUT_GAIN):
                                      # measured, the summed glow sits ~-18 dB
                                      # under the take - ambience, not a duet
# The CONSONANCE TIDE: the ensemble breathes between two states. At LOW tide
# the budget is tight and the LFOs deep - tones compete, ebb, pluck as
# individuals. At HIGH tide the budget opens and the swells steady - the
# tones are ALLOWED to gather into a held CHORD. The tide is a very slow
# seeded cycle (one turn per ~25-50 s) blended with how sustained the voice
# has been, so dense passages bloom chordal and sparse ones scatter.
const DRONE_BUDGET_LO := 1.2          # tight: individuals (the ebb/pluck state)
const DRONE_BUDGET_HI := 3.2          # open: the chord is allowed to stand
# Each tone has a seeded CHARACTER (deterministic per lineage root + note):
#   drone - slow attack, very long release, shallow slow breathing
#   swell - long breathing attack, DEEP slow LFO: it ebbs, recedes, returns
#   pluck - an EVENT, not a glow: fires when the melody ARRIVES on its note,
#           instant attack with brighter partials, ~1 s decay
# (a PASSIVE resonator bank was tried first and barely rang: a 1.6 Hz band
# never accumulates energy from a jittering, vibrato-laden f0. The tones are
# ACTIVE instead - envelope followers on melodic proximity - which is also
# exactly the stated behavior: notes that trigger from the voice and hold.)
const RAW_TRIM := 4.0                 # raw (RAW_MODE) output trim, staged by
                                      # the same measurement pass as OUT_GAIN
# FRICATION - the end of the multi-round "noise injection" saga. Both earlier
# routes were wrong, for OPPOSITE reasons, and the ear was right both times:
#   post-cascade band  - added AFTER the tract, sharing no formant shaping with
#                        the voice, so the ear segregated it as a second source
#                        ("injected static").
#   through-the-cascade - shaped by the tract, so it FUSED - but the cascade is
#                        a DC-normalized all-pole chain whose top pole sits at
#                        ~4.7 kHz, and it removed 42 dB at /f/'s band, 69 dB at
#                        /th/'s and 85 dB at /s/'s. Measured consequence: the
#                        S/SH/F/TH log-spectra correlated at 0.99-1.00, three
#                        of the four peaked in the SAME FFT bin (1367 Hz), and
#                        6-10 kHz contrast against the neighbouring vowel was
#                        -0.2 dB. One sound wearing four labels; "the
#                        enunciation simply isn't there" was literal.
# The answer is the one Klatt 1980 shipped and that this file only ever had
# half of: a PARALLEL branch. Frication and bursts drive their own front-cavity
# resonators (Phonemes.TABLE `par`), summed with the cascade output before the
# radiation stage - shaped like a consonant, fused by sharing the lip
# transform and the phoneme's tract POSTURE, but never lowpassed by the vowel.
# FRIC_LEVEL seats the whole branch; per-phoneme balance lives in the table.
# A static var (not a const) ONLY so tests/pure_say.gd can sweep it.
static var FRIC_LEVEL := 0.075
# Reference bandwidth for the parallel branch's power normalization (see
# _tune_parallel). /s/'s principal pole is 500 Hz wide, so referencing here
# keeps the normalizer near unity for a sibilant and leaves FRIC_LEVEL's
# staging in the decade it was already in.
const PAR_REF_BW := 500.0
# Bursts get their own seat. A burst is a transient and a fricative is a steady
# state; sharing one scaler meant neither could be set correctly, and the
# measured result was every release sitting 24-37 dB under the adjacent vowel
# where natural bursts run 8-15 dB under. Staged from measurement, not taste -
# see measure_voice.py and the stop_voiceless/stop_voiced level gates.
# 0.35 -> 0.7 (2026-08-08): releases measured -16.5 to -18.0 dB re the adjacent
# vowel against a -15 to -5 dB target, and natural bursts run 8-15 dB down. The
# user's report was that the sharpest sound in the take was a spurious click
# rather than any stop release - the releases were simply too quiet to read as
# enunciation.
const BURST_GAIN := 0.7
# RADIATION - lip radiation is a differentiator, realized as `y - RAD_A*y[n-1]`.
# The coefficient is a CORNER FREQUENCY, not a magic number: hardcoding 0.96
# pinned the corner to the sample rate, so the 22050 -> 44100 move silently
# doubled it (143 -> 287 Hz) and tilted +3.95 dB of low end back in, which was
# then compensated with a flat OUT_GAIN lift - a level fix for a tilt problem.
const RAD_CORNER := 140.0             # Hz; SR-independent by construction
static var RAD_A := exp(-TWO_PI * RAD_CORNER / SR)
# How long before a segment ends the tract starts moving toward the NEXT
# posture. Real English formant transitions run 40-80 ms; the locus is reached
# at the boundary and the EMA carries the rest into the following phone.
const LOCUS_TIME := 0.050             # seconds
# ...but never more than this share of the segment, so a short glide or nasal
# still reaches its own target before it starts leaving it.
const LOCUS_SHARE := 0.35
# VIBRATO - the cue that separates a held SUNG note from a held robotic tone.
# 5.5 Hz is the centre of the measured human range (5-7 Hz); depth is in
# semitones and is scaled by `Spec.song`, so a speaking voice has none. It
# ramps in over VIB_ONSET so short notes do not wobble - vibrato on a 90 ms
# syllable reads as a fault, not as singing.
const VIB_RATE := 5.5                 # Hz
const VIB_DEPTH := 0.38               # semitones at song = 1
const VIB_ONSET := 0.18               # seconds for the waver to reach full depth
# How often a WILD roll sings at all. Not 0.5: the belt compounds whatever it is
# given, so the roll has to sit below the rate you actually want to meet.
const SONG_INCIDENCE := 0.28
# A sung syllable that is NOT sustained runs FASTER than it would be spoken.
# This is what makes a cadence a cadence: without it every note is long, the
# reading takes twice as long as the same text spoken, and the result is a drone
# rather than a phrase. Real singing alternates - held notes at the joints, runs
# in between, and the contrast is the music.
const SONG_RUN := 0.72
# The sustain gate: a syllable is held when a slow seeded cycle, its prominence
# and its phrase position add up past this. Sustained fraction lands ~20-30%.
const SUSTAIN_BAR := 0.92
const SONG_RUN_MIN := 0.075          # s; a run note still has to be audible as a pitch
# CASCADE BANDWIDTHS (Klatt 1980 nominal). These were `60 + F1*0.06` and
# `90 + F2*0.05`, which put B2 near 145 Hz for a back vowel - roughly twice
# natural - smearing F2 into F1 and costing 3.4 to 7.1 dB of F2 prominence.
# They are damping values, so they do NOT scale with vocal tract length.
# B5 300 -> 200 to match Klatt Table I, so F4 and F5 can form the deliberate
# narrow cluster described below rather than two separate shallow humps.
const BW := [60.0, 90.0, 150.0, 250.0, 200.0, 500.0]
# The sixth cascade pole. Five poles topping out at 4.7 kHz put the -60 dB/oct
# cliff right where speech still needs energy: measured 27 dB of drop between
# 3150 and 5000 Hz, and above 5 kHz the output was bit-for-bit the dither
# floor. A uniform 17.5 cm tube keeps resonating at ~1.1 kHz intervals; F6
# continues the series and moves the cliff up an octave. This costs high end
# above 10 kHz (each added pole steepens the asymptote) - which is free now
# that frication has left the cascade for the parallel branch.
const F6 := 6000.0


## The speaker's trait axes, each in [-1, 1]. THE TRAIT VECTOR IS THE VOICE:
## the zero vector is the hand-curated default speaker (its concrete centres
## live in Spec.from_traits - tune them there), a seed only *initializes* the
## vector, and any UI modulation edits it directly - so a speaker is replicated
## by replaying the vector, never by replaying the gesture that found it.
const TRAIT_KEYS := ["pitch", "lilt", "tract", "pace", "breath", "grit", "drawl", "air", "song"]

## One voice: a trait vector realized into concrete synthesis parameters.
class Spec:
	var seed_value := 0
	var traits := {}                  # trait key -> [-1, 1]; {} = the curated default
	var reading: Array = []           # the READING's lineage: a linear chain of seeds.
	                                  # [0] samples the prosody genome; each later seed
	                                  # perturbs it with decaying strength (refinement,
	                                  # not a re-roll). Captured seeds ARE the labels.
	var influences: Array = []        # toggled belt lineages blended into the walk
	                                  # (each an Array of seeds); the population PRIOR
	                                  # joins automatically - the 1 of 1+N.
	var adrenochrome := {}            # a FROZEN genome, annealed during the hook
	                                  # (see the editor's reel): when present it
	                                  # replaces the lineage-derived walk genome -
	                                  # the seed's identity (motifs, anchors, gates)
	                                  # still flows from the lineage.
	var f0_base := 130.0              # speaking pitch floor (Hz)
	var f0_accent := 2.8              # accent bump strength (semitones)
	var f0_decl := 3.0                # declination span per sentence (semitones)
	# INFLECTION DEPTH: a single scalar on ALL melodic deviation from f0_base -
	# accents, declination, the wander, the attractor pull, terminals, the lot.
	# 1.0 is the natural speaker; toward 0 the whole contour collapses to a FLAT
	# monotone (a robot/whisper register), above 1 it sings. Decoupled from
	# f0_base, so a high, flat voice and a low, swooping one are both reachable.
	var inflect := 1.0
	var formant_scale := 1.0          # vocal tract length (bright .. dark)
	var rate := 1.0                   # tempo multiplier (>1 = faster)
	var breath := 0.05                # aspiration mixed into voiced frames
	var jitter := 0.012               # per-period f0 noise (organic, not robotic)
	var shimmer := 0.04               # per-period amplitude noise
	var pause_comma := 0.18           # seconds
	var pause_stop := 0.42
	var final_lengthen := 1.25        # phrase-final syllable stretch
	# SONG in [0, 1]: how much this voice SINGS rather than speaks. Not a style
	# layer on top of speech - it swaps four behaviours at once, because that is
	# what the difference actually is:
	#   notes    - vowels stretch onto a BEAT GRID instead of taking their
	#              natural length, and prominent syllables take more beats than
	#              weak ones, which is what makes a cadence rather than a drone
	#   steps    - the f0-continuity glide is switched off, so the melody moves
	#              in discrete steps held flat across a note, and the note
	#              SUSTAINS through the consonants instead of dipping
	#   scale    - the anchor-shelf pull goes to ~1, so pitches land ON notes
	#   vibrato  - a periodic 5.5 Hz waver, which is the cue that separates a
	#              held sung note from a held robotic tone
	# Archaeology: the earliest build (11d3c2a8, "ship it") had no continuity
	# pass, no wander and no attractor shelf - `semitones` was decl + accent,
	# held flat per segment, and silences reset to base. That accidental
	# staircase is what sounded like singing, and every naturalness fix since
	# has been sanding it off. This makes it a capability instead of an
	# accident, so the speaking voice keeps its glide and a sung voice can ask
	# for the staircase on purpose.
	var song := 0.0
	var beat := 0.42                  # seconds per beat when singing
	var sustain_period := 5.0         # syllables per sustain cycle
	var sustain_phase := 0.0          # where in that cycle this voice starts
	# THE SUSTAIN BANK: [period, phase, weight] per contributing voice. One
	# cycle can only decide HOW MUCH a voice sings; several incommensurate
	# cycles decide WHERE it sings, and that is the difference between a belt
	# that makes a stronger signal and a belt that makes a different one. Signed
	# weights, so a contributor can suppress a position another one holds rather
	# than only ever adding notes. Empty = the voice's own single cycle, which
	# is exactly the behaviour before the bank existed.
	var sustain_bank: Array = []
	var air_gain := 0.07              # static-band strength (noise above the air line)
	var air_cut := 3000.0             # the air line: above it the voice goes to static

	## Realize a trait vector. The constants here ARE the curated default
	## speaker (all traits 0); each trait bends one perceptual axis around it,
	## exponentially where perception is log-shaped (pitch, tempo).
	static func from_traits(t: Dictionary, seed_value_ := 0, reading_: Array = []) -> Spec:
		var s := Spec.new()
		s.seed_value = seed_value_
		s.reading = reading_.duplicate() if not reading_.is_empty() else [seed_value_]
		s.traits = t.duplicate()
		var pitch := _tv(t, "pitch")
		var lilt := _tv(t, "lilt")
		var tract := _tv(t, "tract")
		var pace := _tv(t, "pace")
		var breath := _tv(t, "breath")
		var grit := _tv(t, "grit")
		var drawl := _tv(t, "drawl")
		var air := _tv(t, "air")
		var song := _tv(t, "song")
		s.f0_base = 130.0 * pow(2.0, 0.85 * pitch)
		# LILT is now the master inflection depth: it no longer scales the accent
		# and declination in isolation (which left the wander and the attractor
		# pull - the real "sing-song" - untouched); instead it drives `inflect`,
		# a global scale on the whole melodic contour. lilt 0 = the natural
		# speaker (unchanged), toward -1 flattens to a monotone, toward +1 sings.
		s.f0_accent = 2.8
		s.f0_decl = 3.0
		s.inflect = clampf(1.0 + lilt, 0.06, 2.0)
		s.formant_scale = pow(2.0, 0.22 * tract)
		# widened from 0.35: the slow end could only reach 1.28x slower than
		# neutral, which is not slow enough for a deliberate reading
		s.rate = pow(2.0, 0.55 * pace)
		s.breath = 0.05 * pow(2.5, breath)
		s.jitter = 0.012 * pow(2.2, grit)
		# shimmer above ~8% reads as pathological roughness, not character -
		# the old 13% top of range was part of the crackle
		s.shimmer = 0.04 * pow(2.2, grit)
		s.pause_comma = 0.18 * pow(1.6, drawl)
		s.pause_stop = 0.42 * pow(1.6, drawl)
		s.final_lengthen = 1.25 * pow(1.25, drawl)
		# the air trait: how much of the upper spectrum tunes to static, and
		# where that line sits (high air = the line drops, more of the voice
		# is breath-noise - the multi-band harmonic/noise mix). Centre dropped
		# 0.07 -> 0.02 in the 2026-08-04 rebuild: the air band existed to give
		# a five-pole cascade some top, and was measured supplying 96-99% of
		# the vowel's brightness - i.e. the "bright" was hiss, not voice. With
		# F6 on the cascade the harmonics carry their own top, and the hiss was
		# actively MASKING consonants: it sat squarely in /sh/'s 2-5 kHz band
		# only the upper half of the axis sings, so the curated default and half
		# the population stay pure speech and singing is something you FIND
		s.song = clampf(song, 0.0, 1.0)
		# the beat follows the speaker's own tempo and drawl - a slow, drawling
		# voice sings slow - so it needs no separate roll
		s.beat = clampf(0.42 / s.rate * pow(1.45, drawl), 0.16, 1.10)
		# how many syllables per sustain cycle, and where in that cycle this
		# voice starts. Derived from traits the speaker already has, so a
		# drawling singer holds notes further apart than a brisk one and no
		# separate roll is needed.
		var cyc := sustain_cycle(t)
		s.sustain_period = float(cyc[0])
		s.sustain_phase = float(cyc[1])
		s.air_gain = 0.02 * pow(2.6, air)
		s.air_cut = 3000.0 * pow(2.0, -0.7 * air)
		return s

	## A voice's own sustain cycle - [syllables per cycle, phase] - from the
	## traits it already has, so a drawling singer holds notes further apart than
	## a brisk one. Static so the belt can ask what any member's cycle would be
	## without realizing a whole Spec for it.
	static func sustain_cycle(t: Dictionary) -> Array:
		var drawl := _tv(t, "drawl")
		var pace := _tv(t, "pace")
		var lilt := _tv(t, "lilt")
		var grit := _tv(t, "grit")
		return [clampf(5.0 * pow(1.7, drawl) * pow(1.4, -pace), 3.0, 16.0),
			fposmod(lilt * 3.7 + grit * 1.3, 1.0)]


	## The sustain drive at syllable `i`: the weighted sum of every contributing
	## cycle, mapped to [0, 1]. Incommensurate periods mean the held positions
	## form a long non-repeating pattern rather than a metronome, and signed
	## weights mean contributors can cancel - so two seeds do not merely sing
	## louder together, they sing in different PLACES together.
	static func sustain_wave(spec: Spec, i: int) -> float:
		if (spec.sustain_bank as Array).is_empty():
			return 0.5 + 0.5 * sin(TAU * (float(i) / spec.sustain_period + spec.sustain_phase))
		var acc := 0.0
		var wsum := 0.0
		for e in spec.sustain_bank:
			var period: float = maxf(float(e[0]), 1.0)
			acc += float(e[2]) * sin(TAU * (float(i) / period + float(e[1])))
			wsum += absf(float(e[2]))
		return clampf(0.5 + 0.5 * acc / maxf(wsum, 0.001), 0.0, 1.0)


	static func _tv(t: Dictionary, key: String) -> float:
		return clampf(float(t.get(key, 0.0)), -1.0, 1.0)

	## A seeded roll of the trait vector - BIMODAL by register: the roll first
	## picks a speaker register (male / female), which sets correlated pitch
	## and vocal-tract centres far apart, then scatters the remaining traits
	## widely. Rolls are meant to sound like DIFFERENT PEOPLE, not takes of one.
	static func sample(rng: RandomNumberGenerator) -> Spec:
		var seed_value_ := int(rng.seed)
		var register := -0.75 if rng.randf() < 0.5 else 0.75
		var t := {
			"pitch": clampf(register + rng.randfn(0.0, 0.25), -1.0, 1.0),
			"tract": clampf(0.6 * register + rng.randfn(0.0, 0.2), -1.0, 1.0),
		}
		# SONG is a MODE, not a shade, so it is not drawn like the timbre axes.
		# As a plain N(0, 0.55) it put half of all wild rolls above zero, and the
		# belt then compounds that: acceptance-weighted parents plus a trust
		# region centred on the party mean means one kept singer pulls its whole
		# line toward singing, which is how ~90% of found seeds ended up sung.
		# An explicit incidence keeps singing something you FIND rather than the
		# default, and the negative draw is pushed well clear of zero so a
		# non-singer's children do not drift across the line by jitter alone.
		t["song"] = rng.randf_range(0.15, 1.0) if rng.randf() < SONG_INCIDENCE \
			else rng.randf_range(-1.0, -0.25)
		for key in TRAIT_KEYS:
			if not t.has(key):
				t[key] = clampf(rng.randfn(0.0, 0.55), -1.0, 1.0)
		return from_traits(t, seed_value_)


## The ModBank move applied to speech: seeded oscillators stacked at several
## TIMESCALES (phrase ~4s, breath group ~1.4s, word ~0.45s), each timescale a
## couple of incommensurate sinusoids, summed per channel (pitch semitones,
## tempo, loudness). This is the continuous-harmonic dynamics layer: the
## completely-linear reading was declination alone; the field makes the melody
## wander the way attention does - slowly at the phrase scale, faster at the
## word scale - deterministically per voice seed.
class ProsodyField:
	var _osc := {}                   # channel -> [[amp, freq_hz, phase], ...]

	func _init(seed_value: int) -> void:
		var rng := RandomNumberGenerator.new()
		rng.seed = hash("prosody_field") ^ seed_value
		for channel in ["f0", "rate", "amp"]:
			var bank: Array = []
			for scale in [[4.0, 1.0], [1.4, 0.55], [0.45, 0.3]]:  # [period s, weight]
				for _i in 2:
					bank.append([
						scale[1] * rng.randf_range(0.6, 1.2),
						(1.0 / scale[0]) * rng.randf_range(0.7, 1.4),
						rng.randf_range(0.0, TAU),
					])
			_osc[channel] = bank

	func sample(channel: String, t: float) -> float:
		var v := 0.0
		for o in _osc[channel]:
			v += o[0] * sin(TAU * o[1] * t + o[2])
		return v


## The stateful half of the reading - what the oscillator field cannot do,
## because a field has no memory. The walk is advanced word by word through the
## text and everything it does is an approximation of what comes next based on
## limited history, all EMAs:
## - **arousal** comes in hot (the genome's `heat`), decays toward a baseline,
##   and is re-excited by sentence starts. It drives PACE: hot reads fast,
##   settled reads slow - the voice arrives quick, then slows down and breathes.
## - **spent** is a sliding-window EMA of recently spent emphasis: an emphasis
##   raises it, which suppresses the next one until it decays - so emphases
##   space themselves out instead of everything (or nothing) being stressed.
## - **breath debt** accumulates per syllable and forces a real pause (longer
##   when calm) even where the text has no punctuation.
## - **motifs**: a small seeded vocabulary of sentence gestures (tilt, lean,
##   gap) the voice reuses - recognizable habits, not one endless wander.
##
## The genome is sampled through a LINEAGE (a linear chain of seeds): the root
## seed samples every parameter, each later seed perturbs them with decaying
## strength (0.6^generation) - refinement around a kept parent, not a re-roll.
## Deterministic per (lineage, text): a captured lineage IS a reproducible
## reading, which is what the belt collects.
class ProsodyWalk:
	var arousal := 1.0
	var spent := 0.0
	var breath := 0.0
	var motif := {}
	var p := {}                       # the genome
	var _motifs: Array = []
	var _anchors: Array = []          # pitch attractors (semitones), pooled 1+N
	var _refract := {}                # per-channel fast-attack / slow-decay bar
	var _ring_amp := 0.0              # resonance: a firing rings; the ring decays
	var _ring_ph := 0.0
	var _swing := 0.0                 # cadence wobble: activations kick it, it decays -
	                                  # a perturbation folded back into the running pace
	var _gate: RandomNumberGenerator  # order-dependent stochastic gates (deterministic)
	var _mods: Array = []             # the finalized modulator set (blended, damped, pruned)
	var _t := 0.0                     # utterance clock (seconds) the modulators ride

	## The population average, created at initialization and OUTSIDE any
	## lineage's influence: the midpoint of every genome range. It is always
	## one voice in the blend (the "1" of 1+N) - the regularizer that keeps a
	## belt full of extremes from compounding.
	const PRIOR := {
		"heat": 1.35, "baseline": 0.375, "settle": 0.11, "breath_span": 9.5,
		"spend_window": 2.4, "lean": 1.0, "pace_hot": 0.91, "pace_calm": 1.21,
		"act_thr": 1.9, "act_gain": 1.0, "gravity": 0.2, "ring": 0.6,
		"hesit_bias": 0.25, "swing_kick": 0.14, "verve": 0.4, "damp": 0.35,
	}
	const PRIOR_MOTIF_SEED := 314159

	# THE MODULATOR GENES - the alchemy. A lineage does not only refine scalars; it
	# can spawn OSCILLATORS that ride a gene over the utterance, giving the voice a
	# jagged route instead of a held level: a slow sine sway, a cosine that leads by
	# a quarter turn, a triangle that ramps and reverses, a sawtooth that snaps back.
	# Each generation may add one. But you cannot pile them on forever - that is the
	# whole point. Four regularizers turn accumulation into BLENDING:
	#   DAMPEN   - the `damp` gene scales every modulator's depth down; a heavily
	#              damped lineage barely wavers however many it has spawned.
	#   NORMALIZE - one fixed depth budget across ALL of a party's modulators. Add a
	#              new one and it does not stack on top; it STEALS energy from the
	#              rest. The set is renormalized to the budget, so the total motion
	#              is bounded and a new gesture must earn its share by diluting the
	#              others. This is the blend.
	#   SUPPRESS  - once diluted, any modulator under a floor is dropped: the weak
	#              ones the blend pushed under the threshold are pruned, not carried
	#              dead. Bad features get suppressed rather than accumulating.
	#   AGE       - recency decay: every generation that lands AFTER a modulator
	#              multiplies its raw depth by ORN_DECAY, so a distant ancestor's
	#              gesture sinks toward the suppress floor and dies. A bad seed's
	#              ornaments age out of the lineage instead of riding the voice
	#              forever; the root's IDENTITY (the genome, refined at
	#              0.6^generation toward the root) stays kept.
	# The PRIOR contributes NO modulators (it is the calm regularizer), so a fuller
	# party dilutes toward stillness unless the lineages keep earning motion.
	const MOD_SHAPES := ["sine", "cosine", "triangle", "saw"]
	const MOD_TARGETS := ["pace", "gravity"]   # the genes an oscillator can ride
	const MOD_BUDGET := 1.15            # total post-dampen modulation depth a party may carry
	const MOD_SUPPRESS := 0.06          # a normalized depth below this is pruned (bad ones die)
	const MOD_RATE := [0.12, 1.6]       # oscillation rate range (cycles/sec over the utterance)
	const ORN_DECAY := 0.7              # ornament recency: raw depth x this per generation of age

	# ELABORATION - the one tunable scalar, 0..1, along the spectrum the user asked
	# for: at 0 a longer lineage REFINES (each generation a smaller nudge, 0.6^gen,
	# the voice settling toward its parent - the original behaviour); toward 1 a
	# longer lineage ELABORATES (perturbation decays far slower, the melodic shelf
	# grows, activations fire more - the voice getting more creative the deeper it
	# goes). It multiplies each seed's own `verve` gene, so the population still
	# varies: some lineages elaborate, some settle, and this dial sets how far the
	# whole spectrum can swing. 0 reproduces the old voices EXACTLY.
	const ELABORATION := 0.5

	# Sane bounds per gene: an elaborate deep lineage perturbs hard, and a gene
	# driven past these stops being a voice (negative pace, a threshold so low
	# every word fires). Generous enough that refined (low-elaboration) lineages
	# never touch them, so they change nothing at ELABORATION 0.
	const G_BOUNDS := {
		"heat": [0.6, 2.4], "baseline": [0.1, 0.9], "settle": [0.02, 0.35],
		"breath_span": [3.0, 20.0], "spend_window": [0.8, 5.0], "lean": [0.3, 2.2],
		"pace_hot": [0.6, 1.1], "pace_calm": [0.9, 1.7], "act_thr": [0.6, 3.2],
		"act_gain": [0.3, 2.2], "gravity": [0.0, 0.8], "ring": [0.15, 1.4],
		"hesit_bias": [-0.6, 1.2], "swing_kick": [0.02, 0.45], "verve": [0.0, 1.0],
		"damp": [0.0, 0.95],
	}
	# The channels a word can sparsely ACTIVATE on - each independent, each
	# with its own refractory. What firing does: stretch = the word's own
	# timescale pulls long; pitch = a jump toward an attractor; echo = the word
	# rings through the delay line; swell = a crescendo across the word;
	# hesit = a hesitation lands BEFORE the word (unfilled gap, or a filled
	# "um" - the %HESITATION of the transcripts).
	const ACT_CHANNELS := ["stretch", "pitch", "echo", "swell", "hesit"]

	## Blended construction: `lineages` is the working reading first, then any
	## toggled belt influences. The genome is the uniform mean of the PRIOR
	## plus every lineage's genome (1+N voices in the average); the motif
	## vocabulary pools everyone's gestures. Deterministic per lineage set.
	## An `override` genome (adrenochrome - annealed during the hook, frozen at
	## catch) replaces the blend outright: it was already integrated with the
	## party's forces when it froze.
	func _init(lineages: Array, override: Dictionary = {}) -> void:
		if not override.is_empty():
			p = override.duplicate()
			for key in PRIOR:            # a frozen genome from an older build
				if not p.has(key):       # inherits new params from the prior
					p[key] = PRIOR[key]
		else:
			var genomes: Array = [PRIOR.duplicate()]
			for lineage in lineages:
				genomes.append(_lineage_genome(lineage))
			p = {}
			for key in PRIOR:
				var v := 0.0
				for g in genomes:
					v += g[key]
				p[key] = v / genomes.size()
		# G_BOUNDS AT THE POINT OF USE (2026-08-08). The bounds existed to "keep
		# an elaborate lineage a VOICE" but were applied only inside
		# _lineage_genome, which the override branch above never calls - so a
		# frozen annealed genome walked in unclamped. The toll's own clamp is
		# relative to the prior span, not to these bounds, so at difficulty 1.0
		# the 5th percentile of act_thr was -0.898 and of breath_span -3.771:
		# both negative, both meaningless as parameters. Clamping here covers
		# the blend and the override alike.
		for key in G_BOUNDS:
			if p.has(key):
				p[key] = clampf(float(p[key]), float(G_BOUNDS[key][0]), float(G_BOUNDS[key][1]))
		# DEPTH-ELABORATION: how much this READING has earned the right to elaborate -
		# its verve x the global dial x how deep the lineage runs (nothing at depth 1,
		# rising over ~4 generations). Zero when ELABORATION is 0, so it changes
		# nothing there. It fires the sparse activations harder and grows the shelf.
		var read: Array = lineages[0] if not lineages.is_empty() else []
		var depth: int = read.size()
		var elab: float = clampf(float(p.get("verve", 0.4)) * ELABORATION, 0.0, 1.0) \
			* clampf(float(depth - 1) / 3.0, 0.0, 1.0)
		if elab > 0.0:
			p.act_thr = maxf(0.6, float(p.act_thr) * (1.0 - 0.35 * elab))  # fire MORE often
			p.act_gain = float(p.act_gain) * (1.0 + 0.5 * elab)           # ...and harder
			p.ring = float(p.ring) * (1.0 + 0.5 * elab)                   # ...ringing longer
		# THE MODULATOR BLEND: pool every lineage's spawned oscillators (the prior
		# brings none - it is the still centre), then finalize the pile into one
		# budgeted, pruned set. Elaboration eases the dampening, so a deep, high-verve
		# reading keeps more of its jaggedness while a shallow one stays smooth. A
		# frozen override carries no live modulators - it was already integrated.
		if override.is_empty():
			var raw_mods: Array = []
			for lineage in lineages:
				raw_mods.append_array(_lineage_mods(lineage))
			var eff_damp: float = clampf(float(p.get("damp", 0.35)) * (1.0 - 0.5 * elab), 0.0, 0.95)
			_mods = _finalize_mods(raw_mods, eff_damp)
		else:
			_mods = []
		_motifs = _motif_bank(PRIOR_MOTIF_SEED)
		for lineage in lineages:
			_motifs.append_array(_motif_bank(int(lineage[0])))
		# the pitch attractor set: the prior's anchors plus each lineage's own,
		# pooled - the semitone shelf the melody gravitates toward (and jumps
		# to on a pitch activation). This is the musical-quantization quality.
		# MEASURED anchors first: an echoed (recorded) seed carries its
		# source's actual melodic modes - the f0 histogram's peaks - in the
		# frozen genome's reserved "anchors" key. They REPLACE the seeded
		# shelf outright: the recording's own notes are the vocabulary, and
		# nothing rolled rides on top of them.
		var measured: Array = override.get("anchors", []) if not override.is_empty() else []
		if not measured.is_empty():
			_anchors = [0.0]
			for a in measured:
				_anchors.append(float(a))
		else:
			_anchors = [0.0, -2.0, 3.0]
			for lineage in lineages:
				var ar := RandomNumberGenerator.new()
				ar.seed = hash("anchors") ^ int(lineage[0])
				for _i in 3:
					_anchors.append(ar.randf_range(-6.0, 8.0))
			# elaboration widens the melodic vocabulary - as a WINDOW, not an
			# archive: the extra anchors are seeded by the NEWEST generations
			# (and capped low), so each refinement rotates old notes out
			# instead of piling the shelf higher. A dense anchor shelf makes
			# gravity meaningless - there is always a note nearby, so the
			# melody stops quantizing and reads as chaos.
			if elab > 0.0 and depth > 1:
				var extra: int = mini(int(round(elab * float(depth - 1) * 1.2)), 4)
				for k in extra:
					var er := RandomNumberGenerator.new()
					er.seed = hash("elab_anchor") ^ int(read[maxi(1, depth - 1 - k)]) ^ k
					_anchors.append(er.randf_range(-7.0, 9.0))
		for c in ACT_CHANNELS:
			_refract[c] = 0.0
		_gate = RandomNumberGenerator.new()
		_gate.seed = hash(str(lineages))
		arousal = p.heat
		motif = _motifs[0]

	## A deterministic coin for the planner (order-stable per lineage set).
	func gate_chance(chance: float) -> bool:
		return _gate.randf() < chance

	## Every sentence ending gets its own shape - fixed constants were cloning
	## the closings ("the living rooooom" always identical). Questions rise by a
	## varied amount; statements mostly fall, variably deep, occasionally flat;
	## the final lengthening is drawn fresh each sentence.
	func sentence_end(question: bool) -> Dictionary:
		var stretch := _gate.randf_range(0.75, 1.45)
		if question:
			return {"stretch": stretch,
				"f1": _gate.randf_range(3.5, 6.5), "f2": _gate.randf_range(1.0, 3.0)}
		var deep := _gate.randf_range(-4.0, -1.0)
		if _gate.randf() < 0.12:
			deep = _gate.randf_range(-0.5, 0.6)
		return {"stretch": stretch, "f1": deep, "f2": deep * 0.45}

	func nearest_anchor(semis: float) -> float:
		var best := 0.0
		var bd := 1e9
		for a in _anchors:
			var d: float = absf(float(a) - semis)
			if d < bd:
				bd = d
				best = a
		return best

	## One lineage's genome: the root seed samples every parameter, each later
	## seed perturbs by 0.6^generation - refinement, not a re-roll.
	static func _lineage_genome(lineage: Array) -> Dictionary:
		var root := RandomNumberGenerator.new()
		root.seed = hash("walk_root") ^ int(lineage[0])
		var g := {
			"heat": root.randf_range(1.1, 1.6),          # opening arousal
			"baseline": root.randf_range(0.25, 0.5),     # settled arousal
			"settle": root.randf_range(0.06, 0.16),      # arousal decay per second
			"breath_span": root.randf_range(6.0, 13.0),  # syllables per breath
			"spend_window": root.randf_range(1.6, 3.2),  # spent-emphasis EMA seconds
			"lean": root.randf_range(0.7, 1.3),          # emphasis appetite
			"pace_hot": root.randf_range(0.86, 0.96),    # duration mult when hot
			"pace_calm": root.randf_range(1.1, 1.32),    # duration mult when settled
			"act_thr": root.randf_range(1.4, 2.4),       # activation threshold (high = sparse)
			"act_gain": root.randf_range(0.6, 1.4),      # activation strength when fired
			"gravity": root.randf_range(0.0, 0.45),      # continuous pull toward pitch attractors
			"ring": root.randf_range(0.3, 0.9),          # resonance: how hard a firing rings
			"hesit_bias": root.randf_range(-0.3, 0.9),   # extra bar for hesitations (high = fluent)
			"swing_kick": root.randf_range(0.05, 0.28),  # cadence wobble per activation
			"verve": root.randf_range(0.0, 0.9),         # this lineage's drive to elaborate with depth
			"damp": root.randf_range(0.15, 0.55),        # how hard this lineage suppresses its own modulators
		}
		# how fast the per-generation perturbation decays: at 0.6 (elaboration off)
		# each generation is a small refinement that fades to nothing; toward 0.9
		# (this lineage's verve x ELABORATION) it barely fades, so deep generations
		# keep meaningfully developing the voice instead of freezing it.
		var e: float = clampf(float(g.verve) * ELABORATION, 0.0, 1.0)
		var decay: float = lerpf(0.6, 0.9, e)
		for i in range(1, lineage.size()):
			var pr := RandomNumberGenerator.new()
			pr.seed = hash("walk_gen") ^ int(lineage[i]) ^ i
			var scale := pow(decay, i)
			for key in g:
				g[key] *= 1.0 + pr.randfn(0.0, 0.18 * scale)
		for key in G_BOUNDS:                             # keep an elaborate lineage a VOICE
			if g.has(key):
				g[key] = clampf(float(g[key]), float(G_BOUNDS[key][0]), float(G_BOUNDS[key][1]))
		return g

	## One lineage's raw modulator set - the oscillators it has SPAWNED. The root
	## may seed one; each later generation may add another, so a deep lineage carries
	## more raw gestures. Raw, because nothing here is dampened, normalized, or
	## pruned yet - that is _finalize_mods' job, run once the party is pooled. Depths
	## are pre-budget; a generation spawns nothing ~40% of the time, so lineages
	## differ in how much motion they bring. Each depth is scaled by ORN_DECAY per
	## generation of AGE (newest = full strength), so a gesture fades as the lineage
	## grows past it and eventually dies at the suppress floor - effects age out;
	## they do not accumulate. The seeded draws themselves never change, so a
	## gesture keeps its shape/rate/phase for as long as it lives.
	static func _lineage_mods(lineage: Array) -> Array:
		var out: Array = []
		for i in lineage.size():
			var mr := RandomNumberGenerator.new()
			mr.seed = hash("walk_mod") ^ int(lineage[i]) ^ (i * 2654435761)
			if i == 0 and mr.randf() < 0.35:
				continue                                 # some roots start still
			elif i > 0 and mr.randf() < 0.4:
				continue                                 # not every generation adds motion
			out.append({
				"target": MOD_TARGETS[mr.randi() % MOD_TARGETS.size()],
				"shape": MOD_SHAPES[mr.randi() % MOD_SHAPES.size()],
				"rate": mr.randf_range(MOD_RATE[0], MOD_RATE[1]),
				# pre-normalization weight, aged by how many generations landed after
				"depth": mr.randf_range(0.25, 1.0) * pow(ORN_DECAY, lineage.size() - 1 - i),
				"phase": mr.randf_range(0.0, TAU),
			})
		return out

	## The alchemy step: turn a pooled pile of raw modulators into a BLEND. Dampen
	## every depth by `damp`, renormalize the whole set to one budget so a new
	## gesture dilutes rather than stacks, then suppress (drop) anything the dilution
	## pushed under the floor. Returns the surviving modulators with final depths.
	static func _finalize_mods(raw: Array, damp: float) -> Array:
		var keep: float = clampf(1.0 - damp, 0.0, 1.0)
		var out: Array = []
		var total := 0.0
		for m in raw:
			var d: float = float(m.depth) * keep
			if d <= 0.0:
				continue
			var mm: Dictionary = (m as Dictionary).duplicate()
			mm.depth = d
			out.append(mm)
			total += d
		if total > MOD_BUDGET:                           # NORMALIZE: the fixed energy budget
			var s: float = MOD_BUDGET / total
			for m in out:
				m.depth = float(m.depth) * s
		var pruned: Array = []                           # SUPPRESS: the diluted weak ones die
		for m in out:
			if float(m.depth) >= MOD_SUPPRESS:
				pruned.append(m)
		return pruned

	## Evaluate the summed modulation on one target at utterance time `t` (seconds),
	## in roughly [-1, 1] after the budget. Each shape gives a different route: a
	## smooth sine/cosine sway, a triangle that ramps then reverses, a saw that snaps.
	func _mod(target: String, t: float) -> float:
		if _mods.is_empty():
			return 0.0
		var s := 0.0
		for m in _mods:
			if m.target != target:
				continue
			var ph: float = t * float(m.rate) + float(m.phase) / TAU
			var frac: float = ph - floor(ph)
			var v := 0.0
			match m.shape:
				"sine":
					v = sin(TAU * ph)
				"cosine":
					v = cos(TAU * ph)
				"triangle":
					v = 2.0 * absf(2.0 * frac - 1.0) - 1.0
				"saw":
					v = 2.0 * frac - 1.0
			s += float(m.depth) * v
		return s

	static func _motif_bank(seed_value: int) -> Array:
		var m := RandomNumberGenerator.new()
		m.seed = hash("motifs") ^ seed_value
		var bank: Array = []
		for _i in 4:
			bank.append({
				"tilt": m.randf_range(-2.0, 2.0),        # semitone slope across the sentence
				"lean": m.randf_range(0.6, 1.4),         # emphasis strength multiplier
				"gap": m.randf_range(0.8, 1.6),          # inter-word gap multiplier
			})
		return bank

	func begin_sentence(question: bool) -> void:
		arousal = minf(arousal + (0.35 if question else 0.2), p.heat)
		motif = _motifs[_gate.randi() % _motifs.size()]

	## Advance one word (est_dur seconds of speech, nsyll syllables, frac = its
	## position 0..1 in the sentence). Returns the planner's modifiers.
	func word(stressed: bool, nsyll: int, est_dur: float, frac: float, punct: bool,
			prominence := 0.5) -> Dictionary:
		var norm: float = clampf(arousal / p.heat, 0.0, 1.0)
		# sparse activations first: their kicks fold back into this word's pace
		var acts := {}
		var kick := 0.0
		for c in ACT_CHANNELS:
			_refract[c] *= exp(-est_dur / 2.5)
			var bar: float = p.act_thr + _refract[c] \
				+ (p.hesit_bias if c == "hesit" else 0.0)
			var drive: float = _gate.randfn(0.0, 1.0) + _ring_amp * 0.5
			var a: float = maxf(0.0, drive - bar) * p.act_gain
			if a > 0.0:
				_refract[c] += 1.2
				_ring_amp = minf(_ring_amp + p.ring * a * 0.5, 1.5)
				kick += a
			acts[c] = clampf(a, 0.0, 1.5)
		if kick > 0.0:
			# the wobble: any strike knocks the cadence off its line - rushing
			# or dragging by seeded coin - and the offset decays back
			_swing += (1.0 if _gate.randf() < 0.5 else -1.0) * p.swing_kick * kick
		var pace: float = lerpf(p.pace_calm, p.pace_hot, norm) \
			* (1.0 + clampf(_swing, -0.3, 0.45))
		# the modulators ride here: a slow oscillation folded into the running pace,
		# on whatever route (sine/cosine/triangle/saw) survived the blend
		pace *= 1.0 + 0.22 * _mod("pace", _t)
		_swing *= exp(-est_dur / 2.0)
		# EMPHASIS: the BASELINE comes from the text ([Phrasing] - which words
		# this clause leans on), and the walk decides how hard THIS SPEAKER
		# leans on it. Previously this was a coin flip that both invented the
		# accent and placed it, so the same sentence stressed different words
		# on different seeds and no seed stressed them where English does. The
		# walk should colour prosody, never generate it.
		var emph := 0.0
		var pre_pause := 0.0
		if stressed:
			var appetite: float = p.lean * motif.lean * (0.5 + 0.5 * (1.0 - norm))
			# spent-emphasis still spaces the BIG leans out; it can no longer
			# delete an accent the sentence structure requires
			emph = clampf(prominence * appetite * (1.0 - 0.55 * spent), 0.0, 1.6)
			spent += 0.42 * prominence
			# an extra push, still seeded and still self-spacing - this is the
			# speaker's own reading of an already-prominent word
			if prominence > 0.7 and spent < 0.9 and _gate.randf() < 0.3 * appetite:
				emph *= 1.35
				pre_pause = 0.04 + 0.09 * emph * (1.0 - norm)
		var breath_pause := 0.0
		breath += nsyll
		# 1.6x margin: unwritten breaths fired on 7.5% of words with 41% landing
		# mid-phrase, because this is a syllable counter with no reference to
		# syntax. Proper placement needs Phrasing to choose the site from the
		# legal ones; until then, rarer.
		if breath >= p.breath_span * 1.6 and not punct:
			breath_pause = 0.12 + 0.28 * (1.0 - norm)
			breath = 0.0
			arousal = maxf(arousal - 0.12, 0.2)
			spent *= 0.5
		elif punct:
			breath = maxf(breath - p.breath_span * 0.6, 0.0)   # punctuation is half a breath
		var ring_st: float = p.ring * _ring_amp * sin(_ring_ph) * 1.5
		_ring_amp *= exp(-est_dur / 0.9)
		_ring_ph += est_dur * TAU * 1.3
		# the EMAs advance by the word's own duration
		arousal = lerpf(arousal, p.baseline, 1.0 - exp(-p.settle * est_dur))
		spent *= exp(-est_dur / p.spend_window)
		# gravity wavers on its own modulator route, then the clock advances
		var gravity: float = clampf(float(p.gravity) * (1.0 + 0.6 * _mod("gravity", _t)), 0.0, 0.8)
		_t += est_dur
		return {
			"pace": pace, "emph": emph, "pre_pause": pre_pause,
			"breath_pause": breath_pause, "tilt": motif.tilt * (frac - 0.5),
			"gap": motif.gap, "acts": acts, "ring_st": ring_st,
			"gravity": gravity,
		}


# Two-pole resonator (the Klatt building block): y = A x + B y1 + C y2.
class Reso:
	var b := 0.0
	var c := 0.0
	var a := 1.0
	var y1 := 0.0
	var y2 := 0.0

	## CASCADE normalization: unity gain at DC. Correct for a chain of poles
	## modelling one tube, where the product must not drift with tuning.
	func tune(f: float, bw: float) -> void:
		var r := exp(-PI * bw / SR)
		c = -r * r
		b = 2.0 * r * cos(TWO_PI * clampf(f, 50.0, SR * 0.45) / SR)
		# NO HISTORY RESCALING HERE, and the reason is worth keeping.
		#
		# Klatt's KLSYN88 (ch. 3, after Fujisaki and Azemi 1971) rescales y1/y2
		# by sqrt(A'/A) on every retune, naming the symptom "clicks and burps".
		# That correction was tried here on 2026-08-08 and made things audibly
		# WORSE - the user described the result as "electrical", which is
		# exactly right. It is a misapplication:
		#
		# Klatt's resonator and this one are normalized differently. Here
		# `a = 1 - b - c` fixes the DC gain at unity, so the steady-state
		# response is already invariant under retuning and y1/y2 are already in
		# consistent units. Worse, `a` is a small difference of larger numbers -
		# for a narrow pole it can be ~0.01 - so a fraction-of-a-hertz frequency
		# change can swing it by tens of percent. Scaling the state by the root
		# of that ratio, at the 689 Hz frame rate, is amplitude modulation at
		# 689 Hz. That is the electrical buzz, not a fix for it.
		#
		# The genuine per-frame discontinuity is the pole MOVEMENT itself, and
		# the real remedy is finer-grained coefficient interpolation, not state
		# surgery. See next/voice_intelligibility.md.
		a = 1.0 - b - c

	## PARALLEL normalization: unity gain AT RESONANCE, so the amplitude a
	## caller passes alongside the pole is the peak it actually gets. A
	## DC-normalized resonator's peak gain scales with f/bw, which would make
	## every parallel amplitude in the phoneme table a lie about its own level.
	func tune_peak(f: float, bw: float) -> void:
		var w := TWO_PI * clampf(f, 50.0, SR * 0.45) / SR
		var r := exp(-PI * bw / SR)
		c = -r * r
		b = 2.0 * r * cos(w)
		# |1 - b e^-jw - c e^-2jw| at the resonance, inverted
		var re := 1.0 - b * cos(w) - c * cos(2.0 * w)
		var im := b * sin(w) + c * sin(2.0 * w)
		a = sqrt(re * re + im * im)

	func step(x: float) -> float:
		var y := a * x + b * y1 + c * y2
		y2 = y1
		y1 = y
		return y


# Anti-resonator: a biquad NOTCH (zero pair ON the unit circle, pole pair just
# inside it), unity gain at DC and Nyquist by construction. The nasal murmur's
# missing ingredient - a nasal is defined by the energy the side cavity
# REMOVES, and running M/N/NG through poles alone made a buzzy hum. The poles
# are not optional: a bare zero pair normalized at DC amplifies the top octave
# ~30x and sprays spikes through the radiation stage; the notch removes ONLY
# the anti-formant region.
class AntiReso:
	var b1 := 0.0                     # shared cosine term (numerator + poles)
	var pr := 0.0                     # pole radius (notch width)
	var g := 1.0
	var x1 := 0.0
	var x2 := 0.0
	var y1 := 0.0
	var y2 := 0.0

	func tune(f: float, bw: float) -> void:
		var w := TWO_PI * clampf(f, 50.0, SR * 0.45) / SR
		pr = exp(-PI * bw / SR)
		b1 = 2.0 * cos(w)
		g = (1.0 - pr * b1 + pr * pr) / (2.0 - b1)

	func step(x: float) -> float:
		var y := g * (x - b1 * x1 + x2) + pr * b1 * y1 - pr * pr * y2
		x2 = x1
		x1 = x
		y2 = y1
		y1 = y
		return y


## Synthesize a paragraph. Returns:
## `{pcm: PackedFloat32Array, sr, dur, words: [{text,t0,t1,sentence}], phones: [{p,t0,t1,word}]}`.
## Deterministic per (text, spec). Heavy (a few seconds of compute per ten
## seconds of speech in GDScript); callers on the UI thread should chunk via
## render_async in [SynthEditor].
static func render(text: String, spec: Spec) -> Dictionary:
	var segs := plan(text, spec)
	return synth(segs, spec)


# Voiceless obstruents raise the f0 of the following vowel a touch
# (microprosody) - a small cue human ears expect and flat synthesis lacks.
const _VOICELESS := ["P", "T", "K", "F", "TH", "S", "SH", "HH"]


## Plan text into synthesis segments. Beyond phoneme durations and pauses, the
## plan carries the whole prosodic reading:
## - stress: accented vowels lengthen, brighten and rise; unstressed ones
##   shorten, quieten and REDUCE (formants pulled toward schwa);
## - contours: declination across the sentence, final lowering at a period,
##   a rise at a question mark, a continuation rise into a comma;
## - microprosody: vowels after voiceless consonants start slightly higher;
## - the multi-timescale [ProsodyField] wanders pitch, tempo and loudness
##   continuously (seeded per voice).
## Pure data; `synth` realizes it through the EMAs. `events`, if provided, is
## filled with the strike times: `{t, kind, a}` per sparse activation - the
## planner knows exactly when every effect will hit, which is what makes the
## bite dynamics (catch-when-it-strikes) possible downstream.
static func plan(text: String, spec: Spec, events: Array = []) -> Array:
	var segs: Array = []
	var field := ProsodyField.new(int(spec.reading[0]))
	var walk := ProsodyWalk.new([spec.reading] + spec.influences, spec.adrenochrome)
	var t_cursor := 0.12
	# a syllable counter that runs across the WHOLE text, not per sentence, so
	# the sustain cycle is continuous through the reading instead of resetting
	# at every full stop
	var syll_i := 0
	var sentences := Phonemes.parse(text)
	# SENTENCE stress, from the text's own structure - deterministic, unseeded,
	# and the baseline the walk modulates rather than replaces
	Phrasing.annotate(sentences)
	for si in sentences.size():
		var words: Array = sentences[si]
		var vowels_total := 0
		for w in words:
			for p in w.phones:
				if _ptype(p) == "vowel":
					vowels_total += 1
		var vseen := 0
		var question: bool = words.size() > 0 and String(words[-1].get("punct", "")) == "?"
		walk.begin_sentence(question)
		for wi in words.size():
			var w: Dictionary = words[wi]
			var wstress: Array = w.get("stress", [])
			var accent_at := Phonemes.stress_vowel(w.phones, wstress) if w.stressed else -1
			var last_word: bool = wi == words.size() - 1
			var nsyll := 0
			var est_dur := 0.0
			for p in w.phones:
				est_dur += Phonemes.TABLE.get(p, {}).get("dur", 80.0) * 0.001 / spec.rate
				if _ptype(p) == "vowel":
					nsyll += 1
			var prom: float = float(w.get("prominence", 0.5))
			var mods := walk.word(w.stressed, nsyll, est_dur,
				float(wi) / maxf(1.0, float(words.size() - 1)), w.pause_after != "none", prom)
			for c in mods.acts:
				if float(mods.acts[c]) > 0.0:
					events.append({"t": t_cursor, "kind": c, "a": float(mods.acts[c])})
			if RAW_MODE:
				# the strikes above stay real (the bites keep working); the
				# walk's AUDIBLE realization is neutralized wholesale
				mods = {"pace": 1.0, "emph": 0.0, "pre_pause": 0.0,
					"breath_pause": 0.0, "tilt": 0.0, "gap": 1.0,
					"ring_st": 0.0, "gravity": 0.0,
					"acts": {"stretch": 0.0, "pitch": 0.0, "echo": 0.0,
						"swell": 0.0, "hesit": 0.0}}
			# a spontaneous hesitation lands BEFORE the word: an unfilled gap,
			# or (by seeded coin) a filled "um" - low, flat, reduced
			var hes: float = mods.acts.hesit
			if DISFLUENCY and hes > 0.0 and not w.get("hesit", false):
				if walk.gate_chance(0.45):
					var hdur := 0.14 + 0.14 * hes
					t_cursor += hdur + 0.05
					segs.append({"p": "AH", "dur": hdur, "word": -1, "sentence": si,
						"text": "", "word_start": false, "word_end": false,
						"semitones": -2.0, "amp": 0.5, "reduce": 0.6, "echo": 0.0})
					segs.append(_sil(0.05, si))
				else:
					var gdur := 0.1 + 0.25 * hes
					t_cursor += gdur
					segs.append(_sil(gdur, si))
			var fin := {}
			if last_word:
				fin = walk.sentence_end(question)
			if mods.pre_pause > 0.0:
				t_cursor += mods.pre_pause
				segs.append(_sil(mods.pre_pause, si))
			var wsegs: Array = []
			for pi in (w.phones as Array).size():
				var p: String = w.phones[pi]
				var entry: Dictionary = Phonemes.TABLE.get(p, {})
				if entry.is_empty():
					continue
				var dur: float = entry.get("dur", 80.0) * 0.001 / spec.rate
				dur *= mods.pace       # the walk's pacing: hot fast, settled slow
				var acts: Dictionary = mods.acts
				# a stretch activation pulls the whole word's timescale long
				dur *= 1.0 + 0.4 * acts.stretch
				var amp := 1.0
				var reduce := 0.0
				var semis := 0.0
				var decor := 0.0   # ornament budget, see MELODY_DECOR
				var is_vowel := _ptype(p) == "vowel"
				# LEXICAL STRESS (from the dictionary) is what gives the reading
				# a rhythm instead of a chant. English listeners segment running
				# speech BY stress pattern, so a flat reading costs word
				# boundaries, not just naturalness - every syllable arriving at
				# equal length and equal vowel quality is most of what "muddy"
				# means when the phonemes are already right.
				var lex_st: int = int(wstress[pi]) if pi < wstress.size() else 0
				if is_vowel:
					vseen += 1
					syll_i += 1
					if lex_st == 0:
						# unstressed: shorter, quieter, and CENTRALIZED - the
						# vowel loses its identity toward schwa, which is what
						# makes the stressed syllables stand out
						dur *= 0.62
						amp *= 0.78
						# 0.7 was a 70% lerp of ALL THREE formants onto one
						# fixed schwa, which collapsed the unstressed vowel
						# space to 8.5% of the table hull (LPC-measured 1.69
						# Bark^2 against 11.49 stressed). The shortened
						# duration above already supplies most of the
						# centralization a reduced vowel should have, via
						# trajectory undershoot - this lerp was doubling it.
						# 0.3 assumed trajectory undershoot would supply the rest of
						# the centralization. Halving the formant time constants took
						# that away - short vowels now actually ARRIVE - so the explicit
						# lerp is doing all of the work and the reduced hull shrank
						# (0.425 -> 0.364). Less lerp, now that attainment is real.
						reduce = 0.18
					elif lex_st == 2:
						dur *= 0.88
						amp *= 0.92
						reduce = 0.25
					dur *= 1.0 + 0.25 * acts.stretch   # vowels carry most of the pull
					if spec.song > 0.0:
						# THE SUSTAIN GATE. Stretching every vowel onto the beat
						# grid made a drone that took twice as long as speech -
						# a constant cadence is not a cadence. Music alternates:
						# notes are HELD at the joints of a phrase and RUN
						# through in between, and that contrast is the music.
						# A slow seeded cycle over syllables supplies the
						# periodicity, prominence says which words deserve to be
						# held, and a phrase ending is a natural place to land.
						# Same thresholded-drive idiom as the activation
						# channels, so it is sparse and self-spacing by
						# construction rather than by a rate constant.
						var cyc: float = Spec.sustain_wave(spec, syll_i)
						var drive: float = 0.62 * cyc + 0.55 * prom \
							+ (0.28 if w.pause_after != "none" else 0.0)
						if drive > SUSTAIN_BAR:
							var hold: float = clampf(
								(drive - SUSTAIN_BAR) / (1.45 - SUSTAIN_BAR), 0.0, 1.0)
							var beats: float = 1.0 + round(hold * 2.0)
							dur = lerpf(dur, beats * spec.beat, spec.song)
						else:
							# run past it, faster than it would be spoken - but
							# not so fast it stops being a note. Below ~75 ms
							# there is not enough voiced time to hear a pitch,
							# however quickly the f0 arrives.
							dur = maxf(dur * lerpf(1.0, SONG_RUN, spec.song),
								lerpf(dur, SONG_RUN_MIN, spec.song))
					# declination falls across the sentence; the field wanders on top
					semis -= spec.f0_decl * (float(vseen) / maxf(1.0, float(vowels_total)))
					# DECORATIVE terms accumulate separately and share a budget
					# (see MELODY_DECOR). Seven melodic terms used to sum with
					# no ceiling at all, measured at a p50 sentence span of 8.95
					# semitones and a p95 of 13.89 - which is the "inflection is
					# all over the place" report. Structural terms (declination,
					# the prominence-scaled accent, terminal contours, comma
					# rises) are the reading and stay unbounded; the ornaments
					# compete for a fixed share.
					decor += mods.tilt      # the sentence motif's slope
					decor += mods.ring_st   # the resonance ring from recent firings
					if pi == accent_at:
						# the accent's SIZE now tracks how prominent this word
						# is in its clause: a nucleus lands hard, a de-accented
						# repeat barely rises. Every content word getting the
						# same full accent is a list being read, not a sentence.
						# capped: this reached 1.7x, and with phrase-final
						# stretch on top the measured stressed:unstressed
						# duration ratio was 3.07:1 against a published ~1.9:1
						dur *= 1.0 + 0.25 * prom * (1.0 + 0.4 * mods.emph)
						amp *= 1.0 + 0.34 * prom * (1.0 + 0.25 * mods.emph)
						semis += spec.f0_accent * prom + 2.2 * mods.emph
					elif not w.stressed:
						# a function word: reduced as a whole, on top of
						# whatever its own syllables asked for
						dur *= 0.85
						amp *= 0.9
						reduce = maxf(reduce, 0.5)   # vowel reduction: drift toward schwa
					if pi > 0 and _VOICELESS.has(w.phones[pi - 1]):
						decor += 0.8    # microprosody after voiceless consonants
					if not RAW_MODE:
						# the wander is what makes speech sound unstudied and
						# what makes a sung note sound out of tune - a held
						# note has to be STILL for the vibrato to read as
						# vibrato rather than drift
						decor += field.sample("f0", t_cursor) * (1.0 - spec.song)
					# pitch attractors: the melody is continuously pulled toward
					# the voice's anchor shelf (gravity), and a pitch activation
					# JUMPS most of the way there - musical quantization
					# singing is quantized pitch: the shelf stops being a gentle
					# attractor and becomes the scale the melody is confined to
					var pull: float = clampf(0.3 * mods.gravity + 0.55 * acts.pitch, 0.0, 0.85)
					# quantization engages EARLY and fully: a half-applied pull
					# parks the pitch midway between its natural value and the
					# note, which measured worse-tuned than plain speech. Being
					# on a scale is the defining feature, so it is not something
					# to fade in - what fades in over the axis is note LENGTH.
					pull = maxf(pull, clampf(spec.song * 2.0, 0.0, 1.0) * 0.95)
					if pull > 0.0:
						semis = lerpf(semis, walk.nearest_anchor(semis), pull)
					# a swell activation is a crescendo across the word
					if acts.swell > 0.0:
						amp *= 1.0 + acts.swell * lerpf(-0.12, 0.3,
							float(pi) / maxf(1.0, float((w.phones as Array).size() - 1)))
				if last_word and pi >= (w.phones as Array).size() - 2:
					dur *= spec.final_lengthen * float(fin.stretch)
				if not RAW_MODE:
					dur *= 1.0 + 0.12 * field.sample("rate", t_cursor)
					amp *= 1.0 + 0.15 * field.sample("amp", t_cursor)
				# an authored %HESITATION: low, flat, quiet, fully reduced
				if w.get("hesit", false):
					amp *= 0.55
					reduce = 0.6
					semis = -2.0 + (0.0 if RAW_MODE else field.sample("f0", t_cursor) * 0.3)
					dur *= 1.5
				t_cursor += dur
				wsegs.append({
					"p": p, "dur": dur, "word": wi, "sentence": si,
					"text": w.text, "word_start": pi == 0,
					"word_end": pi == (w.phones as Array).size() - 1,
					"semitones": semis + clampf(decor, -MELODY_DECOR, MELODY_DECOR),
					"amp": amp, "reduce": reduce,
					"echo": clampf(0.55 * acts.echo, 0.0, 0.9),
					"display": w.get("display", w.text),
					# Which rewritten SOURCE run this word belongs to, if any -
					# `2009` is three spoken words and one thing on the page.
					# See Phonemes.parse; -1 means the word is its own source.
					"group": w.get("src_span", -1),
				})
			# terminal contours land on the sentence's last word's vowels, with
			# a freshly drawn shape each sentence (see Walk.sentence_end)
			if last_word:
				var vsegs: Array = wsegs.filter(func(s): return _ptype(s.p) == "vowel")
				if vsegs.size() > 0:
					vsegs[-1].semitones += float(fin.f1)
					_resnap(vsegs[-1], walk, spec)
				if vsegs.size() > 1:
					vsegs[-2].semitones += float(fin.f2)
					_resnap(vsegs[-2], walk, spec)
			# a comma word carries a small continuation rise (the "not done yet" cue)
			elif w.pause_after == "comma":
				for k in range(wsegs.size() - 1, -1, -1):
					if _ptype(wsegs[k].p) == "vowel":
						wsegs[k].semitones += 1.8
						_resnap(wsegs[k], walk, spec)
						break
			segs.append_array(wsegs)
			var pause: String = w.pause_after
			if pause != "none":
				var pdur: float = spec.pause_comma if pause == "comma" else spec.pause_stop
				t_cursor += pdur
				segs.append(_sil(pdur, si))
			elif mods.breath_pause > 0.0 and not last_word:
				# breath debt came due mid-sentence: a real pause the text never wrote
				t_cursor += mods.breath_pause
				segs.append(_sil(mods.breath_pause, si))
			elif not last_word:
				# LEGATO: the old 15ms silence between every running word dropped
				# the voice to zero at each boundary - the single loudest robotic
				# tell (a picket fence of gaps). Phase, amplitude and formants all
				# persist across segments through the EMAs, so a much smaller gap
				# lets consecutive words PHONATE THROUGH the boundary (the way a
				# real voice slurs running words) while keeping a hair of
				# articulation space. Real breaths/pauses still come from
				# punctuation and the breath-debt branch above.
				var gap: float = 0.004 * mods.gap
				t_cursor += gap
				segs.append(_sil(gap, si))
	# a breath of silence at both ends so playback and analysis never clip a boundary
	segs.push_front(_sil(0.12, -1))
	segs.append(_sil(0.12, -1))
	# f0 continuity: the melody is a WORD property, not a vowel property.
	# semitones were only ever computed on vowels, so every consonant and
	# silence targeted 0 st and the f0 EMA dived toward the base mid-word - a
	# picket-fence melody (measured: 4-5 st swings INSIDE words). Consonants
	# now sit on the line between their neighbouring vowels, and silences
	# pre-position toward the NEXT vowel (inaudible - the amplitude is zero -
	# but the EMA arrives on pitch instead of gliding in from neutral).
	var next_v := 0.0
	var next_semis := PackedFloat32Array()
	next_semis.resize(segs.size())
	for i in range(segs.size() - 1, -1, -1):
		if _ptype(String(segs[i].p)) == "vowel":
			next_v = float(segs[i].semitones)
		next_semis[i] = next_v
	# ... and at song = 1 the factor goes to 0, so a consonant HOLDS the note it
	# is inside instead of gliding toward the next one. That is the staircase.
	var glide: float = 0.6 * (1.0 - spec.song)
	var prev_v := next_semis[0]
	for i in segs.size():
		var seg: Dictionary = segs[i]
		if _ptype(String(seg.p)) == "vowel":
			prev_v = float(seg.semitones)
		elif String(seg.p) == "SIL":
			seg.semitones = lerpf(prev_v, next_semis[i], 1.0 - spec.song)
		else:
			seg.semitones = lerpf(prev_v, next_semis[i], glide)
	return segs


## Re-quantize a vowel after the terminal contour has moved it. The sentence
## ending and the comma rise are applied AFTER the per-vowel anchor pull, and
## they land on the LONGEST vowels in the reading - and, when singing, on
## exactly the syllables the sustain gate chose to hold, because a phrase
## ending is one of the things that makes a syllable worth holding. So the one
## note a listener has the best chance of hearing as a pitch was guaranteed to
## sit 1 to 6.5 semitones off the shelf. Musically this turns a fall into a
## fall TO A NOTE, which is what a cadence is.
static func _resnap(seg: Dictionary, walk: ProsodyWalk, spec: Spec) -> void:
	if spec.song <= 0.0:
		return
	var pull: float = clampf(spec.song * 2.0, 0.0, 1.0) * 0.95
	seg.semitones = lerpf(float(seg.semitones), walk.nearest_anchor(float(seg.semitones)), pull)


static func _sil(dur: float, si: int) -> Dictionary:
	return {"p": "SIL", "dur": dur, "word": -1, "sentence": si, "text": "",
		"word_start": false, "word_end": false, "semitones": 0.0, "amp": 1.0,
		"reduce": 0.0, "echo": 0.0}


static func _ptype(p: String) -> String:
	return Phonemes.TABLE.get(p, {}).get("type", "sil")


## Realize planned segments into PCM + the timing map. Split out from render()
## so the editor can chunk it across frames and tests can drive it directly.
static func synth(segs: Array, spec: Spec, from_seg := 0, to_seg := -1, state := {}) -> Dictionary:
	if state.is_empty():
		state = synth_state(spec)
	if to_seg < 0:
		to_seg = segs.size()
	var noise_rng: RandomNumberGenerator = state.rng
	var out: PackedFloat32Array = state.pcm
	for i in range(from_seg, to_seg):
		var seg: Dictionary = segs[i]
		var entry: Dictionary = Phonemes.TABLE.get(seg.p, {"type": "sil"})
		var t0 := float(out.size()) / SR
		# where the mouth goes NEXT. A segment's own posture holds for most of
		# its length and then glides toward this over the last LOCUS_TIME - the
		# locus mechanism (Haskins). Retargeting only at segment boundaries put
		# the whole transition inside the FOLLOWING phone, which for a stop is
		# the silent closure, so the listener never heard it - and the F2
		# transition is the primary place cue for exactly the consonants whose
		# own noise is weakest.
		var nxt := _next_formants(segs, i, spec)
		match String(entry.get("type", "sil")):
			"sil":
				_run_frames(out, state, spec, noise_rng, seg, entry, 0.0, 0.0, seg.dur)
			"stop":
				# closure at the stop's OWN locus (voiced stops keep a faint
				# murmur), then a burst through the parallel branch
				var murmur := 0.06 if entry.get("voiced", false) else 0.0
				_retarget(state, _seg_formants(entry, 0.0, spec), spec)
				_run_frames(out, state, spec, noise_rng, seg, entry, murmur, 0.0, seg.dur * 0.6)
				_tune_parallel(state, entry, spec)
				# THE PLACE CUE (2026-08-08). This used to retarget the cascade
				# to the FOLLOWING phone before the release, on the reasoning
				# that a burst should carry the coming vowel's transition. The
				# effect was the opposite: the whole locus-to-vowel excursion
				# was spent during the closure, so every stop released from the
				# vowel's own posture and all six came out identical. Measured
				# onset-F2 spread across the three voiceless places was 35 Hz
				# in /AA/ against a natural 600-800.
				#
				# A stop's place is carried by its burst spectrum filtered by
				# its OWN cavity, plus the transition that follows in the VOWEL.
				# So hold the locus here and hand the forward glide to the run,
				# exactly as voiceless fricatives already do below.
				_run_frames(out, state, spec, noise_rng, seg, entry, murmur,
					float(entry.get("namp", 1.0)) * BURST_GAIN, maxf(0.008, seg.dur * 0.12),
					false, true)
				if not entry.get("voiced", false):
					# voice onset time: a voiceless release leaks aspiration
					# through the ONCOMING vowel's formants before the folds
					# start - snapping from burst straight into full voicing
					# is one of the loudest "synthetic" tells there is.
					#
					# No retarget here either (2026-08-08): aspiration through a
					# tract that has ALREADY arrived at the vowel is what made
					# the transition 95.7% complete before voicing began, which
					# is the same defect as retargeting before the burst. The
					# tract stays at the locus and the FOLLOWING VOWEL's own
					# segment pulls it home through its onset EMA, which is
					# where a listener can actually use it.
					_run_frames(out, state, spec, noise_rng, seg, entry, 0.0, 0.16, seg.dur * 0.3, true)
				else:
					# /b d g/ had no release EVENT at all: 45-48 ms of closure,
					# a 9 ms burst at -42 dB, then the vowel - which measured
					# 22 dB under the adjacent vowel and left voicing at chance.
					# A voiced stop still has a VOT, it is just short and
					# voiced: the folds are already running while the tract is
					# still opening. Give it one, with the glide handed forward
					# so the transition lands in the vowel where it is audible.
					_run_frames(out, state, spec, noise_rng, seg, entry,
						0.30 * float(seg.get("amp", 1.0)), 0.04, seg.dur * 0.14)
			"fric":
				# the posture is the fricative's own (this branch was the ONE
				# that never set one - the whole "F and TH are inaudible" bug),
				# and the frication itself rides the parallel branch.
				# A VOICELESS fricative gets NO forward glide: the tract is
				# silent through it, so moving toward the next vowel here spends
				# the transition where nobody can hear it and delivers the vowel
				# already sitting on its own target. Holding the locus instead
				# makes the VOWEL's onset carry the place cue - which for /f/
				# and /th/ is the primary cue, their own noise being weak.
				# Reported as "the F in father is completely silent": the F was
				# audible as hiss but had no articulatory signature at all.
				_tune_parallel(state, entry, spec)
				var voiced: bool = entry.get("voiced", false)
				# The voice bar is a weak low-frequency murmur, not a second voice.
				# A flat 0.5 put voiced fricatives 12 dB ABOVE their voiceless
				# twins (measured V -5.7 vs F -18.0 dB re vowel) when natural
				# speech puts them a few dB BELOW, and it ignored seg.amp so no
				# stress could reach them.
				var v := (0.10 * float(seg.get("amp", 1.0))) if voiced else 0.0
				_retarget(state, _seg_formants(entry, 0.0, spec), spec)
				_run_frames(out, state, spec, noise_rng, seg, entry, v, entry.namp, seg.dur,
					false, false, nxt if voiced else [])
			"asp":
				# /h/ is glottal: its noise belongs in the CASCADE (the whole
				# tract is its filter), aimed where the formants are heading
				_retarget(state, nxt, spec)
				_run_frames(out, state, spec, noise_rng, seg, entry, 0.0, 0.22, seg.dur, true)
			_:
				# vowel / glide / nasal: periodic source through the cascade
				var amp: float = seg.get("amp", 1.0)
				if entry.type == "glide":
					amp *= 0.75
				elif entry.type == "nasal":
					amp *= 0.45
				_retarget(state, _seg_formants(entry, 0.0, spec, seg.get("reduce", 0.0)), spec)
				_run_frames(out, state, spec, noise_rng, seg, entry, amp, 0.0, seg.dur, false, false, nxt)
		var t1 := float(out.size()) / SR
		_record_timing(state, seg, t0, t1)
	var done: bool = to_seg >= segs.size()
	if done:
		# flush the limiter's pending block (trailing silence by construction)
		var pend: PackedFloat32Array = state.lim_buf
		var gflush: float = state.lim_g
		for k in pend.size():
			out.append(pend[k] * gflush)
		state.lim_buf = PackedFloat32Array()
	state.pcm = out
	return {
		"pcm": out, "sr": SR, "dur": float(out.size()) / SR,
		"words": state.words, "phones": state.phones, "done": done, "state": state,
	}


## Fresh synthesis state (resonators, EMA formants, f0 realization, timing maps).
static func synth_state(spec: Spec) -> Dictionary:
	var rng := RandomNumberGenerator.new()
	rng.seed = spec.seed_value
	var st := {
		"rng": rng, "pcm": PackedFloat32Array(),
		"r1": Reso.new(), "r2": Reso.new(), "r3": Reso.new(),
		"r4": Reso.new(), "r5": Reso.new(), "r6": Reso.new(),
		"anti": AntiReso.new(), "anti_mix": 0.0, "anti_f": 950.0,
		# the PARALLEL branch (see FRIC_LEVEL): front-cavity resonators driven
		# by turbulence, summed with the cascade rather than filtered by it
		"par": [Reso.new(), Reso.new(), Reso.new(), Reso.new(), Reso.new(), Reso.new(), Reso.new()],
		"par_a": _zeroes(7), "par_n": 0, "par_ab": 0.0,
		"fsm": [500.0 * spec.formant_scale, 1400.0 * spec.formant_scale, 2400.0 * spec.formant_scale],
		"ftg": [500.0 * spec.formant_scale, 1400.0 * spec.formant_scale, 2400.0 * spec.formant_scale],
		"f0sm": spec.f0_base * 1.12, "phase": 0.0, "ampsm": 0.0,
		"pulse": _pulse_table(0.4, 0.16), "pulse_lax": _pulse_table(0.58, 0.34),
		"dcb": 0.0, "tension": 0.5, "nlp": 0.0, "tilt_y": 0.0, "prev": 0.0, "nampsm": 0.0,
		"vib_ph": 0.0,
		"nroute": 0,
		"jit": 1.0, "pgain": 1.0,
		"lim_buf": PackedFloat32Array(), "lim_g": 1.0, "lim_need": 1.0, "cenv": 0.0,
		# retune-glided spec scalars: a live retune() swaps the spec between
		# chunks, and these stepping instantly mid-stream was an audible tick
		# per retune - the reel retunes every 2 s. Initialized to the spec, so
		# a fixed-spec render is untouched (the EMA of a constant is itself).
		"brsm": spec.breath, "airgsm": spec.air_gain, "aircsm": spec.air_cut,
		"fssm": spec.formant_scale,
		"ebuf": _zeroes(int(ECHO_DELAY * SR)), "eidx": 0, "elp": 0.0,
		"dr_st": PackedFloat32Array(), "dr_f": PackedFloat32Array(),
		"dr_ph": PackedFloat32Array(), "dr_env": PackedFloat32Array(),
		"dr_kind": PackedInt32Array(), "dr_atk": PackedFloat32Array(),
		"dr_rel": PackedFloat32Array(), "dr_lr": PackedFloat32Array(),
		"dr_lph": PackedFloat32Array(), "dr_p2": PackedFloat32Array(),
		"dr_p3": PackedFloat32Array(), "dr_hi": PackedByteArray(),
		"dr_tr": 0.03, "dr_tph": 0.0, "dr_ve": 0.0,
		"dr_out": 0.0, "dr_alt": false,
		"words": [], "phones": [], "wopen": {},
	}
	# THE RESONANCE bank (see the DRONE consts): one tone per anchor note.
	# The anchors are the walk's pitch attractors - rebuilt here exactly as
	# plan() builds them (same lineage + frozen genome = same notes, incl. a
	# recorded seed's MEASURED melodic modes), so the drone plays the notes
	# the melody actually lands on.
	if DRONE and not RAW_MODE:
		var walk := ProsodyWalk.new([spec.reading] + spec.influences, spec.adrenochrome)
		var seen: Array = []
		# LOCAL arrays, assigned into the state once: appending through a
		# `(st.x as PackedFloat32Array)` cast appends to a CoW COPY that is
		# immediately discarded - the classic packed-array trap (this bank
		# shipped EMPTY that way; measured as a -163 dB "drone")
		var dsts := PackedFloat32Array()
		var dfs := PackedFloat32Array()
		for a in walk._anchors:
			var stq := snappedf(float(a), 0.5)
			if seen.has(stq):
				continue
			seen.append(stq)
			# tune to the pitch the melody will ACTUALLY realize. The strings
			# were tuned to the raw anchor while the melody plays it at
			# anchor * inflect, so every voice with inflect != 1 had its drone
			# a growing interval out of tune with the note it was answering -
			# and `semis_now` below is already in inflect-scaled units, so the
			# proximity test disagreed with the tuning too.
			var a_real: float = float(a) * spec.inflect
			dsts.append(a_real)
			dfs.append(clampf(spec.f0_base * 1.06 * pow(2.0, a_real / 12.0), 40.0, 900.0))
			if dfs.size() >= DRONE_STRINGS:
				break
		st.dr_st = dsts
		st.dr_f = dfs
		var nd := dfs.size()
		var zeros := PackedFloat32Array()
		zeros.resize(nd)
		st.dr_ph = zeros.duplicate()
		st.dr_env = zeros
		var trng := RandomNumberGenerator.new()
		trng.seed = hash("tide") ^ int(spec.reading[0])
		st.dr_tr = trng.randf_range(0.02, 0.045)   # tide rate (Hz)
		st.dr_tph = trng.randf_range(0.0, TAU)
		var kinds := PackedInt32Array()
		var atks := PackedFloat32Array()
		var rels := PackedFloat32Array()
		var lrs := PackedFloat32Array()
		var lphs := PackedFloat32Array()
		var p2s := PackedFloat32Array()
		var p3s := PackedFloat32Array()
		for k in nd:
			var pr := RandomNumberGenerator.new()
			pr.seed = hash("resonance") ^ int(spec.reading[0]) ^ (k * 2654435761)
			var roll := pr.randf()
			if roll < 0.35:              # pluck
				kinds.append(2)
				atks.append(0.004)
				rels.append(pr.randf_range(0.6, 1.3))
				lrs.append(0.0)
				lphs.append(0.0)
				p2s.append(pr.randf_range(0.45, 0.6))
				p3s.append(pr.randf_range(0.2, 0.35))
			elif roll < 0.7:             # swell
				kinds.append(1)
				atks.append(pr.randf_range(1.2, 2.2))
				rels.append(pr.randf_range(2.0, 3.5))
				lrs.append(pr.randf_range(0.07, 0.2))
				lphs.append(pr.randf_range(0.0, TAU))
				p2s.append(pr.randf_range(0.25, 0.4))
				p3s.append(0.0)
			else:                        # drone
				kinds.append(0)
				atks.append(pr.randf_range(0.3, 0.6))
				rels.append(pr.randf_range(5.0, 8.0))
				lrs.append(pr.randf_range(0.04, 0.1))
				lphs.append(pr.randf_range(0.0, TAU))
				p2s.append(pr.randf_range(0.2, 0.4))
				p3s.append(0.0)
		st.dr_kind = kinds
		st.dr_atk = atks
		st.dr_rel = rels
		st.dr_lr = lrs
		st.dr_lph = lphs
		st.dr_p2 = p2s
		st.dr_p3 = p3s
		var hi := PackedByteArray()
		hi.resize(nd)
		st.dr_hi = hi
	return st


static func _zeroes(n: int) -> PackedFloat32Array:
	var a := PackedFloat32Array()
	a.resize(n)
	return a


# ---- internals -------------------------------------------------------------


# Rosenberg glottal pulse, one period in 64 samples: rising open phase (length
# `open_len` of the period), closing phase (`close_len`), closed remainder.
# Two tables are built - tense (sharp closure, bright) and lax (longer, rounder,
# darker) - and every cycle plays a different mix of the two, so no two glottal
# cycles share a spectrum. Differentiated at synth time by the radiation
# first-difference, which is folded into the output stage.
## UNIPOLAR by construction, DC removed downstream instead (2026-08-08).
## The mean subtraction this comment used to describe was right about the
## problem and wrong about the cure. Subtracting a constant put the CLOSED
## phase at -0.30 (tense) / -0.51 (lax) instead of zero, and the source is read
## as `table[phase] * amp * period_gain` with period_gain and the tense/lax mix
## both redrawn at every wrap - so each period multiplied a NON-ZERO pedestal
## by a new number, which is a step, ~130 times a second on a voiced segment.
## That is a discontinuity on every vowel, nasal and voiced stop, which is
## exactly where the residual clicking was measured (AE, T, D, K, AH, M).
## A flow pulse really is unipolar and really is zero while the folds are shut;
## the DC it carries is removed by a blocker on the source instead, which costs
## nothing at the wrap because it is not a function of the per-period gain.
## Old note, still true about the pedestal's cost:
## ZERO-MEAN by construction. A Rosenberg pulse is a FLOW: it is unipolar, and
## its mean (0.404 for the tense table) is a DC term. Every cascade pole passes
## DC at unity, and the only rejection anywhere downstream is the radiation
## zero, so that pedestal survived into the output at 0.0727 and held 43-52% of
## every take's total power - inaudible, but eating 0.07 of the 0.85 peak
## budget, amplitude-modulated at the syllable rate, and feeding the leveler's
## envelope detector. The physical model is not wrong (flow really is unipolar);
## what is wrong is asking a single first-order radiation zero to remove it.
## Subtracting the mean here makes the source what the cascade should see and
## makes every RMS staging number in this file honest.
static func _pulse_table(open_len: float, close_len: float) -> PackedFloat32Array:
	var t := PackedFloat32Array()
	t.resize(PULSE_N)
	for i in PULSE_N:
		var u := float(i) / float(PULSE_N)
		var v := 0.0
		if u < open_len:
			v = 0.5 * (1.0 - cos(PI * u / open_len))
		elif u < open_len + close_len:
			v = cos(PI * (u - open_len) / (2.0 * close_len))
		t[i] = v
	return t


# Schwa - the neutral vowel unstressed vowels reduce toward. This was
# [640, 1190, 2390], which is IDENTICAL to AH's own target, so reducing an AH
# toward "schwa" moved it nowhere - and AH is the vowel English reduces to.
# A true schwa is central: F1 and F2 pulled toward the neutral tube.
const _SCHWA := [500.0, 1500.0, 2500.0]


static func _seg_formants(entry: Dictionary, u: float, spec: Spec, reduce := 0.0) -> Array:
	var f: Array = entry.get("f", [500.0, 1400.0, 2400.0])
	var f2: Array = entry.get("f2", f)
	var out: Array = []
	for k in 3:
		var v := lerpf(f[k], f2[k], u)
		# Reduction is a VOWEL QUALITY move and quality lives in F1/F2. F3 is
		# doing phonemic work for exactly one vowel - ER is the only entry with
		# a low F3 (1690 Hz against 2240-3010 for everything else,
		# phonemes.gd:35) and a low F3 IS American /r/. Reducing it raised ER's
		# F3 to 2257 Hz and deleted every unstressed "-er" in the language.
		if reduce > 0.0 and k < 2:
			v = lerpf(v, _SCHWA[k], reduce)
		out.append(v * spec.formant_scale)
	return out


static func _next_formants(segs: Array, i: int, spec: Spec) -> Array:
	for j in range(i + 1, segs.size()):
		var entry: Dictionary = Phonemes.TABLE.get(segs[j].p, {})
		if entry.has("f"):
			return _seg_formants(entry, 0.0, spec)
	return [500.0 * spec.formant_scale, 1400.0 * spec.formant_scale, 2400.0 * spec.formant_scale]


static func _retarget(state: Dictionary, f: Array, _spec: Spec) -> void:
	state.ftg = f


## Point the PARALLEL branch at this obstruent's front-cavity resonances (the
## `par` triples in Phonemes.TABLE). Centres scale with vocal tract length;
## bandwidths do NOT - a shorter tube raises a resonance, it does not sharpen
## it, and scaling both was silently giving short tracts a higher-Q /s/ than
## long ones. Resonators are PEAK-normalized, so the table's third column is
## the gain it actually gets.
static func _tune_parallel(state: Dictionary, entry: Dictionary, _spec: Spec) -> void:
	# AMPLITUDES ONLY (2026-08-08 rebuild). The parallel branch no longer owns
	# any frequency of its own: slots p1..p6 are retuned every frame to the same
	# (f, bw) the cascade is using, in the frame retune block below. That is
	# Klatt 1980's actual construction - in his FORTRAN there is one SETABC call
	# per formant and the parallel difference equations reuse those very
	# coefficients, only the amplitude differs - and p.981 names the reason:
	# "the poles are the natural resonant frequencies of the entire vocal tract,
	# no matter where the source is located ... (and helpful in preventing the
	# fricative noises from 'dissociating' from the rest of the speech signal)."
	# Ghost's standalone absolute-frequency poles were that dissociation, and
	# the user localized most of the remaining static to the fricative probe.
	#
	# Slot p7 is the exception and is deliberate: Klatt's F6 is a CONSTANT extra
	# resonator with no cascade counterpart, "added to the parallel branch
	# specifically for the synthesis of very high frequency noise in [s, z]".
	# He refuses to move it, naming the cost as clicks and moving energy
	# concentrations - so continuity comes from the pole always EXISTING, not
	# from it tracking. Placed at 7400 Hz rather than his 4900 because ghost is
	# not stuck at a 10 kHz sample rate and that is where real /s/ peaks.
	var pa: Array = entry.get("pa", [])
	var amps: PackedFloat32Array = state.par_a
	for k in 7:
		amps[k] = float(pa[k]) if k < pa.size() else 0.0
	state.par_a = amps
	state.par_n = 7
	state.par_ab = float(entry.get("ab", 0.0))


## The inner loop: run `dur` seconds in FRAME-sized chunks. Per frame: EMA the
## formants toward their targets (coarticulation), EMA f0 toward the segment's
## semitone offset (Fujisaki realization), retune the cascade, then fill samples
## from pulse + noise sources. `vamp` scales the periodic source, `namp` the
## noise path; `asp_cascade` sends noise through the formant cascade (for HH).
static func _run_frames(out: PackedFloat32Array, state: Dictionary, spec: Spec,
		rng: RandomNumberGenerator, seg: Dictionary, entry: Dictionary,
		vamp: float, namp: float, dur: float, asp_cascade := false,
		burst := false, to_f: Array = []) -> void:
	var n := int(round(dur * SR))
	# NOISE ROUTE tracking: aspiration (VOT, /h/) is GLOTTAL noise and excites
	# the whole tract, so it runs through the cascade; frication and bursts are
	# generated at a constriction and excite only the cavity in front of it, so
	# they run through the PARALLEL branch. The envelope (nampsm) must not
	# survive a route switch: re-emitting a leftover on the other bus snapped a
	# burst-tuned band on in ONE sample ~1 ms into the vowel after every stop -
	# measured as the largest remaining click class (|d| ~0.3).
	var route := 1 if asp_cascade else 0
	if int(state.nroute) != route:
		state.nampsm = 0.0
	state.nroute = route
	# BOTH envelopes (noise and voiced amplitude) are RAMPED per sample between
	# frame values (below). As per-frame constants they were 2.9 ms staircases:
	# the burst attack landed as a one-sample cliff (the click inside every
	# P/T/K), and a fast vowel onset stepped ~30% mid-pulse per frame, which
	# the formant cascade then amplified ~3x - measured as a one-sample jump
	# of ~0.2 at stop->vowel boundaries. A ramp has no corners.
	var nsm_from: float = state.nampsm
	var amp_from: float = state.ampsm
	if burst:
		# a plosive burst is a TRANSIENT: instant attack, exponential decay.
		# The EMA's slow-attack/hard-cut envelope was the burst reversed -
		# every T and K landed as a pop instead of a release
		state.nampsm = maxf(float(state.nampsm), namp)
		namp = 0.0
	var r1: Reso = state.r1
	var r2: Reso = state.r2
	var r3: Reso = state.r3
	var r4: Reso = state.r4
	var r5: Reso = state.r5
	var r6: Reso = state.r6
	var anti: AntiReso = state.anti
	var par: Array = state.par
	var par_a: PackedFloat32Array = state.par_a
	var par_n: int = state.par_n
	var par_ab: float = state.par_ab
	var pulse: PackedFloat32Array = state.pulse
	var is_diph: bool = entry.has("f2")
	var ttype := String(entry.get("type", "sil"))
	# inflection depth scales the whole deviation from base: at 0 the melody
	# collapses to a flat monotone at f0_base, above 1 it exaggerates
	var f0_target: float = spec.f0_base * pow(2.0, seg.semitones * spec.inflect / 12.0) * 1.06
	var vib_on: bool = spec.song > 0.0 and ttype != "sil"
	var done := 0
	# radiation memory CONTINUES across segments - resetting it clicked at
	# every phoneme boundary (a pop per segment; the "static")
	var prev: float = state.prev
	# ... and so do the per-period draws: period_gain resetting to 1.0 at every
	# segment boundary was a mid-cycle gain step per phoneme, and the jitter
	# draw being overwritten at every FRAME boundary left the cycle lengths
	# nearly metronomic - the oldest robot-voice cue there is
	var period_gain: float = state.pgain
	var jit: float = state.jit
	# coarticulation speed is articulator-dependent, not one constant: the
	# tract glides slowly into vowels/glides/nasals and releases fast out of a
	# burst - a single fast EMA read as plastic morphs between postures
	# HALVED (2026-08-08). A first-order EMA needs 3 tau to arrive; at 24 ms
	# that is 72 ms against a median vowel of ~95 ms, and replaying state.fsm
	# against the alignment showed only 25.5% of vowels reaching within 5% of
	# their F1 target (56.6% for F2). Under 60 ms, F1 spent 0% of the segment
	# on target. The locus glide is NOT implicated - removing it entirely moved
	# midpoint attainment by 0.00 - so the time constant is the whole story.
	# Glides and nasals additionally had the SLOWEST tau on the shortest
	# segments in the inventory (60-75 ms), which is backwards: a glide is a
	# ballistic gesture, not a slow drift.
	var ftau := 0.009
	if ttype == "vowel":
		ftau = 0.012
	elif ttype == "glide" or ttype == "nasal":
		ftau = 0.010
	elif ttype == "stop":
		ftau = 0.006
	# the nasal zero engages by MIX (the anti-resonator itself runs on every
	# sample so its state never sees a switch-on transient), and its FREQUENCY
	# is the phoneme's own: the side-cavity zero is the entire acoustic
	# difference between m, n and ng, and one shared notch collapsed them.
	var anti_target := 1.0 if ttype == "nasal" else 0.0
	if ttype == "nasal":
		state.anti_f = float(entry.get("zero", 950.0))
	# THE LOCUS GLIDE: hold this segment's own posture, then bend toward the
	# next one over the final LOCUS_TIME. Consonant PLACE lives in the F2
	# transition on the neighbouring vowel, not in the consonant's own steady
	# state - so the movement has to happen while the vowel is still sounding.
	var base_f: Array = (state.ftg as Array).duplicate()
	var glide_n := 0.0
	if not to_f.is_empty():
		# never spend more than LOCUS_SHARE of a segment leaving its own
		# target. At half, a 97 ms /r/ spent 48 ms travelling away from the
		# low F3 that IS an /r/, and with the glide EMA lagging on top it
		# never arrived - reported as the "rs" in "yours" being inaudible.
		glide_n = minf(LOCUS_TIME * SR, float(n) * LOCUS_SHARE)
	while done < n:
		var m := mini(FRAME, n - done)
		var u := float(done) / maxf(1.0, float(n))
		if is_diph:
			base_f = _seg_formants(entry, u, spec, seg.get("reduce", 0.0))
		if glide_n > 0.0:
			# 0 until the glide window opens, reaching 1 exactly at the
			# boundary, so the target is continuous across segments
			var w := clampf((float(done) - (float(n) - glide_n)) / glide_n, 0.0, 1.0)
			var g: Array = []
			for k in 3:
				g.append(lerpf(float(base_f[k]), float(to_f[k]), w))
			state.ftg = g
		else:
			state.ftg = base_f
		# EMAs: formants (per-type tau above), f0 ~35 ms, amplitude ~8 ms
		var fa := 1.0 - exp(-float(m) / (SR * ftau))
		# f0 approach: ~35 ms speaking, ~12 ms singing. A singer ARRIVES on a
		# pitch; a speaker glides onto it. With the speech constant, a 50 ms run
		# note never reached its target at all and the fast passages smeared
		# off-key - measured as on-note falling back to speech levels the moment
		# runs were introduced. It is also the original staircase's character.
		var pa := 1.0 - exp(-float(m) / (SR * lerpf(0.035, 0.012, spec.song)))
		var aa := 1.0 - exp(-float(m) / (SR * 0.008))
		# ... and the retune glide (~60 ms): spec scalars read per frame, so a
		# live retune() bends them instead of stepping them mid-stream
		var ra := 1.0 - exp(-float(m) / (SR * 0.06))
		for k in 3:
			state.fsm[k] = lerpf(state.fsm[k], state.ftg[k], fa)
		var f0_t := f0_target
		if vib_on:
			# time INSIDE this note drives the onset ramp; the phase runs on the
			# take's own clock so the waver is continuous across a held note's
			# internal segment boundaries
			var t_in := float(done) / SR
			var depth: float = VIB_DEPTH * spec.song * clampf(t_in / VIB_ONSET, 0.0, 1.0)
			state.vib_ph = fposmod(float(state.vib_ph) + TAU * VIB_RATE * float(m) / SR, TAU)
			f0_t *= pow(2.0, depth * sin(float(state.vib_ph)) / 12.0)
		state.f0sm = lerpf(state.f0sm, f0_t, pa)
		state.ampsm = lerpf(state.ampsm, vamp, aa)
		# the noise path gets an envelope too: frication switching on/off
		# abruptly was a click per consonant
		# BOUNDED NOISE RELEASE (2026-08-08). A symmetric 8 ms EMA gated only at
		# 1e-4 takes ln(1e4) x 8 = 74 ms to actually reach zero, so frication
		# bleeds most of the way into the following vowel. Measured: above 5 kHz
		# the noise is still within 1-2 dB of the obstruent's own plateau at the
		# vowel's FIRST sample, and 7-8 dB down 10 ms in. It is summed outside
		# the cascade, so it is not shaped by the vowel at all - it just lies on
		# top of the F2 onset, which phonemes.gd:55-57 names as the primary
		# place cue for /f/ and /th/. That is the "muddy, not sharp enough"
		# report: every consonant is smeared into the vowel after it.
		#
		# Klatt 1980 p.978 interpolates noise intensity across the 5 ms frame
		# and is therefore bounded - the noise is GONE 5 ms after the frame
		# says so. Attack keeps the existing ramp (a burst must stay a
		# transient); only the release is bounded, plus a real floor so the
		# tail cannot crawl.
		var nrel := aa if namp > float(state.nampsm) else 1.0 - exp(-float(m) / (SR * 0.005))
		state.nampsm = lerpf(state.nampsm, namp, nrel)
		if namp <= 0.0 and float(state.nampsm) < 0.0005:
			state.nampsm = 0.0
		state.anti_mix = lerpf(state.anti_mix, anti_target, aa)
		state.brsm = lerpf(float(state.brsm), spec.breath, ra)
		state.airgsm = lerpf(float(state.airgsm), spec.air_gain, ra)
		state.aircsm = lerpf(float(state.aircsm), spec.air_cut, ra)
		state.fssm = lerpf(float(state.fssm), spec.formant_scale, ra)
		var nsm: float = state.nampsm
		var amix: float = state.anti_mix
		var breath_g: float = float(state.brsm) * NOISE_TRIM if NOISE_FX else 0.0
		var airg: float = float(state.airgsm) * NOISE_TRIM if NOISE_FX else 0.0
		var fs: float = state.fssm
		r1.tune(state.fsm[0], BW[0])
		r2.tune(state.fsm[1], BW[1])
		r3.tune(state.fsm[2], BW[2])
		# the upper poles: fixed presence formants continuing the uniform-tube
		# series. Three resonators left nothing above 3 kHz but noise (the
		# hollow AM-radio timbre); F4-F6 give the voice a top, the way Klatt's
		# multi-pole cascade did.
		# THE UPPER CLUSTER (2026-08-08). Klatt Table I puts F4/F5 at 3300/3750
		# with B4 250 / B5 200, and p.980 says why: "The particular values
		# chosen for the fourth and fifth formant frequencies produce an energy
		# concentration around 3 to 3.5 kHz and a rapid falloff above about 4
		# kHz, which is a pattern typical of many talkers." Ghost had them 1300
		# Hz apart instead of 450, which is two shallow humps instead of one
		# concentration: modelled 2.5-5 kHz peak-to-valley 36.4 dB for /AA/
		# against 51.9 for the clustered placement, and 3.3 dB less energy in
		# 2-4 kHz. Measured vowel spectra matched the analytic prediction of
		# ghost's own equations within 1.5 dB, so nothing downstream was
		# causing this and nothing downstream could have undone it - this is
		# the muffle, at its source.
		r4.tune(3400.0 * fs, BW[3])
		r5.tune(3800.0 * fs, BW[4])
		r6.tune(F6 * fs, BW[5])
		anti.tune(float(state.anti_f) * fs, 350.0)
		# The parallel bank rides the SAME poles, retuned every frame so the
		# frication maxima move with the tract instead of standing still while
		# the voice glides past them. p7 is the fixed sibilant pole.
		var pbank: Array = state.par
		(pbank[0] as Reso).tune_peak(state.fsm[0], BW[0])
		(pbank[1] as Reso).tune_peak(state.fsm[1], BW[1])
		(pbank[2] as Reso).tune_peak(state.fsm[2], BW[2])
		(pbank[3] as Reso).tune_peak(3400.0 * fs, BW[3])
		(pbank[4] as Reso).tune_peak(3800.0 * fs, BW[4])
		(pbank[5] as Reso).tune_peak(F6 * fs, BW[5])
		(pbank[6] as Reso).tune_peak(7400.0 * fs, 900.0)
		var inc: float = state.f0sm * jit * float(PULSE_N) / SR
		var amp: float = state.ampsm
		var pulse_lax: PackedFloat32Array = state.pulse_lax
		var tension: float = state.tension
		var nlp: float = state.nlp
		var tilt_y: float = state.tilt_y
		# the air line as a one-pole coefficient: what leaks above it is static
		var air_k: float = 1.0 - exp(-TAU * float(state.aircsm) / SR)
		# vocal effort opens the spectral tilt: emphatic frames are brighter,
		# settled ones darker - the walk's dynamics now reach the TIMBRE.
		# Floor raised (0.3 -> 0.45 base, 0.2 -> 0.35 clamp) in the fidelity
		# pass: the old floor lowpassed settled speech into the reported
		# "dull, flat" tone; the dynamic (emphatic = brighter) is unchanged.
		var tilt_k: float = clampf(0.45 + 0.5 * amp, 0.35, 0.98)
		var esend: float = seg.get("echo", 0.0)
		var ebuf: PackedFloat32Array = state.ebuf
		var eidx: int = state.eidx
		var esize := ebuf.size()
		var elp: float = state.elp
		var elp_k: float = 1.0 - exp(-TAU * ECHO_LP / SR)
		var dr_st: PackedFloat32Array = state.dr_st
		var dr_f: PackedFloat32Array = state.dr_f
		var dr_ph: PackedFloat32Array = state.dr_ph
		var dr_env: PackedFloat32Array = state.dr_env
		var dr_n := dr_f.size()
		var dr_out: float = state.dr_out
		var dr_alt: bool = state.dr_alt
		var dr_p2: PackedFloat32Array = state.dr_p2
		var dr_p3: PackedFloat32Array = state.dr_p3
		var dr_gain := PackedFloat32Array()
		if dr_n > 0:
			# per FRAME: the ENSEMBLE (see the DRONE consts). Every tone reads
			# melodic proximity, but each responds in its own character -
			# drones hold, swells breathe on their own LFO, plucks fire as
			# EVENTS when the melody arrives on their note - and the budget
			# makes them compete, so the resonance shifts instead of blending.
			var dr_kind: PackedInt32Array = state.dr_kind
			var dr_atk: PackedFloat32Array = state.dr_atk
			var dr_rel: PackedFloat32Array = state.dr_rel
			var dr_lr: PackedFloat32Array = state.dr_lr
			var dr_lph: PackedFloat32Array = state.dr_lph
			var dr_hi: PackedByteArray = state.dr_hi
			var tnow := float(out.size()) / SR
			var semis_now := 12.0 * log(maxf(state.f0sm, 1.0) / (spec.f0_base * 1.06)) / log(2.0)
			var vgate: float = 1.0 if vamp > 0.1 else 0.0
			# the consonance tide (see the DRONE consts): slow seeded cycle
			# blended with a sustained-voice EMA - high tide = chordal
			state.dr_ve = lerpf(float(state.dr_ve), vgate, 1.0 - exp(-float(m) / (SR * 2.5)))
			var tide := clampf(0.5 + 0.5 * sin(TAU * float(state.dr_tr) * tnow + float(state.dr_tph)), 0.0, 1.0)
			var tide_mix := clampf(0.55 * tide + 0.55 * float(state.dr_ve), 0.0, 1.0)
			var esum := 0.0
			for k in dr_n:
				var prox: float = maxf(0.0, 1.0 - absf(semis_now - dr_st[k]) / DRONE_NEAR)
				if dr_kind[k] == 2:
					# PLUCK: an arrival event - the melody crossing ONTO the
					# note fires it at full strength (phase reset for a clean
					# transient), then it only decays
					var on_note: bool = vgate > 0.0 and prox > 0.6
					if on_note and dr_hi[k] == 0:
						dr_env[k] = 1.0
						dr_ph[k] = 0.0
					dr_hi[k] = 1 if on_note else 0
					dr_env[k] *= exp(-float(m) / (SR * dr_rel[k]))
				else:
					# high tide flattens the LFOs: the swells steady and SYNC
					# into the chord instead of ebbing against it
					var depth: float = (0.25 if dr_kind[k] == 0 else 0.55) * (1.0 - 0.7 * tide_mix)
					var lfo := 1.0 - depth * (0.5 + 0.5 * sin(TAU * dr_lr[k] * tnow + dr_lph[k]))
					var target: float = vgate * prox * lfo
					if target > dr_env[k]:
						dr_env[k] = minf(dr_env[k] + float(m) / (SR * dr_atk[k]), target)
					else:
						dr_env[k] *= exp(-float(m) / (SR * dr_rel[k]))
				esum += dr_env[k]
			# the budget: a chord's total energy is capped, so a newly lit
			# tone pushes the others back - motion, not accumulation
			var budget := lerpf(DRONE_BUDGET_LO, DRONE_BUDGET_HI, tide_mix)
			var scale := 1.0 if esum <= budget else budget / esum
			dr_gain.resize(dr_n)
			for k in dr_n:
				dr_gain[k] = dr_env[k] * scale
			state.dr_hi = dr_hi          # CoW: the pluck edge detector persists
		var phase: float = state.phase
		var dcb: float = state.dcb
		var blk := PackedFloat32Array()
		blk.resize(m)
		for _s in m:
			# the envelopes, ramped across the frame (see the top of the func)
			var env_u := (float(_s) + 1.0) / float(m)
			var nsm_s := lerpf(nsm_from, nsm, env_u)
			var amp_s := lerpf(amp_from, amp, env_u)
			phase += inc
			if phase >= float(PULSE_N):
				phase -= float(PULSE_N)
				# per-period organic variation: jitter the pitch, shimmer the
				# gain, and wander the glottal TENSION - no two cycles alike.
				# The draws live in state and hold until the NEXT period.
				# RAW bypass: a perfectly regular pulse train (jit, gain and
				# tension hold their neutral state defaults).
				if not RAW_MODE:
					jit = 1.0 + rng.randfn(0.0, spec.jitter)
					inc = state.f0sm * jit * float(PULSE_N) / SR
					period_gain = 1.0 + rng.randfn(0.0, spec.shimmer)
					tension = clampf(lerpf(tension, rng.randf(), 0.3), 0.0, 1.0)
			# interpolated wavetable read: the raw int() lookup stair-stepped
			# the pulse - audible as gritty aliasing static
			var pidx := int(phase)
			var pfrac := phase - float(pidx)
			var pnext := pidx + 1 if pidx < PULSE_N - 1 else 0
			var src := lerpf(
				lerpf(pulse_lax[pidx], pulse_lax[pnext], pfrac),
				lerpf(pulse[pidx], pulse[pnext], pfrac), tension) * amp_s * period_gain
			# DC blocker: a one-pole 20 Hz highpass takes the flow pulse's DC
			# out AFTER the per-period gain, so the pedestal costs no peak
			# budget and no step at the wrap. See _pulse_table.
			dcb += (src - dcb) * 0.00285
			src -= dcb
			var hiss := rng.randf() * 2.0 - 1.0
			# aspiration is pitch-synchronous: air leaks during the OPEN phase
			# of the cycle, not as a steady decoupled hiss floor
			src += hiss * breath_g * amp_s * (1.0 if phase < 26.0 else 0.3)
			# stations with static: above the air line the harmonic voice gives
			# way to noise - highpassed hiss joins the excitation itself
			nlp += air_k * (hiss - nlp)
			src += (hiss - nlp) * airg * amp_s
			# effort tilt (one-pole lowpass, coefficient driven by amp)
			tilt_y += tilt_k * (src - tilt_y)
			src = tilt_y
			# turbulence gets its OWN draw. Sharing one `hiss` sample between
			# the aspiration, the air band and the constriction noise made
			# three nominally independent sources add coherently (+6 dB
			# instead of +3) and correlated their spectra.
			# REVERTED 2026-08-08. Klatt's topology (Gaussian source, low-passed
			# so radiation cancels it, summed BEFORE radiation) is architecturally
			# right and measured right - obstruent tilt landed flat - but the
			# implementation made the artifact audibly WORSE, roughly doubling its
			# rate by ear. The compensating integrator (nlpf = turb + RAD_A * nlpf,
			# a 140 Hz pole) runs on EVERY sample regardless of whether frication is
			# active, so it is a random walk carrying a large drifting offset; when
			# a fricative gates on, the envelope multiplies whatever value the walk
			# happens to hold, giving a step at the onset AND at the offset.
			# Any future attempt must gate or reset the walk with the envelope.
			var turb := rng.randf() * 2.0 - 1.0
			var y: float
			if asp_cascade:
				# GLOTTAL noise (/h/, VOT): the source is at the folds, so the
				# whole tract filters it - the cascade is correct here
				y = r6.step(r5.step(r4.step(r3.step(r2.step(r1.step(hiss * nsm_s * 0.5))))))
			else:
				y = r6.step(r5.step(r4.step(r3.step(r2.step(r1.step(src))))))
			# the nasal zero: blend toward the anti-resonated path while a
			# nasal speaks (the murmur is DEFINED by removed energy). This is
			# a property of the VOICED side branch, so it sits before the
			# parallel sum, not after.
			var yz := anti.step(y)
			y = lerpf(y, yz, amix)
			# THE PARALLEL BRANCH (see FRIC_LEVEL): frication and bursts are
			# generated AT a constriction and excite only the cavity in front
			# of it. Their resonators are peak-normalized and summed here -
			# KLATT'S FIXED SIGN PATTERN (1980 p.982, Fig. 13). Signs are no
			# longer data: R2, R4 and R6 are inverted, everything else adds.
			# The rationale ghost recorded was backwards - alternation does not
			# stop a null forming, it deliberately CAUSES one, because Klatt
			# measured that "spectral notches are less perceptible than energy
			# fill in a spectral valley between two formants". With every pole
			# now sitting on a tract formant rather than on its own invented
			# frequency, the old per-phoneme sign workaround has nothing left
			# to work around.
			var pout := par_ab * turb
			pout += (par[0] as Reso).step(turb) * par_a[0]
			pout -= (par[1] as Reso).step(turb) * par_a[1]
			pout += (par[2] as Reso).step(turb) * par_a[2]
			pout -= (par[3] as Reso).step(turb) * par_a[3]
			pout += (par[4] as Reso).step(turb) * par_a[4]
			pout -= (par[5] as Reso).step(turb) * par_a[5]
			pout += (par[6] as Reso).step(turb) * par_a[6]
			# radiation: first difference brightens the spectrum like lips do.
			var rad := y - prev * RAD_A
			prev = y
			# Frication sums AFTER radiation: without a pre-compensating filter
			# this is what keeps its spectrum flat (see the note above), and it
			# is the arrangement the last build the user judged least-bad used.
			if nsm_s > 0.0001 and not asp_cascade:
				rad += pout * nsm_s * FRIC_LEVEL
			# the echo bus: silent until a word is sent into it, then it rings.
			# Damped in the loop (see ECHO_LP) - every repeat comes back darker
			var e := ebuf[eidx]
			elp += elp_k * ((rad * esend + e * ECHO_FB) - elp)
			ebuf[eidx] = elp
			eidx += 1
			if eidx >= esize:
				eidx = 0
			# THE RESONANCE: the anchor tones themselves (see the DRONE
			# consts) - a sine with a soft octave for warmth, each held by
			# its envelope. Half-rate stepped with a zero-order hold; the
			# tones live far below where that matters.
			if dr_n > 0:
				if dr_alt:
					var ssum := 0.0
					for k in dr_n:
						var ev := dr_gain[k]
						if ev > 0.002:
							var ph := dr_ph[k] + TAU * dr_f[k] * 2.0 / SR
							if ph >= TAU:
								ph -= TAU
							dr_ph[k] = ph
							ssum += (sin(ph) + dr_p2[k] * sin(ph * 2.0)
								+ dr_p3[k] * sin(ph * 3.0)) * ev
					dr_out = ssum * DRONE_LEVEL
				dr_alt = not dr_alt
			blk[_s] = (rad + e * 0.8 + dr_out) * OUT_GAIN
		if RAW_MODE:
			# RAW bypass: no leveler, no limiter, no clip, no grain - the
			# block goes out as synthesized, under one fixed measured-safe
			# trim. (lim_buf stays empty, so the end-of-take flush is a no-op.)
			for i in m:
				out.append(blk[i] * RAW_TRIM)
		else:
			# the broadcast stage (see the consts): emit the PREVIOUS block
			# under a linear gain ramp whose endpoint already respects THIS
			# block's peak - lookahead limiting with no corners and no pumping
			var pend: PackedFloat32Array = state.lim_buf
			var pk := 0.0
			var bsum := 0.0
			for i in m:
				var av := absf(blk[i])
				pk = maxf(pk, av)
				bsum += av
			# the leveler: block envelope with fast attack / slow release drives
			# a bounded 2:1 gain; the limiter needs then constrain the SAME ramp
			# endpoint, so peaks are still guaranteed under LIMIT with no corners
			var cenv: float = state.cenv
			var bmean := bsum / maxf(float(m), 1.0)
			if bmean > cenv:
				cenv = lerpf(cenv, bmean, 0.19)      # ~8 ms attack
			else:
				cenv = lerpf(cenv, bmean, 0.012)     # ~120 ms release
			state.cenv = cenv
			var gc := clampf(sqrt(COMP_TARGET / maxf(cenv, 0.02)), COMP_MIN, COMP_MAX)
			var need_new: float = LIMIT / maxf(pk, 0.0001)
			var g0: float = state.lim_g
			var g1: float = minf(gc, minf(float(state.lim_need), need_new))
			g1 = minf(g1, g0 * LIMIT_RELEASE)
			var np := pend.size()
			for i in np:
				var gg := lerpf(g0, g1, (float(i) + 1.0) / float(np))
				# the safety clip: IDENTITY below the knee - the ramp already did
				# the real peak work, this only rounds the rare surviving tip
				var s := pend[i] * gg
				var sa := absf(s)
				if sa > KNEE:
					s = signf(s) * (KNEE + (CLIP_CEIL - KNEE) * tanh((sa - KNEE) / (CLIP_CEIL - KNEE)))
				# the medium's constant grain - drawn even when NOISE_FX is off,
				# so the flag A/Bs the same take rather than rerolling everything
				var grain := (rng.randf() * 2.0 - 1.0) * FLOOR_MIN
				out.append(s + (grain if NOISE_FX else 0.0))
			state.lim_buf = blk
			state.lim_need = need_new
			state.lim_g = g1
		nsm_from = nsm
		amp_from = amp
		state.phase = phase
		state.dcb = dcb   # locals hoisted out of the loop must be written back
		state.tension = tension
		state.nlp = nlp
		state.tilt_y = tilt_y
		state.prev = prev
		state.pgain = period_gain
		state.jit = jit
		state.ebuf = ebuf            # packed arrays are CoW: persist the written copy
		state.eidx = eidx
		state.elp = elp
		state.dr_ph = dr_ph          # CoW again: the ring state must persist
		state.dr_env = dr_env
		state.dr_out = dr_out
		state.dr_alt = dr_alt
		done += m


static func _record_timing(state: Dictionary, seg: Dictionary, t0: float, t1: float) -> void:
	if seg.word < 0:
		return
	# `reduce` rides along because it is the one planner decision a measurement
	# pass cannot recover from the audio: whether a vowel was MEANT to be
	# centralized is the difference between a healthy reduction and a collapsed
	# vowel space, and inferring it from duration would be circular.
	state.phones.append({"p": seg.p, "t0": t0, "t1": t1, "word": seg.word,
		"sentence": seg.sentence, "reduce": seg.get("reduce", 0.0)})
	# KEYED BY THE SOURCE RUN where there is one, so the three words of `two
	# thousand nine` open ONE karaoke entry reading `2009` and the loop below
	# extends it to the last of them. Separate namespaces, because a run index
	# and a word index are both small integers and would otherwise collide.
	var key := "%d:w%d" % [seg.sentence, seg.word]
	if int(seg.get("group", -1)) >= 0:
		key = "%d:g%d" % [seg.sentence, int(seg.group)]
	if seg.word_start and not state.wopen.has(key):
		# a run whose first word was dropped leaves a continuation with nothing
		# to draw; show what is being said rather than an empty card
		var shown := String(seg.get("display", seg.text))
		state.wopen[key] = state.words.size()
		state.words.append({"text": shown if not shown.is_empty() else String(seg.text),
			"t0": t0, "t1": t1, "sentence": seg.sentence})
	if state.wopen.has(key):
		state.words[state.wopen[key]].t1 = t1


## Write PCM16 mono WAV. Returns the globalized path (playable by Spectrum,
## ffmpeg, anything).
##
## ATOMIC: the bytes go to a temp file that is renamed into place at the end.
## A take can be read by another PROCESS (the export bake, the export render)
## while something re-renders the same take here - and FileAccess.WRITE
## truncates on open, so a plain write left a reader holding an empty or
## half-written WAV. That is exactly how an export died: the render process
## opened a truncated take, logged "no audio loaded", and then recorded
## silence forever because a session with no audio never ends.
static func write_wav(path: String, pcm: PackedFloat32Array) -> String:
	var bytes := PackedByteArray()
	bytes.resize(pcm.size() * 2)
	for i in pcm.size():
		var v := int(clampf(pcm[i], -1.0, 1.0) * 32767.0)
		bytes.encode_s16(i * 2, v)
	var tmp := path + ".part"
	var f := FileAccess.open(tmp, FileAccess.WRITE)
	f.store_buffer("RIFF".to_ascii_buffer())
	f.store_32(36 + bytes.size())
	f.store_buffer("WAVE".to_ascii_buffer())
	f.store_buffer("fmt ".to_ascii_buffer())
	f.store_32(16)
	f.store_16(1)                      # PCM
	f.store_16(1)                      # mono
	f.store_32(SR)
	f.store_32(SR * 2)                 # byte rate
	f.store_16(2)                      # block align
	f.store_16(16)                     # bits
	f.store_buffer("data".to_ascii_buffer())
	f.store_32(bytes.size())
	f.store_buffer(bytes)
	f.close()
	# rename over the destination: readers see the old take or the new one,
	# never a partial one
	var abs_tmp := ProjectSettings.globalize_path(tmp)
	var abs_out := ProjectSettings.globalize_path(path)
	if DirAccess.rename_absolute(abs_tmp, abs_out) != OK:
		push_warning("ghost: could not finalize WAV at " + abs_out)
		return abs_tmp
	return abs_out
