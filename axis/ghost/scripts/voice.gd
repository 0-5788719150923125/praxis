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

const SR := 22050
const FRAME := 64                     # samples per parameter update (~2.9 ms)
const TWO_PI := TAU
# Fixed output gain in place of retroactive normalization (streaming cannot
# know the future peak). Calibrated: the cascade's raw peak is ~3.3 and nearly
# invariant across trait extremes (2.6-3.3 measured), so this lands peaks
# around 0.85 - and live playback, the WAV, and loop passes are all identical.
const OUT_GAIN := 0.26
# The echo bus: a feedback delay line the output always passes through. Sends
# are zero except on echo-activated words, so the line is silent until a word
# is thrown into it - then it rings, decaying, through whatever follows
# (including the pauses). Fixed model-agnostic constants, per the house rule.
const ECHO_DELAY := 0.17              # seconds
const ECHO_FB := 0.45                 # feedback per repeat


## The speaker's trait axes, each in [-1, 1]. THE TRAIT VECTOR IS THE VOICE:
## the zero vector is the hand-curated default speaker (its concrete centres
## live in Spec.from_traits - tune them there), a seed only *initializes* the
## vector, and any UI modulation edits it directly - so a speaker is replicated
## by replaying the vector, never by replaying the gesture that found it.
const TRAIT_KEYS := ["pitch", "lilt", "tract", "pace", "breath", "grit", "drawl", "air"]

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
	var f0_base := 130.0              # speaking pitch floor (Hz)
	var f0_accent := 4.0              # accent bump strength (semitones)
	var f0_decl := 3.0                # declination span per sentence (semitones)
	var formant_scale := 1.0          # vocal tract length (bright .. dark)
	var rate := 1.0                   # tempo multiplier (>1 = faster)
	var breath := 0.05                # aspiration mixed into voiced frames
	var jitter := 0.012               # per-period f0 noise (organic, not robotic)
	var shimmer := 0.06               # per-period amplitude noise
	var pause_comma := 0.18           # seconds
	var pause_stop := 0.42
	var final_lengthen := 1.25        # phrase-final syllable stretch
	var air_gain := 0.1               # static-band strength (noise above the air line)
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
		s.f0_base = 130.0 * pow(2.0, 0.85 * pitch)
		s.f0_accent = 4.0 * pow(2.0, 0.8 * lilt)
		s.f0_decl = 3.0 * pow(2.0, 0.5 * lilt)
		s.formant_scale = pow(2.0, 0.22 * tract)
		s.rate = pow(2.0, 0.35 * pace)
		s.breath = 0.05 * pow(2.5, breath)
		s.jitter = 0.012 * pow(2.2, grit)
		s.shimmer = 0.06 * pow(2.2, grit)
		s.pause_comma = 0.18 * pow(1.6, drawl)
		s.pause_stop = 0.42 * pow(1.6, drawl)
		s.final_lengthen = 1.25 * pow(1.25, drawl)
		# the air trait: how much of the upper spectrum tunes to static, and
		# where that line sits (high air = the line drops, more of the voice
		# is breath-noise - the multi-band harmonic/noise mix)
		s.air_gain = 0.1 * pow(3.0, air)
		s.air_cut = 3000.0 * pow(2.0, -0.7 * air)
		return s

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
	var _gate: RandomNumberGenerator  # order-dependent stochastic gates (deterministic)

	## The population average, created at initialization and OUTSIDE any
	## lineage's influence: the midpoint of every genome range. It is always
	## one voice in the blend (the "1" of 1+N) - the regularizer that keeps a
	## belt full of extremes from compounding.
	const PRIOR := {
		"heat": 1.35, "baseline": 0.375, "settle": 0.11, "breath_span": 9.5,
		"spend_window": 2.4, "lean": 1.0, "pace_hot": 0.91, "pace_calm": 1.21,
		"act_thr": 1.9, "act_gain": 1.0, "gravity": 0.2, "ring": 0.6,
	}
	const PRIOR_MOTIF_SEED := 314159
	# The channels a word can sparsely ACTIVATE on - each independent, each
	# with its own refractory. What firing does: stretch = the word's own
	# timescale pulls long; pitch = a jump toward an attractor; echo = the word
	# rings through the delay line; swell = a crescendo across the word.
	const ACT_CHANNELS := ["stretch", "pitch", "echo", "swell"]

	## Blended construction: `lineages` is the working reading first, then any
	## toggled belt influences. The genome is the uniform mean of the PRIOR
	## plus every lineage's genome (1+N voices in the average); the motif
	## vocabulary pools everyone's gestures. Deterministic per lineage set.
	func _init(lineages: Array) -> void:
		var genomes: Array = [PRIOR.duplicate()]
		for lineage in lineages:
			genomes.append(_lineage_genome(lineage))
		p = {}
		for key in PRIOR:
			var v := 0.0
			for g in genomes:
				v += g[key]
			p[key] = v / genomes.size()
		_motifs = _motif_bank(PRIOR_MOTIF_SEED)
		for lineage in lineages:
			_motifs.append_array(_motif_bank(int(lineage[0])))
		# the pitch attractor set: the prior's anchors plus each lineage's own,
		# pooled - the semitone shelf the melody gravitates toward (and jumps
		# to on a pitch activation). This is the musical-quantization quality.
		_anchors = [0.0, -2.0, 3.0]
		for lineage in lineages:
			var ar := RandomNumberGenerator.new()
			ar.seed = hash("anchors") ^ int(lineage[0])
			for _i in 3:
				_anchors.append(ar.randf_range(-6.0, 8.0))
		for c in ACT_CHANNELS:
			_refract[c] = 0.0
		_gate = RandomNumberGenerator.new()
		_gate.seed = hash(str(lineages))
		arousal = p.heat
		motif = _motifs[0]

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
		}
		for i in range(1, lineage.size()):
			var pr := RandomNumberGenerator.new()
			pr.seed = hash("walk_gen") ^ int(lineage[i]) ^ i
			var scale := pow(0.6, i)
			for key in g:
				g[key] *= 1.0 + pr.randfn(0.0, 0.18 * scale)
		return g

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
	func word(stressed: bool, nsyll: int, est_dur: float, frac: float, punct: bool) -> Dictionary:
		var norm: float = clampf(arousal / p.heat, 0.0, 1.0)
		var pace: float = lerpf(p.pace_calm, p.pace_hot, norm)
		var emph := 0.0
		var pre_pause := 0.0
		if stressed and spent < 0.4:
			var appetite: float = p.lean * motif.lean * (0.5 + 0.5 * (1.0 - norm))
			if _gate.randf() < 0.22 * appetite + (0.18 if frac > 0.7 else 0.0):
				emph = clampf(appetite * (1.0 - spent), 0.4, 1.4)
				spent += 0.55
				pre_pause = 0.04 + 0.09 * emph * (1.0 - norm)
		var breath_pause := 0.0
		breath += nsyll
		if breath >= p.breath_span and not punct:
			breath_pause = 0.12 + 0.28 * (1.0 - norm)
			breath = 0.0
			arousal = maxf(arousal - 0.12, 0.2)
			spent *= 0.5
		elif punct:
			breath = maxf(breath - p.breath_span * 0.6, 0.0)   # punctuation is half a breath
		# sparse activations: each channel is a thresholded nonlinearity over a
		# seeded drive plus the resonance ring. Firing raises that channel's own
		# bar (fast attack) which then decays slowly - so events are SPARSE and
		# self-spacing; firing also feeds the ring, so events resonate: one
		# strike colours the next seconds and invites a neighbour.
		var acts := {}
		for c in ACT_CHANNELS:
			_refract[c] *= exp(-est_dur / 2.5)
			var drive: float = _gate.randfn(0.0, 1.0) + _ring_amp * 0.5
			var a: float = maxf(0.0, drive - (p.act_thr + _refract[c])) * p.act_gain
			if a > 0.0:
				_refract[c] += 1.2
				_ring_amp = minf(_ring_amp + p.ring * a * 0.5, 1.5)
			acts[c] = clampf(a, 0.0, 1.5)
		var ring_st: float = p.ring * _ring_amp * sin(_ring_ph) * 1.5
		_ring_amp *= exp(-est_dur / 0.9)
		_ring_ph += est_dur * TAU * 1.3
		# the EMAs advance by the word's own duration
		arousal = lerpf(arousal, p.baseline, 1.0 - exp(-p.settle * est_dur))
		spent *= exp(-est_dur / p.spend_window)
		return {
			"pace": pace, "emph": emph, "pre_pause": pre_pause,
			"breath_pause": breath_pause, "tilt": motif.tilt * (frac - 0.5),
			"gap": motif.gap, "acts": acts, "ring_st": ring_st,
			"gravity": p.gravity,
		}


# Two-pole resonator (the Klatt building block): y = A x + B y1 + C y2.
class Reso:
	var b := 0.0
	var c := 0.0
	var a := 1.0
	var y1 := 0.0
	var y2 := 0.0

	func tune(f: float, bw: float) -> void:
		var r := exp(-PI * bw / SR)
		c = -r * r
		b = 2.0 * r * cos(TWO_PI * clampf(f, 50.0, SR * 0.45) / SR)
		a = 1.0 - b - c

	func step(x: float) -> float:
		var y := a * x + b * y1 + c * y2
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
## Pure data; `synth` realizes it through the EMAs.
static func plan(text: String, spec: Spec) -> Array:
	var segs: Array = []
	var field := ProsodyField.new(int(spec.reading[0]))
	var walk := ProsodyWalk.new([spec.reading] + spec.influences)
	var t_cursor := 0.12
	var sentences := Phonemes.parse(text)
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
			var accent_at := Phonemes.stress_vowel(w.phones) if w.stressed else -1
			var last_word: bool = wi == words.size() - 1
			var nsyll := 0
			var est_dur := 0.0
			for p in w.phones:
				est_dur += Phonemes.TABLE.get(p, {}).get("dur", 80.0) * 0.001 / spec.rate
				if _ptype(p) == "vowel":
					nsyll += 1
			var mods := walk.word(w.stressed, nsyll, est_dur,
				float(wi) / maxf(1.0, float(words.size() - 1)), w.pause_after != "none")
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
				var is_vowel := _ptype(p) == "vowel"
				if is_vowel:
					vseen += 1
					dur *= 1.0 + 0.25 * acts.stretch   # vowels carry most of the pull
					# declination falls across the sentence; the field wanders on top
					semis -= spec.f0_decl * (float(vseen) / maxf(1.0, float(vowels_total)))
					semis += mods.tilt      # the sentence motif's slope
					semis += mods.ring_st   # the resonance ring from recent firings
					if pi == accent_at:
						dur *= 1.25 * (1.0 + 0.4 * mods.emph)
						amp = 1.15 * (1.0 + 0.25 * mods.emph)
						semis += spec.f0_accent + 2.2 * mods.emph
					elif not w.stressed:
						dur *= 0.8
						amp = 0.85
						reduce = 0.35   # vowel reduction: drift toward schwa
					if pi > 0 and _VOICELESS.has(w.phones[pi - 1]):
						semis += 0.8    # microprosody after voiceless consonants
					semis += field.sample("f0", t_cursor)
					# pitch attractors: the melody is continuously pulled toward
					# the voice's anchor shelf (gravity), and a pitch activation
					# JUMPS most of the way there - musical quantization
					var pull: float = clampf(0.3 * mods.gravity + 0.55 * acts.pitch, 0.0, 0.85)
					if pull > 0.0:
						semis = lerpf(semis, walk.nearest_anchor(semis), pull)
					# a swell activation is a crescendo across the word
					if acts.swell > 0.0:
						amp *= 1.0 + acts.swell * lerpf(-0.12, 0.3,
							float(pi) / maxf(1.0, float((w.phones as Array).size() - 1)))
				if last_word and pi >= (w.phones as Array).size() - 2:
					dur *= spec.final_lengthen
				dur *= 1.0 + 0.12 * field.sample("rate", t_cursor)
				amp *= 1.0 + 0.15 * field.sample("amp", t_cursor)
				t_cursor += dur
				wsegs.append({
					"p": p, "dur": dur, "word": wi, "sentence": si,
					"text": w.text, "word_start": pi == 0,
					"word_end": pi == (w.phones as Array).size() - 1,
					"semitones": semis, "amp": amp, "reduce": reduce,
					"echo": clampf(0.55 * acts.echo, 0.0, 0.9),
				})
			# terminal contours land on the sentence's last word's vowels:
			# a question RISES into the end, a statement falls further
			if last_word:
				var vsegs: Array = wsegs.filter(func(s): return _ptype(s.p) == "vowel")
				if vsegs.size() > 0:
					vsegs[-1].semitones += 5.0 if question else -2.5
				if vsegs.size() > 1:
					vsegs[-2].semitones += 2.0 if question else -1.2
			# a comma word carries a small continuation rise (the "not done yet" cue)
			elif w.pause_after == "comma":
				for k in range(wsegs.size() - 1, -1, -1):
					if _ptype(wsegs[k].p) == "vowel":
						wsegs[k].semitones += 1.8
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
				# a whisper of articulation space between running words
				var gap: float = 0.015 * mods.gap
				t_cursor += gap
				segs.append(_sil(gap, si))
	# a breath of silence at both ends so playback and analysis never clip a boundary
	segs.push_front(_sil(0.12, -1))
	segs.append(_sil(0.12, -1))
	return segs


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
		match String(entry.get("type", "sil")):
			"sil":
				_run_frames(out, state, spec, noise_rng, seg, entry, 0.0, 0.0, seg.dur)
			"stop":
				# closure (voiced stops keep a faint murmur), then an 8 ms burst
				var murmur := 0.06 if entry.get("voiced", false) else 0.0
				_run_frames(out, state, spec, noise_rng, seg, entry, murmur, 0.0, seg.dur * 0.65)
				state.noise.tune(entry.burst_f * spec.formant_scale, entry.burst_bw)
				_run_frames(out, state, spec, noise_rng, seg, entry, murmur, 0.5, maxf(0.008, seg.dur * 0.12))
			"fric":
				state.noise.tune(entry.noise_f * spec.formant_scale, entry.noise_bw)
				var v := 0.5 if entry.get("voiced", false) else 0.0
				_run_frames(out, state, spec, noise_rng, seg, entry, v, entry.namp, seg.dur)
			"asp":
				# aspiration through wherever the formants are heading (the EMA is
				# already moving toward the next phone's targets)
				_retarget(state, _next_formants(segs, i, spec), spec)
				_run_frames(out, state, spec, noise_rng, seg, entry, 0.0, 0.22, seg.dur, true)
			_:
				# vowel / glide / nasal: periodic source through the cascade
				var amp: float = seg.get("amp", 1.0)
				if entry.type == "glide":
					amp *= 0.75
				elif entry.type == "nasal":
					amp *= 0.45
				_retarget(state, _seg_formants(entry, 0.0, spec, seg.get("reduce", 0.0)), spec)
				_run_frames(out, state, spec, noise_rng, seg, entry, amp, 0.0, seg.dur)
		var t1 := float(out.size()) / SR
		_record_timing(state, seg, t0, t1)
	state.pcm = out
	var done: bool = to_seg >= segs.size()
	return {
		"pcm": out, "sr": SR, "dur": float(out.size()) / SR,
		"words": state.words, "phones": state.phones, "done": done, "state": state,
	}


## Fresh synthesis state (resonators, EMA formants, f0 realization, timing maps).
static func synth_state(spec: Spec) -> Dictionary:
	var rng := RandomNumberGenerator.new()
	rng.seed = spec.seed_value
	return {
		"rng": rng, "pcm": PackedFloat32Array(),
		"r1": Reso.new(), "r2": Reso.new(), "r3": Reso.new(), "noise": Reso.new(),
		"fsm": [500.0 * spec.formant_scale, 1400.0 * spec.formant_scale, 2400.0 * spec.formant_scale],
		"ftg": [500.0 * spec.formant_scale, 1400.0 * spec.formant_scale, 2400.0 * spec.formant_scale],
		"f0sm": spec.f0_base * 1.12, "phase": 0.0, "ampsm": 0.0,
		"pulse": _pulse_table(0.4, 0.16), "pulse_lax": _pulse_table(0.58, 0.34),
		"tension": 0.5, "nlp": 0.0, "tilt_y": 0.0,
		"ebuf": _zeroes(int(ECHO_DELAY * SR)), "eidx": 0,
		"words": [], "phones": [], "wopen": {},
	}


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
static func _pulse_table(open_len: float, close_len: float) -> PackedFloat32Array:
	var t := PackedFloat32Array()
	t.resize(64)
	for i in 64:
		var u := float(i) / 64.0
		var v := 0.0
		if u < open_len:
			v = 0.5 * (1.0 - cos(PI * u / open_len))
		elif u < open_len + close_len:
			v = cos(PI * (u - open_len) / (2.0 * close_len))
		t[i] = v
	return t


# Schwa - the neutral vowel unstressed vowels reduce toward.
const _SCHWA := [640.0, 1190.0, 2390.0]


static func _seg_formants(entry: Dictionary, u: float, spec: Spec, reduce := 0.0) -> Array:
	var f: Array = entry.get("f", [500.0, 1400.0, 2400.0])
	var f2: Array = entry.get("f2", f)
	var out: Array = []
	for k in 3:
		var v := lerpf(f[k], f2[k], u)
		if reduce > 0.0:
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


## The inner loop: run `dur` seconds in FRAME-sized chunks. Per frame: EMA the
## formants toward their targets (coarticulation), EMA f0 toward the segment's
## semitone offset (Fujisaki realization), retune the cascade, then fill samples
## from pulse + noise sources. `vamp` scales the periodic source, `namp` the
## noise path; `asp_cascade` sends noise through the formant cascade (for HH).
static func _run_frames(out: PackedFloat32Array, state: Dictionary, spec: Spec,
		rng: RandomNumberGenerator, seg: Dictionary, entry: Dictionary,
		vamp: float, namp: float, dur: float, asp_cascade := false) -> void:
	var n := int(round(dur * SR))
	var r1: Reso = state.r1
	var r2: Reso = state.r2
	var r3: Reso = state.r3
	var nr: Reso = state.noise
	var pulse: PackedFloat32Array = state.pulse
	var is_diph: bool = entry.has("f2")
	var f0_target: float = spec.f0_base * pow(2.0, seg.semitones / 12.0) * 1.06
	var done := 0
	var prev := 0.0
	var period_gain := 1.0
	while done < n:
		var m := mini(FRAME, n - done)
		var u := float(done) / maxf(1.0, float(n))
		if is_diph:
			state.ftg = _seg_formants(entry, u, spec, seg.get("reduce", 0.0))
		# EMAs: formants ~18 ms, f0 ~35 ms, amplitude ~8 ms time constants
		var fa := 1.0 - exp(-float(m) / (SR * 0.018))
		var pa := 1.0 - exp(-float(m) / (SR * 0.035))
		var aa := 1.0 - exp(-float(m) / (SR * 0.008))
		for k in 3:
			state.fsm[k] = lerpf(state.fsm[k], state.ftg[k], fa)
		state.f0sm = lerpf(state.f0sm, f0_target, pa)
		state.ampsm = lerpf(state.ampsm, vamp, aa)
		r1.tune(state.fsm[0], 60.0 + state.fsm[0] * 0.06)
		r2.tune(state.fsm[1], 90.0 + state.fsm[1] * 0.05)
		r3.tune(state.fsm[2], 150.0)
		var inc: float = state.f0sm * 64.0 / SR
		var amp: float = state.ampsm
		var pulse_lax: PackedFloat32Array = state.pulse_lax
		var tension: float = state.tension
		var nlp: float = state.nlp
		var tilt_y: float = state.tilt_y
		# the air line as a one-pole coefficient: what leaks above it is static
		var air_k: float = 1.0 - exp(-TAU * spec.air_cut / SR)
		# vocal effort opens the spectral tilt: emphatic frames are brighter,
		# settled ones darker - the walk's dynamics now reach the TIMBRE
		var tilt_k: float = clampf(0.3 + 0.5 * amp, 0.2, 0.95)
		var esend: float = seg.get("echo", 0.0)
		var ebuf: PackedFloat32Array = state.ebuf
		var eidx: int = state.eidx
		var esize := ebuf.size()
		var phase: float = state.phase
		for _s in m:
			phase += inc
			if phase >= 64.0:
				phase -= 64.0
				# per-period organic variation: jitter the pitch, shimmer the
				# gain, and wander the glottal TENSION - no two cycles alike
				inc = state.f0sm * (1.0 + rng.randfn(0.0, spec.jitter)) * 64.0 / SR
				period_gain = 1.0 + rng.randfn(0.0, spec.shimmer)
				tension = clampf(lerpf(tension, rng.randf(), 0.3), 0.0, 1.0)
			var pidx := int(phase)
			var src := lerpf(pulse_lax[pidx], pulse[pidx], tension) * amp * period_gain
			var hiss := rng.randf() * 2.0 - 1.0
			# aspiration is pitch-synchronous: air leaks during the OPEN phase
			# of the cycle, not as a steady decoupled hiss floor
			src += hiss * spec.breath * amp * (1.0 if phase < 26.0 else 0.3)
			# stations with static: above the air line the harmonic voice gives
			# way to noise - highpassed hiss joins the excitation itself
			nlp += air_k * (hiss - nlp)
			src += (hiss - nlp) * spec.air_gain * amp
			# effort tilt (one-pole lowpass, coefficient driven by amp)
			tilt_y += tilt_k * (src - tilt_y)
			src = tilt_y
			var y: float
			if asp_cascade:
				y = r3.step(r2.step(r1.step(hiss * namp * 0.5)))
			else:
				y = r3.step(r2.step(r1.step(src)))
				if namp > 0.0:
					y += nr.step(hiss) * namp
			# radiation: first difference brightens the spectrum like lips do
			var rad := y - prev * 0.96
			prev = y
			# the echo bus: silent until a word is sent into it, then it rings
			var e := ebuf[eidx]
			ebuf[eidx] = rad * esend + e * ECHO_FB
			eidx += 1
			if eidx >= esize:
				eidx = 0
			out.append((rad + e * 0.8) * OUT_GAIN)
		state.phase = phase
		state.tension = tension
		state.nlp = nlp
		state.tilt_y = tilt_y
		state.ebuf = ebuf            # packed arrays are CoW: persist the written copy
		state.eidx = eidx
		done += m


static func _record_timing(state: Dictionary, seg: Dictionary, t0: float, t1: float) -> void:
	if seg.word < 0:
		return
	state.phones.append({"p": seg.p, "t0": t0, "t1": t1, "word": seg.word, "sentence": seg.sentence})
	var key := "%d:%d" % [seg.sentence, seg.word]
	if seg.word_start and not state.wopen.has(key):
		state.wopen[key] = state.words.size()
		state.words.append({"text": seg.text, "t0": t0, "t1": t1, "sentence": seg.sentence})
	if state.wopen.has(key):
		state.words[state.wopen[key]].t1 = t1


## Write PCM16 mono WAV. Returns the globalized path (playable by Spectrum,
## ffmpeg, anything).
static func write_wav(path: String, pcm: PackedFloat32Array) -> String:
	var bytes := PackedByteArray()
	bytes.resize(pcm.size() * 2)
	for i in pcm.size():
		var v := int(clampf(pcm[i], -1.0, 1.0) * 32767.0)
		bytes.encode_s16(i * 2, v)
	var f := FileAccess.open(path, FileAccess.WRITE)
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
	return ProjectSettings.globalize_path(path)
