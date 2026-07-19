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


## The speaker's trait axes, each in [-1, 1]. THE TRAIT VECTOR IS THE VOICE:
## the zero vector is the hand-curated default speaker (its concrete centres
## live in Spec.from_traits - tune them there), a seed only *initializes* the
## vector, and any UI modulation edits it directly - so a speaker is replicated
## by replaying the vector, never by replaying the gesture that found it.
const TRAIT_KEYS := ["pitch", "lilt", "tract", "pace", "breath", "grit", "drawl"]

## One voice: a trait vector realized into concrete synthesis parameters.
class Spec:
	var seed_value := 0
	var traits := {}                  # trait key -> [-1, 1]; {} = the curated default
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

	## Realize a trait vector. The constants here ARE the curated default
	## speaker (all traits 0); each trait bends one perceptual axis around it,
	## exponentially where perception is log-shaped (pitch, tempo).
	static func from_traits(t: Dictionary, seed_value_ := 0) -> Spec:
		var s := Spec.new()
		s.seed_value = seed_value_
		s.traits = t.duplicate()
		var pitch := _tv(t, "pitch")
		var lilt := _tv(t, "lilt")
		var tract := _tv(t, "tract")
		var pace := _tv(t, "pace")
		var breath := _tv(t, "breath")
		var grit := _tv(t, "grit")
		var drawl := _tv(t, "drawl")
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
	var field := ProsodyField.new(spec.seed_value)
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
		for wi in words.size():
			var w: Dictionary = words[wi]
			var accent_at := Phonemes.stress_vowel(w.phones) if w.stressed else -1
			var last_word: bool = wi == words.size() - 1
			var wsegs: Array = []
			for pi in (w.phones as Array).size():
				var p: String = w.phones[pi]
				var entry: Dictionary = Phonemes.TABLE.get(p, {})
				if entry.is_empty():
					continue
				var dur: float = entry.get("dur", 80.0) * 0.001 / spec.rate
				var amp := 1.0
				var reduce := 0.0
				var semis := 0.0
				var is_vowel := _ptype(p) == "vowel"
				if is_vowel:
					vseen += 1
					# declination falls across the sentence; the field wanders on top
					semis -= spec.f0_decl * (float(vseen) / maxf(1.0, float(vowels_total)))
					if pi == accent_at:
						dur *= 1.25
						amp = 1.15
						semis += spec.f0_accent
					elif not w.stressed:
						dur *= 0.8
						amp = 0.85
						reduce = 0.35   # vowel reduction: drift toward schwa
					if pi > 0 and _VOICELESS.has(w.phones[pi - 1]):
						semis += 0.8    # microprosody after voiceless consonants
					semis += field.sample("f0", t_cursor)
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
			elif not last_word:
				# a whisper of articulation space between running words
				t_cursor += 0.015
				segs.append(_sil(0.015, si))
	# a breath of silence at both ends so playback and analysis never clip a boundary
	segs.push_front(_sil(0.12, -1))
	segs.append(_sil(0.12, -1))
	return segs


static func _sil(dur: float, si: int) -> Dictionary:
	return {"p": "SIL", "dur": dur, "word": -1, "sentence": si, "text": "",
		"word_start": false, "word_end": false, "semitones": 0.0, "amp": 1.0, "reduce": 0.0}


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
		"pulse": _pulse_table(), "words": [], "phones": [], "wopen": {},
	}


# ---- internals -------------------------------------------------------------


# Rosenberg glottal pulse, one period in 64 samples: rising open phase, sharp
# closing phase, closed remainder. Differentiated at synth time by the radiation
# first-difference, which is folded into the output stage.
static func _pulse_table() -> PackedFloat32Array:
	var t := PackedFloat32Array()
	t.resize(64)
	for i in 64:
		var u := float(i) / 64.0
		var v := 0.0
		if u < 0.4:
			v = 0.5 * (1.0 - cos(PI * u / 0.4))
		elif u < 0.56:
			v = cos(PI * (u - 0.4) / 0.32)
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
		for _s in m:
			var ph: float = state.phase
			ph += inc
			if ph >= 64.0:
				ph -= 64.0
				# per-period organic variation: jitter the pitch, shimmer the gain
				inc = state.f0sm * (1.0 + rng.randfn(0.0, spec.jitter)) * 64.0 / SR
				period_gain = 1.0 + rng.randfn(0.0, spec.shimmer)
			state.phase = ph
			var src := pulse[int(ph)] * amp * period_gain
			var hiss := rng.randf() * 2.0 - 1.0
			src += hiss * spec.breath * amp
			var y: float
			if asp_cascade:
				y = r3.step(r2.step(r1.step(hiss * namp * 0.5)))
			else:
				y = r3.step(r2.step(r1.step(src)))
				if namp > 0.0:
					y += nr.step(hiss) * namp
			# radiation: first difference brightens the spectrum like lips do
			out.append((y - prev * 0.96) * OUT_GAIN)
			prev = y
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
