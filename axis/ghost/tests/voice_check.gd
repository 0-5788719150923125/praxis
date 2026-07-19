extends SceneTree

## Headless check for the Voice synthesizer (rung 0 of next/voice.md, automated).
##
## Run: godot --headless --path axis/ghost --script tests/voice_check.gd
##
## 1. Formant check (the objective half of "does the vowel read"): synthesize
##    sustained /AA/ and /IY/ and measure spectral energy near their F1/F2
##    targets with Goertzel probes. /AA/ must beat /IY/ around 730 Hz and /IY/
##    must beat /AA/ around 2290 Hz, by a clear factor - if the filter layer
##    works, this cannot fail; if it fails, nothing else matters.
## 2. Sentence check: render a sentence, assert monotonic word timings that
##    cover the audio, and write the WAV to /tmp/ghost_scratch/ for the human
##    half of the check (listening).

const Voice_ := preload("res://scripts/voice.gd")
const Phonemes_ := preload("res://scripts/phonemes.gd")

const OUT_DIR := "/tmp/ghost_scratch"


func _init() -> void:
	var failures := 0
	failures += _check_vowels()
	failures += _check_traits()
	failures += _check_sentence()
	if failures == 0:
		print("voice_check: ALL OK")
	else:
		print("voice_check: %d FAILURE(S)" % failures)
	quit(failures)


# The gate runs on the CURATED DEFAULT speaker (the zero trait vector) - the
# one voice whose quality is a constant, not a roll. Sampled voices vary and
# are auditioned by ear, not asserted here.
func _spec() -> Voice_.Spec:
	return Voice_.Spec.from_traits({})


func _hold(phone: String, spec: Voice_.Spec) -> PackedFloat32Array:
	var segs := [
		{"p": "SIL", "dur": 0.05, "word": -1, "sentence": 0, "text": "",
			"word_start": false, "word_end": false, "semitones": 0.0},
		{"p": phone, "dur": 0.5, "word": 0, "sentence": 0, "text": phone,
			"word_start": true, "word_end": true, "semitones": 0.0},
	]
	return Voice_.synth(segs, spec).pcm


## Goertzel magnitude around f (probe f-50, f, f+50 and take the max), over the
## steady middle of the vowel.
func _energy_near(pcm: PackedFloat32Array, f: float) -> float:
	var n := pcm.size()
	var from := int(n * 0.35)
	var to := int(n * 0.85)
	var best := 0.0
	for probe: float in [f - 50.0, f, f + 50.0]:
		var w := TAU * probe / Voice_.SR
		var coeff := 2.0 * cos(w)
		var s1 := 0.0
		var s2 := 0.0
		for i in range(from, to):
			var s := pcm[i] + coeff * s1 - s2
			s2 = s1
			s1 = s
		best = maxf(best, s1 * s1 + s2 * s2 - coeff * s1 * s2)
	return best


## Within-signal spectral balance: for each vowel, energy near /AA/'s F1 region
## over energy near /IY/'s F2 region - then compare the two vowels' balances.
## Robust to harmonic placement (numerator and denominator share a signal);
## the separation for real formants is orders of magnitude.
func _check_vowels() -> int:
	var spec := _spec()
	var aa := _hold("AA", spec)
	var iy := _hold("IY", spec)
	var scale := spec.formant_scale
	var bal_aa := _energy_near(aa, 730.0 * scale) / maxf(_energy_near(aa, 2290.0 * scale), 0.000001)
	var bal_iy := _energy_near(iy, 730.0 * scale) / maxf(_energy_near(iy, 2290.0 * scale), 0.000001)
	var sep := bal_aa / maxf(bal_iy, 0.000001)
	print("voice_check: vowel balance separation /AA/ vs /IY/: x%.0f" % sep)
	if sep < 100.0:
		print("voice_check: FAIL - /AA/ and /IY/ are not spectrally distinct")
		return 1
	return 0


## The trait vector IS the voice: the same vector must reproduce the identical
## take byte for byte, and a moved trait must change it.
func _check_traits() -> int:
	var text := "the voice is the vector"
	var a := Voice_.render(text, Voice_.Spec.from_traits({"pitch": 0.3, "drawl": -0.2}))
	var b := Voice_.render(text, Voice_.Spec.from_traits({"pitch": 0.3, "drawl": -0.2}))
	var c := Voice_.render(text, Voice_.Spec.from_traits({"pitch": -0.3, "drawl": -0.2}))
	var bad := 0
	if a.pcm != b.pcm:
		print("voice_check: FAIL - identical trait vectors produced different audio")
		bad += 1
	if a.pcm == c.pcm:
		print("voice_check: FAIL - a moved trait did not change the audio")
		bad += 1
	print("voice_check: trait determinism ok (%d samples)" % a.pcm.size())
	return bad


func _check_sentence() -> int:
	var spec := _spec()
	var text := "Once upon a time, a small voice spoke from the machine. It was not human, but it was alive."
	var result := Voice_.render(text, spec)
	var words: Array = result.words
	var bad := 0
	if words.size() < 15:
		print("voice_check: FAIL - expected ~19 words, got %d" % words.size())
		bad += 1
	var prev_end := 0.0
	for w in words:
		if w.t0 < prev_end - 0.001 or w.t1 <= w.t0:
			print("voice_check: FAIL - non-monotonic timing at '%s'" % w.text)
			bad += 1
			break
		prev_end = w.t1
	if words.size() > 0 and absf(words[words.size() - 1].t1 - result.dur) > 1.0:
		print("voice_check: FAIL - last word ends %.2fs but audio is %.2fs" % [words[words.size() - 1].t1, result.dur])
		bad += 1
	DirAccess.make_dir_recursive_absolute(OUT_DIR)
	var path := Voice_.write_wav(OUT_DIR + "/voice_sentence.wav", result.pcm)
	# the sidecar timing map, exactly as the synth editor writes it - so a
	# session booted on this WAV attaches the karaoke overlay
	var side := FileAccess.open(OUT_DIR + "/voice_sentence.json", FileAccess.WRITE)
	side.store_string(JSON.stringify({"words": words}))
	side.close()
	print("voice_check: sentence %.2fs, %d words -> %s" % [result.dur, words.size(), path])
	# a second voice from a different seed must differ (the population axis)
	var rng := RandomNumberGenerator.new()
	rng.seed = 99
	var other := Voice_.render(text, Voice_.Spec.sample(rng))
	Voice_.write_wav(OUT_DIR + "/voice_sentence_alt.wav", other.pcm)
	if absf(other.dur - result.dur) < 0.001:
		print("voice_check: FAIL - two sampled voices produced identical durations")
		bad += 1
	# audition set: the question contour and a handful of rolled speakers
	# (bimodal register - expect clearly different PEOPLE, not takes)
	var q := Voice_.render("Is it alive? It is alive.", spec)
	Voice_.write_wav(OUT_DIR + "/voice_question.wav", q.pcm)
	for s in [3, 12, 31, 47]:
		var vr := RandomNumberGenerator.new()
		vr.seed = s
		var vspec := Voice_.Spec.sample(vr)
		var take := Voice_.render("The city listened, and the lights began to move.", vspec)
		Voice_.write_wav(OUT_DIR + "/voice_roll_%d.wav" % s, take.pcm)
		print("voice_check: roll %d  pitch=%.2f tract=%.2f -> f0 %.0f Hz" % [
			s, vspec.traits.pitch, vspec.traits.tract, vspec.f0_base])
	return bad
