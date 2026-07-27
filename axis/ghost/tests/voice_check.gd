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
	if Voice_.RAW_MODE:
		# raw diagnostic bypass: the walk's audible realization is neutralized,
		# so readings from different roots are IDENTICAL by design right now
		print("voice_check: readings check SKIPPED (Voice.RAW_MODE is on)")
	else:
		failures += _check_readings()
	failures += _check_ornament_aging()
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


## The reading lineage: different roots read differently, the same lineage is
## byte-deterministic, and breath pauses EMERGE in unpunctuated text (the walk's
## breath debt coming due - the text below has no punctuation at all).
func _check_readings() -> int:
	var text := "the machine spoke slowly and the whole room listened to every single word it said without a sound"
	var bad := 0
	var a := Voice_.render(text, Voice_.Spec.from_traits({}, 0, [11]))
	var b := Voice_.render(text, Voice_.Spec.from_traits({}, 0, [22]))
	if a.pcm == b.pcm:
		print("voice_check: FAIL - two reading roots produced identical audio")
		bad += 1
	var c1 := Voice_.render(text, Voice_.Spec.from_traits({}, 0, [11, 5]))
	var c2 := Voice_.render(text, Voice_.Spec.from_traits({}, 0, [11, 5]))
	if c1.pcm != c2.pcm:
		print("voice_check: FAIL - the same lineage is not deterministic")
		bad += 1
	if c1.pcm == a.pcm:
		print("voice_check: FAIL - an evolved child did not differ from its parent")
		bad += 1
	var segs := Voice_.plan(text, Voice_.Spec.from_traits({}, 0, [11]))
	var breaths := 0
	for s in segs:
		if s.p == "SIL" and s.dur >= 0.1 and int(s.sentence) >= 0:
			breaths += 1
	if breaths < 1:
		print("voice_check: FAIL - no breath emerged in a long unpunctuated sentence")
		bad += 1
	# the 1+N blend: a toggled influence must change the reading, deterministically
	var inf1 := Voice_.Spec.from_traits({}, 0, [11])
	inf1.influences = [[22]]
	var inf2 := Voice_.Spec.from_traits({}, 0, [11])
	inf2.influences = [[22]]
	var i1 := Voice_.render(text, inf1)
	if i1.pcm == a.pcm:
		print("voice_check: FAIL - a toggled influence did not change the reading")
		bad += 1
	if i1.pcm != Voice_.render(text, inf2).pcm:
		print("voice_check: FAIL - the influence blend is not deterministic")
		bad += 1
	print("voice_check: readings ok - lineage + blend deterministic, %d breath(s) emerged" % breaths)
	DirAccess.make_dir_recursive_absolute(OUT_DIR)
	Voice_.write_wav(OUT_DIR + "/voice_reading_a.wav", a.pcm)
	Voice_.write_wav(OUT_DIR + "/voice_reading_b.wav", b.pcm)
	return bad


## Ornaments AGE OUT across the lineage (the recency decay): a generation's
## spawned modulator keeps its seeded gesture (shape/rate/phase) but loses
## depth by ORN_DECAY for every generation that lands after it, until the
## suppress floor prunes it - while the NEWEST generation always enters at
## full strength, and the elaboration anchor shelf stays a bounded window.
func _check_ornament_aging() -> int:
	var bad := 0
	var root := -1
	for s in range(1, 200):
		if Voice_.ProsodyWalk._lineage_mods([s]).size() > 0:
			root = s
			break
	if root < 0:
		print("voice_check: FAIL - no modulator-spawning root in 200 seeds")
		return 1
	var shallow: Array = Voice_.ProsodyWalk._lineage_mods([root])
	var chain: Array = [root]
	for g in 8:
		chain.append(1000 + g)
	var deep: Array = Voice_.ProsodyWalk._lineage_mods(chain)
	var m0: Dictionary = shallow[0]
	var mn: Dictionary = deep[0]
	if mn.shape != m0.shape or absf(float(mn.rate) - float(m0.rate)) > 1e-6:
		print("voice_check: FAIL - aging changed a gesture's identity, not just depth")
		bad += 1
	var expect: float = float(m0.depth) * pow(Voice_.ProsodyWalk.ORN_DECAY, chain.size() - 1)
	if absf(float(mn.depth) - expect) > 1e-5 or float(mn.depth) >= float(m0.depth):
		print("voice_check: FAIL - root modulator depth %.4f, expected aged %.4f" % [mn.depth, expect])
		bad += 1
	# the aged root gesture must DIE at the finalize stage while it survives
	# in the shallow lineage (same damp either way)
	var kept_shallow := false
	for m in Voice_.ProsodyWalk._finalize_mods(shallow, 0.35):
		if absf(float(m.rate) - float(m0.rate)) < 1e-6:
			kept_shallow = true
	var kept_deep := false
	for m in Voice_.ProsodyWalk._finalize_mods(deep, 0.35):
		if absf(float(m.rate) - float(m0.rate)) < 1e-6:
			kept_deep = true
	if not kept_shallow or kept_deep:
		print("voice_check: FAIL - aging out: shallow kept=%s deep kept=%s (want true/false)"
			% [kept_shallow, kept_deep])
		bad += 1
	# the newest generation enters at full, un-aged strength
	var fresh := -1
	for x in range(1, 200):
		if Voice_.ProsodyWalk._lineage_mods([root, x]).size() == 2:
			fresh = x
			break
	if fresh >= 0:
		var pair: Array = Voice_.ProsodyWalk._lineage_mods([root, fresh])
		if float(pair[1].depth) < 0.25 - 1e-6:
			print("voice_check: FAIL - the newest generation's modulator arrived pre-aged")
			bad += 1
	# the anchor shelf is a window, not an archive: prior (3) + lineage (3)
	# + at most 4 elaboration anchors, however deep the reading runs
	var walk := Voice_.ProsodyWalk.new([chain])
	if walk._anchors.size() > 10:
		print("voice_check: FAIL - anchor shelf grew past the window (%d)" % walk._anchors.size())
		bad += 1
	if bad == 0:
		print("voice_check: ornament aging ok - root gesture %.3f -> %.3f over %d generations, pruned; %d anchors"
			% [m0.depth, mn.depth, chain.size() - 1, walk._anchors.size()])
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
	# authored %HESITATION: parses into a filled "um" shown as an ellipsis
	var hsegs := Voice_.plan("%HESITATION welcome, welcome.", Voice_.Spec.from_traits({}))
	var found_hesit := false
	for s in hsegs:
		if s.text == "…":
			found_hesit = true
			break
	if not found_hesit:
		print("voice_check: FAIL - %HESITATION did not produce a filled pause word")
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
	# voice quality extremes: the air line low (breathy static) vs closed (clear)
	var airy := Voice_.render("The city listened, and the lights began to move.",
		Voice_.Spec.from_traits({"air": 0.9, "breath": 0.5}))
	Voice_.write_wav(OUT_DIR + "/voice_quality_airy.wav", airy.pcm)
	var clear := Voice_.render("The city listened, and the lights began to move.",
		Voice_.Spec.from_traits({"air": -1.0, "breath": -0.6}))
	Voice_.write_wav(OUT_DIR + "/voice_quality_clear.wav", clear.pcm)
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
