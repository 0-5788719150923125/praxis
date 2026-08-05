extends SceneTree

## The bare-minimum synthesis repro: text in, WAVs out. No Ghost boot, no
## autoloads, no scenes, no stream, no push path - just [Voice.render] and
## [Voice.write_wav].
##
## LADDER v4 (2026-08-04) - the parallel branch, and a calibrator that can
## actually see the defect it exists to catch.
##
## v3's metric was BROADBAND fricative/vowel RMS against a target ratio. That
## number is orthogonal to the thing that was wrong: run against the old
## through-cascade synth it converged, reported a healthy 0.35, and declared
## success on fricatives that had 0.02-0.06% of their energy above 4 kHz, whose
## S/SH/F/TH spectra correlated at 0.99-1.00, and which carried -0.2 dB of
## 6-10 kHz contrast against the neighbouring vowel. Two whole RCA rounds
## passed on a metric that could not fail. A level meter cannot detect a
## timbre collapse.
##
## So v4 measures the two things that define an audible fricative:
##   CONTRAST  - energy of each fricative against the adjacent vowel, in dB,
##               inside the band its own poles declare. This makes /f/ and /th/
##               audible despite being quiet: they live where vowels are
##               empty. Natural /s/ runs +15 to +25 dB, /f/ and /th/ +8 to +15.
##   DISTINCTNESS - the pairwise correlation of the S / SH / F / TH log
##               spectra. Four phonemes that correlate above ~0.9 are one
##               sound wearing four labels, whatever their levels say.
##
## Run:  godot --headless --path axis/ghost --script tests/pure_say.gd
##       (optionally: -- "your own text here")
## Listen:  mpv /tmp/ghost_scratch/pure_say_<name>.wav

const Voice_ := preload("res://scripts/voice.gd")
const Phonemes := preload("res://scripts/phonemes.gd")

const OUT_DIR := "/tmp/ghost_scratch"
const VOICELESS_FRIC := ["S", "SH", "F", "TH"]
const VOWELS := ["IY", "IH", "EH", "AE", "AA", "AO", "UH", "UW", "AH", "ER",
	"AY", "EY", "OY", "AW", "OW"]

# The contrast band is PER PHONEME, read from its parallel poles (_fric_band).
# Probed on a 250 Hz grid.
const BAND_STEP := 250.0
# The shape grid for the distinctness check: log-spaced, ABOVE the voice.
const SHAPE_BINS := 20
const SHAPE_LO := 1500.0
const SHAPE_HI := 12000.0
# What a healthy result looks like. These are ACCEPTANCE numbers from the
# phonetics literature, not tuning knobs - they do not move to make a run pass.
#
# THREE gates, because no single number can diagnose a fricative:
#   CONTRAST is only diagnostic where the phoneme's band lies ABOVE the vowel's
#     formant range. /s/ (5.0-7.8 kHz), /f/ (3.4-10.1) and /th/ (4.4-10.7) sit
#     above F3, so a buried one shows up immediately. /sh/ (2.2-3.9 kHz) sits
#     ON F3/F4 by physical fact, so its contrast is genuinely low and a
#     threshold there would measure the vowel, not the consonant.
#   LEVEL catches the failure that recurred twice in this file's history:
#     frication seated far louder than the voice. It applies to everything.
#   DISTINCTNESS catches the collapse that a level meter cannot see.
# /sh/ is therefore gated on level and distinctness, where it scores strongly
# (-10.7 dB, and -0.41 correlation against /s/ - maximally distinct).
const CONTRAST_FLOOR_HZ := 3000.0     # gate contrast only above this band edge
const WANT_CONTRAST_DB := 8.0         # minimum in-band contrast vs the vowel
const WANT_LEVEL_DB := [-18.0, -4.0]  # broadband fricative vs vowel, natural range
const WANT_MAX_CORR := 0.90           # above this, the fricatives have collapsed


func _init() -> void:
	var text := "The thin fox thought of five soft things. She shall wish for shelter. Once upon a time, the river spoke softly; stones and static kept the signal safe."
	var args := OS.get_cmdline_user_args()
	if args.size() > 0:
		text = " ".join(args)
	var spec := Voice_.Spec.from_traits({}, 1, [1])
	var kept: float = Voice_.FRIC_LEVEL
	DirAccess.make_dir_recursive_absolute(OUT_DIR)
	var ladder := [
		{"name": "off", "lvl": 0.0},
		{"name": "soft", "lvl": kept * 0.5},
		{"name": "seated", "lvl": kept},
		{"name": "firm", "lvl": kept * 2.0},
	]
	var failures := 0
	for rung in ladder:
		Voice_.FRIC_LEVEL = float(rung.lvl)
		var res: Dictionary = Voice_.render(text, spec)
		var pcm: PackedFloat32Array = res.pcm
		var peak := 0.0
		for v in pcm:
			peak = maxf(peak, absf(v))
		var rms := _rms(pcm, 0, pcm.size())
		print("\npure_say: --- %s (FRIC_LEVEL %.3f)  peak %.3f  RMS %.1f dBFS" % [
			rung.name, rung.lvl, peak, 20.0 * log(maxf(rms, 1e-9)) / log(10.0)])
		var rep := _report_contrast(pcm, res.phones)
		var corr := _report_distinctness(pcm, res.phones)
		# only the shipped seat is a gate; the other rungs are for the ear
		if rung.name == "seated":
			for row in rep:
				if float(row.band_lo) >= CONTRAST_FLOOR_HZ and float(row.contrast) < WANT_CONTRAST_DB:
					print("  FAIL: /%s/ in-band contrast %.1f dB < %.1f dB - buried" % [
						String(row.p).to_lower(), row.contrast, WANT_CONTRAST_DB])
					failures += 1
				if float(row.level) < WANT_LEVEL_DB[0] or float(row.level) > WANT_LEVEL_DB[1]:
					print("  FAIL: /%s/ broadband level %+.1f dB outside %+.0f..%+.0f dB" % [
						String(row.p).to_lower(), row.level, WANT_LEVEL_DB[0], WANT_LEVEL_DB[1]])
					failures += 1
			if corr > WANT_MAX_CORR:
				print("  FAIL: fricative spectra correlate at %.3f > %.3f - they have collapsed" % [corr, WANT_MAX_CORR])
				failures += 1
		var out := pcm.duplicate()
		var scale := 0.7 / maxf(peak, 0.001)
		for i in out.size():
			out[i] *= scale
		var path := Voice_.write_wav(OUT_DIR + "/pure_say_%s.wav" % rung.name, out)
		print("  -> %s" % path)
	Voice_.FRIC_LEVEL = kept
	print("\npure_say: %s" % ("ALL OK" if failures == 0 else "%d FAILURE(S)" % failures))
	quit(failures)


## Per-fricative in-band energy against the ADJACENT vowel's, in dB. Returns
## the weakest contrast found (the number that decides whether the quietest
## consonant in the language is audible).
func _report_contrast(pcm: PackedFloat32Array, phones: Array) -> Array:
	var rows: Array = []
	for p in VOICELESS_FRIC:
		var vals: Array = []
		var band := _fric_band(p)
		for i in phones.size():
			if String(phones[i].p) != p:
				continue
			var vi := _adjacent_vowel(phones, i)
			if vi < 0:
				continue
			var fe := _band(pcm, phones[i], band)
			var ve := _band(pcm, phones[vi], band)
			if fe <= 0.0 or ve <= 0.0:
				continue
			vals.append(10.0 * log(fe / ve) / log(10.0))
		if vals.is_empty():
			continue
		vals.sort()
		var med: float = vals[int(vals.size() / 2.0)]
		# BROADBAND level alongside the band contrast. Contrast says "can you
		# hear it"; this says "is it too loud". Natural voiceless fricatives
		# sit 8-15 dB BELOW the vowels overall even while towering over them
		# in their own band - both numbers must be right, and watching only one of
		# them is how each previous round went wrong in a different direction.
		var wide: Array = []
		for i in phones.size():
			if String(phones[i].p) != p:
				continue
			var vi := _adjacent_vowel(phones, i)
			if vi < 0:
				continue
			var fr := _rms(pcm, int(float(phones[i].t0) * Voice_.SR), mini(int(float(phones[i].t1) * Voice_.SR), pcm.size()))
			var vr := _rms(pcm, int(float(phones[vi].t0) * Voice_.SR), mini(int(float(phones[vi].t1) * Voice_.SR), pcm.size()))
			if fr > 0.0 and vr > 0.0:
				wide.append(20.0 * log(fr / vr) / log(10.0))
		wide.sort()
		var wmed: float = wide[int(wide.size() / 2.0)] if not wide.is_empty() else 0.0
		var gated: bool = float(band[0]) >= CONTRAST_FLOOR_HZ
		print("  /%s/  %.1f-%.1f kHz vs vowel: %+6.1f dB%s    broadband vs vowel: %+6.1f dB   (n=%d)" % [
			p.to_lower(), band[0] / 1000.0, band[1] / 1000.0, med,
			"" if gated else " (overlaps formants, not gated)", wmed, vals.size()])
		rows.append({"p": p, "contrast": med, "level": wmed, "band_lo": band[0]})
	return rows


## Pairwise correlation of the fricative log spectra. The regression test for
## "one sound wearing four labels" - reports the WORST (most collapsed) pair.
func _report_distinctness(pcm: PackedFloat32Array, phones: Array) -> float:
	var shapes := {}
	for p in VOICELESS_FRIC:
		var acc := PackedFloat32Array()
		acc.resize(SHAPE_BINS)
		var n := 0
		for ph in phones:
			if String(ph.p) != p:
				continue
			var s := _shape(pcm, ph)
			if s.is_empty():
				continue
			for k in SHAPE_BINS:
				acc[k] += s[k]
			n += 1
		if n > 0:
			for k in SHAPE_BINS:
				acc[k] /= float(n)
			shapes[p] = acc
	var worst := -1.0
	var worst_pair := ""
	var line := ""
	var keys: Array = shapes.keys()
	for a in keys.size():
		for b in range(a + 1, keys.size()):
			var pair := "%s/%s" % [keys[a], keys[b]]
			var c := _corr(shapes[keys[a]], shapes[keys[b]])
			line += "  %s %.3f%s" % [pair, c, "*" if _lenient(pair) else ""]
			# /f/ and /th/ are exempt from the collapse gate: they are the one
			# genuinely confusable pair in English frication (Jongman et al.
			# classify them near chance from the noise alone), and their real
			# cue is the F2 locus on the neighbouring vowel - 900 Hz vs 1400 Hz
			# in the table - which a frication-only spectrum cannot see.
			if _lenient(pair):
				continue
			if c > worst:
				worst = c
				worst_pair = pair
	print("  spectral distinctness (* = locus-cued pair, not gated):%s" % line)
	if worst >= 0.0:
		print("    worst gated pair %s correlates %.3f" % [worst_pair, worst])
	return worst


func _lenient(pair: String) -> bool:
	return pair == "F/TH" or pair == "TH/F"


func _adjacent_vowel(phones: Array, i: int) -> int:
	for j in [i - 1, i + 1]:
		if j >= 0 and j < phones.size() and VOWELS.has(String(phones[j].p)):
			return j
	return -1


## Energy in the phoneme's OWN band, derived from its declared parallel poles:
## an octave centred on its highest front-cavity resonance. A fixed 3-8 kHz
## window is wrong for /sh/, whose energy sits at 2.4-3.6 kHz - it scored the
## correctly-built /sh/ at -2.9 dB and would have sent me tuning a phoneme that
## was already right. Reading the band from the table means the metric follows
## the data instead of having to be maintained alongside it.
func _fric_band(p: String) -> Array:
	var par: Array = Phonemes.TABLE.get(p, {}).get("par", [])
	# span the poles that actually CARRY the phoneme (amplitude >= 0.5), each
	# widened by half its bandwidth. An octave centred on the topmost pole was
	# the first attempt and it overshot: it gave /sh/ a 2.2-5.0 kHz window when
	# its poles stop at 3.85 kHz, so a fifth of the window was pure vowel and
	# the ratio read 3 dB low. The band has to be the phoneme's own support.
	var lo := 1e9
	var hi := 0.0
	for t in par:
		if float(t[2]) < 0.5:
			continue
		lo = minf(lo, float(t[0]) - float(t[1]) * 0.5)
		hi = maxf(hi, float(t[0]) + float(t[1]) * 0.5)
	if hi <= 0.0:
		return [3000.0, 8000.0]
	return [maxf(lo, 1500.0), hi]


func _band(pcm: PackedFloat32Array, ph: Dictionary, band: Array) -> float:
	var a := int(float(ph.t0) * Voice_.SR)
	var b := mini(int(float(ph.t1) * Voice_.SR), pcm.size())
	if b - a < 128:
		return 0.0
	var acc := 0.0
	var f: float = band[0]
	while f <= float(band[1]):
		acc += _goertzel(pcm, a, b, f)
		f += BAND_STEP
	return acc / float(b - a)


## Log-magnitude spectrum on a log-spaced grid, mean-removed so the
## correlation measures SHAPE and not level. The grid starts ABOVE the voice:
## below ~1.5 kHz every fricative window carries the same leakage from the
## neighbouring vowel and the cascade ring, and including it made two
## completely different fricatives correlate at 0.91 with frication switched
## OFF entirely - the metric was measuring the shared tail, not the sound.
func _shape(pcm: PackedFloat32Array, ph: Dictionary) -> PackedFloat32Array:
	var a := int(float(ph.t0) * Voice_.SR)
	var b := mini(int(float(ph.t1) * Voice_.SR), pcm.size())
	if b - a < 128:
		return PackedFloat32Array()
	var out := PackedFloat32Array()
	out.resize(SHAPE_BINS)
	var ratio: float = pow(SHAPE_HI / SHAPE_LO, 1.0 / float(SHAPE_BINS - 1))
	var f := SHAPE_LO
	var mean := 0.0
	for k in SHAPE_BINS:
		var v := 10.0 * log(maxf(_goertzel(pcm, a, b, f) / float(b - a), 1e-12)) / log(10.0)
		out[k] = v
		mean += v
		f *= ratio
	mean /= float(SHAPE_BINS)
	for k in SHAPE_BINS:
		out[k] -= mean
	return out


func _corr(x: PackedFloat32Array, y: PackedFloat32Array) -> float:
	var sxy := 0.0
	var sxx := 0.0
	var syy := 0.0
	for k in x.size():
		sxy += x[k] * y[k]
		sxx += x[k] * x[k]
		syy += y[k] * y[k]
	return sxy / maxf(sqrt(sxx * syy), 1e-12)


## Goertzel power at f over pcm[a, b).
func _goertzel(pcm: PackedFloat32Array, a: int, b: int, f: float) -> float:
	var cw := 2.0 * cos(TAU * f / float(Voice_.SR))
	var s1 := 0.0
	var s2 := 0.0
	for i in range(a, b):
		var s0: float = pcm[i] + cw * s1 - s2
		s2 = s1
		s1 = s0
	return s1 * s1 + s2 * s2 - cw * s1 * s2


func _rms(pcm: PackedFloat32Array, a: int, b: int) -> float:
	var acc := 0.0
	for i in range(a, b):
		acc += pcm[i] * pcm[i]
	return sqrt(acc / maxf(float(b - a), 1.0))
