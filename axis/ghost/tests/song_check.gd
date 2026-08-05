extends SceneTree

## Does the sung voice actually sing? Three measurements that separate singing
## from speaking, run across the `song` axis so the trend is visible rather than
## a single pass/fail:
##
##   NOTE LENGTH  - a sung note is 300-1000 ms. Speech vowels are ~100 ms, and
##                  the earlier audit found only 0.17% of planned vowels reached
##                  400 ms, which is why no seed could ever sing.
##   PITCH REST   - the fraction of voiced time within 25 cents of a shelf note.
##                  Singing sits ON pitches; speech glides through them.
##   FLATNESS     - median |df0/dt| in cents per second over voiced frames,
##                  excluding vibrato by measuring the trend not the waver. A
##                  held note is still; a spoken vowel is always moving.
##
## Run: godot --headless --path axis/ghost --script tests/song_check.gd

const Voice_ := preload("res://scripts/voice.gd")

const TEXT := "The far end is yours. It has been yours for so long that the cushion has surrendered."
const OUT_DIR := "/tmp/ghost_scratch"


func _init() -> void:
	DirAccess.make_dir_recursive_absolute(OUT_DIR)
	print("song  vowels  median  p90     max     on-note held  |df0/dt|     total   spread   sustained")
	for song in [0.0, 0.5, 1.0]:
		var spec := Voice_.Spec.from_traits({"song": song}, 1, [1])
		var segs := Voice_.plan(TEXT, spec)
		var durs: Array = []
		for seg in segs:
			if Phonemes.TABLE.get(String(seg.p), {}).get("type", "") == "vowel":
				durs.append(float(seg.dur) * 1000.0)
		durs.sort()
		var res: Dictionary = Voice_.synth(segs, spec)
		var f0 := _track(res.pcm, spec.f0_base)
		var on := _on_note(f0, spec, false)
		var on_held := _on_note(f0, spec, true)
		var mv := _motion(f0)
		# how long the WHOLE reading takes relative to the same text spoken, and
		# how uneven the syllables are - a cadence needs contrast, a drone does
		# not have any. CV near 0 is a metronome; speech runs ~0.55.
		var total := 0.0
		for seg in segs:
			total += float(seg.dur)
		var mean := 0.0
		for d in durs:
			mean += d
		mean /= maxf(float(durs.size()), 1.0)
		var sd := 0.0
		for d in durs:
			sd += (d - mean) * (d - mean)
		sd = sqrt(sd / maxf(float(durs.size()), 1.0))
		var held := 0
		for d in durs:
			if d > 300.0:
				held += 1
		print("%.1f   %4d    %5.0f   %5.0f   %5.0f   %5.1f%% %5.1f%%   %6.0f cents/s   %5.1fs  CV %.2f  held %2d%%" % [
			song, durs.size(), durs[durs.size() / 2], durs[9 * durs.size() / 10],
			durs[durs.size() - 1], on * 100.0, on_held * 100.0, mv, total, sd / maxf(mean, 1.0),
			int(round(100.0 * float(held) / maxf(float(durs.size()), 1.0)))])
		var pcm: PackedFloat32Array = res.pcm
		var pk := 0.0
		for v in pcm:
			pk = maxf(pk, absf(v))
		for i in pcm.size():
			pcm[i] *= 0.7 / maxf(pk, 0.001)
		print("        -> %s" % Voice_.write_wav(OUT_DIR + "/song_%.0f.wav" % (song * 10.0), pcm))
	# how often a WILD roll sings at all, and how strongly. The belt compounds
	# whatever the roll gives it - acceptance-weighted parents inside a trust
	# region centred on the party mean - so the roll has to sit BELOW the rate
	# you want to meet in play. Reported here because ~90% of found seeds
	# singing was the symptom that sent us looking.
	var rng := RandomNumberGenerator.new()
	var sings := 0
	var strong := 0
	var n := 4000
	for i in n:
		rng.seed = i * 2654435761
		var sp := Voice_.Spec.sample(rng)
		if sp.song > 0.0:
			sings += 1
		if sp.song > 0.5:
			strong += 1
	print("\nwild rolls: %.0f%% sing at all, %.0f%% sing strongly (song > 0.5)" % [
		100.0 * float(sings) / float(n), 100.0 * float(strong) / float(n)])
	quit()


## Autocorrelation f0 per 20 ms hop, semitones above f0_base; NAN when unvoiced.
func _track(pcm: PackedFloat32Array, base: float) -> PackedFloat32Array:
	var sr := Voice_.SR
	var hop := int(sr * 0.02)
	var win := int(sr * 0.045)
	var lo := int(sr / 400.0)
	var hi := int(sr / 60.0)
	var out := PackedFloat32Array()
	var a := 0
	while a + win < pcm.size():
		var e := 0.0
		for i in range(a, a + win):
			e += pcm[i] * pcm[i]
		if e / float(win) < 1e-6:
			out.append(NAN)
			a += hop
			continue
		var best := 0.0
		var best_lag := 0
		for lag in range(lo, hi):
			var c := 0.0
			for i in range(a, a + win - lag):
				c += pcm[i] * pcm[i + lag]
			if c > best:
				best = c
				best_lag = lag
		if best_lag <= 0:
			out.append(NAN)
		else:
			out.append(12.0 * log(float(sr) / float(best_lag) / base) / log(2.0))
		a += hop
	return out


## Fraction of voiced time within 25 cents of one of the voice's anchors, with
## the VIBRATO SMOOTHED OUT FIRST. Measuring raw frames scores a perfectly
## quantized note at ~40%, because a +-38 cent waver spends most of its cycle
## outside a 25 cent window - the metric would have been measuring the vibrato
## and reporting it as bad tuning.
## `held_only` restricts to frames that are actually STILL (|df0/dt| under
## 50 cents/s). Once most syllables became short runs, the overall figure was
## dominated by transitions - which are not where a tune lives. This asks the
## musically meaningful question: when this voice holds a note, is it on one?
func _on_note(f0: PackedFloat32Array, spec: Voice_.Spec, held_only: bool) -> float:
	f0 = _smooth(f0, 9)
	var walk := Voice_.ProsodyWalk.new([spec.reading] + spec.influences, spec.adrenochrome)
	var n := 0
	var hit := 0
	for i in f0.size():
		var v: float = f0[i]
		if is_nan(v):
			continue
		if held_only:
			if i < 3 or is_nan(f0[i - 3]):
				continue
			if absf(v - f0[i - 3]) * 100.0 / 0.06 > 50.0:
				continue
		n += 1
		# the synth realizes f0_base * 2^(semis*inflect/12) * 1.06, so the track
		# carries a fixed +1.01 st offset that has to come off before the value
		# means anything to the shelf
		var semis: float = (v - 1.009) / maxf(spec.inflect, 0.01)
		var near: float = walk.nearest_anchor(semis)
		if absf(semis - near) < 0.25:
			hit += 1
	return float(hit) / maxf(float(n), 1.0)


## Running median over `w` frames (~180 ms at w=9), NAN-preserving: one full
## vibrato period, so the trend survives and the waver does not.
func _smooth(f0: PackedFloat32Array, w: int) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	for i in f0.size():
		if is_nan(f0[i]):
			out.append(NAN)
			continue
		var win: Array = []
		for k in range(maxi(0, i - w / 2), mini(f0.size(), i + w / 2 + 1)):
			if not is_nan(f0[k]):
				win.append(f0[k])
		win.sort()
		out.append(win[win.size() / 2] if not win.is_empty() else NAN)
	return out


## Median absolute f0 slope over voiced runs, in cents per second. Measured over
## a 3-frame span so the vibrato waver does not count as motion.
func _motion(f0: PackedFloat32Array) -> float:
	f0 = _smooth(f0, 9)
	var d: Array = []
	for i in range(3, f0.size()):
		if is_nan(f0[i]) or is_nan(f0[i - 3]):
			continue
		d.append(absf(f0[i] - f0[i - 3]) * 100.0 / 0.06)
	if d.is_empty():
		return 0.0
	d.sort()
	return d[d.size() / 2]
