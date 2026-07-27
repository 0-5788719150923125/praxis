extends SceneTree

## The bare-minimum synthesis repro: text in, WAVs out. No Ghost boot, no
## autoloads, no scenes, no stream, no push path - just [Voice.render] and
## [Voice.write_wav].
##
## Ladder v3 - through-tract frication at CALIBRATED drive. v2's verdict:
## the through-cascade route improved the character ("voice-like, not
## static") but the guessed drives (2.2 / 4.5) seated fricatives ABOVE the
## voice again - the original level crime through a better pipe. This run
## self-calibrates: it renders once at drive 1.0, measures the voiceless-
## fricative vs vowel RMS ratio from its own timing map, solves the drive
## that seats fricatives at TARGET_RATIO (~-12 dB, where real /s/ lives),
## and builds the ladder around that point:
##   off        - no frication (the validated baseline)
##   thru_soft  - half the calibrated drive
##   thru       - the calibrated drive
##   thru_firm  - twice the calibrated drive
##
## Run:  godot --headless --path axis/ghost --script tests/pure_say.gd
##       (optionally: -- "your own text here")
## Listen:  mpv /tmp/ghost_scratch/pure_say_<name>.wav

const Voice_ := preload("res://scripts/voice.gd")

const OUT_DIR := "/tmp/ghost_scratch"
const VOICELESS_FRIC := ["S", "SH", "F", "TH"]
const VOWELS := ["IY", "IH", "EH", "AE", "AA", "AO", "UH", "UW", "AH", "ER",
	"AY", "EY", "OY", "AW", "OW"]
const TARGET_RATIO := 0.35            # fricative/vowel median RMS - a step
                                      # crisper than the first -12 dB seat, per
                                      # the "consonants more clear" request


func _init() -> void:
	var text := "Once upon a time, the river spoke softly. Stones and static kept the signal safe. It was not human, but it was alive."
	var args := OS.get_cmdline_user_args()
	if args.size() > 0:
		text = " ".join(args)
	var spec := Voice_.Spec.from_traits({}, 1, [1])
	var kept_trim: float = Voice_.FRIC_TRIM
	var kept_thru: float = Voice_.FRIC_THRU
	DirAccess.make_dir_recursive_absolute(OUT_DIR)
	# calibration render: drive 1.0, measure, solve (frication scales
	# linearly with the drive; the voiced material does not move)
	Voice_.FRIC_TRIM = 0.0
	Voice_.FRIC_THRU = 1.0
	var cal: Dictionary = Voice_.render(text, spec)
	var r1 := _ratio(cal.pcm, cal.phones)
	var drive: float = TARGET_RATIO / maxf(r1, 0.0001)
	print("pure_say: calibration - fric/vowel %.3f at drive 1.0 -> calibrated drive %.2f" % [r1, drive])
	var ladder := [
		{"name": "off", "thru": 0.0},
		{"name": "thru_soft", "thru": drive * 0.5},
		{"name": "thru", "thru": drive},
		{"name": "thru_firm", "thru": drive * 2.0},
	]
	for rung in ladder:
		Voice_.FRIC_THRU = rung.thru
		var res: Dictionary = Voice_.render(text, spec)
		var pcm: PackedFloat32Array = res.pcm
		var achieved := _ratio(pcm, res.phones)
		var peak := 0.0
		for v in pcm:
			peak = maxf(peak, absf(v))
		var scale := 0.7 / maxf(peak, 0.001)
		for i in pcm.size():
			pcm[i] *= scale
		var path := Voice_.write_wav(OUT_DIR + "/pure_say_%s.wav" % rung.name, pcm)
		print("pure_say: %-9s thru %.2f  fric/vowel %.3f  raw peak %.3f -> %s" % [
			rung.name, rung.thru, achieved, peak, path])
	Voice_.FRIC_TRIM = kept_trim
	Voice_.FRIC_THRU = kept_thru
	print("pure_say: RAW_MODE=%s NOISE_FX=%s - does `thru` read as the voice making an s?" % [
		str(Voice_.RAW_MODE), str(Voice_.NOISE_FX)])
	quit()


## Median per-phone RMS of the voiceless fricatives over the vowels.
func _ratio(pcm: PackedFloat32Array, phones: Array) -> float:
	var fr := _phone_rms(pcm, phones, VOICELESS_FRIC)
	var vo := _phone_rms(pcm, phones, VOWELS)
	return fr / maxf(vo, 0.0001)


func _phone_rms(pcm: PackedFloat32Array, phones: Array, wanted: Array) -> float:
	var vals: Array = []
	for ph in phones:
		if not wanted.has(String(ph.p)):
			continue
		var a := int(float(ph.t0) * Voice_.SR)
		var b := mini(int(float(ph.t1) * Voice_.SR), pcm.size())
		if b - a < 40:
			continue
		var acc := 0.0
		for i in range(a, b):
			acc += pcm[i] * pcm[i]
		vals.append(sqrt(acc / float(b - a)))
	if vals.is_empty():
		return 0.0
	vals.sort()
	return vals[int(vals.size() / 2.0)]
