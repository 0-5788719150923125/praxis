extends SceneTree

## Per-phone diagnostic dump: renders a text and prints every phoneme with its
## position, duration, broadband RMS and three band energies, so a specific
## reported word ("the F in father is silent") can be checked against what the
## synthesizer actually emitted rather than against a whole-take average.
##
## Run: godot --headless --path axis/ghost --script tests/phone_dump.gd -- "your text"

const Voice_ := preload("res://scripts/voice.gd")

const OUT_DIR := "/tmp/ghost_scratch"


func _init() -> void:
	var text := "Father. Yours. My father gave yours away."
	var args := OS.get_cmdline_user_args()
	if args.size() > 0:
		text = " ".join(args)
	var spec := Voice_.Spec.from_traits({}, 1, [1])
	DirAccess.make_dir_recursive_absolute(OUT_DIR)
	var res: Dictionary = Voice_.render(text, spec)
	var pcm: PackedFloat32Array = res.pcm
	print("text: %s" % text)
	print("%-5s %-6s %-7s %8s  %8s %8s %8s   %s" % [
		"phone", "t0", "dur_ms", "rms_dB", "0-1k", "1-3k", "3-8k", "word"])
	for ph in res.phones:
		var a := int(float(ph.t0) * Voice_.SR)
		var b := mini(int(float(ph.t1) * Voice_.SR), pcm.size())
		if b - a < 8:
			continue
		var r := _rms(pcm, a, b)
		print("%-5s %-6.3f %-7.1f %8.1f  %8.1f %8.1f %8.1f   %s" % [
			ph.p, ph.t0, (float(ph.t1) - float(ph.t0)) * 1000.0,
			_db(r), _db(sqrt(_band(pcm, a, b, 200.0, 1000.0))),
			_db(sqrt(_band(pcm, a, b, 1000.0, 3000.0))),
			_db(sqrt(_band(pcm, a, b, 3000.0, 8000.0))), ph.get("word", -1)])
	Voice_.write_wav(OUT_DIR + "/phone_dump.wav", pcm)
	print("-> %s/phone_dump.wav" % OUT_DIR)
	quit()


func _db(v: float) -> float:
	return 20.0 * log(maxf(v, 1e-9)) / log(10.0)


func _band(pcm: PackedFloat32Array, a: int, b: int, lo: float, hi: float) -> float:
	var acc := 0.0
	var f := lo
	var n := 0
	while f <= hi:
		acc += _goertzel(pcm, a, b, f)
		f += 200.0
		n += 1
	return acc / (float(b - a) * maxf(float(n), 1.0))


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
