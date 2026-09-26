extends SceneTree

## Gate for the AMBIENCE DIAL AS SEVERITY: the bed plays at full level at any setting, and
## the dial is how often a bass swell answers the reader's pitch moving.
##
## Asserted on what is PLAYED: the bottom of the dial is the bed ALONE (the chance is zero,
## not merely unlikely), the bed is identical across the dial (events draw from their own
## generator), every swell lands on a bar line in the key, and what the top adds stays under
## the voice. (Three melodic layers were built here and removed - see voice_fx.gd.)
##
## Run: tests/run_quiet.sh ambience_severity_check

const SR := 22050
const SECS := 90.0

var _fails: Array = []


func _init() -> void:
	var fx := VoiceFX.new()
	for sev in [0.05, 0.1]:
		fx.pad = sev
		var c := fx._chance(VoiceFX.BASS_ONSET, VoiceFX.BASS_CHANCE)
		_ok(c == 0.0, "at %.2f a cue can still be taken (summed chance %.4f)" % [sev, c])
	var dry := _voice()
	var low := _run(dry, 0.1)
	var high := _run(dry, 1.0)
	_ok(low["swells"].is_empty(), "0.1 should be the bed alone, got %d swells" % low["swells"].size())
	_check_swells(high)
	# Before the first phrase has ended there is no cue, so the two renders are the SAME bed.
	var a: PackedFloat32Array = low["out"]
	var b: PackedFloat32Array = high["out"]
	var same := 0.0
	for i in int(2.5 * SR):
		same = maxf(same, absf(a[i] - b[i]))
	_ok(same < 1e-6, "the dial changed the bed itself before any cue (max diff %.6f)" % same)
	var ev := 0.0
	var vo := 0.0
	for i in a.size():
		ev += (b[i] - a[i]) * (b[i] - a[i])
		vo += dry[i] * dry[i]
	ev = sqrt(ev / a.size())
	vo = sqrt(vo / a.size())
	_ok(ev > 0.0003, "the events at 1.0 are inaudible (RMS %.5f)" % ev)
	_ok(ev < vo * 0.25, "the events at 1.0 compete with the voice (RMS %.4f vs voice %.4f)" % [ev, vo])
	print("ambience_severity_check: 1.0 plays %d swells, events RMS %.4f vs voice %.4f"
		% [high["swells"].size(), ev, vo])
	if _fails.is_empty():
		print("ambience_severity_check: ALL OK")
		quit(0)
	else:
		for f in _fails:
			print("ambience_severity_check: FAIL  " + f)
		quit(1)


func _check_swells(run: Dictionary) -> void:
	var swells: Array = run["swells"]
	_ok(swells.size() >= 1, "1.0 answered none of the pitch moves in %ds" % SECS)
	var bar := int(run["eighth"]) * 8
	var key := float(run["key"])
	for sw in swells:
		_ok(int(sw[0]) % bar == 0, "a swell began off the bar line (sample %d, bar %d)" % [sw[0], bar])
		var semis := 12.0 * log(float(sw[1]) / key) / log(2.0)
		var pc := posmod(int(round(semis)), 12)
		_ok(absf(semis - round(semis)) < 0.02 and (pc == 0 or pc == 7),
			"a swell at %.2f Hz is neither the key's root nor its fifth" % sw[1])


## Phrases: 3 s of a buzzing "voice" with a syllable envelope and a pitch glide, a 1.5 s pause.
func _voice() -> PackedFloat32Array:
	var buf := PackedFloat32Array()
	buf.resize(int(SECS * SR))
	var rng := RandomNumberGenerator.new()
	rng.seed = 7
	var ph := 0.0
	var base := 120.0
	var glide := 0.0
	var last := -1
	for i in buf.size():
		var t := float(i) / SR
		if int(t / 4.5) != last:
			last = int(t / 4.5)
			base = rng.randf_range(105.0, 135.0)
			glide = rng.randf_range(-0.2, 0.2)
		var into := fmod(t, 4.5)
		ph += TAU * base * (1.0 + glide * into / 3.0) / SR
		var syl := 0.5 + 0.5 * sin(TAU * 4.0 * t)
		buf[i] = (sin(ph) + 0.5 * sin(2.0 * ph) + 0.3 * sin(3.0 * ph)) * 0.05 * syl \
			if into < 3.0 else 0.0
	return buf


func _run(dry: PackedFloat32Array, sev: float) -> Dictionary:
	var fx := VoiceFX.new()
	fx.pad_seed = 12345
	fx.setup(SR)
	fx.pad = sev
	fx.prime_key(dry)
	var out := PackedFloat32Array()
	var swells: Array = []
	var pos := 0
	while pos < dry.size():
		var c := dry.slice(pos, mini(pos + 64, dry.size()))
		var waiting: int = fx._bass_at
		c = fx.process(c)
		if waiting >= 0 and fx._bass_at < 0:
			swells.append([waiting, fx._bass_hz])
		out.append_array(c)
		pos += 64
	return {"out": out, "swells": swells, "eighth": fx._eighth, "key": fx._key_hz()}


func _ok(cond: bool, msg: String) -> void:
	if not cond:
		_fails.append(msg)
