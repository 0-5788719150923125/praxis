extends SceneTree

## Gate for the PAD-LED INTRO - the ambience bed playing alone, before the first word.
##
## This exists because the feature has a silent failure mode that looks exactly like the
## feature being off. [VoiceFX] is a filter, not a source, and its pad is key-slaved: the
## tonic comes from the tracked pitch of the voice, and [method VoiceFX._estimate_pitch]
## returns 0 for any window under its energy floor. So over a leading pad of digital
## silence the tonic never rises off zero, [method VoiceFX._start_tone] refuses to
## schedule anything, and the intro is silent - with nothing anywhere to say so.
##
## [method VoiceFX.prime_key] is what fixes it, and its FIRST version was wrong in a way
## only measurement caught: it scanned from sample 0 and gave up after two seconds, so on
## a five second intro it examined nothing but the padding and always came back empty.
## The test below is written against the observable - is there sound in the intro - which
## is why it caught that; a test of "was prime_key called" would have passed.
##
## Run: godot --headless --path axis/ghost --script tests/pad_intro_check.gd

const SR := 22050
const INTRO := 5.0
const VOICE := 3.0
## Measured reference: real narration through this chain sits at RMS ~0.047 (see
## VoiceFX.pad_level's staging notes). A bed should be clearly under the voice and
## clearly above nothing.
const FLOOR := 0.0015
const CEIL := 0.030

var _fails: Array = []


func _init() -> void:
	var n_intro := int(INTRO * SR)

	var bare := _run(false, n_intro)
	var primed := _run(true, n_intro)

	_ok(bare["intro"] < 1e-6,
		"unprimed: the intro must be silent, since that is the bug being fixed (got %.6f)"
		% bare["intro"])
	_ok(primed["intro"] > FLOOR,
		"primed: the pad must actually SOUND through the intro - got RMS %.6f, needs > %.4f"
		% [primed["intro"], FLOOR])
	_ok(primed["intro"] < CEIL,
		"primed: the bed must sit UNDER the voice, not compete with it - got RMS %.6f, max %.3f"
		% [primed["intro"], CEIL])
	_ok(primed["voice"] > 0.1,
		"primed: the narration itself must survive the chain (got %.4f)" % primed["voice"])
	# The bed swells; it must not arrive at full level. A pad that is already flat out at
	# the top of the intro reads as a cut, not as a fade-in.
	_ok(primed["early"] < primed["late"],
		"primed: the bed must still be SWELLING across the intro - first half %.6f, second half %.6f"
		% [primed["early"], primed["late"]])

	print("pad_intro_check: intro RMS  unprimed=%.6f  primed=%.6f  (voice %.4f)"
		% [bare["intro"], primed["intro"], primed["voice"]])
	print("pad_intro_check: swell      first half=%.6f -> second half=%.6f"
		% [primed["early"], primed["late"]])
	if _fails.is_empty():
		print("pad_intro_check: ALL OK")
		quit()
		return
	print("pad_intro_check: %d FAILURE(S)" % _fails.size())
	for f in _fails:
		print("   ", f)
	quit(1)


func _ok(cond: bool, msg: String) -> void:
	if not cond:
		_fails.append(msg)


func _run(primed: bool, n_intro: int) -> Dictionary:
	var fx := VoiceFX.new()
	fx.pad_seed = 12345
	fx.setup(SR)
	fx.pad = 1.0
	fx.resonance = 0.6
	var buf := _signal(n_intro)
	if primed:
		fx.prime_key(buf)
	buf = fx.process(buf)
	var half := n_intro / 2
	return {
		"intro": _rms(buf, 0, n_intro),
		"early": _rms(buf, 0, half),
		"late": _rms(buf, half, n_intro),
		"voice": _rms(buf, n_intro, buf.size()),
	}


## Silence, then a crude voiced signal - a 120 Hz fundamental with two harmonics, which
## is what the autocorrelation tracker is built to find.
func _signal(n_intro: int) -> PackedFloat32Array:
	var n_voice := int(VOICE * SR)
	var b := PackedFloat32Array()
	b.resize(n_intro + n_voice)
	for i in n_voice:
		var t := float(i) / float(SR)
		b[n_intro + i] = 0.30 * (sin(TAU * 120.0 * t)
			+ 0.5 * sin(TAU * 240.0 * t) + 0.25 * sin(TAU * 360.0 * t))
	return b


func _rms(b: PackedFloat32Array, a: int, z: int) -> float:
	if z <= a:
		return 0.0
	var s := 0.0
	for i in range(a, z):
		s += b[i] * b[i]
	return sqrt(s / float(z - a))
