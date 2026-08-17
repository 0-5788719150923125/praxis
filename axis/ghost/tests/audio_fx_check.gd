extends SceneTree

## The Masking audio effect's bus, and that its knobs actually reach it.
##   godot --headless --path . --script res://tests/audio_fx_check.gd
##
## This exists because of the failure mode a bus chain has: assigning a property
## Godot's class does not have raises at runtime and ABORTS THE REST OF THE
## FUNCTION, and the audible result is simply that some of the knobs do nothing.
## Exactly that happened here - AudioEffectCompressor takes `release_ms` while its
## attack is `attack_us`, so a `release_us` assignment killed every line after it
## and the delay and reverb silently kept their defaults while the EQ and
## compressor worked. Nothing in the editor would have shown that; you would just
## have moved the Ambience slider and heard nothing.
##
## So: build the bus, drive it at neutral / full / half, and assert each stage
## moved. Half is checked as well as full because the layer's ENVELOPE scales all
## of it, and an effect that ignores the envelope would pass a full-vs-neutral
## test while popping on instead of fading in.

var _fails: PackedStringArray = []


func _initialize() -> void:
	var ed: Node = load("res://scripts/mask_editor.gd").new()
	root.add_child(ed)
	ed._ensure_audio_bus()

	var bi := AudioServer.get_bus_index(ed.MASK_BUS)
	_expect(bi >= 0, "the effects bus was not created")
	_expect(AudioServer.get_bus_send(bi) == "Master",
		"the bus does not send to Master - the export would record silence from it")
	var chain := []
	for i in AudioServer.get_bus_effect_count(bi):
		chain.append(AudioServer.get_bus_effect(bi, i).get_class())
	print("bus %d -> %s : %s" % [bi, AudioServer.get_bus_send(bi), ", ".join(chain)])
	# ORDER MATTERS and is not cosmetic: tone before dynamics (so the compressor
	# hears the bass you asked for), dynamics before echo, and the room last -
	# around all of it, as a room is.
	_expect(chain == ["AudioEffectEQ6", "AudioEffectCompressor",
			"AudioEffectDelay", "AudioEffectReverb"],
		"the chain is in the wrong order: %s" % ", ".join(chain))

	var neutral := _read(ed, {})
	var full := _read(ed, _layer(1.0))
	var half := _read(ed, _layer(0.5))
	print("%-8s %s" % ["neutral", _fmt(neutral)])
	print("%-8s %s" % ["full", _fmt(full)])
	print("%-8s %s" % ["half", _fmt(half)])

	# NEUTRAL MUST BE INAUDIBLE. A session with no audio marker has to sound like
	# the clip, not like the clip in a room.
	_expect(absf(neutral.bass) < 0.01, "neutral lifts the bass by %.2f dB" % neutral.bass)
	_expect(neutral.ratio <= 1.001, "neutral compresses (ratio %.2f)" % neutral.ratio)
	_expect(neutral.wet < 0.01, "neutral is wet (%.2f)" % neutral.wet)
	_expect(neutral.tap1_db < -40.0, "neutral echoes (tap1 %.1f dB)" % neutral.tap1_db)

	# EVERY STAGE MOVES. One entry per knob, so a stage that silently kept its
	# defaults - the reported failure - is named rather than averaged away.
	for probe in [["bass EQ", full.bass, neutral.bass], ["compressor", full.ratio, neutral.ratio],
			["echo level", full.tap1_db, neutral.tap1_db],
			["echo time", full.tap1_ms, neutral.tap1_ms],
			["feedback tone", full.fb_lp, neutral.fb_lp],
			["reverb wet", full.wet, neutral.wet],
			["reverb room", full.room, neutral.room]]:
		_expect(absf(float(probe[1]) - float(probe[2])) > 0.001,
			"%s did not respond (%.3f at full, %.3f at neutral) - a setter that "
			% [probe[0], probe[1], probe[2]] + "raises aborts every line after it")

	# The second tap must NOT be a multiple of the first, or the repeats land on
	# top of each other as one loud slap instead of decaying into each other.
	var mult: float = full.tap2_ms / maxf(full.tap1_ms, 1e-4)
	print("second tap is %.3fx the first" % mult)
	_expect(absf(mult - round(mult)) > 0.15,
		"the second echo tap is %.2fx the first - near a whole multiple, so the "
		% mult + "repeats stack instead of interleaving")

	# THE ENVELOPE SCALES IT. Half must land between, or the effect pops on.
	for probe in [["bass", half.bass, neutral.bass, full.bass],
			["reverb wet", half.wet, neutral.wet, full.wet]]:
		var lo: float = minf(float(probe[2]), float(probe[3]))
		var hi: float = maxf(float(probe[2]), float(probe[3]))
		_expect(float(probe[1]) > lo + (hi - lo) * 0.15
			and float(probe[1]) < lo + (hi - lo) * 0.85,
			"%s ignores the marker's envelope (%.3f at half, between %.3f and %.3f)"
			% [probe[0], probe[1], lo, hi])

	ed.free()
	print("")
	if _fails.is_empty():
		print("audio_fx_check: PASS - the chain is built, neutral is inaudible, ",
			"every stage responds, and the envelope scales it.")
		quit(0)
	else:
		for f in _fails:
			print("audio_fx_check: FAIL - ", f)
		quit(1)


static func _layer(env: float) -> Dictionary:
	return {"env": env, "intensity_a": 1.0, "fx_stick": 1.0, "fx_smooth": 0.8,
		"fx_lag": 0.7, "fx_density": 0.9, "fx_scale": 1.6, "fx_contrast": 0.85}


func _read(ed: Node, l: Dictionary) -> Dictionary:
	ed._apply_audio_fx(l)
	return {
		"bass": ed._fx_eq.get_band_gain_db(0), "ratio": ed._fx_comp.ratio,
		"tap1_db": ed._fx_delay.tap1_level_db, "tap1_ms": ed._fx_delay.tap1_delay_ms,
		"tap2_ms": ed._fx_delay.tap2_delay_ms, "fb_lp": ed._fx_delay.feedback_lowpass,
		"wet": ed._fx_reverb.wet, "room": ed._fx_reverb.room_size,
	}


static func _fmt(d: Dictionary) -> String:
	return ("bass %+5.1fdB  ratio %.2f  tap1 %6.1fms @%+7.1fdB  fb_lp %6.0fHz  wet %.2f  room %.2f"
		% [d.bass, d.ratio, d.tap1_ms, d.tap1_db, d.fb_lp, d.wet, d.room])


func _expect(ok: bool, msg: String) -> void:
	if not ok:
		_fails.append(msg)
