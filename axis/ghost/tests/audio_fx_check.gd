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

	_check_room()
	_check_bed_under_voice()

	print("")
	if _fails.is_empty():
		print("audio_fx_check: PASS - the chain is built, neutral is inaudible, ",
			"every stage responds, the envelope scales it, and one room serves both modes.")
		quit(0)
	else:
		for f in _fails:
			print("audio_fx_check: FAIL - ", f)
		quit(1)


## THE BED STAYS UNDER THE VOICE, AND STILL. Reported as "basically 80% of all words a kind of
## echo effect" at Ambience 1.0, and it was two faults: the top octave of the bed's stack landed
## ON the reader's pitch (a tone in the voice's own key, humming along), and the speech envelope
## that ducks the bed followed each word (~0.11 s, written as a per-sample constant meant to be
## ~1 s), so the bed swelled back in every gap. A synthetic reader at 220 Hz speaking in word-
## length bursts must get a bed with almost nothing in its register, and no louder between words
## than during them.
func _check_bed_under_voice() -> void:
	var sr := 22050
	var dry := PackedFloat32Array()
	dry.resize(sr * 30)
	for i in dry.size():
		var t := float(i) / float(sr)
		var on := fmod(t, 0.6) < 0.35                     # a word, then a gap
		var v := 0.0
		if on:
			for h in range(1, 8):
				v += sin(TAU * 220.0 * float(h) * t) / float(h)
		dry[i] = v * 0.05
	var bed := _bed_of(dry, sr)
	# 1) the register: a steep high-pass at 180 Hz (two cascaded biquads) keeps the voice's band
	var hp := _highpass(_highpass(bed, sr, 180.0), sr, 180.0)
	var e_all := 0.0
	var e_hi := 0.0
	for i in range(sr * 5, bed.size()):                # past the bed's own attack
		e_all += bed[i] * bed[i]
		e_hi += hp[i] * hp[i]
	var share := e_hi / maxf(e_all, 1e-12)
	_expect(share < 0.15, "the ambience bed sits in the reader's register: %.0f%% of its energy above 180 Hz under a 220 Hz voice" % (share * 100.0))
	# 2) the pumping: the bed in the gaps against the bed during words
	var in_word := 0.0
	var in_gap := 0.0
	var nw := 0
	var ng := 0
	for i in range(sr * 5, bed.size()):
		var t := fmod(float(i) / float(sr), 0.6)
		if t > 0.05 and t < 0.30:
			in_word += bed[i] * bed[i]
			nw += 1
		elif t > 0.40 and t < 0.58:
			in_gap += bed[i] * bed[i]
			ng += 1
	var pump := sqrt((in_gap / float(ng)) / maxf(in_word / float(nw), 1e-12))
	_expect(pump < 1.2, "the ambience bed swells between words: %.2fx louder in the gaps than under them" % pump)
	print("  bed: %.1f%% above 180 Hz, gaps %.2fx the level under words" % [share * 100.0, pump])


## What Ambience 1.0 adds to [param dry]: the chain at 1.0 minus the chain at 0, everything else off.
func _bed_of(dry: PackedFloat32Array, sr: int) -> PackedFloat32Array:
	var outs: Array = []
	for pad in [0.0, 1.0]:
		var fx := VoiceFX.new()
		fx.pad_seed = 4321
		fx.setup(sr)
		fx.echo_wet = 0.0
		fx.resonance = 0.0
		fx.presence = 1.0
		fx.pad = pad
		fx.room.from_dial(0.0, 0.0)
		fx.prime_key(dry)
		var o := PackedFloat32Array()
		var i := 0
		while i < dry.size():
			o.append_array(fx.process(dry.slice(i, i + 4096)))
			i += 4096
		outs.append(o)
	var bed := PackedFloat32Array()
	bed.resize(dry.size())
	for i in dry.size():
		bed[i] = float(outs[1][i]) - float(outs[0][i])
	return bed


## A Butterworth high-pass biquad at [param fc].
static func _highpass(x: PackedFloat32Array, sr: int, fc: float) -> PackedFloat32Array:
	var w := TAU * fc / float(sr)
	var al := sin(w) / (2.0 * 0.7071)
	var a0 := 1.0 + al
	var b0 := (1.0 + cos(w)) * 0.5 / a0
	var b1 := -(1.0 + cos(w)) / a0
	var a1 := -2.0 * cos(w) / a0
	var a2 := (1.0 - al) / a0
	var y := PackedFloat32Array()
	y.resize(x.size())
	var x1 := 0.0
	var x2 := 0.0
	var y1 := 0.0
	var y2 := 0.0
	for i in x.size():
		var v := b0 * x[i] + b1 * x1 + b0 * x2 - a1 * y1 - a2 * y2
		x2 = x1
		x1 = x[i]
		y2 = y1
		y1 = v
		y[i] = v
	return y


## ONE ROOM, TWO RENDERERS. Masking's room is Godot's [AudioEffectReverb] on the bus
## above; Generative bakes its effects into a take WAV and so cannot use a bus at all,
## and runs the same comb/allpass network per sample instead ([RoomFX.tick]). The dial
## is shared, the engines are not, and this checks the half a bus test cannot reach.
##
## The failure mode being guarded is not a raising setter this time, it is a room that
## is never asked: a dial wired to a settings object nobody ticks changes nothing, and
## sounds exactly like a room turned down. So the assertions are about AUDIO - that a
## tail exists after the input stops, that it lasts longer as the dial opens, and that
## it reaches the Generative chain at all.
const ROOM_SR := 22050
const ROOM_BURST := 0.1          # seconds of excitation
const ROOM_LISTEN := 6.0         # seconds of silence to listen through afterwards


func _check_room() -> void:
	# The two renderers agree about NOTHING BEING THERE. A room at the bottom of the
	# dial must be inaudible in both, or every take that never touched the slider
	# quietly gains a space it did not ask for.
	var off := RoomFX.new()
	off.from_dial(0.0, 0.0)
	_expect(not off.is_active(), "the room is active at dial 0")
	var rv := AudioEffectReverb.new()
	off.to_reverb(rv)
	_expect(rv.wet < 0.001, "dial 0 renders %.3f wet onto the bus reverb" % rv.wet)
	var flat := _burst()
	off.setup(ROOM_SR)
	off.prepare()
	var moved := 0.0
	for i in flat.size():
		moved = maxf(moved, absf(off.tick(flat[i]) - flat[i]))
	_expect(moved < 1e-6, "dial 0 still changes the samples (by %.6f)" % moved)

	# ...and both open with the dial. The bus half is a property read; the DSP half
	# is measured off the tail it actually produces.
	var small := _room_tail(0.25)
	var big := _room_tail(1.0)
	print("room tail: dial 0.25 -> %.2fs (peak %.3f)   dial 1.00 -> %.2fs (peak %.3f)"
		% [small.tail, small.peak, big.tail, big.peak])
	_expect(big.tail > 2.0, "the top of the dial rings for only %.2fs - that is not a hall"
		% big.tail)
	_expect(small.tail < 1.5, "the bottom of the dial rings for %.2fs - a small room's tail "
		% small.tail + "should be gone before the next word")
	_expect(big.tail > small.tail * 2.0,
		"the dial changes the LEVEL but not the SIZE (%.2fs against %.2fs) - the same "
		% [big.tail, small.tail] + "trap the echo dial was in before it shortened its tap")
	# Stability, since the top of the travel sits at 0.98 feedback: a network that grows
	# instead of decaying is silent for a second and then unusable.
	_expect(big.peak <= 1.0 and big.late < big.peak * 0.5,
		"the tail is not decaying (peak %.3f, late %.3f)" % [big.peak, big.late])

	# THE CHAIN, not just the class. This is the wiring the dial depends on: VoiceFX has
	# to actually tick the room, in a position where the room hears the voice.
	var dry_fx := VoiceFX.new()
	dry_fx.setup(ROOM_SR)
	var wet_fx := VoiceFX.new()
	wet_fx.setup(ROOM_SR)
	wet_fx.room.from_dial(1.0, 0.5)
	var dry_tail := _late_energy(dry_fx.process(_burst_then_silence()))
	var wet_tail := _late_energy(wet_fx.process(_burst_then_silence()))
	print("VoiceFX late energy: room off %.6f, room on %.6f" % [dry_tail, wet_tail])
	_expect(dry_tail < 1e-6, "VoiceFX rings with the room off (%.6f)" % dry_tail)
	_expect(wet_tail > 1e-4, "VoiceFX does not ring with the room on (%.6f) - the dial "
		% wet_tail + "is set but nothing ticks it")


## Excite the room, then listen. Returns the tail's length (to -40 dB of its own peak),
## the peak itself, and how much is left at the very end.
func _room_tail(dial: float) -> Dictionary:
	var fx := RoomFX.new()
	fx.setup(ROOM_SR)
	fx.from_dial(dial, 0.5)
	fx.prepare()
	var peak := 0.0
	var out := PackedFloat32Array()
	var burst := _burst()
	for i in burst.size():
		fx.tick(burst[i])
	var n := int(ROOM_LISTEN * ROOM_SR)
	out.resize(n)
	for i in n:
		out[i] = fx.tick(0.0)
		peak = maxf(peak, absf(out[i]))
	var tail := 0.0
	if peak > 0.0:
		for i in range(n - 1, -1, -1):
			if absf(out[i]) > peak * 0.01:
				tail = float(i) / float(ROOM_SR)
				break
	var late := 0.0
	for i in range(n - ROOM_SR, n):
		late = maxf(late, absf(out[i]))
	return {"tail": tail, "peak": peak, "late": late}


## A short tone. Deterministic on purpose - a room measured against noise measures the
## noise as much as the room.
static func _burst() -> PackedFloat32Array:
	var n := int(ROOM_BURST * ROOM_SR)
	var buf := PackedFloat32Array()
	buf.resize(n)
	for i in n:
		buf[i] = sin(TAU * 220.0 * float(i) / float(ROOM_SR)) * 0.5
	return buf


static func _burst_then_silence() -> PackedFloat32Array:
	var buf := _burst()
	buf.resize(int((ROOM_BURST + 1.0) * ROOM_SR))
	return buf


## Peak level over the last half second, which is silence at the input.
static func _late_energy(buf: PackedFloat32Array) -> float:
	var e := 0.0
	for i in range(maxi(0, buf.size() - ROOM_SR / 2), buf.size()):
		e = maxf(e, absf(buf[i]))
	return e


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
