extends RefCounted
class_name RoomFX

## RoomFX - THE ROOM, decided once and rendered twice.
##
## Masking got a room first, as the last stage of its audio effect's bus chain,
## and it is the knob in that panel people reach for most. Generative had none at
## all. Writing a second one from scratch would have shipped two dials that say
## Room and do not mean the same thing - the same failure as two sliders called
## "pace" doing unrelated jobs, which this project has already talked itself out
## of once.
##
## So everything ABOVE the engine lives here: the dial's range, its taper, what
## Resonance does to the tail's colour, what "no room" means, and how a single
## dial collapses into size-plus-wet when a panel only has room for one. Both
## modes ask this class and get the same answer.
##
## THE ENGINE IS THE ONE THING THEY CANNOT SHARE, and the reason is what each
## mode's audio actually is:
##
##   [method to_reverb]  Masking plays its clip through a real [AudioStreamPlayer],
##                       so its room is Godot's own [AudioEffectReverb] on a bus -
##                       sample-exact, free on the main thread, and rebuilt
##                       identically by the export relaunch (see MASK_BUS).
##   [method tick]       Generative pushes samples into a generator and BAKES its
##                       effects into the take WAV, because the render process
##                       boots with nothing but that file. A bus effect cannot be
##                       run offline over a buffer, so this half IS the reverb,
##                       in GDScript.
##
## They are not sample-identical and are not trying to be. They are close because
## they are the same algorithm: Godot's reverb is derived from Jezar at
## Dreampoint's Freeverb (public domain) - eight damped comb filters in parallel
## into four allpasses in series - and so is the network below, from the same
## published tunings. That is what lets one set of numbers drive both.

# --- the dial ------------------------------------------------------------------

## The top of the Room dial. 2 rather than 1 because Masking's neutral sits at
## 1.0: a dial whose rest position is the middle can be opened as well as closed,
## and a session with no audio marker has to keep sounding exactly as it did.
const SIZE_MAX := 2.0
## A single-dial panel's taper, for [method from_dial]. Above 1 so the bottom of
## the travel is genuinely a small close room rather than a quiet big one - the
## same reasoning as [constant VoiceFX.ECHO_TAPER], and for the same material.
const WET_TAPER := 1.5

# --- what the numbers mean -----------------------------------------------------

## Feedback across the dial: a small room's tail dies in a breath, a big one rings
## for seconds. Freeverb's roomsize1 = size * 0.28 + 0.7, which is also how Godot
## maps its own `room_size`, so one number really does drive both engines.
const FB_MIN := 0.7
const FB_SPAN := 0.28
## Damping is what makes a tail sound like a room instead of a metal box, and it
## is RESONANCE that moves it: a resonant space is bright and rings on, a dead one
## swallows the top of every repeat immediately. Same mapping both sides, so the
## Resonance slider colours the room the same way in either mode.
const DAMP_DEAD := 0.9
const DAMP_RING := 0.2
const DAMP_SCALE := 0.4          # Freeverb's scaledamp
## Wet and dry are taken from ONE "how much of the room reaches you" number, but
## they are not each other's complement: a room is added to the sound rather than
## swapped for it, so the dry only steps back part of the way. Full wet against
## fully removed dry is the preset sound this avoids.
const WET_GAIN := 0.8
const DRY_DUCK := 0.35

# --- Freeverb's tunings, at the rate they were published for --------------------
const COMB := [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
const ALLPASS := [556, 441, 341, 225]
const REF_RATE := 44100.0
## The comb bank sums eight lines each with a DC gain of 1/(1 - feedback), so at
## the top of the dial the network multiplies by ~50 before anything else happens.
## Freeverb's own input trim, chosen against the feedback range above rather than
## independently of it.
const FIXED_GAIN := 0.015
const ALLPASS_FB := 0.5

# --- the settings --------------------------------------------------------------

## How big the space is, on the same 0..[constant SIZE_MAX] dial Masking's slider
## uses. 1.0 is neutral.
var size := 1.0
## How much of that space reaches the listener, 0..1. Masking drives this from its
## own Ambience slider; a panel with one dial gets it from [method from_dial].
var wet := 0.0
## The tail's colour, 0..1, from whatever the mode calls Resonance. 0 is a dead
## room, 1 a ringing one.
var resonance := 0.5

var sample_rate := 22050

# --- the DSP half's state ------------------------------------------------------
#
# ONE flat buffer with per-line offsets rather than twelve arrays in an Array.
# Nested indexing (`_lines[k][i] = v`) on packed arrays goes through set_indexed
# and is markedly slower per sample, and this runs inside VoiceFX's per-sample
# loop over every sample of a whole chapter.
#
# The positions are ABSOLUTE indices into that buffer rather than per-line offsets,
# and the bound each wraps at is stored beside them, so the innermost loop reads two
# packed values per line instead of four. The start is only needed on the wrap, which
# happens once per line per delay.
var _buf := PackedFloat32Array()
var _off := PackedInt32Array()
var _end := PackedInt32Array()
var _len := PackedInt32Array()
var _pos := PackedInt32Array()
var _store := PackedFloat32Array()   # each comb's damping lowpass state
# Resolved from the settings whenever they move (the sliders are live), never per
# sample.
var _k_size := -1.0
var _k_wet := -1.0
var _k_res := -1.0
var _fb := 0.0
var _damp1 := 0.0
var _damp2 := 1.0
var _wet_g := 0.0
var _dry_g := 1.0


# --- the shared model ----------------------------------------------------------


## The dial, as the fraction Freeverb and Godot both call room_size.
func size01() -> float:
	return clampf(size / SIZE_MAX, 0.0, 1.0)


func wet_gain() -> float:
	return clampf(wet, 0.0, 1.0) * WET_GAIN


func dry_gain() -> float:
	return 1.0 - clampf(wet, 0.0, 1.0) * DRY_DUCK


func damping() -> float:
	return lerpf(DAMP_DEAD, DAMP_RING, clampf(resonance, 0.0, 1.0))


## Nothing of the room reaches the listener, so a caller may skip it entirely.
## Note this asks about WET and not about size: a big room nobody is standing in
## is silence, and evaluating a twelve-line network to hear silence is the cost
## this saves on every take that leaves the dial at zero.
##
## That cost is worth knowing, because the DSP half is GDScript in a per-sample
## loop: measured at 22050 Hz, the room adds about 6% of realtime on top of
## VoiceFX's own 2.5% - roughly a millisecond per frame while a reading plays, and
## about a minute of extra baking on a twenty-minute chapter. Only paid when the
## dial is up, which is why this gate exists rather than a room that always runs
## at whatever level.
func is_active() -> bool:
	return wet_gain() > 0.0005


## Collapse one 0..1 dial into a room. Used where a panel has a single Room
## slider (Generative), rather than Masking's separate size and Ambience.
##
## Size and wet move TOGETHER here, which Masking deliberately does not do - a big
## dry room and a small wet one are different sounds and one slider cannot say
## both. It is the right trade only because the mode that does this already has
## the other axis under a different name: Presence is distance from the source,
## which is exactly the question a separate wet control answers. So the two dials
## between them still span the same space, they just cut it differently.
func from_dial(k: float, res: float) -> void:
	var t := clampf(k, 0.0, 1.0)
	size = t * SIZE_MAX
	wet = pow(t, WET_TAPER)
	resonance = res


# --- renderer 1: Godot's own reverb, on a bus ----------------------------------


## Push the settings onto a live [AudioEffectReverb]. Every property is assigned,
## including the ones that do not move, so a bus rebuilt from stale state cannot
## keep a value nobody chose.
func to_reverb(fx: AudioEffectReverb) -> void:
	if fx == null:
		return
	fx.wet = wet_gain()
	fx.dry = dry_gain()
	fx.room_size = size01()
	fx.damping = damping()
	fx.spread = 1.0


# --- renderer 2: the same network, in GDScript ---------------------------------


func setup(sr: int) -> void:
	sample_rate = maxi(sr, 1)
	var scale := float(sample_rate) / REF_RATE
	var n := COMB.size() + ALLPASS.size()
	_off.resize(n)
	_end.resize(n)
	_len.resize(n)
	_pos.resize(n)
	_store.resize(COMB.size())
	_store.fill(0.0)
	var total := 0
	for i in n:
		var base: int = int(COMB[i]) if i < COMB.size() else int(ALLPASS[i - COMB.size()])
		var l := maxi(1, int(round(float(base) * scale)))
		_off[i] = total
		_len[i] = l
		_pos[i] = total
		total += l
		_end[i] = total
	_buf.resize(total)
	_buf.fill(0.0)
	_k_size = -1.0                 # force a resolve on the next sample


## Cheap and idempotent, like VoiceFX's echo resolve: the sliders are live, so
## this runs once per buffer and returns immediately unless something moved.
func _resolve() -> void:
	if is_equal_approx(size, _k_size) and is_equal_approx(wet, _k_wet) \
			and is_equal_approx(resonance, _k_res):
		return
	_k_size = size
	_k_wet = wet
	_k_res = resonance
	_fb = FB_MIN + FB_SPAN * size01()
	_damp1 = damping() * DAMP_SCALE
	_damp2 = 1.0 - _damp1
	_wet_g = wet_gain()
	_dry_g = dry_gain()


## One sample through the room. Inlined by the caller's loop rather than exposed
## as a buffer pass, because Generative's chain puts the room BETWEEN the echo and
## the presence filter - the same place the bus chain puts it - and a whole-buffer
## pass could not sit there without splitting that loop in three.
##
## Call [method prepare] once per buffer first: re-resolving the settings per
## sample would cost more than the network does, and they cannot move mid-buffer.
## The 8 and the 12 are the tuning tables' own sizes, read as literals rather than
## as COMB.size() because this is the innermost loop in the audio path.
func tick(x: float) -> float:
	if _buf.is_empty():
		setup(sample_rate)
		_resolve()
	var input := x * FIXED_GAIN
	var acc := 0.0
	for k in 8:
		var i: int = _pos[k]
		var y := _buf[i]
		var s: float = y * _damp2 + _store[k] * _damp1
		_store[k] = s
		_buf[i] = input + s * _fb
		i += 1
		_pos[k] = i if i < _end[k] else _off[k]
		acc += y
	for k in range(8, 12):
		var i: int = _pos[k]
		var y := _buf[i]
		_buf[i] = acc + y * ALLPASS_FB
		acc = y - acc
		i += 1
		_pos[k] = i if i < _end[k] else _off[k]
	return x * _dry_g + acc * _wet_g


## A whole buffer through the room, in place. For callers that want nothing but
## the room over some audio; VoiceFX uses [method tick] directly so the room keeps
## its place in the chain.
func process(buf: PackedFloat32Array) -> PackedFloat32Array:
	if _buf.is_empty():
		setup(sample_rate)
	prepare()
	if not is_active():
		return buf
	for i in buf.size():
		buf[i] = tick(buf[i])
	return buf


## Re-resolve the settings. Call once per buffer before a run of [method tick].
func prepare() -> void:
	_resolve()
