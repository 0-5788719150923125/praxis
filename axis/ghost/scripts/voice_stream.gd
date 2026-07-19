extends Node
class_name VoiceStream

## VoiceStream - real-time speech: synthesis races just ahead of playback.
##
## The answer to "why wait for a render": [Voice] synthesizes ~30x faster than
## real time, so audio can start the moment a small lead exists and the rest is
## produced in a **sliding window** ahead of the playhead. Each frame the pump
## synthesizes (budgeted, so the live show never drops frames) until the
## buffered lead reaches its target, converts the fresh PCM, and pushes it into
## the [AudioStreamGeneratorPlayback] that [Spectrum] opened on the analyzed
## bus - so the scenes react to the voice as it is being made. The word timing
## map grows live for the karaoke, `completed` fires once the take is fully
## synthesized (WAV + sidecar written for the exporter), and after that the
## pump keeps re-pushing the finished PCM from the top: a streamed take is an
## **endless looping session** by construction, like manual mode.
##
## Onset alignment: the prebuffer is synthesized and pushed in the same frame
## the player starts, so pushed sample N plays at N / SR - the timing map and
## the audio cannot drift apart. Deterministic per (text, trait vector).

signal completed(dur: float, wav_path: String)

const PREBUFFER := 0.35             # seconds pushed before playback starts
const TARGET_LEAD := 1.2            # keep this much synthesized ahead of the playhead
const BUDGET_USEC := 6000           # per-frame synthesis budget (the show keeps its fps)
const SEG_CHUNK := 4                # segments per synth() call inside the budget loop

var words: Array = []               # live-growing [{text, t0, t1, sentence}] (shared with Subtitles)
var take_base := ""                 # user://synth/take_<id> (wav + json written on completion)

var _segs: Array = []
var _spec: Voice.Spec
var _state := {}
var _next_seg := 0
var _pcm := PackedFloat32Array()    # everything synthesized so far
var _pushed := 0                    # frames handed to the generator (includes loop passes)
var _playback: AudioStreamGeneratorPlayback
var _done := false                  # synthesis finished (looping continues)
var _wav_path := ""


func setup(text: String, spec: Voice.Spec, base: String) -> void:
	_spec = spec
	take_base = base
	_segs = Voice.plan(text, spec)
	_state = Voice.synth_state(spec)
	words = _state.words            # same Array instance the synthesizer appends to


## The session seed source: same text + same trait vector = the same show.
func fingerprint() -> int:
	var trait_sig := ""
	for key in Voice.TRAIT_KEYS:
		trait_sig += "%s=%.3f;" % [key, float(_spec.traits.get(key, 0.0))]
	return hash(trait_sig + JSON.stringify(_segs.size()) + str(_segs))


## Synthesize and push the prebuffer synchronously (a few ms of compute), so the
## caller can start the player and hand us its playback in the same frame.
func attach_playback(pb: AudioStreamGeneratorPlayback) -> void:
	_playback = pb
	_synth_until(int(PREBUFFER * Voice.SR), 10 * BUDGET_USEC)
	_push_available()


func _process(_delta: float) -> void:
	if _playback == null:
		return
	if not _done:
		# lead = synthesized audio not yet played. The generator's ring holds
		# (capacity - frames_available) queued frames; consumed = pushed - queued.
		var queued := _generator_capacity() - int(_playback.get_frames_available())
		var consumed := maxi(0, _pushed - queued)
		var lead := float(_pcm.size() - consumed) / Voice.SR
		if lead < TARGET_LEAD:
			_synth_until(_pcm.size() + int((TARGET_LEAD - lead) * Voice.SR), BUDGET_USEC)
	_push_available()


func _generator_capacity() -> int:
	# frames in the generator's internal ring (buffer_length * mix rate);
	# frames_available counts FREE space, so capacity - available = queued
	return int(Spectrum.STREAM_BUFFER * Voice.SR)


func _synth_until(target_frames: int, budget_usec: int) -> void:
	var t0 := Time.get_ticks_usec()
	while _next_seg < _segs.size() and _pcm.size() < target_frames \
			and Time.get_ticks_usec() - t0 < budget_usec:
		var j := mini(_next_seg + SEG_CHUNK, _segs.size())
		var result := Voice.synth(_segs, _spec, _next_seg, j, _state)
		_next_seg = j
		_pcm = result.pcm
	if _next_seg >= _segs.size() and not _done:
		_finish()


## Push whatever the generator has room for: fresh frames while synthesizing,
## then loop passes over the finished take forever.
func _push_available() -> void:
	if _playback == null or _pcm.is_empty():
		return
	var room := int(_playback.get_frames_available())
	while room > 0:
		var src := _pushed if not _done else _pushed % _pcm.size()
		if not _done and src >= _pcm.size():
			break                    # synthesis hasn't caught up; push again next frame
		var n := mini(room, _pcm.size() - src)
		if n <= 0:
			break
		var buf := PackedVector2Array()
		buf.resize(n)
		for i in n:
			var v := _pcm[src + i]
			buf[i] = Vector2(v, v)
		_playback.push_buffer(buf)
		_pushed += n
		room -= n


func _finish() -> void:
	_done = true
	var dur := float(_pcm.size()) / Voice.SR
	DirAccess.make_dir_recursive_absolute("user://synth")
	_wav_path = Voice.write_wav(take_base + ".wav", _pcm)
	var side := FileAccess.open(take_base + ".json", FileAccess.WRITE)
	side.store_string(JSON.stringify({"words": words}))
	side.close()
	completed.emit(dur, _wav_path)


## Duration so far (final once `completed` has fired).
func duration() -> float:
	return float(_pcm.size()) / Voice.SR
