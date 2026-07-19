extends Node
class_name VoiceStream

## VoiceStream - real-time speech on its own thread: render lag cannot break it.
##
## Synthesis and audio delivery run on a **worker thread**, fully decoupled from
## the render loop: a heavy scene can drop frames and the voice keeps flowing,
## because the thread keeps a sliding lead (~[constant TARGET_LEAD]s) of PCM
## synthesized ahead of the playhead and pushed into the
## [AudioStreamGeneratorPlayback] that [Spectrum] opened on the analyzed bus
## (the generator's push API is designed for exactly this threaded use). The
## main thread only drains word-timing snapshots for the karaoke and relays
## completion signals - a few microseconds a frame.
##
## The stream is a **live instrument**, not a render job:
## - [method retune] swaps the voice spec mid-stream (an atomic reference swap;
##   the worker reads it per chunk), so timbre traits bend the voice WHILE it
##   speaks, landing ~[constant TARGET_LEAD]s later - no restart.
## - [method restart] replaces the content in place (new text or a re-planned
##   voice): worker joined, generator buffer cleared, timing rebased via
##   [member time_base] - the session, scene, and Director stay untouched.
## The session seed comes from the FIRST content's fingerprint; the take's WAV
## + sidecar are written when synthesis completes (the WAV is exactly what was
## heard, retunes included), and finished takes loop endlessly on the thread.

signal completed(dur: float, wav_path: String)
signal restarted(base: float)

const PREBUFFER := 0.3              # synthesized + pushed before the thread takes over
# Deep lead: the ring holds seconds of finished audio, so even a machine
# saturated by a heavy scene (which can preempt this worker AND the audio
# callback) has a fat cushion. Nothing needs low mid-stream latency anymore -
# the sliders that once retuned live are gone; the loop is throw/restart.
const TARGET_LEAD := 2.5
const SEG_CHUNK := 4                # segments per synth() call on the worker

var words: Array = []               # main-thread copy for Subtitles (shared by reference)
var events: Array = []              # planned strike times [{t, kind, a}] - the bites
var take_base := ""                 # user://synth/take_<id> (wav + json on completion)
var time_base := 0.0                # playback time at the current content's start

var _spec: Voice.Spec               # swapped whole by retune() - worker reads per chunk
var _segs: Array = []
var _state := {}
var _next_seg := 0
var _pcm := PackedFloat32Array()
var _pushed := 0                    # frames handed to the generator (worker-owned)
var _playback: AudioStreamGeneratorPlayback
var _thread: Thread
var _mutex := Mutex.new()
var _cancel := false
var _words_snapshot: Array = []     # worker-written duplicates, mutex-guarded
var _words_dirty := false
var _done := false
var _finish_info: Array = []        # [dur, wav_path] once, mutex-guarded
var _emitted_done := false
var _underruns := 0                 # telemetry: times the ring went dry (see worker)


func setup(text: String, spec: Voice.Spec, base: String) -> void:
	_spec = spec
	take_base = base
	events.clear()
	_segs = Voice.plan(text, spec, events)
	_state = Voice.synth_state(spec)


## The session seed source: same text + same trait vector = the same show.
func fingerprint() -> int:
	var trait_sig := str(_spec.reading) + str(_spec.influences) + ";"
	for key in Voice.TRAIT_KEYS:
		trait_sig += "%s=%.3f;" % [key, float(_spec.traits.get(key, 0.0))]
	return hash(trait_sig + JSON.stringify(_segs.size()) + str(_segs))


## Synthesize and push the prebuffer synchronously (a few ms), so the caller can
## start the player in this same frame (pushed sample N plays at N / SR), then
## hand production to the worker thread.
func attach_playback(pb: AudioStreamGeneratorPlayback) -> void:
	_playback = pb
	_synth_chunks(int(PREBUFFER * Voice.SR))
	_push_available()
	_snapshot_words()
	_start_worker()


## Swap the voice mid-stream (timbre traits: pitch, tract, breath, grit). An
## atomic reference swap read by the worker at its next chunk - the bend lands
## about TARGET_LEAD seconds after the gesture. Plan-baked traits (pace, drawl,
## lilt) and text changes need restart() instead.
func retune(spec: Voice.Spec) -> void:
	_spec = spec


## Replace the content in place: new plan, cleared generator buffer, timing
## rebased - the session and scene continue, only the voice's content changes.
func restart(text: String, spec: Voice.Spec) -> void:
	_stop_worker()
	_spec = spec
	events.clear()
	_segs = Voice.plan(text, spec, events)
	_state = Voice.synth_state(spec)
	_next_seg = 0
	_pcm = PackedFloat32Array()
	_pushed = 0
	_done = false
	_emitted_done = false
	_finish_info = []
	_words_snapshot = []
	words.clear()                    # same Array instance Subtitles holds
	if _playback != null:
		# a generator's ring can't be cleared while active: cycle the player
		# instead, which rebases playback time to 0 and yields a fresh playback
		_playback = Spectrum.restart_stream()
	time_base = 0.0
	_synth_chunks(int(PREBUFFER * Voice.SR))
	_push_available()
	_snapshot_words()
	restarted.emit(time_base)
	_start_worker()


## Main thread, per frame: drain the worker's word snapshots into the shared
## array and relay the completion signal. Deliberately tiny.
func _process(_delta: float) -> void:
	if _words_dirty:
		_mutex.lock()
		words.clear()
		words.append_array(_words_snapshot)
		_words_dirty = false
		_mutex.unlock()
	if not _emitted_done:
		_mutex.lock()
		var info := _finish_info
		_mutex.unlock()
		if info.size() == 2:
			_emitted_done = true
			completed.emit(info[0], info[1])


func _exit_tree() -> void:
	_stop_worker()


# ---- worker ----------------------------------------------------------------


func _start_worker() -> void:
	_cancel = false
	_thread = Thread.new()
	# HIGH priority: when a heavy scene saturates every core, the voice must
	# win the scheduler - the render is ALLOWED to lag; the audio is not.
	_thread.start(_worker_loop, Thread.PRIORITY_HIGH)


func _stop_worker() -> void:
	if _thread != null:
		_cancel = true
		_thread.wait_to_finish()
		_thread = null
	_cancel = false


func _worker_loop() -> void:
	var started := false
	while not _cancel:
		var worked := false
		var queued := _generator_capacity() - int(_playback.get_frames_available())
		if _next_seg < _segs.size():
			var consumed := maxi(0, _pushed - queued)
			if float(_pcm.size() - consumed) / Voice.SR < TARGET_LEAD:
				_synth_one_chunk()
				_snapshot_words()
				worked = true
				if _next_seg >= _segs.size():
					_finish_take()
		# underrun telemetry: the generator ring going dry mid-take is the one
		# thing that must never happen silently - if you ever see this line,
		# the audio stuttered and we know exactly which layer failed
		if started and queued <= 0 and _pushed > 0:
			_underruns += 1
			if _underruns == 1 or _underruns % 50 == 0:
				print("ghost: VOICE UNDERRUN x%d (ring dry - report this)" % _underruns)
		if queued > 0:
			started = true
		_push_available()
		if not worked:
			OS.delay_msec(4)


func _synth_chunks(target_frames: int) -> void:
	while _next_seg < _segs.size() and _pcm.size() < target_frames:
		_synth_one_chunk()


func _synth_one_chunk() -> void:
	var j := mini(_next_seg + SEG_CHUNK, _segs.size())
	var result := Voice.synth(_segs, _spec, _next_seg, j, _state)
	_next_seg = j
	_pcm = result.pcm


func _snapshot_words() -> void:
	var live: Array = _state.words
	_mutex.lock()
	_words_snapshot = []
	for w in live:
		_words_snapshot.append((w as Dictionary).duplicate())
	_words_dirty = true
	_mutex.unlock()


## Push whatever the generator has room for: fresh frames while synthesizing,
## then loop passes over the finished take forever.
func _push_available() -> void:
	if _playback == null or _pcm.is_empty():
		return
	var room := int(_playback.get_frames_available())
	while room > 0:
		var src := _pushed if not _done else _pushed % _pcm.size()
		if not _done and src >= _pcm.size():
			break                    # synthesis hasn't caught up; next pass
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


func _generator_capacity() -> int:
	# frames in the generator's internal ring (buffer_length * mix rate);
	# frames_available counts FREE space, so capacity - available = queued
	return int(Spectrum.STREAM_BUFFER * Voice.SR)


## Worker-side: the take is fully synthesized. Write the WAV + sidecar here
## (FileAccess is fine off-thread); the main thread emits `completed`.
func _finish_take() -> void:
	_done = true
	var dur := float(_pcm.size()) / Voice.SR
	DirAccess.make_dir_recursive_absolute("user://synth")
	var wav := Voice.write_wav(take_base + ".wav", _pcm)
	var side := FileAccess.open(take_base + ".json", FileAccess.WRITE)
	side.store_string(JSON.stringify({"words": _state.words}))
	side.close()
	_mutex.lock()
	_finish_info = [dur, wav]
	_mutex.unlock()
