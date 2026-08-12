extends Node

## Spectrum - the audio front end (autoload).
##
## Owns the [AudioStreamPlayer] and the [AudioEffectSpectrumAnalyzer] sitting on
## the Master bus. Every frame it samples the analyzer across a log-spaced set
## of frequency bands, packs the result into a typed [AudioFeatures], and stores
## it on [member current]. Scenes read [member current]; they never see the
## analyzer. This is the one place that knows audio exists.
##
## Two backends: the live analyzer (default, real-time, what you author in) and a
## baked timeline ([SpectrumBake], enabled with --use-bake) read from pre-computed
## frames for deterministic, analyzer-independent Movie Maker exports.

## Number of log-spaced bands in [member AudioFeatures.bands].
const BAND_COUNT := 64
const FREQ_MIN := 30.0
const FREQ_MAX := 16000.0

## dB window mapped onto 0..1. Magnitudes quieter than -DB_FLOOR read as 0.
const DB_FLOOR := 60.0

## Frames per second the offline bake is sampled at (must match bake_runner.gd).
const BAKE_FPS := 30
const Bake := preload("res://scripts/bake.gd")

## Named-band frequency splits (Hz), low edge inclusive.
const NAMED := {
	"bass": [30.0, 150.0],
	"low_mid": [150.0, 500.0],
	"mid": [500.0, 2000.0],
	"high": [2000.0, 6000.0],
	"treble": [6000.0, 16000.0],
}

## The latest frame. Read this from anywhere; it is replaced every frame.
var current: AudioFeatures = AudioFeatures.new()

## A stable hash of the loaded audio's path - scenes seed from this so the same
## song always renders the same video. 0 when nothing is loaded.
var song_hash: int = 0
var _sig: HarmonicSignature = null       # rolling perceptual harmonic descriptor + content seed
var _sig_fast: HarmonicSignature = null  # short-memory twin for the Echo re-localizer: recognizing
                                         # that the audio moved (a loop seam) must not wait out the
                                         # seconds of context the seeding descriptor integrates

## Emitted when a loaded song reaches its end (not in idle mode). main listens to
## return to the splash. Looping streams never end, so this never fires for them.
signal song_finished

var _player: AudioStreamPlayer
var _analyzer: AudioEffectSpectrumAnalyzerInstance
var _tap: AudioEffectCapture          # the LIVE_TAP recorder (see the const)
var _tap_ring := PackedFloat32Array()
var _tap_pos := 0
var _tap_filled := 0
## THE BOOKEND - held silence before the content and after it.
##
## A show used to begin on its first sample and end on its last, which reads as abrupt
## however gracefully the picture fades: [method Director._bookend_fade] eases the
## image up from black over the opening seconds, but those are the opening seconds OF
## THE NARRATION, so the first words are spoken behind a dark screen and the last are
## still landing as it goes out. Nothing was ever added to the timeline.
##
## These two add it. [member lead_in] holds playback for that many seconds while the
## session clock runs anyway; [member tail] keeps the session alive that long after the
## audio ends, before [signal song_finished] is emitted.
##
## THE CLOCK IS THE WHOLE DESIGN. [member current].time reads `_hold` during the hold,
## `lead_in + playback_position` while playing, and `lead_in + length + _tail_t` during
## the tail - ONE continuous timeline across all three. That is not a convenience; it is
## what makes everything downstream correct for free. The bookend fade, Echo's cursor,
## the subtitle time base and the exporter's progress all read this clock, and the
## obvious alternative - simply delaying `play()` - DOUBLE-FADES, because during the
## hold `time` would be the idle clock ramping the fade up, and then `play()` snaps it
## back to zero and it fades from black a second time.
##
## Why not simply write silence into the audio file, which sounds simpler: the session
## seed is derived from the audio's own fingerprint, so padding the file makes the same
## script render a DIFFERENT show, and it invalidates the spectrum bake cache. It also
## could not work for auto mode, where the song belongs to the user.
var lead_in := 0.0
var tail := 0.0

## Whether the loaded audio ALREADY CONTAINS the bookend silence.
##
## Two mechanisms are needed, because the two modes own their audio differently:
##
##   AUTO / MANUAL play a song that belongs to the user and must not be rewritten, so
##   the hold is performed here at playback time and the intro is genuinely silent.
##
##   SYNTHESIS renders its own take, so the silence is written INTO the PCM - and that
##   is strictly better there, because the ambience pad is a filter over the buffer:
##   given real samples to write into, it swells during the intro and the analyzer
##   HEARS it, so the scenes have something to move to instead of idling through a
##   dead five seconds. A held player cannot produce that; there is nothing to filter.
##
## When this is true the clock needs no offset (the file's own position already spans
## the bookend) and playback must not be held or deferred - but [member lead_in] and
## [member tail] are still set, because the fade windows are read from them.
var bookend_baked := false

## Seconds over which the AUDIO is ramped in and out, alongside the picture. Applied to
## the MASTER BUS rather than the player, deliberately: the analyzer is a bus EFFECT and
## runs BEFORE the bus's output volume, so a master ramp is heard and recorded without
## dimming what the scenes see. Fading the player instead would dim the visuals a second
## time on top of the bookend's own fade, and the picture would sink twice as fast as
## the sound.
var fade_audio := true

## SCRUB HOOKS, for a session whose audio is not a seekable file.
##
## The synthesis path pushes PCM into a generator a chunk at a time, so there is no file to
## seek and the player's own position is meaningless across a restart. But the editor
## driving it holds the decoded take and can re-push it from any offset, which is a
## perfectly good seek - it just cannot be expressed through an AudioStreamPlayer. A
## session that can do this registers these three; anything else leaves them empty and the
## file path above applies.
var scrub_pos := Callable()      # () -> float, seconds into the content
var scrub_len := Callable()      # () -> float, total seconds currently known
var scrub_seek := Callable()     # (float) -> void

var _hold := 0.0                # counts up through the lead-in
var _held := false              # lead-in running: the player has NOT been started yet
var _tail_t := 0.0              # counts up through the tail
var _tailing := false           # audio has ended; running out the tail before signalling
var _bus_db := 0.0              # master trim we applied, so it can be put back exactly
var _last_time := 0.0           # last published clock, held when the player stops mid-session

var _has_audio := false
var _idle_time := 0.0
var _override_path := ""        # song chosen on the splash; wins over CLI/default
var _loaded_path := ""          # the resolved path actually loaded (for re-launch/export)

# Baked backend: a pre-computed per-frame spectrum timeline (one PackedFloat32Array
# of BAND_COUNT per frame). Used by the export render (--use-bake) so a recorded
# video's reactivity is correct and analyzer-independent. Live sessions leave it off.
var _baked := false
var _baked_frames: Array = []

# Smoothing / beat state.
var _energy_avg := 0.0          # slow moving average, for onset comparison
var _beat := 0.0
var _onset_high := false        # tempo: were we above the onset threshold last frame (edge detect)
var _last_beat_t := -1.0        # tempo: playback time of the previous beat onset
var _beat_period := 0.5         # tempo: smoothed seconds between onsets (~120 BPM default)

# Spectral-flux / movement state (drives audio-triggered scene changes).
var _prev_bands := PackedFloat32Array()
var _flux_fast := 0.0           # short EMA of flux - the current "agitation"
var _flux_slow := 0.0           # long EMA - the passage's baseline

# Per-band smoothing (anti-jitter). Lerp factor toward the new value each frame.
const SMOOTH := 0.4
var _sm_bands := PackedFloat32Array()
var _sm_named := {}
var _band_lo := PackedFloat32Array()   # precomputed per-band edges
var _band_hi := PackedFloat32Array()


func _ready() -> void:
	_setup_analyzer()
	_precompute_bands()
	_player = AudioStreamPlayer.new()
	_player.bus = "Master"
	_player.finished.connect(_on_player_finished)
	add_child(_player)
	# Audio is no longer loaded here - main decides when (immediately on a direct
	# boot, or after the user picks a song on the splash). See begin().


## Load the chosen audio and start playback. Called once the session begins: with
## the splash's picked path, or with "" on a direct boot (which falls back to the
## --audio flag, then res://audio/song.wav). Idempotent enough to re-point the song.
func begin(path := "") -> void:
	if not path.is_empty():
		_override_path = path
	_load_audio()
	if _has_audio:
		var bake_file := _arg_value("--bake-file")
		if not bake_file.is_empty():
			# Pre-built cache (the export render's normal path): load it and start -
			# NO in-process baking, so the render never blocks on a grey frame.
			_baked_frames = Bake.load_cache(bake_file, BAND_COUNT)
			_baked = not _baked_frames.is_empty()
			if _baked:
				print("ghost: loaded bake file - %d frames" % _baked_frames.size())
			else:
				push_warning("ghost: bake file missing/empty: %s" % bake_file)
		elif OS.get_cmdline_user_args().has("--use-bake"):
			_bake()                              # direct-CLI fallback (bakes in-process)
		# A synthesis take writes its own bookend into its PCM (so the ambience pad has
		# real samples to swell into) and records the fact in its sidecar. Believe the
		# file over the request: holding playback for a file that already opens with
		# five seconds of silence would give ten.
		_adopt_baked_bookend()
		# The lead-in holds the FIRST SAMPLE, not the session: the clock below starts
		# running immediately, so scenes animate and the picture fades up through the
		# hold. _process starts playback when the hold is spent.
		_hold = 0.0
		_tail_t = 0.0
		_tailing = false
		_held = lead_in > 0.001 and not bookend_baked
		if not _held:
			_player.play()


# Pre-analyze the loaded song into a spectrum timeline (export render only). Blocks
# until done - fine for a non-interactive render, and Movie Maker only starts
# recording once it is ready. Cached per song, so a re-export skips the analysis.
func _bake() -> void:
	var cache := _bake_cache_path()
	_baked_frames = Bake.load_cache(cache, BAND_COUNT)
	if not _baked_frames.is_empty():
		print("ghost: loaded cached bake - %d frames" % _baked_frames.size())
	else:
		_baked_frames = Bake.bake(_loaded_path, BAKE_FPS, BAND_COUNT, FREQ_MIN, FREQ_MAX, DB_FLOOR)
		if not _baked_frames.is_empty():
			Bake.save_cache(cache, _baked_frames, BAND_COUNT)
			print("ghost: baked spectrum - %d frames (cached)" % _baked_frames.size())
	_baked = not _baked_frames.is_empty()
	if not _baked:
		push_warning("ghost: bake failed; the render will fall back to the live analyzer")


# Cache key: the song's path + byte size (so replacing the file invalidates it).
func _bake_cache_path() -> String:
	var p := _loaded_path
	if p.begins_with("res://") or p.begins_with("user://"):
		p = ProjectSettings.globalize_path(p)
	var sz := 0
	if FileAccess.file_exists(p):
		var fa := FileAccess.open(p, FileAccess.READ)
		if fa != null:
			sz = fa.get_length()
			fa.close()
	return "user://bake_%d.spec" % hash(p + "_" + str(sz))


## Generator ring size (seconds) for streamed sessions - see begin_stream().
const STREAM_BUFFER := 4.0

## LIVE TAP (diagnostic, 2026-07-26): record the MASTER BUS - everything the
## mixer actually hears (takes, loop seams, restarts, generator gaps) - into
## a rolling ring, written to user://synth/live_tap.wav on quit. Context:
## offline takes measure clean (silence floors below -70 dBFS, zero
## saturation) while the ear still hears noise bursts live, so the open
## question is WHERE between the synthesized block and the speaker the
## artifact enters. If this capture CONTAINS the bursts, they are inside the
## mix and analyzable; if it is clean while the ear hears them, the artifact
## is post-mix (audio driver / OS - e.g. PipeWire xruns under scene load).
## Flip off when the hunt is over. (Off since 2026-07-26 - the hunt ended:
## the noise was frication routing, voice_rca.md sections 11-16. Re-arm any
## time an ear/meter mismatch needs the actual mixer output on disk.)
const LIVE_TAP := false
const TAP_SECONDS := 120.0            # rolling window: the LAST two minutes

var _streaming := false
var _stream_length := 0.0


## Begin a STREAMED session: no file - the caller pushes PCM into the returned
## generator playback as it produces it ([VoiceStream]), and the analyzer bus
## hears it exactly like a song, so features/signature/scenes all just work.
## `fp` stands in for the file fingerprint as the session seed source (the
## caller derives it from its own content - e.g. text + voice traits - so the
## same input replays the same show). The audio path / length are unknown at
## start; the caller reports them via set_stream_info() once the take is done.
func begin_stream(fp: int, sr: int) -> AudioStreamGeneratorPlayback:
	stop()
	var gen := AudioStreamGenerator.new()
	gen.mix_rate = sr
	gen.buffer_length = STREAM_BUFFER
	_player.stream = gen
	song_hash = fp
	_loaded_path = ""
	_streaming = true
	_has_audio = true
	# play() first, then the caller pushes its prebuffer in this same frame -
	# so pushed sample N plays at N / sr and timing maps cannot drift.
	_player.play()
	return _player.get_stream_playback()


## The streamed take is fully synthesized: it now has a real file (for the
## exporter's relaunch) and a known length (for the exporter's progress math).
func set_stream_info(path: String, length: float) -> void:
	_loaded_path = path
	_stream_length = length


## True while the session is a streamed take (synthesis mode).
func is_streaming() -> bool:
	return _streaming


## Did audio actually load for this session? An export render checks this: a
## render with no audio never ends (song_finished cannot fire) and would
## record silence indefinitely.
func has_audio() -> bool:
	return _has_audio


## Fade the streaming player out over a breath (~two frames) so a restart's
## stop/play cycle never truncates the waveform mid-cycle - the raw cut WAS an
## audible pop on every throw and edit. restart_stream() restores the volume.
func fade_stream(dur := 0.04) -> void:
	if _player == null or not _streaming:
		return
	var tw := create_tween()
	tw.tween_property(_player, "volume_db", -60.0, dur)
	await tw.finished


## Cycle the streaming generator for an in-place content restart. A generator's
## ring buffer cannot be cleared while playback is active (clear_buffer errors),
## so the restart is a stop/play cycle: playback position rebases to 0 and a
## FRESH playback object comes back - the old one is dead, push to this one.
func restart_stream() -> AudioStreamGeneratorPlayback:
	_stream_length = 0.0
	_loaded_path = ""    # the old take's file no longer matches what is playing
	_player.stop()
	_player.volume_db = 0.0          # undo any fade_stream() cut
	_player.play()
	return _player.get_stream_playback()


## Can this session be scrubbed at all?
##
## A file can be seeked; a GENERATOR cannot. In synthesis the audio is being pushed into a
## ring buffer a chunk at a time and nothing behind the playhead still exists, so there is
## no position to seek to - which is also why the scrubber hides itself rather than
## offering a control that would silently do nothing.
func seekable() -> bool:
	if scrub_seek.is_valid():
		return true
	return _has_audio and not _streaming and _content_length() > 0.0


## Where the scrub bar should sit, and how long the thing is. A STREAM cannot answer
## either from the player: restarting a generator resets its playback position to zero, so
## after one seek the player's own clock says nothing about where the content is. The
## session that owns the audio answers instead when it can (see [member scrub_seek]).
func scrub_position() -> float:
	return float(scrub_pos.call()) if scrub_pos.is_valid() else current.time


func scrub_length() -> float:
	return float(scrub_len.call()) if scrub_len.is_valid() else song_length()


## Move the playhead to [param t] on the SESSION clock - the same clock
## [member current].time runs on, bookend included, so a caller can hand back a position
## it read from there without knowing whether the silence is held or baked.
##
## WHAT THIS DOES NOT DO is re-derive the visuals. Everything that READS the clock follows
## correctly and immediately: the baked spectrum is a timeline lookup, the karaoke reads
## `current.time - time_base`, the bookend fade is a function of position. But the
## [Director] is a SIMULATION, not a function of t - its scene choice, its hold schedule
## and its RNG stream evolve from the sequence of events it has actually seen. So after a
## seek the show carries on from the scene that is up, rather than jumping to the scene a
## from-the-start playthrough would have been showing. ([Echo] then pulls it back toward
## the content on its own, since it re-localizes against the harmonic signature rather
## than against elapsed time - but that is a drift back into alignment, not a guarantee.)
##
## Reproducing the exact visual state of an arbitrary t would mean replaying the Director
## from zero with drawing off. That is a real thing this architecture could do, and it is
## not what a scrub bar is for.
func seek(t: float) -> void:
	if scrub_seek.is_valid():
		scrub_seek.call(t)
		return
	if not seekable():
		return
	var content := _content_length()
	var want := clampf(t, 0.0, lead_in + content + tail)
	_tailing = false
	_tail_t = 0.0
	if not bookend_baked and want < lead_in:
		# Landed inside the held silence: hold there rather than starting the audio early.
		_held = true
		_hold = want
		_player.stop()
		return
	_held = false
	var pos := clampf(want - _clock_offset(), 0.0, maxf(0.0, content - 0.05))
	if not _player.playing:
		_player.play(pos)
	else:
		_player.seek(pos)


## Restart the loaded song from the top WITHOUT touching any session state - no
## reseed, no reload, the fingerprint and analyzers carry straight on. Manual mode
## loops the audio endlessly with this; whether the VISUALS restart is the
## storyboard's own business (its `loop` / `tail` fields), not the session's.
func replay() -> void:
	if _has_audio and _player != null and _player.stream != null:
		_player.play()


## Stop playback and reset to a clean, songless state, so the next begin() starts
## fresh. Called when a session ends (the song finished, or we returned home).
func stop() -> void:
	if _player != null:
		_player.stop()
		_player.volume_db = 0.0      # a mid-fade session end must not mute the next
	_has_audio = false
	_idle_time = 0.0
	_last_time = 0.0
	_override_path = ""
	_streaming = false
	_stream_length = 0.0
	song_hash = 0
	current = AudioFeatures.new()
	# Put the master back where we found it. The bookend trims a GLOBAL bus, so a
	# session that ended mid-fade would otherwise hand the next one a quiet mixer -
	# and on the home screen there is nothing playing to make that audible until it
	# is far too late to work out why.
	_held = false
	_tailing = false
	_hold = 0.0
	_tail_t = 0.0
	if absf(_bus_db) > 0.0001:
		_bus_db = 0.0
		AudioServer.set_bus_volume_db(0, 0.0)


func _on_player_finished() -> void:
	if not _has_audio:
		return
	# Run the tail out before telling anyone the song ended - the export quits on this
	# signal (main.gd), so the deferral IS the outro. Godot's movie writer records the
	# audio bus, so the held silence is captured with the picture and the two stay
	# locked without any container-level padding.
	if tail > 0.001 and not bookend_baked and not _tailing:
		_tailing = true
		_tail_t = 0.0
		return
	song_finished.emit()


## Read the loaded take's sidecar and adopt the bookend it was rendered with, if any.
##
## The sidecar is the same JSON the karaoke subtitles come from, so this costs one small
## file read on a path that already existed. A take that carries `bookend` was padded at
## render time and its word timings are ALREADY shifted to match, which is why this sets
## `bookend_baked` rather than trying to reconcile two offsets.
func _adopt_baked_bookend() -> void:
	bookend_baked = false
	if _loaded_path.is_empty():
		return
	var side := _loaded_path.get_basename() + ".json"
	if not FileAccess.file_exists(side):
		return
	var parsed = JSON.parse_string(FileAccess.get_file_as_string(side))
	if typeof(parsed) != TYPE_DICTIONARY:
		return
	var bk = (parsed as Dictionary).get("bookend")
	if typeof(bk) != TYPE_DICTIONARY:
		return
	lead_in = maxf(0.0, float((bk as Dictionary).get("in", 0.0)))
	tail = maxf(0.0, float((bk as Dictionary).get("out", 0.0)))
	bookend_baked = lead_in > 0.001 or tail > 0.001


## How far the session clock runs AHEAD of the player's own position. Zero when the
## bookend is baked into the file, because then the file's position already spans it.
func _clock_offset() -> float:
	return 0.0 if bookend_baked else lead_in


## The audio's own length, without the bookend - what the player is actually playing.
func _content_length() -> float:
	if _streaming:
		return _stream_length
	if _has_audio and _player != null and _player.stream != null:
		return _player.stream.get_length()
	return 0.0


## Ramp the master bus in over the lead-in and out over the tail, so the sound arrives
## and leaves with the picture. Equal-power rather than linear: a linear ramp on a
## sustained bed audibly dips through its middle, because perceived loudness follows
## roughly the square root of power.
func _apply_bookend_gain(t: float) -> void:
	if not _has_audio:
		return
	var g := 1.0
	if lead_in > 0.001:
		g = minf(g, clampf(t / lead_in, 0.0, 1.0))
	if tail > 0.001:
		var total := song_length()
		if total > 0.0:
			g = minf(g, clampf((total - t) / tail, 0.0, 1.0))
	var want := -80.0 if g <= 0.0005 else linear_to_db(sqrt(g))
	if absf(want - _bus_db) < 0.05:
		return
	_bus_db = want
	AudioServer.set_bus_volume_db(0, want)


## The filesystem path of the song actually loaded (or "" if idle). The exporter
## re-passes it to the Movie Maker render so it renders the same track.
func audio_path() -> String:
	return _loaded_path


## Length of the loaded song in seconds (0 when idle / unknown). The exporter uses
## it with the playback position ([member current].time) to know it is near the end.
func song_length() -> float:
	# The WHOLE timeline, bookend included, because that is the clock `current.time`
	# runs on. Everything that compares a position against this - the bookend fade,
	# Echo's arc roll, the exporter's percentage - would be wrong by the bookend
	# otherwise, and the fade in particular would go to black before the tail.
	var content := _content_length()
	if content <= 0.0:
		return 0.0                       # streaming and not yet measured: unknown
	if bookend_baked:
		return content                   # the file already spans the bookend
	return lead_in + content + tail


## The current perceptual harmonic descriptor (12 chroma + coarse shape, normalised). For
## SMOOTH content-driven modulation of a scene's dynamics. Empty until the analyzer is up.
func harmonic_signature() -> PackedFloat32Array:
	return _sig.vector() if _sig != null else PackedFloat32Array()

## The FAST descriptor (same shape, ~0.7s of context instead of 2.5s): for listeners that
## must notice a content change quickly - the [Echo] re-localizer - at the cost of jitter.
func harmonic_signature_fast() -> PackedFloat32Array:
	return _sig_fast.vector() if _sig_fast != null else PackedFloat32Array()

## A coarse content seed (SimHash bucket) from the harmonics RIGHT NOW - the same for the same
## music even re-encoded / cut up, drifting only as the content does. `bits` sets bucket width
## (fewer = wider/more robust). For DISCRETE choices (which scene / behavior).
func harmonic_bucket(bits := 10) -> int:
	return _sig.bucket(bits) if _sig != null else 0

## The full content seed from the live harmonics.
func harmonic_seed() -> int:
	return _sig.seed() if _sig != null else 0


## A live, harmonic-derived SEED BIAS, meant to be XOR-mixed into any seed expression - it does
## not REPLACE the existing seed (session identity, scene index, history all stay); it BIASES it,
## so the harmonic channels themselves continuously steer the sampled randomness everywhere this
## is threaded. Read it AT THE MOMENT a thing is instanced (it samples the current spectrum).
## Same music -> same bias trajectory -> same show; the bias is coarse + smoothed, so it survives
## re-encoding and a cut-out segment carries its own.
func seed_bias() -> int:
	if _sig == null:
		return 0
	return _sig.bucket(12) * 0x2545F4914F6CDD1D   # spread the coarse harmonic bucket across the bits


func _exit_tree() -> void:
	_write_tap()


## Dump the LIVE_TAP ring (oldest -> newest) as a WAV at the BUS mix rate -
## this is the actual mixer output, not the synthesized take.
func _write_tap() -> void:
	if _tap == null or _tap_filled == 0:
		return
	var rate := int(AudioServer.get_mix_rate())
	var n := _tap_filled
	var start := (_tap_pos - n + _tap_ring.size()) % _tap_ring.size()
	var bytes := PackedByteArray()
	bytes.resize(n * 2)
	for i in n:
		var v := clampf(_tap_ring[(start + i) % _tap_ring.size()], -1.0, 1.0)
		bytes.encode_s16(i * 2, int(v * 32767.0))
	DirAccess.make_dir_recursive_absolute("user://synth")
	var f := FileAccess.open("user://synth/live_tap.wav", FileAccess.WRITE)
	f.store_buffer("RIFF".to_ascii_buffer())
	f.store_32(36 + bytes.size())
	f.store_buffer("WAVE".to_ascii_buffer())
	f.store_buffer("fmt ".to_ascii_buffer())
	f.store_32(16)
	f.store_16(1)
	f.store_16(1)
	f.store_32(rate)
	f.store_32(rate * 2)
	f.store_16(2)
	f.store_16(16)
	f.store_buffer("data".to_ascii_buffer())
	f.store_32(bytes.size())
	f.store_buffer(bytes)
	f.close()
	print("ghost: live tap written -> %s (%.1fs @ %d Hz)" % [
		ProjectSettings.globalize_path("user://synth/live_tap.wav"),
		float(n) / float(rate), rate])


# Install the analyzer on the Master bus and grab its instance.
func _setup_analyzer() -> void:
	var bus := AudioServer.get_bus_index("Master")
	var fx := AudioEffectSpectrumAnalyzer.new()
	fx.buffer_length = 0.1   # short window - tighter reaction to transients
	AudioServer.add_bus_effect(bus, fx)
	var idx := AudioServer.get_bus_effect_count(bus) - 1
	_analyzer = AudioServer.get_bus_effect_instance(bus, idx)
	if LIVE_TAP:
		_tap = AudioEffectCapture.new()
		_tap.buffer_length = 0.5
		AudioServer.add_bus_effect(bus, _tap)
		_tap_ring.resize(int(TAP_SECONDS * AudioServer.get_mix_rate()))
		print("ghost: LIVE TAP armed - last %ds of the Master bus -> synth/live_tap.wav on quit (mix %d Hz, out latency %.1f ms, device '%s')"
			% [int(TAP_SECONDS), int(AudioServer.get_mix_rate()),
				AudioServer.get_output_latency() * 1000.0, AudioServer.get_output_device()])


# Log-spaced band edges, computed once.
func _precompute_bands() -> void:
	_band_lo.resize(BAND_COUNT)
	_band_hi.resize(BAND_COUNT)
	var ratio := FREQ_MAX / FREQ_MIN
	var centres := PackedFloat32Array()
	centres.resize(BAND_COUNT)
	for i in BAND_COUNT:
		_band_lo[i] = FREQ_MIN * pow(ratio, float(i) / float(BAND_COUNT))
		_band_hi[i] = FREQ_MIN * pow(ratio, float(i + 1) / float(BAND_COUNT))
		centres[i] = sqrt(_band_lo[i] * _band_hi[i])     # geometric centre (log-spaced)
	_sig = HarmonicSignature.new(centres)
	_sig_fast = HarmonicSignature.new(centres, 0.7)


func _process(delta: float) -> void:
	# drain the LIVE_TAP capture into the rolling ring (mono-averaged)
	if _tap != null:
		while _tap.get_frames_available() > 0:
			var got := _tap.get_buffer(mini(_tap.get_frames_available(), 4096))
			if got.is_empty():
				break
			for v in got:
				_tap_ring[_tap_pos] = (v.x + v.y) * 0.5
				_tap_pos = (_tap_pos + 1) % _tap_ring.size()
			_tap_filled = mini(_tap_filled + got.size(), _tap_ring.size())

	var f := AudioFeatures.new()

	# --- the bookend clock (see lead_in / tail) ---
	# Held silence counts as session time so the picture can fade up through it, but the
	# bands stay zero, which is what the intro is FOR: the scenes idle, the Director's
	# cut trigger sits under its silence floor and holds one shot, and nothing lurches.
	if _held:
		_hold += delta
		if _hold >= lead_in:
			_player.play()
			_held = false
	elif _tailing:
		_tail_t += delta
		if _tail_t >= tail:
			_tailing = false
			song_finished.emit()

	if _has_audio and _player.playing:
		f.time = _clock_offset() + _player.get_playback_position()
		if _baked:
			_fill_bands_baked(f)
		else:
			_fill_bands(f)
	elif _held or _tailing:
		# Bookend silence: the clock is continuous with the playing branch above, so
		# nothing downstream can tell the difference except that it is quiet.
		f.time = _hold if _held else lead_in + _content_length() + _tail_t
	elif _has_audio:
		# A LOADED SESSION NEVER FALLS BACK TO THE IDLE CLOCK. The player can be stopped
		# while audio is still loaded - it has run past the end, or a seek landed on the
		# last fraction of a second - and reverting to `_idle_time` there restarts the
		# session clock from zero, which reads downstream as the whole show jumping back
		# to its first frame. Caught by seek_check: a seek past the end reported 0.01 s.
		# Hold the last position instead; the tail and the finish signal own what happens
		# next, and neither of them wants the clock moving backwards first.
		f.time = _last_time
	else:
		_idle_time += delta
		f.time = _idle_time
		# bands stay zero; scenes idle-animate on f.time
	_last_time = f.time
	if fade_audio:
		_apply_bookend_gain(f.time)

	# Overall energy: mean of the spectrum, lightly smoothed.
	var sum := 0.0
	for v in f.bands:
		sum += v
	var raw_energy := sum / float(max(1, f.bands.size()))
	f.energy = raw_energy

	# Beat: pulse when energy jumps above its slow average.
	_energy_avg = lerpf(_energy_avg, raw_energy, 0.08)
	var onset := raw_energy > _energy_avg * 1.4 + 0.02
	# Tempo: on the RISING edge of an onset, measure the interval since the last one and fold it
	# into a smoothed beat period. Reject implausible gaps (~30..270 BPM) so double-triggers and
	# missed beats don't corrupt the estimate. This is the "how fast is the music" signal.
	if onset and not _onset_high:
		if _last_beat_t >= 0.0:
			var ibi := f.time - _last_beat_t
			if ibi > 0.22 and ibi < 2.0:
				_beat_period = lerpf(_beat_period, ibi, 0.2)
		_last_beat_t = f.time
	_onset_high = onset
	if onset:
		_beat = 1.0
	else:
		_beat = maxf(0.0, _beat - delta * 4.0)
	f.beat = _beat
	f.beat_period = _beat_period

	_compute_movement(f)
	current = f

	# Roll the perceptual harmonic descriptor (chroma + coarse shape) and its content seed. This
	# tracks WHAT the music is, robustly, so scenes can be seeded from the harmonics themselves
	# rather than the file - see HarmonicSignature / next/harmonic_seeding.md.
	if _sig != null:
		_sig.update(f.bands, f.bass + f.low_mid, f.mid, f.high + f.treble, f.flux, delta)
	if _sig_fast != null:
		_sig_fast.update(f.bands, f.bass + f.low_mid, f.mid, f.high + f.treble, f.flux, delta)


# Spectral flux + a sliding-window "movement" score. Flux is how much new
# frequency content arrived this frame; movement is the short-term flux measured
# against the passage's own baseline, so it spikes at section changes (a drop, a
# build, a new instrument) and stays low through a steady groove.
func _compute_movement(f: AudioFeatures) -> void:
	var flux := 0.0
	if _prev_bands.size() == f.bands.size() and f.bands.size() > 0:
		for i in f.bands.size():
			flux += maxf(0.0, f.bands[i] - _prev_bands[i])
		flux /= float(f.bands.size())
	_prev_bands = f.bands.duplicate()
	f.flux = flux

	_flux_fast = lerpf(_flux_fast, flux, 0.18)
	_flux_slow = lerpf(_flux_slow, flux, 0.012)
	# How far the recent agitation sits above the running baseline.
	var ratio := _flux_fast / (_flux_slow + 0.0008)
	f.movement = clampf((ratio - 1.3) * 0.7, 0.0, 1.0)


# Sample the analyzer into f.bands and the named convenience fields. The raw
# analyzer magnitudes jitter frame to frame, so each value is EMA-smoothed
# against the previous frame - this calms every scene at the source.
func _fill_bands(f: AudioFeatures) -> void:
	if _sm_bands.size() != BAND_COUNT:
		_sm_bands.resize(BAND_COUNT)
	f.bands.resize(BAND_COUNT)
	for i in BAND_COUNT:
		var raw := _band_energy(_band_lo[i], _band_hi[i])
		_sm_bands[i] = lerpf(_sm_bands[i], raw, SMOOTH)
		f.bands[i] = _sm_bands[i]
	f.bass = _smooth_named("bass", NAMED.bass)
	f.low_mid = _smooth_named("low_mid", NAMED.low_mid)
	f.mid = _smooth_named("mid", NAMED.mid)
	f.high = _smooth_named("high", NAMED.high)
	f.treble = _smooth_named("treble", NAMED.treble)


# Baked counterpart of _fill_bands: read the band frame at the current playback
# time from the timeline and apply the same EMA smoothing, so the baked replay
# tracks the live look. Named bands are aggregated from the 64 baked bands.
func _fill_bands_baked(f: AudioFeatures) -> void:
	if _sm_bands.size() != BAND_COUNT:
		_sm_bands.resize(BAND_COUNT)
	f.bands.resize(BAND_COUNT)
	var idx := clampi(int(f.time * BAKE_FPS), 0, _baked_frames.size() - 1)
	var raw: PackedFloat32Array = _baked_frames[idx]
	for i in BAND_COUNT:
		_sm_bands[i] = lerpf(_sm_bands[i], raw[i], SMOOTH)
		f.bands[i] = _sm_bands[i]
	f.bass = _named_baked("bass", raw, NAMED.bass)
	f.low_mid = _named_baked("low_mid", raw, NAMED.low_mid)
	f.mid = _named_baked("mid", raw, NAMED.mid)
	f.high = _named_baked("high", raw, NAMED.high)
	f.treble = _named_baked("treble", raw, NAMED.treble)


# A named band averaged from the baked log bands over its frequency range, smoothed.
func _named_baked(key: String, raw: PackedFloat32Array, pair: Array) -> float:
	var ratio := FREQ_MAX / FREQ_MIN
	var b0 := clampi(int(BAND_COUNT * log(float(pair[0]) / FREQ_MIN) / log(ratio)), 0, BAND_COUNT - 1)
	var b1 := clampi(int(BAND_COUNT * log(float(pair[1]) / FREQ_MIN) / log(ratio)), 0, BAND_COUNT - 1)
	var s := 0.0
	for b in range(b0, b1 + 1):
		s += raw[b]
	var rawv := s / float(maxi(1, b1 - b0 + 1))
	var prev: float = _sm_named.get(key, 0.0)
	var v := lerpf(prev, rawv, SMOOTH)
	_sm_named[key] = v
	return v


# A named band, EMA-smoothed like the spectrum.
func _smooth_named(key: String, pair: Array) -> float:
	var raw := _band_energy(pair[0], pair[1])
	var prev: float = _sm_named.get(key, 0.0)
	var v := lerpf(prev, raw, SMOOTH)
	_sm_named[key] = v
	return v


# One band: magnitude over a frequency range, mapped from dB to 0..1.
func _band_energy(lo: float, hi: float) -> float:
	var mag := _analyzer.get_magnitude_for_frequency_range(
		lo, hi, AudioEffectSpectrumAnalyzerInstance.MAGNITUDE_MAX)
	var db := linear_to_db(mag.length())
	return clampf((db + DB_FLOOR) / DB_FLOOR, 0.0, 1.0)


# Resolve a stream from `--audio <path>` or res://audio/song.wav and load it.
func _load_audio() -> void:
	var path := _audio_path_from_args()
	var stream: AudioStream = null

	if not path.is_empty():
		stream = _load_external(path)
		if stream == null:
			push_warning("ghost: could not load audio at %s" % path)
	if stream == null and ResourceLoader.exists("res://audio/song.wav"):
		path = "res://audio/song.wav"
		stream = load(path)

	if stream != null:
		_player.stream = stream
		_has_audio = true
		song_hash = _fingerprint(path)
		_loaded_path = path
	else:
		print("ghost: no audio loaded - scenes will idle-animate.")


# A content fingerprint of the audio file, so the seed is a true *file* match - the same
# sound yields the same show regardless of the file's name or location (rename-proof),
# which a path hash is not. Samples up to ~768 KB from the start / middle / end plus the
# byte length, rather than hashing a whole multi-MB file, which is plenty to distinguish
# tracks. Falls back to the path hash if the bytes can't be read. (Phase 1 of spectral
# determinism - exact file. A perceptual signature that also matches re-encodes / lossy
# copies is the planned phase 2; see the README roadmap.)
func _fingerprint(path: String) -> int:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return hash(path)
	var size := f.get_length()
	var chunk := 262144
	var acc := PackedByteArray()
	acc.append_array(f.get_buffer(chunk))                 # start
	if size > chunk * 2:
		f.seek(size / 2)
		acc.append_array(f.get_buffer(chunk))             # middle
	if size > chunk:
		f.seek(maxi(0, size - chunk))
		acc.append_array(f.get_buffer(chunk))             # end
	f.close()
	return hash(acc) ^ int(size * 0x9E3779B1)


# Value following a `--flag` in the user args, or "".
func _arg_value(flag: String) -> String:
	var args := OS.get_cmdline_user_args()
	for i in args.size():
		if args[i] == flag and i + 1 < args.size():
			return args[i + 1]
	return ""


func _audio_path_from_args() -> String:
	if not _override_path.is_empty():
		return _override_path
	var args := OS.get_cmdline_user_args()
	for i in args.size():
		if args[i] == "--audio" and i + 1 < args.size():
			return args[i + 1]
	return ""


# External (non-res://) files via the runtime loaders, by extension.
func _load_external(path: String) -> AudioStream:
	var lower := path.to_lower()
	if lower.ends_with(".wav"):
		return AudioStreamWAV.load_from_file(path)
	if lower.ends_with(".mp3"):
		return AudioStreamMP3.load_from_file(path)
	if lower.ends_with(".ogg") or lower.ends_with(".oga"):
		return AudioStreamOggVorbis.load_from_file(path)
	if lower.ends_with(".flac"):
		return _load_flac(path)
	if ResourceLoader.exists(path):
		return load(path)
	return null


# Godot 4.6 has no runtime FLAC loader (FLAC is editor-import only), so transcode
# to a temp WAV with ffmpeg. Degrades gracefully if ffmpeg isn't on PATH.
func _load_flac(path: String) -> AudioStream:
	var tmp := ProjectSettings.globalize_path("user://ghost_flac.wav")
	var code := OS.execute("ffmpeg", ["-y", "-loglevel", "error", "-i", path, tmp])
	if code == 0 and FileAccess.file_exists(tmp):
		return AudioStreamWAV.load_from_file(tmp)
	push_warning("ghost: FLAC playback needs ffmpeg on PATH to decode (%s)" % path)
	return null
