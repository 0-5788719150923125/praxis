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
var _sig: HarmonicSignature = null   # rolling perceptual harmonic descriptor + content seed

## Emitted when a loaded song reaches its end (not in idle mode). main listens to
## return to the splash. Looping streams never end, so this never fires for them.
signal song_finished

var _player: AudioStreamPlayer
var _analyzer: AudioEffectSpectrumAnalyzerInstance
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
	_has_audio = false
	_idle_time = 0.0
	_override_path = ""
	song_hash = 0
	current = AudioFeatures.new()


func _on_player_finished() -> void:
	if _has_audio:
		song_finished.emit()


## The filesystem path of the song actually loaded (or "" if idle). The exporter
## re-passes it to the Movie Maker render so it renders the same track.
func audio_path() -> String:
	return _loaded_path


## Length of the loaded song in seconds (0 when idle / unknown). The exporter uses
## it with the playback position ([member current].time) to know it is near the end.
func song_length() -> float:
	if _has_audio and _player != null and _player.stream != null:
		return _player.stream.get_length()
	return 0.0


## The current perceptual harmonic descriptor (12 chroma + coarse shape, normalised). For
## SMOOTH content-driven modulation of a scene's dynamics. Empty until the analyzer is up.
func harmonic_signature() -> PackedFloat32Array:
	return _sig.vector() if _sig != null else PackedFloat32Array()

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


# Install the analyzer on the Master bus and grab its instance.
func _setup_analyzer() -> void:
	var bus := AudioServer.get_bus_index("Master")
	var fx := AudioEffectSpectrumAnalyzer.new()
	fx.buffer_length = 0.1   # short window - tighter reaction to transients
	AudioServer.add_bus_effect(bus, fx)
	var idx := AudioServer.get_bus_effect_count(bus) - 1
	_analyzer = AudioServer.get_bus_effect_instance(bus, idx)


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


func _process(delta: float) -> void:
	var f := AudioFeatures.new()

	if _has_audio and _player.playing:
		f.time = _player.get_playback_position()
		if _baked:
			_fill_bands_baked(f)
		else:
			_fill_bands(f)
	else:
		_idle_time += delta
		f.time = _idle_time
		# bands stay zero; scenes idle-animate on f.time

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
