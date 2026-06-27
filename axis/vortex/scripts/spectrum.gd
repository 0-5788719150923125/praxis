extends Node

## Spectrum - the audio front end (autoload).
##
## Owns the [AudioStreamPlayer] and the [AudioEffectSpectrumAnalyzer] sitting on
## the Master bus. Every frame it samples the analyzer across a log-spaced set
## of frequency bands, packs the result into a typed [AudioFeatures], and stores
## it on [member current]. Scenes read [member current]; they never see the
## analyzer. This is the one place that knows audio exists.
##
## Today this is a live backend (reacts to playback). A baked backend - read
## from a pre-computed spectrum timeline for deterministic Movie Maker exports -
## is the planned sibling (see README).

## Number of log-spaced bands in [member AudioFeatures.bands].
const BAND_COUNT := 64
const FREQ_MIN := 30.0
const FREQ_MAX := 16000.0

## dB window mapped onto 0..1. Magnitudes quieter than -DB_FLOOR read as 0.
const DB_FLOOR := 60.0

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

var _player: AudioStreamPlayer
var _analyzer: AudioEffectSpectrumAnalyzerInstance
var _has_audio := false
var _idle_time := 0.0

# Smoothing / beat state.
var _energy_avg := 0.0          # slow moving average, for onset comparison
var _beat := 0.0

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
	add_child(_player)
	_load_audio()
	if _has_audio:
		_player.play()


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
	for i in BAND_COUNT:
		_band_lo[i] = FREQ_MIN * pow(ratio, float(i) / float(BAND_COUNT))
		_band_hi[i] = FREQ_MIN * pow(ratio, float(i + 1) / float(BAND_COUNT))


func _process(delta: float) -> void:
	var f := AudioFeatures.new()

	if _has_audio and _player.playing:
		f.time = _player.get_playback_position()
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
	if raw_energy > _energy_avg * 1.4 + 0.02:
		_beat = 1.0
	else:
		_beat = maxf(0.0, _beat - delta * 4.0)
	f.beat = _beat

	_compute_movement(f)
	current = f


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
			push_warning("vortex: could not load audio at %s" % path)
	if stream == null and ResourceLoader.exists("res://audio/song.wav"):
		path = "res://audio/song.wav"
		stream = load(path)

	if stream != null:
		_player.stream = stream
		_has_audio = true
		song_hash = hash(path)
	else:
		print("vortex: no audio loaded - scenes will idle-animate.")


func _audio_path_from_args() -> String:
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
	var tmp := ProjectSettings.globalize_path("user://vortex_flac.wav")
	var code := OS.execute("ffmpeg", ["-y", "-loglevel", "error", "-i", path, tmp])
	if code == 0 and FileAccess.file_exists(tmp):
		return AudioStreamWAV.load_from_file(tmp)
	push_warning("vortex: FLAC playback needs ffmpeg on PATH to decode (%s)" % path)
	return null
