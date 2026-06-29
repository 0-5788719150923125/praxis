extends RefCounted
class_name SpectrumBake

## SpectrumBake - offline analysis of a song into a spectrum timeline.
##
## The live [Spectrum] reads an analyzer on the audio bus; that only works while
## the song plays in real time, and it goes unreliable under Movie Maker's offline
## audio - which is why a recorded video can't trust it. This bakes the spectrum
## *ahead of time*, straight from the samples: decode the song to mono PCM, slide a
## windowed FFT across it, and aggregate each window into the same 64 log-spaced
## bands the analyzer produces. The result is a per-frame timeline the Spectrum can
## replay deterministically, identically every time, independent of playback - so a
## render driven from it is correct and reproducible. (Live sessions still use the
## analyzer; only the exporter bakes.)

const RATE := 44100      # samples/sec we decode to
const WIN := 2048        # FFT window (≈21 Hz bins - enough low-end for bass/beat)

## Cross-process progress channel: the bake and the render write their 0..1 progress
## here, and the exporter (a different process) reads it for the status notification.
## They share the project's user:// dir, so this works across the separate processes.
const STATUS_PATH := "user://export_status.txt"


static func write_progress(frac: float) -> void:
	# Write to a temp file then rename: rename is atomic, so a reader in another
	# process never catches a half-written file (which errored get_as_text).
	var tmp := STATUS_PATH + ".tmp"
	var f := FileAccess.open(tmp, FileAccess.WRITE)
	if f == null:
		return
	f.store_string("%f" % clampf(frac, 0.0, 1.0))
	f.close()
	DirAccess.rename_absolute(
		ProjectSettings.globalize_path(tmp), ProjectSettings.globalize_path(STATUS_PATH))


# 0..1 progress, or -1 if absent/unreadable. Reads raw bytes (not get_as_text, which
# asserts on a length mismatch) so a read landing mid-write degrades to -1, not an error.
static func read_progress() -> float:
	if not FileAccess.file_exists(STATUS_PATH):
		return -1.0
	var f := FileAccess.open(STATUS_PATH, FileAccess.READ)
	if f == null:
		return -1.0
	var s := f.get_buffer(f.get_length()).get_string_from_utf8().strip_edges()
	f.close()
	return float(s) if s.is_valid_float() else -1.0


## Bake `song_path` into an Array of PackedFloat32Array(band_count) - one band frame
## per 1/fps second. Returns [] on failure (e.g. ffmpeg missing). Matches the live
## analyzer's band edges so the baked look tracks the live preview.
static func bake(song_path: String, fps: int, band_count: int,
		freq_min: float, freq_max: float, db_floor: float) -> Array:
	var samples := _load_samples(song_path)
	if samples.is_empty():
		return []

	var hann := _hann(WIN)
	# Precompute the FFT-bin span of each log band.
	var lo_bin := PackedInt32Array()
	var hi_bin := PackedInt32Array()
	lo_bin.resize(band_count)
	hi_bin.resize(band_count)
	var ratio := freq_max / freq_min
	for b in band_count:
		var f_lo := freq_min * pow(ratio, float(b) / band_count)
		var f_hi := freq_min * pow(ratio, float(b + 1) / band_count)
		lo_bin[b] = maxi(1, int(f_lo * WIN / RATE))
		hi_bin[b] = clampi(int(f_hi * WIN / RATE), lo_bin[b] + 1, WIN / 2)

	var frames := []
	var hop := int(RATE / fps)
	var n := samples.size()
	var total := maxi(1, int(n / hop))
	var pos := 0
	var next_log := 0.25
	print("ghost bake: analyzing %.1fs of audio (%d frames)…" % [float(n) / RATE, total])
	while pos < n:
		var done := float(frames.size()) / total
		if done >= next_log:
			print("ghost bake: %d%%" % int(next_log * 100))
			next_log += 0.25
		if frames.size() % 16 == 0:
			write_progress(done)
		var re := PackedFloat32Array()
		var im := PackedFloat32Array()
		re.resize(WIN)
		im.resize(WIN)
		for k in WIN:
			var s := pos + k - WIN / 2          # window centred on the frame time
			re[k] = (samples[s] if (s >= 0 and s < n) else 0.0) * hann[k]
			im[k] = 0.0
		_fft(re, im)

		var bands := PackedFloat32Array()
		bands.resize(band_count)
		for b in band_count:
			var sum := 0.0
			var c := 0
			for bin in range(lo_bin[b], hi_bin[b]):
				sum += sqrt(re[bin] * re[bin] + im[bin] * im[bin])
				c += 1
			var mag := (sum / float(maxi(1, c))) / float(WIN / 2)
			var db := linear_to_db(maxf(mag, 1e-9))
			bands[b] = clampf((db + db_floor) / db_floor, 0.0, 1.0)
		frames.append(bands)
		pos += hop
	write_progress(1.0)
	return frames


# Decode any supported audio to mono float samples at RATE via ffmpeg (already a
# soft dependency, used for FLAC). f32le → to_float32_array is a fast native read.
static func _load_samples(song_path: String) -> PackedFloat32Array:
	var src := song_path
	if src.begins_with("res://") or src.begins_with("user://"):
		src = ProjectSettings.globalize_path(src)
	var tmp := ProjectSettings.globalize_path("user://ghost_bake.f32")
	var out := []
	var code := OS.execute("ffmpeg",
		["-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", str(RATE), "-f", "f32le", tmp],
		out, true)
	if code != 0 or not FileAccess.file_exists(tmp):
		push_warning("ghost bake: ffmpeg could not decode %s (is ffmpeg on PATH?)" % src)
		return PackedFloat32Array()
	var f := FileAccess.open(tmp, FileAccess.READ)
	if f == null:
		return PackedFloat32Array()
	var bytes := f.get_buffer(f.get_length())
	f.close()
	return bytes.to_float32_array()


const CACHE_MAGIC := 0x42414B45     # "BAKE"


## Where the bake for a given song is cached - keyed by its path + byte size, so
## replacing the file invalidates it. Shared by the exporter (which writes it) and
## Spectrum (which reads it via --bake-file).
static func cache_path(song_path: String) -> String:
	var p := song_path
	if p.begins_with("res://") or p.begins_with("user://"):
		p = ProjectSettings.globalize_path(p)
	var sz := 0
	if FileAccess.file_exists(p):
		var fa := FileAccess.open(p, FileAccess.READ)
		if fa != null:
			sz = fa.get_length()
			fa.close()
	return "user://bake_%d.spec" % hash(p + "_" + str(sz))


## Write a baked timeline to a cache file (so a re-export of the same song skips the
## analysis). Keyed by the caller (see Spectrum) on the song's path + size.
static func save_cache(path: String, frames: Array, band_count: int) -> void:
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f == null:
		return
	f.store_32(CACHE_MAGIC)
	f.store_32(band_count)
	f.store_32(frames.size())
	for fr in frames:
		f.store_buffer((fr as PackedFloat32Array).to_byte_array())
	f.close()


## Load a cached timeline, or [] if absent / stale / mismatched band count.
static func load_cache(path: String, band_count: int) -> Array:
	if not FileAccess.file_exists(path):
		return []
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null or f.get_32() != CACHE_MAGIC or f.get_32() != band_count:
		if f != null:
			f.close()
		return []
	var count := f.get_32()
	var frames := []
	for i in count:
		frames.append(f.get_buffer(band_count * 4).to_float32_array())
	f.close()
	return frames


static func _hann(n: int) -> PackedFloat32Array:
	var w := PackedFloat32Array()
	w.resize(n)
	for i in n:
		w[i] = 0.5 - 0.5 * cos(TAU * float(i) / float(n - 1))
	return w


# In-place iterative radix-2 Cooley-Tukey FFT (n must be a power of two).
static func _fft(re: PackedFloat32Array, im: PackedFloat32Array) -> void:
	var n := re.size()
	# Bit-reversal permutation.
	var j := 0
	for i in range(1, n):
		var bit := n >> 1
		while j & bit:
			j ^= bit
			bit >>= 1
		j |= bit
		if i < j:
			var tr := re[i]; re[i] = re[j]; re[j] = tr
			var ti := im[i]; im[i] = im[j]; im[j] = ti
	# Butterflies.
	var length := 2
	while length <= n:
		var ang := -TAU / float(length)
		var wr := cos(ang)
		var wi := sin(ang)
		var i := 0
		while i < n:
			var cr := 1.0
			var ci := 0.0
			for k in range(length / 2):
				var a := i + k
				var b := i + k + length / 2
				var tr := cr * re[b] - ci * im[b]
				var ti := cr * im[b] + ci * re[b]
				re[b] = re[a] - tr
				im[b] = im[a] - ti
				re[a] += tr
				im[a] += ti
				var ncr := cr * wr - ci * wi
				ci = cr * wi + ci * wr
				cr = ncr
			i += length
		length <<= 1
