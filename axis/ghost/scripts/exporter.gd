extends CanvasLayer
class_name Exporter

## Exporter - render the visualization to a video, in the background, in two steps.
##
## Persistent (main creates it once, never frees it), so an in-flight export and its
## status survive the song ending and the return to the home screen. In a song's
## window it shows an Export button; clicking it asks for a quality (720p@30 /
## 1080p@60 / 4K@60) and a path, then:
##
##   1. BAKE (headless, no window): a one-shot `bake_runner` process analyzes the song
##      into a spectrum cache. This is the slow part, and it now runs windowless and
##      off the render's critical path - so there is no grey, frozen render window.
##      Cached per song, so only the first export of a song pays for it.
##   2. RENDER (Movie Maker): a second process loads that cache (`--bake-file`, no
##      in-process baking) and draws immediately, recording the visualization + audio
##      to a scratch AVI. Its window is moved off-screen by main.
##   3. TRANSCODE (ffmpeg): the scratch AVI is re-encoded to the chosen MP4 (H.264 + AAC),
##      then deleted. Godot only writes AVI, and AVI is a 32-bit/RIFF container that
##      corrupts past ~4 GB (the 4K exports had a broken index + glitchy audio); the MP4
##      we ship uses 64-bit offsets, is ~10-20x smaller, and plays everywhere.
##
## All three steps are separate processes, polled by PID; status ("Analyzing… / Rendering… /
## Finalizing… / Saved ✓") shows here in the main window. Nothing to watch, nothing to force-quit.

const Bake := preload("res://scripts/bake.gd")

# Output quality presets, offered when the Export button is pressed. The render renders
# natively at this resolution (the viewport is resized in export mode) and Movie Maker
# records at this fps, so the file is exactly what is chosen here. 1080p@60 is the
# project's native size, so it is the default.
const QUALITIES := [
	{"label": "720p · 30 fps  (HD, smaller file)", "w": 1280, "h": 720, "fps": 30, "tag": "720p"},
	{"label": "1080p · 60 fps  (Full HD)", "w": 1920, "h": 1080, "fps": 60, "tag": "1080p"},
	{"label": "4K · 60 fps  (UHD, full resolution)", "w": 3840, "h": 2160, "fps": 60, "tag": "4k"},
]
const DEFAULT_QUALITY := 1

var _btn: Button
var _status: Label
var _dialog: FileDialog
var _quality_menu: PopupMenu
var _state := "idle"     # idle | baking | rendering | transcoding | done
var _bake_pid := -1
var _render_pid := -1
var _transcode_pid := -1
var _out := ""           # the final file the user chose (.mp4)
var _avi := ""           # the intermediate Movie Maker AVI (transcoded away, then deleted)
var _song := ""
var _cache := ""
var _done_t := 0.0
var _pct := 0            # last progress read from the render/bake process
var _song_dur := 0.0     # song length captured at export START (the live song may end mid-transcode)
var _quality: Dictionary = QUALITIES[DEFAULT_QUALITY]
var _announced := false  # one-shot console note the first time export becomes ready
var _note_t := 0.0       # countdown for self-clearing informational notes
var _prepping := false   # a provider take is rendering (async) - ignore re-clicks
var _stall_t := 0.0      # seconds since the render last showed ANY sign of life
var _stall_frac := -1.0  # high-water fractional progress the watchdog has seen
var _stall_size := 0     # high-water size of the movie file being written

# The watchdog exists for ONE failure: a render that can never finish (the
# audio failed to load, so the session has no end and Movie Maker records
# silence until the disk fills). It must never fire on a render that is merely
# SLOW - heavy scenes at 4K can spend minutes on a few seconds of video.
#
# The first version measured INTEGER PERCENT and killed a healthy 720p render
# of a 344 s take: one percent point is 3.4 s of video there, which a heavy
# scene can easily take longer than a minute to produce. Percent is far too
# coarse to mean "alive". Liveness is now two independent fine-grained
# signals - the FRACTIONAL playback position and the movie file GROWING on
# disk - and either one counts, with a much longer fuse.
const STALL_LIMIT := 300.0
const STALL_MIN_GROWTH := 65536   # bytes; below this the file is not really moving

## Synthesis hook: a mode whose audio is REPRODUCIBLE ON DEMAND (the voice is
## a pure function of the text and the seed) registers a provider that renders
## the current take to disk and returns its path. Export then works from the
## very first moment - the click IS the trigger that makes the audio.
var take_provider := Callable()

## Optional companion to [member take_provider]: returns true when the provider
## has something worth rendering (in synthesis: at least one seed on the belt,
## or a voice already cast). When it returns false the button greys out - a
## click could only produce a video of nothing.
var take_ready := Callable()


func _ready() -> void:
	layer = 250          # above the splash (200), so status shows on the home screen too
	_clear_override()    # remove a stale override.cfg left by a crashed/killed render
	_build_ui()


func _build_ui() -> void:
	_btn = Button.new()
	_btn.text = "⤓"                    # icon-only - matches assistant.gd's chat-bubble toggle
	_btn.tooltip_text = "Render this visualization + audio to a video file (in the background)"
	_btn.focus_mode = Control.FOCUS_NONE
	_btn.custom_minimum_size = Vector2(40, 40)
	_btn.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	# Same 40x40 box, same row (-28/-68), as assistant.gd's toggle, right of this one -
	# see that file's _TOGGLE_SIZE/_TOGGLE_ROW_BOTTOM doc for why the numbers match.
	_btn.offset_left = -112
	_btn.offset_top = -68
	_btn.offset_right = -72
	_btn.offset_bottom = -28
	_btn.visible = false
	_btn.modulate.a = 0.0       # fade in elegantly when it becomes eligible
	_btn.pressed.connect(_on_export)
	add_child(_btn)

	_status = Label.new()
	_status.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	# Shifted left in step with _btn above, so right-aligned text ends just clear of
	# the (now icon-sized) button instead of the old wide button's edge.
	_status.offset_left = -572
	_status.offset_top = -64
	_status.offset_right = -116
	_status.offset_bottom = -28
	_status.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	_status.add_theme_color_override("font_shadow_color", Color(0, 0, 0, 0.7))
	_status.add_theme_constant_override("shadow_offset_x", 1)
	_status.add_theme_constant_override("shadow_offset_y", 1)
	_status.visible = false
	add_child(_status)

	# Quality picker - shown first when Export is pressed, before the save dialog.
	_quality_menu = PopupMenu.new()
	for i in QUALITIES.size():
		_quality_menu.add_item(QUALITIES[i].label, i)
	_quality_menu.id_pressed.connect(_on_quality)
	add_child(_quality_menu)

	_dialog = FileDialog.new()
	_dialog.file_mode = FileDialog.FILE_MODE_SAVE_FILE
	_dialog.access = FileDialog.ACCESS_FILESYSTEM
	_dialog.use_native_dialog = true
	_dialog.title = "Export video"
	_dialog.filters = PackedStringArray(["*.mp4 ; Video (MP4, H.264 + AAC)"])
	_dialog.current_file = "ghost.mp4"
	# Default to the Downloads folder so an export lands somewhere predictable (the native dialog
	# otherwise opens in its last-used directory, which is easy to lose track of).
	var downloads := OS.get_system_dir(OS.SYSTEM_DIR_DOWNLOADS)
	if not downloads.is_empty():
		_dialog.current_dir = downloads
	_dialog.size = Vector2i(800, 560)
	_dialog.file_selected.connect(_on_path)
	add_child(_dialog)


func _process(dt: float) -> void:
	match _state:
		"baking":
			if OS.is_process_running(_bake_pid):
				_poll_pct()
				_set_status("⏳  Analyzing audio…  %d%%" % _pct, Color(0.95, 0.92, 0.7))
			elif FileAccess.file_exists(_cache):
				_start_render()                  # bake finished -> render from the cache
			else:
				_fail("⚠  Bake failed (is ffmpeg on PATH?)")
		"rendering":
			if OS.is_process_running(_render_pid):
				_poll_pct()
				# STALL WATCHDOG (see the consts): alive = the playback
				# position advanced at ALL, or the movie file grew. A slow
				# render satisfies both; only a render that cannot finish
				# satisfies neither.
				_stall_t += dt
				var frac := Bake.read_progress()
				if frac > _stall_frac:
					_stall_frac = frac
					_stall_t = 0.0
				var sz := _file_size(_avi)
				if sz > _stall_size + STALL_MIN_GROWTH:
					_stall_size = sz
					_stall_t = 0.0
				if _stall_t > STALL_LIMIT:
					OS.kill(_render_pid)
					_clear_override()
					DirAccess.remove_absolute(_avi)
					_fail("⚠  Render stalled at %d%% - frozen for %d min (see console)"
						% [_pct, int(STALL_LIMIT / 60.0)])
					push_warning("ghost export: render stalled (no frames, no progress); killed pid %d"
						% _render_pid)
				else:
					# one decimal, deliberately: whole percents on a long take
					# sit still for minutes on heavy scenes, which reads as a
					# freeze. A moving number is the difference between "slow"
					# and "hung" for the person watching it.
					_set_status("⏺  Rendering %s …  %.1f%%" % [
						_out.get_file(), maxf(_stall_frac, 0.0) * 100.0],
						Color(0.95, 0.92, 0.7))
			else:
				_clear_override()                # render finished -> restore live resolution
				# The render only reports success by PID exit; make sure it actually produced the AVI
				# (a crashed Movie Maker exits too) before spending minutes transcoding nothing.
				if FileAccess.file_exists(_avi) and _file_size(_avi) > 65536:
					_repair_avi_sizes(_avi)
					_start_transcode()
				else:
					_fail("⚠  Render produced no file (see console)")
		"transcoding":
			if OS.is_process_running(_transcode_pid):
				_set_status("⏳  Finalizing %s …  %d%%" % [_out.get_file(), _read_transcode_pct()], Color(0.95, 0.92, 0.7))
			elif FileAccess.file_exists(_out) and _file_size(_out) > 4096:
				DirAccess.remove_absolute(_avi)   # transcode ok -> drop the scratch AVI
				_state = "done"
				_done_t = 30.0
				_set_status("✓  Saved  %s" % _out, Color(0.82, 0.95, 0.86))
				print("ghost: exported -> ", _out)
			else:
				# Transcode failed (ffmpeg missing/errored). Keep the raw AVI so the work isn't lost.
				_state = "done"
				_done_t = 30.0
				_set_status("⚠  Transcode failed; raw file kept: %s" % _avi, Color(1.0, 0.7, 0.6))
				push_warning("ghost export: transcode failed; kept AVI at " + _avi)
		"done":
			_done_t -= dt
			if _done_t <= 0.0:
				_status.visible = false
				_state = "idle"
	# The button fades in once eligible (idle, not mid-export, past the delay) and
	# fades out otherwise - never a hard pop.
	# an informational note (e.g. "nothing to export yet") clears itself
	if _note_t > 0.0 and _state == "idle":
		_note_t -= dt
		if _note_t <= 0.0:
			_status.visible = false
	var want := _state == "idle" and _can_export()
	if want and not _announced:
		_announced = true
		print("ghost: export ready (⤓ bottom-right)")
	_btn.modulate.a = lerpf(_btn.modulate.a, 1.0 if want else 0.0, 1.0 - exp(-6.0 * dt))
	_btn.visible = _btn.modulate.a > 0.02
	# greyed, never gone: nothing to render yet (see _can_export's history)
	var content := _has_content()
	_btn.disabled = not content
	_btn.tooltip_text = ("Render this visualization + audio to a video file (in the background)"
		if content else
		"Nothing to render yet - catch a seed (or play a song) first")


# THE BUTTON IS ALWAYS VISIBLE. History, because this exact gate regressed
# repeatedly: eligibility was time-gated (30s of playback / half the song),
# then length-gated for streams, then time-gated again by a concurrent edit
# whose premise (that synthesis paces to real time) was wrong - and in
# synthesis sessions EVERY throw/edit resets the playback clock via the
# stream restart cycle, so a time gate can NEVER be satisfied while the user
# iterates. Each version HID the button from someone.
#
# So the rule is: never hide it, and never gate it on TIMING. It may only be
# DISABLED (greyed, still there, with a tooltip that says why) for the one
# honest reason - there is no content: no song loaded, and no provider that
# could make one (synthesis with an empty belt and nothing cast).
func _can_export() -> bool:
	return true


func _has_content() -> bool:
	if not Spectrum.audio_path().is_empty():
		return true
	if not take_provider.is_valid():
		return false
	return not take_ready.is_valid() or bool(take_ready.call())


# Refresh the cached percentage from the worker process (ignore mid-write misreads).
func _poll_pct() -> void:
	var p := Bake.read_progress()
	if p >= 0.0:
		_pct = int(round(p * 100.0))


# Step 0: pick the output resolution / fps. Pop the menu up by the button.
# THE PICKERS COME FIRST - the same order as every other session: quality,
# then path, then the background pipeline. A synthesis take that still needs
# rendering is made in _on_path, AFTER the user commits - doing it here made
# the click sit on "Rendering the take…" with no dialog in sight, which read
# as a button that never asks anything (and a cancel cost a wasted render).
# A click with nothing to export and no way to make it explains itself
# instead of opening a doomed dialog.
func _on_export() -> void:
	if _prepping:
		return
	_song = Spectrum.audio_path()
	if _song.is_empty() and not take_provider.is_valid():
		_note_t = 4.0
		_set_status("⚠  Nothing to export yet - play or speak something first",
			Color(1.0, 0.85, 0.6))
		return
	var btn_rect := _btn.get_global_rect()
	_quality_menu.reset_size()
	var pos := Vector2i(btn_rect.position) + Vector2i(0, -int(_quality_menu.get_contents_minimum_size().y) - 8)
	_quality_menu.position = pos
	_quality_menu.popup()


func _on_quality(id: int) -> void:
	_quality = QUALITIES[id]
	_dialog.current_file = "ghost_%s.mp4" % _quality.tag
	_dialog.popup_centered()


func _on_path(out_path: String) -> void:
	# RE-ENTRANCY GUARD, and it is load-bearing: this method awaits a take
	# render, and the native file dialog can deliver file_selected more than
	# once (and the user can re-export while a take is still rendering). Two
	# overlapping runs rendered the SAME take file from two threads - a reader
	# in another process (the export render itself) opened it mid-write and
	# saw a truncated WAV, which is how a render ended up recording silence
	# forever. write_wav is atomic now; this keeps the work from doubling too.
	if _prepping:
		return
	# _song was resolved at click time (_on_export) - the live audio path. A
	# synthesis session without a finished take renders one NOW, after the
	# quality and path are committed: the provider is a coroutine that runs
	# the synthesis on a worker thread (rendering it on the main thread froze
	# the window while the audio kept playing), and awaiting a coroutine
	# through a Callable is verified to work - the await is required.
	if _song.is_empty() and take_provider.is_valid():
		_prepping = true
		_set_status("⏳  Rendering the take…", Color(0.95, 0.92, 0.7))
		_song = String(await take_provider.call())
		_prepping = false
	if _song.is_empty():
		_song = Spectrum.audio_path()
	if _song.is_empty():
		_fail("⚠  No song to export")
		return
	_out = out_path
	if _out.get_extension().to_lower() != "mp4":
		_out += ".mp4"
	# Capture the duration NOW, while the song is loaded: the transcode (esp. 4K) runs for minutes, by
	# which point the live song may have ended/unloaded and Spectrum.song_length() would read 0.
	_song_dur = Spectrum.song_length()
	if _song_dur <= 0.0 and _song.get_extension().to_lower() == "wav":
		# a provider-rendered take Spectrum hasn't finished streaming: PCM16
		# mono WAV, so the file itself says how long it is
		_song_dur = maxf(0.0, float(_file_size(_song) - 44) / (2.0 * float(Voice.SR)))
	# Movie Maker records to this intermediate AVI (beside the final file, on the same disk); we then
	# transcode it to the chosen .mp4 and delete it. The AVI is only ever scratch - it never ships,
	# so its 4 GB/RIFF index limit (which corrupts 4K exports) can't reach the user.
	_avi = _out.get_basename() + ".render.avi"
	_cache = Bake.cache_path(_song)
	if FileAccess.file_exists(_cache):
		_start_render()                          # already analyzed -> straight to render
	else:
		_start_bake()


# Step 1: headless analysis (no window). Writes the spectrum cache, then exits.
func _start_bake() -> void:
	_pct = 0
	Bake.write_progress(0.0)
	var exe := OS.get_executable_path()
	var project := ProjectSettings.globalize_path("res://")
	_bake_pid = OS.create_process(exe, PackedStringArray([
		"--headless", "--path", project, "--script", "res://scripts/bake_runner.gd",
		"--", "--bake-song", _song, "--bake-out", _cache]))
	if _bake_pid > 0:
		_state = "baking"
		print("ghost: analyzing audio (pid ", _bake_pid, ") -> ", _cache)
	else:
		_fail("⚠  Could not start the analysis process")


# Step 2: Movie Maker render that loads the cache (--bake-file) and draws at once.
func _start_render() -> void:
	_pct = 0
	Bake.write_progress(0.0)
	# Movie Maker locks its output resolution to the project's viewport size at engine
	# startup, before any script runs - so the only way to drive it is override.cfg, which
	# Godot reads from the project root at boot. We write the chosen resolution there (in
	# "viewport" stretch mode, so the render is an offscreen buffer of exactly that size,
	# independent of the physical display - true 4K on a 1080p monitor). It is removed
	# when the render finishes, restoring the live window to its native canvas_items mode.
	_write_override(int(_quality.w), int(_quality.h))
	var exe := OS.get_executable_path()
	var project := ProjectSettings.globalize_path("res://")
	var args := PackedStringArray([
		"--path", project, "--write-movie", _avi, "--fixed-fps", str(_quality.fps),
		"--", "--export", "--bake-file", _cache, "--audio", _song])
	# THE SEED: pin it only when there is a live session to reproduce. Without
	# one (exporting a take straight from the belt - no cast, no show watched
	# yet) passing session_seed() would pin the render to 0, which is not a
	# session, just a constant. Omitting --seed lets the render resolve the
	# seed the way every songless boot does: from the AUDIO'S OWN fingerprint
	# (see Director._resolve_seed) - spectral determinism, so the same take
	# always renders the same show, and the show belongs to the voice.
	var seed_val := Director.session_seed()
	if seed_val != 0:
		args.append("--seed")
		args.append(str(seed_val))
	if Director.is_manual():
		args.append("--storyboard")
		args.append(Director.storyboard_source())   # the loadable name/path, NOT the display name
	_render_pid = OS.create_process(exe, args)
	if _render_pid > 0:
		_state = "rendering"
		_stall_t = 0.0
		_stall_frac = -1.0
		_stall_size = 0
		print("ghost: rendering %dx%d @ %d fps (pid %d) -> %s" % [
			_quality.w, _quality.h, _quality.fps, _render_pid, _avi])
	else:
		_clear_override()
		_fail("⚠  Could not start the render process")


# Step 3: transcode the scratch AVI into the chosen MP4 (H.264 + AAC) via ffmpeg. This is what the
# user actually keeps: MP4 uses 64-bit offsets so its index is valid at any size (Godot's AVI is
# 32-bit and corrupts past 4 GB, which is why 4K exports had a broken index and glitchy audio), and
# H.264 is ~10-20x smaller than the MJPEG intermediate. `-fflags +genpts` re-derives timestamps so a
# damaged AVI index is bypassed; audio is re-encoded from decoded PCM, so it comes out clean.
func _start_transcode() -> void:
	var dur := _song_dur
	_pct = 0                  # reset from the render's 100% so "Finalizing" starts fresh, not stuck full
	_progress_reset()
	var args := PackedStringArray([
		"-y", "-fflags", "+genpts", "-i", _avi,
		# `fast` preset: 4K@60 with `medium` was punishingly slow; `fast` roughly halves encode time for a
		# few % larger file - a good trade for a visualizer. crf 20 keeps it visually clean.
		"-c:v", "libx264", "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p",
		"-c:a", "aac", "-b:a", "192k",
		"-progress", ProjectSettings.globalize_path(_PROGRESS_FILE), "-nostats", "-loglevel", "error",
		_out])
	_transcode_pid = OS.create_process("ffmpeg", args)
	if _transcode_pid > 0:
		_state = "transcoding"
		print("ghost: transcoding (pid %d, %.0fs) %s -> %s" % [_transcode_pid, dur, _avi, _out])
	else:
		# No ffmpeg: we can't produce the MP4. Leave the raw AVI so the render isn't wasted.
		_state = "done"
		_done_t = 30.0
		_set_status("⚠  ffmpeg not found; raw file kept: %s" % _avi, Color(1.0, 0.7, 0.6))


const _PROGRESS_FILE := "user://transcode_progress.txt"


func _progress_reset() -> void:
	var f := FileAccess.open(_PROGRESS_FILE, FileAccess.WRITE)
	if f != null:
		f.store_string("")
		f.close()


# ffmpeg writes `out_time_us=<microseconds>` lines to the progress file; the fraction of the song's
# duration (captured at export start) it has reached is the transcode percent. Robust to partial/mid-
# write reads, and to the live song having ended (we use the captured `_song_dur`, not a live read).
func _read_transcode_pct() -> int:
	if _song_dur <= 0.0 or not FileAccess.file_exists(_PROGRESS_FILE):
		return _pct
	var text := FileAccess.get_file_as_string(_PROGRESS_FILE)
	var best := -1.0
	for line in text.split("\n"):
		if line.begins_with("out_time_us="):
			best = maxf(best, line.substr(12).to_float() / 1_000_000.0)
		elif line.begins_with("out_time_ms="):     # older ffmpeg (value is microseconds despite the name)
			best = maxf(best, line.substr(12).to_float() / 1_000_000.0)
	if best >= 0.0:
		_pct = clampi(int(round(best / _song_dur * 100.0)), 0, 99)
	return _pct


# Godot's AVI writer keeps 32-bit RIFF/LIST size fields, and a 4K render crosses
# 4 GiB in a few minutes of video - past that the written sizes WRAP (mod 2^32) and
# the container lies about where the frame data ends, even though every 00db/01wb
# chunk after it is written correctly all the way to EOF (verified by walking a 5 GB
# artifact chunk-by-chunk). Demuxers that trust those fields (players especially -
# their seeks also hit the equally-wrapped idx1 offsets) stall or repeat frames.
# The repair is two words: RIFF size and the movi LIST size become 0 - "size
# unknown, read to end of file" - turning any demux into a clean sequential walk of
# the intact chunks. No-op for files under 4 GiB (their sizes are correct).
func _repair_avi_sizes(path: String) -> void:
	var f := FileAccess.open(path, FileAccess.READ_WRITE)
	if f == null:
		return
	if f.get_length() < 4294967296:
		f.close()
		return
	f.seek(4)
	f.store_32(0)                       # RIFF size -> unknown
	var pos := 12
	for i in 64:                        # walk top-level chunks to the movi LIST
		f.seek(pos)
		var tag := f.get_buffer(4).get_string_from_ascii()
		var csize := f.get_32()
		if tag == "LIST" and f.get_buffer(4).get_string_from_ascii() == "movi":
			f.seek(pos + 4)
			f.store_32(0)               # movi size -> unknown
			print("ghost export: repaired wrapped >4GiB AVI sizes in ", path.get_file())
			break
		if csize <= 0 or tag.is_empty():
			break
		pos += 8 + csize + (csize & 1)
	f.close()


func _file_size(path: String) -> int:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return 0
	var s := f.get_length()
	f.close()
	return s


# override.cfg lives in the project root only for the duration of a render; Godot reads
# it at startup to override project.godot (here: the export's output resolution + stretch
# mode). _ready() also clears any stale copy left by a render that never exited cleanly.
func _override_path() -> String:
	return ProjectSettings.globalize_path("res://override.cfg")


func _write_override(w: int, h: int) -> void:
	var f := FileAccess.open(_override_path(), FileAccess.WRITE)
	if f == null:
		push_warning("ghost export: could not write override.cfg (resolution may default)")
		return
	# viewport_* set the RENDERED (recorded) resolution; window_*_override shrink the
	# OS window itself to an unobtrusive floater. In "viewport" stretch mode the two
	# are independent - true 4K on any monitor, tiny window. The window must stay a
	# normal, drawable window: minimizing it makes Godot skip rendering and the movie
	# records frozen frames (see boot.gd).
	f.store_string("[display]\n\nwindow/size/viewport_width=%d\nwindow/size/viewport_height=%d\nwindow/size/window_width_override=480\nwindow/size/window_height_override=270\nwindow/stretch/mode=\"viewport\"\n" % [w, h])
	f.close()


func _clear_override() -> void:
	var path := _override_path()
	if FileAccess.file_exists(path):
		DirAccess.remove_absolute(path)


func _fail(msg: String) -> void:
	_clear_override()
	_state = "done"
	_done_t = 8.0
	_set_status(msg, Color(1.0, 0.7, 0.6))
	push_warning("ghost export: " + msg)


func _set_status(text: String, color: Color) -> void:
	_status.text = text
	_status.add_theme_color_override("font_color", color)
	_status.visible = true
