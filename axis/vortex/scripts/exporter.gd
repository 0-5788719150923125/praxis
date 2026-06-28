extends CanvasLayer
class_name Exporter

## Exporter - render the visualization to a video, in the background, in two steps.
##
## Persistent (main creates it once, never frees it), so an in-flight export and its
## status survive the song ending and the return to the home screen. In a song's
## window it shows an Export button; clicking it asks for a path and then:
##
##   1. BAKE (headless, no window): a one-shot `bake_runner` process analyzes the song
##      into a spectrum cache. This is the slow part, and it now runs windowless and
##      off the render's critical path - so there is no grey, frozen render window.
##      Cached per song, so only the first export of a song pays for it.
##   2. RENDER (Movie Maker): a second process loads that cache (`--bake-file`, no
##      in-process baking) and draws immediately, recording the visualization + audio
##      to the file. Its window is moved off-screen by main.
##
## Both steps are separate processes, polled by PID; status ("Baking… / Rendering… /
## Exported ✓") shows here in the main window. Nothing to watch, nothing to force-quit.

# Show the Export button after this many seconds of playback - no need to watch the
# whole thing. For songs too short to reach that, show it partway through instead
# (SHORT_FRACTION), so short clips still get a button.
const EXPORT_DELAY := 30.0
const SHORT_FRACTION := 0.5
const Bake := preload("res://scripts/bake.gd")

var _btn: Button
var _status: Label
var _dialog: FileDialog
var _state := "idle"     # idle | baking | rendering | done
var _bake_pid := -1
var _render_pid := -1
var _out := ""
var _song := ""
var _cache := ""
var _done_t := 0.0
var _pct := 0            # last progress read from the render/bake process


func _ready() -> void:
	layer = 250          # above the splash (200), so status shows on the home screen too
	_build_ui()


func _build_ui() -> void:
	_btn = Button.new()
	_btn.text = "⤓  Export video"
	_btn.tooltip_text = "Render this visualization + audio to a video file (in the background)"
	_btn.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	_btn.offset_left = -210
	_btn.offset_top = -72
	_btn.offset_right = -28
	_btn.offset_bottom = -28
	_btn.visible = false
	_btn.modulate.a = 0.0       # fade in elegantly when it becomes eligible
	_btn.pressed.connect(_on_export)
	add_child(_btn)

	_status = Label.new()
	_status.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	_status.offset_left = -560
	_status.offset_top = -64
	_status.offset_right = -28
	_status.offset_bottom = -28
	_status.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	_status.add_theme_color_override("font_shadow_color", Color(0, 0, 0, 0.7))
	_status.add_theme_constant_override("shadow_offset_x", 1)
	_status.add_theme_constant_override("shadow_offset_y", 1)
	_status.visible = false
	add_child(_status)

	_dialog = FileDialog.new()
	_dialog.file_mode = FileDialog.FILE_MODE_SAVE_FILE
	_dialog.access = FileDialog.ACCESS_FILESYSTEM
	_dialog.use_native_dialog = true
	_dialog.title = "Export video"
	_dialog.filters = PackedStringArray(["*.avi ; Video (AVI, MJPEG + audio)"])
	_dialog.current_file = "vortex.avi"
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
				_set_status("⏺  Rendering %s …  %d%%" % [_out.get_file(), _pct], Color(0.95, 0.92, 0.7))
			else:
				_state = "done"
				_done_t = 24.0
				_set_status("✓  Exported  %s" % _out.get_file(), Color(0.82, 0.95, 0.86))
		"done":
			_done_t -= dt
			if _done_t <= 0.0:
				_status.visible = false
				_state = "idle"
	# The button fades in once eligible (idle, not mid-export, past the delay) and
	# fades out otherwise - never a hard pop.
	var want := _state == "idle" and _can_export()
	_btn.modulate.a = lerpf(_btn.modulate.a, 1.0 if want else 0.0, 1.0 - exp(-6.0 * dt))
	_btn.visible = _btn.modulate.a > 0.02


# Eligible once playback passes the delay (30s), or partway through a song too short
# to reach it.
func _can_export() -> bool:
	var length := Spectrum.song_length()
	if length <= 0.0:
		return false
	return Spectrum.current.time >= minf(EXPORT_DELAY, length * SHORT_FRACTION)


# Refresh the cached percentage from the worker process (ignore mid-write misreads).
func _poll_pct() -> void:
	var p := Bake.read_progress()
	if p >= 0.0:
		_pct = int(round(p * 100.0))


func _on_export() -> void:
	_dialog.popup_centered()


func _on_path(out_path: String) -> void:
	_song = Spectrum.audio_path()
	if _song.is_empty():
		_fail("⚠  No song to export")
		return
	_out = out_path
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
		print("vortex: analyzing audio (pid ", _bake_pid, ") -> ", _cache)
	else:
		_fail("⚠  Could not start the analysis process")


# Step 2: Movie Maker render that loads the cache (--bake-file) and draws at once.
func _start_render() -> void:
	_pct = 0
	Bake.write_progress(0.0)
	var exe := OS.get_executable_path()
	var project := ProjectSettings.globalize_path("res://")
	var args := PackedStringArray([
		"--path", project, "--write-movie", _out, "--fixed-fps", "60",
		"--", "--export", "--bake-file", _cache, "--seed", str(Director.session_seed()),
		"--audio", _song])
	if Director.is_manual():
		args.append("--runbook")
		args.append(Director.runbook_name())
	_render_pid = OS.create_process(exe, args)
	if _render_pid > 0:
		_state = "rendering"
		print("vortex: rendering (pid ", _render_pid, ") -> ", _out)
	else:
		_fail("⚠  Could not start the render process")


func _fail(msg: String) -> void:
	_state = "done"
	_done_t = 8.0
	_set_status(msg, Color(1.0, 0.7, 0.6))
	push_warning("vortex export: " + msg)


func _set_status(text: String, color: Color) -> void:
	_status.text = text
	_status.add_theme_color_override("font_color", color)
	_status.visible = true
