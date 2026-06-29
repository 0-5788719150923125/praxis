extends CanvasLayer
class_name Splash

## Splash - the start screen.
##
## The session's front door: import a song from disk, then pick how the show is
## driven by clicking one of two buttons (there is no separate "start" - the mode
## button *is* start):
##   Auto    - the Director picks scenes by novelty and cuts on the music (today's
##             visualizer, unchanged).
##   Manual  - open the workspace (a left-side storyboard panel over a running scene)
##             to orchestrate the show by hand. Scaffolding for now; see workspace.gd.
##
## The last imported song is remembered (user://ghost.cfg) and pre-selected on the
## next launch. Clicking a mode calls back into main as start_session.call(audio,
## manual: bool), then frees the splash. Built in code (no .tscn).

const CFG_PATH := "user://ghost.cfg"

## Set by main before the splash enters the tree: called as
## start_session.call(audio_path: String, manual: bool) when a mode is clicked.
var start_session: Callable

var _audio_path := ""
var _audio_label: Label
var _file_dialog: FileDialog


func _ready() -> void:
	layer = 200
	_load_last_song()
	_build_ui()


func _build_ui() -> void:
	var bg := ColorRect.new()
	bg.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	bg.color = Color(0.03, 0.03, 0.05, 1.0)
	add_child(bg)

	var center := CenterContainer.new()
	center.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(center)

	var col := VBoxContainer.new()
	col.alignment = BoxContainer.ALIGNMENT_CENTER
	col.add_theme_constant_override("separation", 16)
	col.custom_minimum_size = Vector2(560, 0)
	center.add_child(col)

	var title := Label.new()
	title.text = "ghost"
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	title.add_theme_font_size_override("font_size", 72)
	title.add_theme_color_override("font_color", Color(0.92, 0.95, 1.0))
	col.add_child(title)

	var sub := Label.new()
	sub.text = "a spectral music visualizer"
	sub.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	sub.add_theme_color_override("font_color", Color(0.55, 0.62, 0.75))
	col.add_child(sub)

	col.add_child(_spacer(12))

	# --- Song import (pre-filled with the remembered song) ---
	var song_row := HBoxContainer.new()
	song_row.alignment = BoxContainer.ALIGNMENT_CENTER
	song_row.add_theme_constant_override("separation", 12)
	col.add_child(song_row)

	var load_btn := Button.new()
	load_btn.text = "Import song…"
	load_btn.custom_minimum_size = Vector2(150, 40)
	load_btn.pressed.connect(_open_file_dialog)
	song_row.add_child(load_btn)

	_audio_label = Label.new()
	_audio_label.text = _audio_caption()
	_audio_label.add_theme_color_override("font_color", Color(0.7, 0.78, 0.9))
	song_row.add_child(_audio_label)

	col.add_child(_spacer(16))

	# --- Mode buttons (each one starts the session) ---
	var btn_row := HBoxContainer.new()
	btn_row.alignment = BoxContainer.ALIGNMENT_CENTER
	btn_row.add_theme_constant_override("separation", 20)
	col.add_child(btn_row)

	var auto_btn := Button.new()
	auto_btn.text = "Auto  ▶"
	auto_btn.tooltip_text = "Scenes chosen for you, cut on the music"
	auto_btn.custom_minimum_size = Vector2(190, 52)
	auto_btn.pressed.connect(_start_auto)
	btn_row.add_child(auto_btn)

	var manual_btn := Button.new()
	manual_btn.text = "Manual  ▶"
	manual_btn.tooltip_text = "Open the workspace to orchestrate scenes by hand"
	manual_btn.custom_minimum_size = Vector2(190, 52)
	manual_btn.pressed.connect(_start_manual)
	btn_row.add_child(manual_btn)

	var hint := Label.new()
	hint.text = "Space next · F11 fullscreen · ` feedback · Esc quit"
	hint.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	hint.add_theme_color_override("font_color", Color(0.4, 0.46, 0.56))
	col.add_child(hint)

	# Native file picker (falls back to Godot's built-in if no native dialog).
	_file_dialog = FileDialog.new()
	_file_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	_file_dialog.access = FileDialog.ACCESS_FILESYSTEM
	_file_dialog.use_native_dialog = true
	_file_dialog.filters = PackedStringArray([
		"*.wav, *.mp3, *.ogg, *.oga, *.flac ; Audio"])
	_file_dialog.size = Vector2i(800, 560)
	_file_dialog.file_selected.connect(_on_file_selected)
	add_child(_file_dialog)


func _spacer(h: int) -> Control:
	var s := Control.new()
	s.custom_minimum_size = Vector2(0, h)
	return s


func _audio_caption() -> String:
	if not _audio_path.is_empty():
		return "♪ " + _audio_path.get_file()
	if ResourceLoader.exists("res://audio/song.wav"):
		return "(using bundled audio/song.wav)"
	return "(no song — Auto will idle-animate)"


func _open_file_dialog() -> void:
	_file_dialog.popup_centered()


func _on_file_selected(path: String) -> void:
	_audio_path = path
	_audio_label.text = "♪ " + path.get_file()
	_save_last_song(path)


# --- Start (a mode button click) --------------------------------------------

func _start_auto() -> void:
	_start(false)


func _start_manual() -> void:
	_start(true)


func _start(manual: bool) -> void:
	if not _audio_path.is_empty():
		_save_last_song(_audio_path)
	if start_session.is_valid():
		start_session.call(_audio_path, manual)
	queue_free()


# --- Remembered song (user://ghost.cfg) ------------------------------------

func _load_last_song() -> void:
	var cfg := ConfigFile.new()
	if cfg.load(CFG_PATH) != OK:
		return
	var p := String(cfg.get_value("audio", "last", ""))
	if not p.is_empty() and FileAccess.file_exists(p):
		_audio_path = p     # still on disk - pre-select it


func _save_last_song(path: String) -> void:
	var cfg := ConfigFile.new()
	cfg.load(CFG_PATH)      # keep any other keys; ignore "missing file"
	cfg.set_value("audio", "last", path)
	cfg.save(CFG_PATH)
