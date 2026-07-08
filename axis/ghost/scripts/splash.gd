extends CanvasLayer
class_name Splash

## Splash - the start screen.
##
## The session's front door: import a clip from disk, then pick how the show is
## driven by clicking a mode button (there is no separate "start" - the mode button
## *is* start). One Import dialog accepts audio OR video - the extension it sees
## picks the KIND (see VIDEO_EXTS), and the kind picks which buttons are even shown:
##   audio kind -> Auto (the Director picks scenes, cuts on the music) and Manual
##                 (open the workspace to orchestrate by hand).
##   video kind -> Mask only (see mask_editor.gd: key two colors apart, place
##                 markers, export). Auto/Manual don't apply to a video import and
##                 Mask doesn't apply to an audio one, so only the relevant set ever
##                 shows - no dead buttons that error if clicked for the wrong kind.
##
## Both the audio and video path (and which kind was active) are remembered
## independently (user://ghost.cfg) and restored on the next launch. Clicking a mode
## calls back into main - start_session.call(audio, manual: bool) for Auto/Manual,
## start_mask.call(video_path) for Mask - then frees the splash. Built in code (no
## .tscn).

const CFG_PATH := "user://ghost.cfg"
const VIDEO_EXTS := ["mp4", "mov", "mkv", "webm", "avi"]

## Set by main before the splash enters the tree.
var start_session: Callable    # start_session.call(audio_path: String, manual: bool)
var start_mask: Callable       # start_mask.call(video_path: String)

var _audio_path := ""
var _video_path := ""
var _kind := "audio"           # "audio" or "video" - which import is active
var _caption: Label
var _audio_buttons: Control
var _mask_buttons: Control
var _file_dialog: FileDialog


func _ready() -> void:
	layer = 200
	_load_last_song()
	_load_last_video()
	_load_last_kind()
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

	# --- Import (audio OR video - the extension picks the slot; both remembered) ---
	var song_row := HBoxContainer.new()
	song_row.alignment = BoxContainer.ALIGNMENT_CENTER
	song_row.add_theme_constant_override("separation", 12)
	col.add_child(song_row)

	var load_btn := Button.new()
	load_btn.text = "Import…"
	load_btn.tooltip_text = "A song for Auto/Manual, or a video clip for Mask"
	load_btn.custom_minimum_size = Vector2(150, 40)
	load_btn.pressed.connect(_open_file_dialog)
	song_row.add_child(load_btn)

	_caption = Label.new()
	_caption.add_theme_color_override("font_color", Color(0.7, 0.78, 0.9))
	song_row.add_child(_caption)

	col.add_child(_spacer(16))

	# --- Mode buttons (each one starts the session) - only the group matching the
	# active import kind is ever visible, so there's never a button on screen that
	# doesn't apply to what's loaded (see _refresh_mode).
	var btn_slot := CenterContainer.new()
	col.add_child(btn_slot)

	_audio_buttons = HBoxContainer.new()
	_audio_buttons.alignment = BoxContainer.ALIGNMENT_CENTER
	_audio_buttons.add_theme_constant_override("separation", 20)
	btn_slot.add_child(_audio_buttons)

	var auto_btn := Button.new()
	auto_btn.text = "Auto  ▶"
	auto_btn.tooltip_text = "Scenes chosen for you, cut on the music"
	auto_btn.custom_minimum_size = Vector2(190, 52)
	auto_btn.pressed.connect(_start_auto)
	_audio_buttons.add_child(auto_btn)

	var manual_btn := Button.new()
	manual_btn.text = "Manual  ▶"
	manual_btn.tooltip_text = "Open the workspace to orchestrate scenes by hand"
	manual_btn.custom_minimum_size = Vector2(190, 52)
	manual_btn.pressed.connect(_start_manual)
	_audio_buttons.add_child(manual_btn)

	_mask_buttons = HBoxContainer.new()
	_mask_buttons.alignment = BoxContainer.ALIGNMENT_CENTER
	btn_slot.add_child(_mask_buttons)

	var mask_btn := Button.new()
	mask_btn.text = "Mask  ▶"
	mask_btn.tooltip_text = "Key two colors apart in an imported clip and place markers"
	mask_btn.custom_minimum_size = Vector2(190, 52)
	mask_btn.pressed.connect(_start_mask)
	_mask_buttons.add_child(mask_btn)

	_refresh_mode()

	var hint := Label.new()
	hint.text = "F11 fullscreen · ` feedback · Esc quit"
	hint.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	hint.add_theme_color_override("font_color", Color(0.4, 0.46, 0.56))
	col.add_child(hint)

	# Native file picker (falls back to Godot's built-in if no native dialog).
	_file_dialog = FileDialog.new()
	_file_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	_file_dialog.access = FileDialog.ACCESS_FILESYSTEM
	_file_dialog.use_native_dialog = true
	_file_dialog.filters = PackedStringArray([
		"*.wav, *.mp3, *.ogg, *.oga, *.flac, *.mp4, *.mov, *.mkv, *.webm, *.avi ; Audio or video"])
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


func _video_caption() -> String:
	if not _video_path.is_empty():
		return "🎬 " + _video_path.get_file()
	return "(no clip imported yet)"


## Show the caption and button group for the active kind; hide the other entirely -
## the point being there is never a visible button that doesn't apply to what's
## loaded (see class doc).
func _refresh_mode() -> void:
	_caption.text = _video_caption() if _kind == "video" else _audio_caption()
	_audio_buttons.visible = _kind == "audio"
	_mask_buttons.visible = _kind == "video"


func _open_file_dialog() -> void:
	_file_dialog.popup_centered()


func _on_file_selected(path: String) -> void:
	if VIDEO_EXTS.has(path.get_extension().to_lower()):
		_video_path = path
		_save_last_video(path)
		_kind = "video"
	else:
		_audio_path = path
		_save_last_song(path)
		_kind = "audio"
	_save_last_kind(_kind)
	_refresh_mode()


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


func _start_mask() -> void:
	if not _video_path.is_empty():
		_save_last_video(_video_path)
	if start_mask.is_valid():
		start_mask.call(_video_path)     # "" is fine - mask_editor prompts its own dialog
	queue_free()


# --- Remembered song / clip (user://ghost.cfg) ------------------------------

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


func _load_last_video() -> void:
	var cfg := ConfigFile.new()
	if cfg.load(CFG_PATH) != OK:
		return
	var p := String(cfg.get_value("video", "last", ""))
	if not p.is_empty() and FileAccess.file_exists(p):
		_video_path = p


func _save_last_video(path: String) -> void:
	var cfg := ConfigFile.new()
	cfg.load(CFG_PATH)
	cfg.set_value("video", "last", path)
	cfg.save(CFG_PATH)


func _load_last_kind() -> void:
	var cfg := ConfigFile.new()
	if cfg.load(CFG_PATH) != OK:
		return
	var k := String(cfg.get_value("ui", "kind", "audio"))
	# Only honor a remembered "video" kind if that clip is still actually on disk -
	# otherwise Mask would be the only button shown with nothing for it to open.
	if k == "video" and not _video_path.is_empty():
		_kind = "video"
	else:
		_kind = "audio"


func _save_last_kind(kind: String) -> void:
	var cfg := ConfigFile.new()
	cfg.load(CFG_PATH)
	cfg.set_value("ui", "kind", kind)
	cfg.save(CFG_PATH)
