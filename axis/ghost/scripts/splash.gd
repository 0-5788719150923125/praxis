extends CanvasLayer
class_name Splash

## Splash - the start screen: every mode, always visible.
##
## The front door lists all four instruments as a column of mode rows - name,
## one-line description, and what each consumes - so nothing is hidden behind
## an import kind (an earlier design showed only the buttons matching the last
## import, which read as "modes missing" once there were four):
##   Auto      - the seeded show; the Director picks scenes, cuts on the music.
##   Manual    - the workspace + storyboards; orchestrate the show by hand.
##   Synthesis - write a script, ghost speaks it; the show reacts to the voice.
##   Mask Lab  - chroma-key effects over an imported video clip.
## A mode button *is* start - there is no separate start button.
##
## One Import dialog accepts audio OR video; the extension routes it to the
## right slot (VIDEO_EXTS), both slots are remembered independently
## (user://ghost.cfg) and shown side by side. Auto/Manual use the song slot,
## Mask Lab uses the clip slot, Synthesis needs no import at all. Clicking a
## mode calls back into main (start_session / start_synth / start_mask), then
## frees the splash. Built in code (no .tscn).

const CFG_PATH := "user://ghost.cfg"
const VIDEO_EXTS := ["mp4", "mov", "mkv", "webm", "avi"]

## The assistant dropdown: display name -> the persisted key (see
## assistant_backend()). "" (Off) means no assistant at all - main.gd and
## mask_editor.gd both gate creating an Assistant node on this being non-empty.
## Only one real backend exists yet; the dropdown shape is here so a second one
## (a different CLI, an API-direct backend, whatever) is a new entry, not a
## redesign - see assistant.gd, which is NOT backend-pluggable internally yet
## because there is only one backend to plug.
const ASSISTANT_BACKENDS := ["Off", "Claude Code CLI"]
const ASSISTANT_KEYS := ["", "claude_cli"]

## Set by main before the splash enters the tree.
var start_session: Callable    # start_session.call(audio_path: String, manual: bool)
var start_mask: Callable       # start_mask.call(video_path: String)
var start_synth: Callable      # start_synth.call()

var _audio_path := ""
var _video_path := ""
var _assistant_backend := ""
var _caption: Label
var _file_dialog: FileDialog


func _ready() -> void:
	layer = 200
	_load_last_song()
	_load_last_video()
	_assistant_backend = assistant_backend()
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
	col.add_theme_constant_override("separation", 14)
	col.custom_minimum_size = Vector2(640, 0)
	center.add_child(col)

	var title := Label.new()
	title.text = "ghost"
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	title.add_theme_font_size_override("font_size", 72)
	title.add_theme_color_override("font_color", Color(0.92, 0.95, 1.0))
	col.add_child(title)

	var sub := Label.new()
	sub.text = "a spectral audio-visual instrument"
	sub.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	sub.add_theme_color_override("font_color", Color(0.55, 0.62, 0.75))
	col.add_child(sub)

	col.add_child(_spacer(10))

	# --- Import (audio OR video - the extension picks the slot; both remembered
	# and both shown, since different modes below consume different slots) ---
	var song_row := HBoxContainer.new()
	song_row.alignment = BoxContainer.ALIGNMENT_CENTER
	song_row.add_theme_constant_override("separation", 12)
	col.add_child(song_row)

	var load_btn := Button.new()
	load_btn.text = "Import…"
	load_btn.tooltip_text = "A song for Auto/Manual, or a video clip for Mask Lab"
	load_btn.custom_minimum_size = Vector2(150, 40)
	load_btn.pressed.connect(_open_file_dialog)
	song_row.add_child(load_btn)

	_caption = Label.new()
	_caption.add_theme_color_override("font_color", Color(0.7, 0.78, 0.9))
	song_row.add_child(_caption)
	_refresh_caption()

	col.add_child(_spacer(12))

	# --- The mode list: one row per instrument, all always visible ---
	_add_mode_row(col, "Auto  ▶", "The seeded show - scenes chosen for you, cut on the music.",
		"uses the song", _start_auto)
	_add_mode_row(col, "Manual  ▶", "Orchestrate by hand - the workspace, storyboards, dials.",
		"uses the song", _start_manual)
	_add_mode_row(col, "Synthesis  ▶", "Write a script; ghost speaks it and the show reacts to the voice.",
		"no import needed", _start_synth)
	_add_mode_row(col, "Mask Lab  ▶", "Chroma-key effects over a video - markers, tracks, renders.",
		"uses the clip", _start_mask)

	col.add_child(_spacer(6))
	var asst_row := HBoxContainer.new()
	asst_row.alignment = BoxContainer.ALIGNMENT_CENTER
	asst_row.add_theme_constant_override("separation", 10)
	col.add_child(asst_row)

	var asst_label := Label.new()
	asst_label.text = "Assistant:"
	asst_label.add_theme_color_override("font_color", Color(0.55, 0.62, 0.75))
	asst_row.add_child(asst_label)

	var asst_option := OptionButton.new()
	asst_option.focus_mode = Control.FOCUS_NONE
	asst_option.tooltip_text = "Feedback left with ` gets handed to this, one-shot, the moment you submit it"
	for name in ASSISTANT_BACKENDS:
		asst_option.add_item(name)
	asst_option.select(maxi(0, ASSISTANT_KEYS.find(_assistant_backend)))
	asst_option.item_selected.connect(_on_assistant_selected)
	asst_row.add_child(asst_option)

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


## One mode row: the start button on the left, description + input hint beside
## it. Every mode is always clickable - a missing import is handled by the mode
## itself (Auto idles without a song, Mask Lab prompts for a clip).
func _add_mode_row(col: VBoxContainer, name: String, desc: String, uses: String, action: Callable) -> void:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 16)
	col.add_child(row)

	var btn := Button.new()
	btn.text = name
	btn.custom_minimum_size = Vector2(180, 52)
	btn.pressed.connect(action)
	row.add_child(btn)

	var text_col := VBoxContainer.new()
	text_col.alignment = BoxContainer.ALIGNMENT_CENTER
	text_col.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row.add_child(text_col)

	var desc_label := Label.new()
	desc_label.text = desc
	desc_label.add_theme_color_override("font_color", Color(0.7, 0.78, 0.9))
	text_col.add_child(desc_label)

	var uses_label := Label.new()
	uses_label.text = uses
	uses_label.add_theme_font_size_override("font_size", 12)
	uses_label.add_theme_color_override("font_color", Color(0.42, 0.48, 0.58))
	text_col.add_child(uses_label)


func _spacer(h: int) -> Control:
	var s := Control.new()
	s.custom_minimum_size = Vector2(0, h)
	return s


func _refresh_caption() -> void:
	var parts: Array[String] = []
	if not _audio_path.is_empty():
		parts.append("♪ " + _audio_path.get_file())
	elif ResourceLoader.exists("res://audio/song.wav"):
		parts.append("♪ bundled audio/song.wav")
	else:
		parts.append("♪ none (Auto will idle-animate)")
	if not _video_path.is_empty():
		parts.append("🎬 " + _video_path.get_file())
	_caption.text = "   ".join(parts)


func _open_file_dialog() -> void:
	_file_dialog.popup_centered()


func _on_file_selected(path: String) -> void:
	if VIDEO_EXTS.has(path.get_extension().to_lower()):
		_video_path = path
		_save_last_video(path)
	else:
		_audio_path = path
		_save_last_song(path)
	_refresh_caption()


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


func _start_synth() -> void:
	if start_synth.is_valid():
		start_synth.call()
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


func _on_assistant_selected(idx: int) -> void:
	_assistant_backend = ASSISTANT_KEYS[idx]
	var cfg := ConfigFile.new()
	cfg.load(CFG_PATH)
	cfg.set_value("assistant", "backend", _assistant_backend)
	cfg.save(CFG_PATH)


## The persisted assistant choice ("" = Off, "claude_cli" = Claude Code CLI) -
## a STATIC reader over the same user://ghost.cfg this instance writes, so
## main.gd and mask_editor.gd can both read it directly rather than have it
## threaded through start_session/start_mask. That matters because splash
## isn't even in the tree for a direct CLI --mask-edit launch - the setting
## still has to apply there, from whatever a PREVIOUS run last chose.
static func assistant_backend() -> String:
	var cfg := ConfigFile.new()
	if cfg.load(CFG_PATH) != OK:
		return ""
	return String(cfg.get_value("assistant", "backend", ""))
