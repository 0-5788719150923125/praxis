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
##   Masking  - chroma-key effects over an imported video clip.
## A mode button *is* start - there is no separate start button.
##
## ONE source row imports everything: the picker button plus a free-form field
## that accepts a URL or a raw file path. Whatever arrives is routed to the
## right slot - http(s) URLs fill the clip slot (Masking's mask_editor
## downloads them itself: yt-dlp in ghost's own venv, see mask_editor.gd
## _start_url_import - a YouTube link is a first-class clip source, no manual
## ripping), local paths route by extension (audio -> song, video -> clip).
## Both slots are remembered independently (user://ghost.cfg), but they are
## NOT displayed side by side anymore - a song caption next to a clip URL
## read as "both will import". Instead each mode row below captions the
## source IT consumes (Auto/Manual show the song, Masking shows the clip),
## so the context is on the button that uses it. Auto/Manual use the song slot,
## Masking uses the clip slot, Synthesis needs no import at all. Clicking a
## mode calls back into main (start_session / start_synth / start_mask), then
## frees the splash. Built in code (no .tscn).

const VIDEO_EXTS := ["mp4", "mov", "mkv", "webm", "avi"]
const AUDIO_EXTS := ["wav", "mp3", "ogg", "oga", "flac"]

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
var start_synth: Callable      # start_synth.call(mode: "fishing"|"neural")

var _audio_path := ""
var _video_path := ""
var _assistant_backend := ""
var _file_dialog: FileDialog
var _env: Control                       # the Environment panel (deps_panel.gd)
var _source_edit: LineEdit
var _uses_auto: Label     # per-mode source captions - see _refresh_sources
var _uses_manual: Label
var _uses_mask: Label


func _ready() -> void:
	layer = 200
	_load_last_song()
	_load_last_video()
	_assistant_backend = assistant_backend()
	_build_ui()
	# THE ⤓ EXPORT BUTTON HAS NO BUSINESS HERE. It is session furniture - it renders the
	# running show to video - and the home screen is not a session: there is nothing on
	# screen to render, so the button could only ever be the greyed "nothing to render
	# yet" stub, sitting on top of the Environment panel and eating its clicks.
	#
	# This does NOT reopen the "never hide the export button" rule in exporter.gd. That
	# rule is about never hiding it DURING a session on timing or eligibility grounds,
	# which is what kept regressing; it says nothing about a screen where no session
	# exists. The ` feedback console is absent here for exactly the same reason, and by
	# the same means.
	var ch := _chrome()
	if ch != null:
		ch.suppress_export(&"splash")


## Hand back everything this screen claimed of the shared furniture.
##
## KEYED (see Chrome._bottom_claims), and here that is essential rather than careful:
## the mode buttons call into main and only THEN queue_free this node, so by the time
## this runs the incoming mode has already made its own claims. Releasing by writing 0 /
## false would undo them.
func _exit_tree() -> void:
	var ch := _chrome()
	if ch != null:
		ch.release_export(&"splash")
		ch.release_bottom(&"splash")


func _chrome() -> Node:
	if get_tree() == null:
		return null
	return get_tree().get_first_node_in_group("ghost_chrome")


## Step the shared ⤓/💬/>_ toggle row up over the Environment panel. Called on every
## resize of that panel, because its height is its own business and changes as rows open.
func _claim_room() -> void:
	var ch := _chrome()
	if ch == null or _env == null or not is_instance_valid(_env):
		return
	# The panel is inset 18 px from the corner, so the room it needs is its height plus
	# that inset plus a small gap - measured off the panel rather than written down, or
	# the two drift apart the first time either number is touched.
	ch.claim_bottom(&"splash", _env.size.y + 18.0 + 10.0)


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

	# --- ONE source row: the picker + a free-form URL/path field. What each
	# mode will actually consume is captioned on its own row below, not here.
	var src_row := HBoxContainer.new()
	src_row.alignment = BoxContainer.ALIGNMENT_CENTER
	src_row.add_theme_constant_override("separation", 12)
	col.add_child(src_row)

	var load_btn := Button.new()
	load_btn.text = "Import…"
	load_btn.tooltip_text = "A song for Auto/Manual, or a video clip for Masking"
	load_btn.custom_minimum_size = Vector2(150, 40)
	load_btn.pressed.connect(_open_file_dialog)
	src_row.add_child(load_btn)

	_source_edit = LineEdit.new()
	_source_edit.placeholder_text = "…or paste a URL / file path and press Enter"
	_source_edit.tooltip_text = "A YouTube (or any http) URL fills the clip slot - Masking downloads it itself.\n" + \
		"A local path routes by extension: audio → song, video → clip."
	_source_edit.custom_minimum_size = Vector2(440, 34)
	_source_edit.text_submitted.connect(_on_source_submitted)
	# An invalid entry turns the text red (see _on_source_submitted); any
	# editing clears the mark.
	_source_edit.text_changed.connect(func(_t: String) -> void:
		_source_edit.remove_theme_color_override("font_color"))
	src_row.add_child(_source_edit)

	col.add_child(_spacer(12))

	# --- The mode list: one row per instrument, all always visible. Each row's
	# small caption is ITS source's live state (see _refresh_sources).
	_uses_auto = _add_mode_row(col, "Auto  ▶",
		"The seeded show - scenes chosen for you, cut on the music.", "", _start_auto)
	_uses_manual = _add_mode_row(col, "Manual  ▶",
		"Orchestrate by hand - the workspace, storyboards, dials.", "", _start_manual)
	# Two separate modes rather than one with a sub-choice: the paths are not
	# variants of each other. Synthesis has a genome, a belt and a fishing loop;
	# Generative has a speaker id and three scalars. See VOICE_PLAN.md section 6.
	_add_mode_row(col, "Synthesis  ▶",
		"Write a script; ghost speaks it and the show reacts to the voice.",
		"no import needed", func() -> void: _choose_synth("fishing"))
	_add_mode_row(col, "Generative  ▶",
		"The same, in a small local neural voice - clearer, at the cost of "
		+ "a downloaded model.", "downloads a voice once",
		func() -> void: _choose_synth("neural"))
	_uses_mask = _add_mode_row(col, "Masking  ▶",
		"Chroma-key effects over a video - markers, tracks, renders.", "", _start_mask)
	_refresh_sources()

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
	# BOUND, so the choice is stored and restored by construction - see Settings.bind. The
	# keys are passed so the setting holds the backend NAME, not a row number that a new
	# backend would silently reassign.
	Settings.bind(asst_option, "assistant", "backend", "", ASSISTANT_KEYS)
	asst_option.item_selected.connect(_on_assistant_selected)
	asst_row.add_child(asst_option)

	var hint := Label.new()
	hint.text = "F11 fullscreen · ` feedback · Esc quit"
	hint.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	hint.add_theme_color_override("font_color", Color(0.4, 0.46, 0.56))
	col.add_child(hint)

	# THE ENVIRONMENT PANEL, bottom-right and out of the column's way. ghost's
	# optional half is all subprocesses (ffmpeg, python, a JS runtime), and a
	# missing one used to surface as a mode failing deep inside itself. Added
	# LAST so it sits above the full-rect CenterContainer in pick order.
	_env = preload("res://scripts/deps_panel.gd").new()
	add_child(_env)
	# ...and the shared toggle row steps up over it. The panel is anchored to the same
	# bottom-right corner the ⤓/💬/>_ toggles are, so left alone they draw ON it - and
	# because the toggles sit on a higher CanvasLayer they also ate the clicks meant for
	# the Environment rows underneath. Its height is not a constant (a row's detail pane
	# opens, a rescan finds more, and it remembers being collapsed), so the claim tracks
	# the panel's actual size rather than a number written down here.
	_env.resized.connect(_claim_room)
	_claim_room()

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


## One mode row: the start button on the left, description + source caption
## beside it. Every mode is always clickable - a missing import is handled by
## the mode itself (Auto idles without a song, Masking prompts for a clip).
## Returns the caption label so _refresh_sources can keep it current.
func _add_mode_row(col: VBoxContainer, name: String, desc: String, uses: String, action: Callable) -> Label:
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
	return uses_label


func _spacer(h: int) -> Control:
	var s := Control.new()
	s.custom_minimum_size = Vector2(0, h)
	return s


## Each mode row captions the source IT will consume - the whole point of the
## merged source row (a song caption sitting beside a clip URL read as "both
## will import", which is never true: Auto/Manual take the song, Masking takes
## the clip).
func _refresh_sources() -> void:
	var song: String
	if not _audio_path.is_empty():
		song = "song: " + _audio_path.get_file()
	elif ResourceLoader.exists("res://audio/song.wav"):
		song = "song: bundled audio/song.wav"
	else:
		song = "no song - idle-animates"
	_uses_auto.text = song
	_uses_manual.text = song
	_uses_mask.text = ("clip: " + _short_source(_video_path)) if not _video_path.is_empty() \
		else "no clip - asks on start"


static func _short_source(p: String) -> String:
	if p.begins_with("http"):
		var s := p.trim_prefix("https://").trim_prefix("http://").trim_prefix("www.")
		return s.substr(0, 40) + ("…" if s.length() > 40 else "")
	return p.get_file()


func _open_file_dialog() -> void:
	_file_dialog.popup_centered()


## The free-form field: a URL or a raw file path, routed to the right slot.
## Anything that is neither an existing file nor URL-shaped turns the entry
## red instead of silently landing in a slot it can't fill.
func _on_source_submitted(text: String) -> void:
	var s := text.strip_edges()
	if s.is_empty():
		return
	var is_url := s.begins_with("http://") or s.begins_with("https://")
	if not is_url and not FileAccess.file_exists(s) and (s.begins_with("www.")
			or s.to_lower().contains("youtu.be") or s.to_lower().contains("youtube.com")):
		s = "https://" + s   # pasted without a scheme
		is_url = true
	if is_url:
		_video_path = s
		_save_last_video(s)
		_source_edit.text = s
		_refresh_sources()
		return
	var ext := s.get_extension().to_lower()
	if FileAccess.file_exists(s) and (VIDEO_EXTS.has(ext) or AUDIO_EXTS.has(ext)):
		_on_file_selected(s)
	else:
		_source_edit.add_theme_color_override("font_color", Color(1.0, 0.45, 0.4))


func _on_file_selected(path: String) -> void:
	if VIDEO_EXTS.has(path.get_extension().to_lower()):
		_video_path = path
		_save_last_video(path)
	else:
		_audio_path = path
		_save_last_song(path)
	_source_edit.text = path   # the field always shows the last import, either origin
	_refresh_sources()


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


func _choose_synth(mode: String) -> void:
	if start_synth.is_valid():
		start_synth.call(mode)
	queue_free()


func _start_mask() -> void:
	if not _video_path.is_empty():
		_save_last_video(_video_path)
	if start_mask.is_valid():
		start_mask.call(_video_path)     # "" is fine - mask_editor prompts its own dialog
	queue_free()


# --- Remembered song / clip (user://ghost.cfg) ------------------------------

func _load_last_song() -> void:
	var p := String(Settings.read("audio", "last", ""))
	if not p.is_empty() and FileAccess.file_exists(p):
		_audio_path = p     # still on disk - pre-select it


func _save_last_song(path: String) -> void:
	Settings.write("audio", "last", path)


func _load_last_video() -> void:
	var p := String(Settings.read("video", "last", ""))
	# URLs are remembered too - "exists" only means anything for local files.
	if not p.is_empty() and (p.begins_with("http") or FileAccess.file_exists(p)):
		_video_path = p


func _save_last_video(path: String) -> void:
	Settings.write("video", "last", path)


func _on_assistant_selected(idx: int) -> void:
	_assistant_backend = ASSISTANT_KEYS[idx]


## The persisted assistant choice ("" = Off, "claude_cli" = Claude Code CLI) -
## a STATIC reader over the same user://ghost.cfg this instance writes, so
## main.gd and mask_editor.gd can both read it directly rather than have it
## threaded through start_session/start_mask. That matters because splash
## isn't even in the tree for a direct CLI --mask-edit launch - the setting
## still has to apply there, from whatever a PREVIOUS run last chose.
static func assistant_backend() -> String:
	var tree := Engine.get_main_loop() as SceneTree
	if tree == null:
		return ""
	var st := tree.root.get_node_or_null("Settings")
	return String(st.read("assistant", "backend", "")) if st != null else ""
