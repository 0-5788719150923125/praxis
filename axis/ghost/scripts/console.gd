extends CanvasLayer
class_name ConsoleView

## The in-app console - a live tail of Godot's own log file. print(),
## push_warning(), push_error() and engine SCRIPT ERRORs all land in
## user://logs/godot*.log via the engine's default file logging, but anyone
## launching ghost as a compiled app or from the Godot launcher never sees
## that stream - it used to die in a terminal nobody had open. The `>_`
## toggle (icon-only, on the shared bottom-right button row: console, export
## ⤓, assistant bubble) opens a panel tailing the newest log live.
##
## Part of the shared Chrome (see chrome.gd), so every mode carries it.
## Subprocess output does NOT land here by itself - a subprocess writes its
## own log file, and its owner echoes the interesting lines into godot's log
## when a step finishes (see mask_editor._yt_echo_log for the yt-dlp
## example). That convention is what makes this window worth opening when a
## download misbehaves.

const _TAIL_BYTES := 16384
const _REFRESH_EVERY := 0.5

var _toggle: Button
var _panel: PanelContainer
var _text: RichTextLabel
var _log_path := ""
var _accum := 0.0
var _last_shown := ""


func _ready() -> void:
	layer = 251   # one above the exporter's furniture (250) - same row, never under it
	_toggle = Button.new()
	_toggle.text = ">_"
	_toggle.tooltip_text = "Console - the app's own log output (print, warnings, errors), live"
	_toggle.focus_mode = Control.FOCUS_NONE
	_toggle.custom_minimum_size = Vector2(40, 40)
	_toggle.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	# The shared bottom-right row (see assistant.gd's _TOGGLE_SIZE doc):
	# assistant -68..-28, export -112..-72, console here at -156..-116. The
	# status labels' right edges moved to -160 to stay clear of this slot.
	_toggle.offset_left = -156
	_toggle.offset_top = -68
	_toggle.offset_right = -116
	_toggle.offset_bottom = -28
	_toggle.pressed.connect(_on_toggle)
	add_child(_toggle)

	_panel = PanelContainer.new()
	_panel.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	_panel.offset_left = -28 - 640
	_panel.offset_right = -28
	_panel.offset_top = -76 - 380
	_panel.offset_bottom = -76
	_panel.visible = false
	add_child(_panel)

	_text = RichTextLabel.new()
	_text.selection_enabled = true
	_text.scroll_following = true   # pinned to the newest lines, like a real terminal
	# Ctrl+C on a selection, which is the whole point of being able to make one.
	_text.shortcut_keys_enabled = true
	_text.add_theme_font_size_override("normal_font_size", 12)
	_text.custom_minimum_size = Vector2(620, 360)
	_panel.add_child(_text)


func _on_toggle() -> void:
	_panel.visible = not _panel.visible
	if _panel.visible:
		_accum = _REFRESH_EVERY   # first refresh lands this frame, not half a second in


func _process(dt: float) -> void:
	if not _panel.visible:
		return
	_accum += dt
	if _accum < _REFRESH_EVERY:
		return
	_accum = 0.0
	# A SELECTION IS A CONVERSATION IN PROGRESS. Rewriting the label's text drops
	# it, and this refreshes twice a second, so a busy log made it impossible to
	# highlight anything for long enough to copy it - reported exactly that way,
	# while the log was the only place the answer to a bug lived. While something
	# is selected the view holds; the moment it is cleared the backlog lands.
	if not _text.get_selected_text().is_empty():
		return
	var tail := _read_tail()
	if tail == _last_shown:
		return
	# APPEND WHAT IS NEW rather than replacing the lot. Besides keeping a
	# selection alive between refreshes, this stops the scroll position jumping
	# on every tick. The window slides as the file grows, so the new tail is not
	# always an extension of the old one - when it is not (a rotation, a seek
	# past the window, the first read) the whole thing is redrawn, which is the
	# only case that legitimately loses the view.
	if not _last_shown.is_empty() and tail.begins_with(_last_shown):
		_text.add_text(tail.substr(_last_shown.length()))
	else:
		_text.text = tail
	_last_shown = tail


## The CURRENT session's log. The engine always writes to the CONFIGURED path
## (user://logs/godot.log by default) and renames PREVIOUS runs to timestamped
## backups (godot2026-07-27T12.35.22.log) - so "newest by filename" picks a
## STALE rotated log every time ("godot2026..." sorts above "godot.log"; the
## first cut of this file did exactly that and the console showed a frozen old
## session). Ask the project setting for the live file; fall back to newest by
## mtime only if the setting is missing.
func _resolve_log() -> String:
	var configured := String(ProjectSettings.get_setting(
		"debug/file_logging/log_path", "user://logs/godot.log"))
	if FileAccess.file_exists(configured):
		return configured
	var dir := DirAccess.open("user://logs")
	if dir == null:
		return ""
	var best := ""
	var best_mtime := 0
	for f in dir.get_files():
		if not f.ends_with(".log"):
			continue
		var p := "user://logs/" + f
		var mt := FileAccess.get_modified_time(p)
		if mt > best_mtime:
			best_mtime = mt
			best = p
	return best


func _read_tail() -> String:
	if _log_path.is_empty():
		_log_path = _resolve_log()
	if _log_path.is_empty():
		return "(no log file - is debug/file_logging/enable_file_logging off?)"
	var f := FileAccess.open(_log_path, FileAccess.READ)
	if f == null:
		return "(could not open %s)" % _log_path
	var sz := f.get_length()
	if sz > _TAIL_BYTES:
		f.seek(sz - _TAIL_BYTES)
	var s := f.get_buffer(mini(int(sz), _TAIL_BYTES)).get_string_from_utf8()
	# A window that starts mid-line reads as a torn first line - drop it.
	if sz > _TAIL_BYTES:
		var nl := s.find("\n")
		if nl >= 0:
			s = s.substr(nl + 1)
	return s


## Lift the toggle - and the panel above it - clear of whatever the mode has put
## along the bottom of the frame. See Chrome.bottom_inset.
func set_bottom_inset(v: float) -> void:
	if _toggle != null:
		_toggle.offset_top = -68.0 - v
		_toggle.offset_bottom = -28.0 - v
	if _panel != null:
		_panel.offset_top = -76.0 - 380.0 - v
		_panel.offset_bottom = -76.0 - v
