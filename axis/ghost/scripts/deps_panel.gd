extends PanelContainer
class_name DepsPanel

## DepsPanel - the environment readout in the home screen's bottom-right corner.
##
## ghost's optional half is all subprocesses: ffmpeg for Masking, python for the
## voice and the clown, a JS runtime for URL imports. When one of them is absent
## the failure used to surface deep inside a mode - "the clip won't open", "the
## voice host exited unexpectedly" - with the actual cause (this machine has no
## ffmpeg) never named anywhere. This panel names it before anything is clicked.
##
## It renders [Deps] and adds nothing of its own. That matters: a status panel with
## private detection logic drifts from the code that actually launches things and
## then reports green while the launch fails. Every row here comes from the same
## [method Deps.resolve] that [Subprocess] uses.
##
## WHAT IT COSTS. A full [method Deps.report] runs one `--version` per installed
## program - a few hundred milliseconds, cold - so it runs on a [Thread] and the
## panel says "checking…" until the results land. The home screen must not hitch
## while listing the things that make it work.
##
## THE HINTS ARE THE POINT. Knowing ffmpeg is missing is half an answer; the other
## half is the command for THIS platform, which is why clicking a row opens the
## detail pane. Copy takes the whole report to the clipboard in plain text, for
## pasting into a bug report - the fastest way to answer "what does your machine
## have on it".

const CFG_PATH := "user://ghost.cfg"

const COL_OK := Color(0.44, 0.84, 0.56)
const COL_BAD := Color(1.0, 0.50, 0.42)
const COL_IDLE := Color(0.40, 0.46, 0.56)
const COL_TEXT := Color(0.70, 0.78, 0.90)
const COL_DIM := Color(0.50, 0.57, 0.68)

var _rows: Array = []
var _thread: Thread
var _list: VBoxContainer
var _body: VBoxContainer
var _title: Label
var _detail: VBoxContainer
var _detail_text: Label
var _detail_link: LinkButton
var _open_key := ""
var _collapsed := false


func _ready() -> void:
	_collapsed = _load_collapsed()
	_build_ui()
	_start_probe()


## A probe in flight owns a thread, and Godot is loud about one that is still
## joinable at free time. The splash frees this the instant a mode starts, which is
## exactly when a cold probe is likely to still be running.
func _exit_tree() -> void:
	_join()


func _join() -> void:
	if _thread != null and _thread.is_started():
		_thread.wait_to_finish()
	_thread = null


# --- layout ------------------------------------------------------------------

func _build_ui() -> void:
	# Pinned to the bottom-right corner and grown UP-LEFT from it, so the panel's
	# height is free to change - a row's detail pane opening, a rescan finding more -
	# without ever moving the corner it is anchored to. The grow directions are set
	# explicitly because the preset leaves them at GROW_DIRECTION_END, which sends a
	# minimum-size-driven panel off the bottom-right of the screen (it did).
	anchor_left = 1.0
	anchor_top = 1.0
	anchor_right = 1.0
	anchor_bottom = 1.0
	offset_left = -18
	offset_top = -18
	offset_right = -18
	offset_bottom = -18
	grow_horizontal = Control.GROW_DIRECTION_BEGIN
	grow_vertical = Control.GROW_DIRECTION_BEGIN
	custom_minimum_size = Vector2(360, 0)
	# Own panel style rather than the theme's: the splash is nearly black and the
	# default StyleBoxFlat is a mid grey slab that reads as a modal dialog.
	var sb := StyleBoxFlat.new()
	sb.bg_color = Color(0.07, 0.08, 0.11, 0.92)
	sb.border_color = Color(0.18, 0.21, 0.27)
	sb.set_border_width_all(1)
	sb.set_corner_radius_all(6)
	sb.content_margin_left = 12
	sb.content_margin_right = 12
	sb.content_margin_top = 8
	sb.content_margin_bottom = 8
	add_theme_stylebox_override("panel", sb)

	var col := VBoxContainer.new()
	col.add_theme_constant_override("separation", 4)
	add_child(col)

	# --- header: the whole thing is the collapse toggle, with the two actions
	# pulled out to the right so a click on either is unambiguous.
	var head := HBoxContainer.new()
	head.add_theme_constant_override("separation", 6)
	col.add_child(head)

	var toggle := Button.new()
	toggle.flat = true
	toggle.focus_mode = Control.FOCUS_NONE
	toggle.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	toggle.alignment = HORIZONTAL_ALIGNMENT_LEFT
	toggle.tooltip_text = "What ghost found on this machine. Click to collapse."
	toggle.pressed.connect(_toggle_collapsed)
	head.add_child(toggle)

	_title = Label.new()
	_title.text = "Environment"
	_title.add_theme_font_size_override("font_size", 13)
	_title.add_theme_color_override("font_color", COL_TEXT)
	_title.mouse_filter = Control.MOUSE_FILTER_IGNORE
	# Clipped, because the verdict is a list of names and a long one would otherwise
	# draw straight over the two buttons beside it (it did).
	_title.clip_text = true
	_title.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	toggle.add_child(_title)
	# The button still has to be tall enough for a label it does not measure.
	toggle.custom_minimum_size = Vector2(0, 20)

	# Words, not glyphs. This panel is the one part of ghost most likely to be read
	# on a machine whose default font is not this one, and an icon that renders as a
	# tofu box on Windows is worse than a six-letter word anywhere.
	head.add_child(_action_button("rescan", "Look again - for after installing something",
		_rescan))
	head.add_child(_action_button("copy", "Copy the whole report as text, for a bug report",
		_copy_report))

	_body = VBoxContainer.new()
	_body.add_theme_constant_override("separation", 2)
	col.add_child(_body)

	var rule := HSeparator.new()
	_body.add_child(rule)

	_list = VBoxContainer.new()
	_list.add_theme_constant_override("separation", 1)
	_body.add_child(_list)

	_detail = VBoxContainer.new()
	_detail.add_theme_constant_override("separation", 3)
	_detail.visible = false
	_body.add_child(_detail)

	var dsep := HSeparator.new()
	_detail.add_child(dsep)

	_detail_text = Label.new()
	_detail_text.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_detail_text.custom_minimum_size = Vector2(336, 0)
	_detail_text.add_theme_font_size_override("font_size", 11)
	_detail_text.add_theme_color_override("font_color", COL_DIM)
	_detail.add_child(_detail_text)

	_detail_link = LinkButton.new()
	_detail_link.focus_mode = Control.FOCUS_NONE
	_detail_link.add_theme_font_size_override("font_size", 11)
	_detail_link.add_theme_color_override("font_color", Color(0.50, 0.70, 1.0))
	_detail_link.pressed.connect(func() -> void:
		if not _detail_link.text.is_empty():
			OS.shell_open(_detail_link.text))
	_detail.add_child(_detail_link)

	_body.visible = not _collapsed
	_placeholder("checking…")


func _action_button(label: String, tip: String, action: Callable) -> Button:
	var b := Button.new()
	b.flat = true
	b.text = label
	b.focus_mode = Control.FOCUS_NONE
	b.tooltip_text = tip
	b.custom_minimum_size = Vector2(0, 20)
	b.add_theme_font_size_override("font_size", 10)
	b.add_theme_color_override("font_color", COL_DIM)
	b.pressed.connect(action)
	return b


func _placeholder(text: String) -> void:
	for c in _list.get_children():
		c.queue_free()
	var l := Label.new()
	l.text = text
	l.add_theme_font_size_override("font_size", 11)
	l.add_theme_color_override("font_color", COL_DIM)
	_list.add_child(l)


# --- the probe ---------------------------------------------------------------

func _start_probe() -> void:
	_join()
	_placeholder("checking…")
	_thread = Thread.new()
	# `Deps.report` only reads the filesystem and spawns short-lived `--version`
	# processes; nothing it touches is a Godot resource, so no main-thread hop is
	# needed until the results come back.
	_thread.start(_probe_worker)


func _probe_worker() -> void:
	var rows := Deps.report()
	_apply.call_deferred(rows)


func _apply(rows: Array) -> void:
	_rows = rows
	_render()
	# A missing FEATURE dependency overrides a remembered collapse: the one moment
	# this panel exists to serve is the one where the user does not yet know to look.
	if _collapsed and not Deps.verdict(rows).is_empty():
		_collapsed = false
		_body.visible = true


func _rescan() -> void:
	Deps.forget_all()
	_open_key = ""
	_detail.visible = false
	_start_probe()


func _copy_report() -> void:
	DisplayServer.clipboard_set(Deps.format_report(_rows))
	_title.text = "Environment  · copied"
	get_tree().create_timer(1.5).timeout.connect(_refresh_title)


# --- rendering ---------------------------------------------------------------

func _render() -> void:
	for c in _list.get_children():
		c.queue_free()
	var managed_started := false
	for r in _rows:
		if int(r.get("kind", Deps.KIND_TOOL)) == Deps.KIND_MANAGED and not managed_started:
			managed_started = true
			_list.add_child(_group_label("ghost's own · installed on first use"))
		_list.add_child(_row_button(r))
	_refresh_title()


func _group_label(text: String) -> Control:
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 2)
	box.add_child(HSeparator.new())
	var l := Label.new()
	l.text = text
	l.add_theme_font_size_override("font_size", 10)
	l.add_theme_color_override("font_color", COL_IDLE)
	box.add_child(l)
	return box


func _refresh_title() -> void:
	var bad := Deps.verdict(_rows)
	if _rows.is_empty():
		_title.text = "Environment"
		_title.add_theme_color_override("font_color", COL_TEXT)
	elif bad.is_empty():
		_title.text = "Environment  ·  all present"
		_title.add_theme_color_override("font_color", COL_OK)
	else:
		_title.text = "Environment  ·  " + bad
		_title.add_theme_color_override("font_color", COL_BAD)
	if _collapsed:
		_title.text += "   ▸"


## One row. A flat [Button] rather than an [HBoxContainer] with a `gui_input`
## handler, so hover highlighting and keyboard focus come from the theme; the three
## labels inside ignore the mouse so the whole strip stays one click target.
func _row_button(r: Dictionary) -> Button:
	var found := bool(r.get("found", false))
	var feature := int(r.get("tier", Deps.TIER_EXTRA)) == Deps.TIER_FEATURE
	var glyph := "●" if found else ("▲" if feature else "○")
	var tint := COL_OK if found else (COL_BAD if feature else COL_IDLE)

	var b := Button.new()
	b.flat = true
	b.focus_mode = Control.FOCUS_NONE
	b.custom_minimum_size = Vector2(0, 17)
	b.tooltip_text = String(r.get("used_for", ""))
	b.pressed.connect(_toggle_detail.bind(String(r.get("key", ""))))

	var row := HBoxContainer.new()
	row.mouse_filter = Control.MOUSE_FILTER_IGNORE
	row.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	row.add_theme_constant_override("separation", 6)
	b.add_child(row)

	row.add_child(_cell(glyph, tint, 11, 12, HORIZONTAL_ALIGNMENT_CENTER))
	var name_cell := _cell(String(r.get("name", "?")),
		COL_TEXT if found else COL_DIM, 11, 0, HORIZONTAL_ALIGNMENT_LEFT)
	name_cell.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row.add_child(name_cell)

	var value := String(r.get("version", ""))
	if value.is_empty():
		value = String(r.get("note", "not found"))
	row.add_child(_cell(value, tint if not found else COL_DIM, 11, 132,
		HORIZONTAL_ALIGNMENT_RIGHT))
	return b


func _cell(text: String, tint: Color, size: int, min_w: int,
		align: int) -> Label:
	var l := Label.new()
	l.text = text
	l.mouse_filter = Control.MOUSE_FILTER_IGNORE
	l.horizontal_alignment = align
	l.clip_text = true
	l.add_theme_font_size_override("font_size", size)
	l.add_theme_color_override("font_color", tint)
	if min_w > 0:
		l.custom_minimum_size = Vector2(min_w, 0)
	return l


## Clicking a row opens what it is for, where it was found, and - the reason this
## pane exists - the install command for THIS platform. Clicking the same row again
## closes it, so the panel returns to its compact height.
func _toggle_detail(key: String) -> void:
	if key == _open_key:
		_open_key = ""
		_detail.visible = false
		return
	var entry := {}
	for r in _rows:
		if String(r.get("key", "")) == key:
			entry = r
			break
	if entry.is_empty():
		return
	_open_key = key
	var lines: PackedStringArray = [String(entry.get("name", "?"))]
	lines.append(String(entry.get("used_for", "")))
	var path := String(entry.get("path", ""))
	if bool(entry.get("found", false)) and not path.is_empty():
		lines.append("found at: " + path)
	elif int(entry.get("kind", Deps.KIND_TOOL)) == Deps.KIND_MANAGED:
		lines.append("not installed yet · %s · will be created at: %s"
			% [String(entry.get("size", "")), path])
	else:
		var hint := Deps.install_hint(entry)
		if not hint.is_empty():
			lines.append("install:  " + hint)
	_detail_text.text = "\n".join(lines)
	_detail_link.text = String(entry.get("site", ""))
	_detail_link.visible = not _detail_link.text.is_empty()
	_detail.visible = true


# --- collapse state (user://ghost.cfg, beside the remembered song and clip) ---

func _toggle_collapsed() -> void:
	_collapsed = not _collapsed
	_body.visible = not _collapsed
	_refresh_title()
	var cfg := ConfigFile.new()
	cfg.load(CFG_PATH)
	cfg.set_value("deps", "collapsed", _collapsed)
	cfg.save(CFG_PATH)


func _load_collapsed() -> bool:
	var cfg := ConfigFile.new()
	if cfg.load(CFG_PATH) != OK:
		return false
	return bool(cfg.get_value("deps", "collapsed", false))
