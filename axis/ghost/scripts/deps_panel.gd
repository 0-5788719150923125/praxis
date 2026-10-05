extends PanelContainer
class_name DepsPanel

## DepsPanel - the environment readout in the home screen's bottom-right corner.
##
## ghost runs on a handful of things it does not draw itself: FFmpeg for video, a Python and its
## per-feature environments for the voice, the trackers, URL import and page capture. ghost
## installs and updates all of those on its own; this panel is where that is visible - what is
## there, what version, what is downloading right now and how far along, and why something
## failed. The few things that stay the machine's own (Linux utilities, the AI CLIs) are listed
## with the command that installs them.
##
## It renders [Deps] and [Provision] and adds nothing of its own. That matters: a status panel with
## private detection logic drifts from the code that actually launches things and then reports
## green while the launch fails. Every row here comes from the same resolver [Subprocess] uses, and
## every progress bar from the job that is actually running.
##
## WHAT IT COSTS. A full [method Deps.report] runs one `--version` per installed program - a few
## hundred milliseconds, cold - so it runs on a [Thread] and the panel says "checking…" until the
## results land; it runs again when a job finishes, to pick up the new version. Progress between
## those comes from the Provisioner's signals and costs nothing.

const COL_OK := Color(0.44, 0.84, 0.56)
const COL_BAD := Color(1.0, 0.50, 0.42)
const COL_RUN := Color(0.45, 0.66, 1.0)
const COL_IDLE := Color(0.40, 0.46, 0.56)
const COL_TEXT := Color(0.70, 0.78, 0.90)
const COL_DIM := Color(0.50, 0.57, 0.68)

const GROUPS := ["kept current by ghost", "built the first time a feature needs it",
	"from this machine"]

var _rows: Array = []
var _thread: Thread
var _list: VBoxContainer
var _body: VBoxContainer
var _title: Label
var _detail: VBoxContainer
var _detail_text: Label
var _detail_link: LinkButton
var _retry: Button
var _checked: Label
var _check_now: Button
var _open_key := ""
var _ui := {}                    # key -> {button, glyph, name, value, bar}
var _collapsed := false
var _reprobe := false            # a job finished while a probe was running
var _tick := 0.0


func _ready() -> void:
	_collapsed = _load_collapsed()
	_build_ui()
	var agent := Provision.agent()
	if agent != null:
		agent.connect("changed", _on_changed)
		agent.connect("finished", _on_finished)
	_start_probe()


## A probe in flight owns a thread, and Godot is loud about one that is still joinable at free
## time. The splash frees this the instant a mode starts, which is exactly when a cold probe is
## likely to still be running.
func _exit_tree() -> void:
	_join()


func _join() -> void:
	if _thread != null and _thread.is_started():
		_thread.wait_to_finish()
	_thread = null


# --- layout ------------------------------------------------------------------

func _build_ui() -> void:
	# Pinned to the bottom-right corner and grown UP-LEFT from it, so the panel's height is free to
	# change - a row's detail pane opening, a rescan finding more - without ever moving the corner
	# it is anchored to. The grow directions are set explicitly because the preset leaves them at
	# GROW_DIRECTION_END, which sends a minimum-size-driven panel off the bottom-right of the screen.
	#
	# THE CORNER ITSELF BELONGS TO THE SHARED TOGGLE ROW (💬 assistant, >_ console), which sits
	# there in every mode: the panel stands on top of it, right edges aligned, rather than pushing
	# it up. A row that moved on this one screen read as wrong, and it is the screen people start on.
	anchor_left = 1.0
	anchor_top = 1.0
	anchor_right = 1.0
	anchor_bottom = 1.0
	offset_left = -28
	offset_top = -Chrome.ROW_TOP - 8
	offset_right = -28
	offset_bottom = -Chrome.ROW_TOP - 8
	grow_horizontal = Control.GROW_DIRECTION_BEGIN
	grow_vertical = Control.GROW_DIRECTION_BEGIN
	custom_minimum_size = Vector2(360, 0)
	# Own panel style rather than the theme's: the splash is nearly black and the default
	# StyleBoxFlat is a mid gray slab that reads as a modal dialog.
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

	# --- header: the whole thing is the collapse toggle, with the two actions pulled out to the
	# right so a click on either is unambiguous.
	var head := HBoxContainer.new()
	head.add_theme_constant_override("separation", 6)
	col.add_child(head)

	var toggle := Button.new()
	toggle.flat = true
	toggle.focus_mode = Control.FOCUS_NONE
	toggle.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	toggle.alignment = HORIZONTAL_ALIGNMENT_LEFT
	toggle.tooltip_text = "What ghost runs on, and what it is installing. Click to collapse."
	toggle.pressed.connect(_toggle_collapsed)
	head.add_child(toggle)

	_title = Label.new()
	_title.text = "Environment"
	_title.add_theme_font_size_override("font_size", 13)
	_title.add_theme_color_override("font_color", COL_TEXT)
	_title.mouse_filter = Control.MOUSE_FILTER_IGNORE
	# Clipped, because the summary is a list of names and a long one would otherwise draw straight
	# over the two buttons beside it.
	_title.clip_text = true
	_title.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	toggle.add_child(_title)
	# The button still has to be tall enough for a label it does not measure.
	toggle.custom_minimum_size = Vector2(0, 20)

	# Words, not glyphs. This panel is the one part of ghost most likely to be read on a machine
	# nobody here has tried, and a six-letter word reads everywhere.
	head.add_child(_action_button("rescan", "Look again - for after installing something", _rescan))
	head.add_child(_action_button("copy", "Copy the whole report as text, for a bug report",
		_copy_report))

	_body = VBoxContainer.new()
	_body.add_theme_constant_override("separation", 2)
	col.add_child(_body)

	_body.add_child(HSeparator.new())

	# SAY THE ROWS CAN BE CLICKED. The detail pane is where a failure's reason and an install
	# command live, and nothing about a flat row says it opens one.
	var hint := Label.new()
	hint.text = "ghost installs and updates these itself. Click one for what it does."
	hint.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	hint.custom_minimum_size = Vector2(336, 0)
	hint.add_theme_font_size_override("font_size", 10)
	hint.add_theme_color_override("font_color", COL_IDLE)
	_body.add_child(hint)

	_list = VBoxContainer.new()
	_list.add_theme_constant_override("separation", 1)
	_body.add_child(_list)

	_detail = VBoxContainer.new()
	_detail.add_theme_constant_override("separation", 3)
	_detail.visible = false
	_body.add_child(_detail)

	_detail.add_child(HSeparator.new())

	_detail_text = Label.new()
	_detail_text.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_detail_text.custom_minimum_size = Vector2(336, 0)
	_detail_text.add_theme_font_size_override("font_size", 11)
	_detail_text.add_theme_color_override("font_color", COL_DIM)
	_detail.add_child(_detail_text)

	var links := HBoxContainer.new()
	links.add_theme_constant_override("separation", 10)
	_detail.add_child(links)
	_retry = _action_button("retry", "Try the install again now", _on_retry)
	_retry.add_theme_color_override("font_color", COL_RUN)
	links.add_child(_retry)
	_detail_link = LinkButton.new()
	_detail_link.focus_mode = Control.FOCUS_NONE
	_detail_link.add_theme_font_size_override("font_size", 11)
	_detail_link.add_theme_color_override("font_color", Color(0.50, 0.70, 1.0))
	_detail_link.pressed.connect(_open_link)
	links.add_child(_detail_link)

	# --- footer: the update policy, where the reader of the list would look for it.
	_body.add_child(HSeparator.new())
	var foot := HBoxContainer.new()
	foot.add_theme_constant_override("separation", 6)
	_body.add_child(foot)
	var auto := CheckBox.new()
	auto.text = "keep up to date"
	auto.focus_mode = Control.FOCUS_NONE
	auto.tooltip_text = "Once a day at launch, bring everything above to its newest release."
	auto.add_theme_font_size_override("font_size", 10)
	auto.add_theme_color_override("font_color", COL_DIM)
	Settings.bind(auto, "deps", "auto_update", true)
	foot.add_child(auto)
	_checked = Label.new()
	_checked.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_checked.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	_checked.add_theme_font_size_override("font_size", 10)
	_checked.add_theme_color_override("font_color", COL_IDLE)
	foot.add_child(_checked)
	_check_now = _action_button("check now", "Look for newer releases of everything installed",
		_on_check_now)
	foot.add_child(_check_now)

	_body.visible = not _collapsed
	_placeholder("checking…")
	_refresh_footer()


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
	_ui = {}
	var l := Label.new()
	l.text = text
	l.add_theme_font_size_override("font_size", 11)
	l.add_theme_color_override("font_color", COL_DIM)
	_list.add_child(l)


func _open_link() -> void:
	if not _detail_link.text.is_empty():
		OS.shell_open(_detail_link.text)


# --- the probe ---------------------------------------------------------------

func _start_probe() -> void:
	if _thread != null and _thread.is_alive():
		_reprobe = true
		return
	_join()
	if _rows.is_empty():
		_placeholder("checking…")
	_thread = Thread.new()
	# `Deps.report` only reads the filesystem and spawns short-lived `--version` processes; nothing
	# it touches is a Godot resource, so no main-thread hop is needed until the results come back.
	_thread.start(_probe_worker)


func _probe_worker() -> void:
	var rows := Deps.report()
	_apply.call_deferred(rows)


func _apply(rows: Array) -> void:
	_join()
	_rows = rows
	_render()
	# Something a feature needs that is missing or failed overrides a remembered collapse: the one
	# moment this panel exists to serve is the one where the user does not yet know to look.
	if _collapsed and not _problem().is_empty():
		_collapsed = false
		_body.visible = true
		_refresh_title()
	if _reprobe:
		_reprobe = false
		_start_probe()


func _rescan() -> void:
	Deps.forget_all()
	var agent := Provision.agent()
	if agent != null:
		agent.call("forget")
	_start_probe()


func _copy_report() -> void:
	DisplayServer.clipboard_set(Deps.format_report(_rows))
	_title.text = "Environment  · copied"
	get_tree().create_timer(1.5).timeout.connect(_refresh_title)


# --- live progress -------------------------------------------------------------

func _on_changed(key: String) -> void:
	if key.is_empty():
		_refresh_footer()
		return
	_live(key)
	_refresh_title()
	if key == _open_key:
		_fill_detail(key)
	set_process(not Provision.active().is_empty())


func _on_finished(_key: String, _ok: bool, _message: String) -> void:
	_start_probe()


func _process(dt: float) -> void:
	# A running job's line carries its elapsed time, which moves even when the job says nothing.
	_tick += dt
	if _tick < 0.5:
		return
	_tick = 0.0
	var running := Provision.active()
	for key in running:
		_live(key)
	_refresh_title()
	if running.is_empty():
		set_process(false)


## Paint one row from what is on disk and what its job is doing right now.
func _live(key: String) -> void:
	var ui: Dictionary = _ui.get(key, {})
	if ui.is_empty():
		return
	var r := _row(key)
	var s := Provision.state(key)
	var found := bool(r.get("found", false))
	# Only what the user has to install is a problem while it is absent. What ghost installs
	# is simply not there YET - it is gray until it is downloading (blue) or has failed (red).
	var yours := Deps.group_of(r) == 2 and int(r.get("tier", Deps.TIER_EXTRA)) == Deps.TIER_FEATURE
	var glyph := "●" if found else ("▲" if yours else "○")
	var tint := COL_OK if found else (COL_BAD if yours else COL_IDLE)
	var value := String(r.get("version", ""))
	if value.is_empty():
		value = String(r.get("note", "not found"))
	var bar: ProgressBar = ui["bar"]
	if bool(s.get("running", false)):
		glyph = "◌"
		tint = COL_RUN
		var f := float(s.get("fraction", -1.0))
		value = String(s.get("phase", "working"))
		if f >= 0.0:
			# The number first: the column clips from the right, and the percent is the part
			# that moves.
			value = "%d%% · %s" % [int(f * 100.0), value]
		bar.visible = true
		bar.indeterminate = f < 0.0
		if f >= 0.0:
			bar.value = f * 100.0
	else:
		bar.visible = false
		if s.has("error") and not found:
			glyph = "▲"
			tint = COL_BAD
			value = "failed - click for why"
	(ui["glyph"] as Label).text = glyph
	(ui["glyph"] as Label).add_theme_color_override("font_color", tint)
	(ui["name"] as Label).add_theme_color_override("font_color", COL_TEXT if found else COL_DIM)
	(ui["value"] as Label).text = value
	(ui["value"] as Label).add_theme_color_override("font_color",
		COL_DIM if found and not bool(s.get("running", false)) else tint)


func _row(key: String) -> Dictionary:
	for r in _rows:
		if String(r.get("key", "")) == key:
			return r
	return {}


# --- rendering ---------------------------------------------------------------

func _render() -> void:
	for c in _list.get_children():
		c.queue_free()
	_ui = {}
	var group := -1
	for r in _rows:
		var g := Deps.group_of(r)
		if g != group:
			group = g
			_list.add_child(_group_label(GROUPS[g]))
		_list.add_child(_row_widget(r))
		_live(String(r.get("key", "")))
	_refresh_title()
	if not _open_key.is_empty():
		_fill_detail(_open_key)
	set_process(not Provision.active().is_empty())


func _group_label(text: String) -> Control:
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 2)
	if _list.get_child_count() > 0:
		box.add_child(HSeparator.new())
	var l := Label.new()
	l.text = text
	l.add_theme_font_size_override("font_size", 10)
	l.add_theme_color_override("font_color", COL_IDLE)
	box.add_child(l)
	return box


## The one-line state for the header. In order of what the reader most needs to know: a failure,
## something a feature needs that is missing, an install in progress, all well.
func _problem() -> String:
	for r in _rows:
		var key := String(r.get("key", ""))
		var s := Provision.state(key)
		if s.has("error") and not bool(s.get("running", false)) and not bool(r.get("found", false)) \
				and Provision.unsupported(key).is_empty():
			return "%s failed" % String(r.get("name", key))
	var missing := PackedStringArray()
	for r in _rows:
		if Deps.group_of(r) == 2 and int(r.get("tier", Deps.TIER_EXTRA)) == Deps.TIER_FEATURE \
				and not bool(r.get("found", false)):
			missing.append(String(r.get("name", "?")))
	if not missing.is_empty():
		return "missing: " + ", ".join(missing)
	return ""


## How many of the things ghost installs at launch are not there yet (and not failing).
func _pending() -> int:
	var n := 0
	for r in _rows:
		if Deps.group_of(r) == 0 and not bool(r.get("found", false)) \
				and Provision.unsupported(String(r.get("key", ""))).is_empty():
			n += 1
	return n


func _refresh_title() -> void:
	var running := Provision.active()
	var problem := _problem()
	if _rows.is_empty():
		_title.text = "Environment"
		_title.add_theme_color_override("font_color", COL_TEXT)
	elif not running.is_empty():
		var s := Provision.state(running[0])
		var name := String(Deps.entry(running[0]).get("name", running[0]))
		var f := float(s.get("fraction", -1.0))
		_title.text = "Environment  ·  %s %s%s" % [
			"updating" if String(s.get("action", "")) == "update" else "installing", name,
			("  %d%%" % int(f * 100.0)) if f >= 0.0 else ""]
		if running.size() > 1:
			_title.text += "  +%d" % (running.size() - 1)
		_title.add_theme_color_override("font_color", COL_RUN)
	elif not problem.is_empty():
		_title.text = "Environment  ·  " + problem
		_title.add_theme_color_override("font_color", COL_BAD)
	elif _pending() > 0:
		_title.text = "Environment  ·  %d to download" % _pending()
		_title.add_theme_color_override("font_color", COL_DIM)
	else:
		_title.text = "Environment  ·  all present"
		_title.add_theme_color_override("font_color", COL_OK)
	if _collapsed:
		_title.text += "   ▸"


func _refresh_footer() -> void:
	if _checked == null:
		return
	var agent := Provision.agent()
	var busy := agent != null and bool(agent.call("checking"))
	_check_now.disabled = busy or agent == null or not bool(agent.call("enabled"))
	if busy:
		_checked.text = "checking for updates…"
		return
	var at := int(agent.call("last_checked")) if agent != null else 0
	if at <= 0:
		_checked.text = "never checked"
		return
	var ago := int(Time.get_unix_time_from_system()) - at
	_checked.text = "checked " + ("just now" if ago < 120 else "%d min ago" % (ago / 60)
		if ago < 7200 else "%d h ago" % (ago / 3600) if ago < 172800 else "%d days ago" % (ago / 86400))


func _on_check_now() -> void:
	var agent := Provision.agent()
	if agent != null:
		agent.call("check_updates")
	_refresh_footer()


func _on_retry() -> void:
	var agent := Provision.agent()
	if agent != null and not _open_key.is_empty():
		agent.call("retry", _open_key)
		_fill_detail(_open_key)


## One row: the button the whole strip is, and a thin progress bar under it while a job runs. A
## flat [Button] rather than an [HBoxContainer] with a `gui_input` handler, so hover highlighting
## and keyboard focus come from the theme; the labels inside ignore the mouse so the whole strip
## stays one click target.
func _row_widget(r: Dictionary) -> Control:
	var key := String(r.get("key", ""))
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 0)

	var b := Button.new()
	b.focus_mode = Control.FOCUS_NONE
	b.custom_minimum_size = Vector2(0, 17)
	b.tooltip_text = String(r.get("used_for", "")) + "\n(click for more)"
	b.mouse_default_cursor_shape = Control.CURSOR_POINTING_HAND
	b.pressed.connect(_toggle_detail.bind(key))
	_style_row(b, key == _open_key)
	box.add_child(b)

	var row := HBoxContainer.new()
	row.mouse_filter = Control.MOUSE_FILTER_IGNORE
	row.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	row.add_theme_constant_override("separation", 6)
	b.add_child(row)

	var glyph := _cell("", COL_IDLE, 11, 12, HORIZONTAL_ALIGNMENT_CENTER)
	row.add_child(glyph)
	var name_cell := _cell(String(r.get("name", "?")), COL_TEXT, 11, 0, HORIZONTAL_ALIGNMENT_LEFT)
	name_cell.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row.add_child(name_cell)
	var value := _cell("", COL_DIM, 11, 156, HORIZONTAL_ALIGNMENT_RIGHT)
	row.add_child(value)

	var bar := ProgressBar.new()
	bar.mouse_filter = Control.MOUSE_FILTER_IGNORE
	bar.show_percentage = false
	bar.custom_minimum_size = Vector2(0, 3)
	var bg := StyleBoxFlat.new()
	bg.bg_color = Color(0.14, 0.17, 0.22)
	var fill := StyleBoxFlat.new()
	fill.bg_color = COL_RUN
	bar.add_theme_stylebox_override("background", bg)
	bar.add_theme_stylebox_override("fill", fill)
	bar.visible = false
	box.add_child(bar)

	_ui[key] = {"button": b, "glyph": glyph, "name": name_cell, "value": value, "bar": bar}
	return box


## A row's look: nothing at rest, a faint band under the pointer - so a row reads as something to
## click - and a stronger band for the row whose detail pane is OPEN, which stays until it is closed.
const ROW_HOVER := Color(0.16, 0.19, 0.25, 0.9)
const ROW_OPEN := Color(0.19, 0.26, 0.38, 0.95)

func _style_row(b: Button, open: bool) -> void:
	var rest := StyleBoxFlat.new()
	rest.bg_color = ROW_OPEN if open else Color(0, 0, 0, 0)
	rest.set_corner_radius_all(3)
	var over := StyleBoxFlat.new()
	over.bg_color = ROW_OPEN if open else ROW_HOVER
	over.set_corner_radius_all(3)
	if open:
		over.bg_color = ROW_OPEN.lightened(0.08)
	b.add_theme_stylebox_override("normal", rest)
	b.add_theme_stylebox_override("hover", over)
	b.add_theme_stylebox_override("pressed", over)
	b.add_theme_stylebox_override("hover_pressed", over)


func _cell(text: String, tint: Color, size: int, min_w: int, align: int) -> Label:
	var l := Label.new()
	l.text = text
	l.mouse_filter = Control.MOUSE_FILTER_IGNORE
	l.horizontal_alignment = align
	l.clip_text = true
	l.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
	l.add_theme_font_size_override("font_size", size)
	l.add_theme_color_override("font_color", tint)
	if min_w > 0:
		l.custom_minimum_size = Vector2(min_w, 0)
	return l


## Clicking a row opens what it is for, where it lives, and what is happening to it: a job's
## progress, a failure's reason and a retry, or - for the machine's own programs - the install
## command for THIS platform. Clicking the same row again closes it.
func _toggle_detail(key: String) -> void:
	var was := _open_key
	if key == _open_key:
		_open_key = ""
		_detail.visible = false
		_restyle(was)
		return
	if _row(key).is_empty():
		return
	_open_key = key
	_fill_detail(key)
	_detail.visible = true
	_restyle(was)
	_restyle(key)


func _fill_detail(key: String) -> void:
	var r := _row(key)
	if r.is_empty():
		return
	var s := Provision.state(key)
	var group := Deps.group_of(r)
	var lines: PackedStringArray = [String(r.get("name", "?")), String(r.get("used_for", ""))]
	var path := String(r.get("path", ""))
	var why := Provision.unsupported(key)
	if not why.is_empty():
		lines.append("Not available on this machine: %s." % why)
	elif bool(s.get("running", false)):
		lines.append(Provision.describe(key, s))
	elif s.has("error") and not bool(r.get("found", false)):
		lines.append("The last attempt failed: %s" % String(s["error"]))
		lines.append("Its log: " + Provision.log_path(key))
	if group == 2:
		if bool(r.get("found", false)):
			lines.append("found at: " + path)
		else:
			var hint := Deps.install_hint(r)
			if not hint.is_empty():
				lines.append("install:  " + hint)
	elif bool(r.get("found", false)):
		if group == 0 and not bool(r.get("own", true)):
			lines.append("Using this machine's copy (%s) until ghost's own has downloaded." % path)
		else:
			lines.append("at: " + path)
	elif why.is_empty() and not bool(s.get("running", false)):
		lines.append(("Downloads at launch" if group == 0 else "Installed the first time it is needed")
			+ (" · %s" % String(r.get("size", "")) if not String(r.get("size", "")).is_empty() else "")
			+ ((" · into " + path) if not path.is_empty() else "") + ".")
	_detail_text.text = "\n".join(lines)
	_retry.visible = group != 2 and s.has("error") and not bool(s.get("running", false)) \
		and why.is_empty() and Provision.can_install()
	_detail_link.text = String(r.get("site", ""))
	_detail_link.visible = not _detail_link.text.is_empty()


func _restyle(key: String) -> void:
	var ui: Dictionary = _ui.get(key, {})
	if not ui.is_empty() and is_instance_valid(ui["button"]):
		_style_row(ui["button"] as Button, key == _open_key)


# --- collapse state ([Settings] owns the file; see settings.gd) ---

func _toggle_collapsed() -> void:
	_collapsed = not _collapsed
	_body.visible = not _collapsed
	_refresh_title()
	Settings.write("deps", "collapsed", _collapsed)


func _load_collapsed() -> bool:
	return bool(Settings.read("deps", "collapsed", false))
