extends CanvasLayer
class_name Workspace

## Workspace - the manual-mode authoring surface (scaffolding).
##
## Opened by the splash's Manual button, over a session already running the default
## storyboard. A left-side panel lists the storyboards in storyboards/; clicking one switches
## the Director to it (manual mode) and jumps to its first scene. The visualizer
## keeps playing behind the panel - this is the canvas the future manual editor will
## live in (per-entry params, reordering, timeline, save). For now it is just the
## storyboard list + the scene behind it; everything else is intentionally stubbed.

const PANEL_W := 300

var _panel: PanelContainer
var _reopen: Button
var _list: VBoxContainer
var _buttons := {}            # storyboard name -> Button
var _active := ""


func _ready() -> void:
	layer = 100               # above the scene, below the feedback console (128)
	_build_ui()
	# Reflect the storyboard the session started on (default), if any.
	_active = Director.storyboard_name()
	_restyle()


func _build_ui() -> void:
	_panel = PanelContainer.new()
	_panel.set_anchors_preset(Control.PRESET_LEFT_WIDE)
	_panel.offset_right = PANEL_W
	add_child(_panel)

	var margin := MarginContainer.new()
	for side in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 14)
	_panel.add_child(margin)

	var col := VBoxContainer.new()
	col.add_theme_constant_override("separation", 10)
	margin.add_child(col)

	var header := HBoxContainer.new()
	col.add_child(header)
	var title := Label.new()
	title.text = "Workspace"
	title.add_theme_font_size_override("font_size", 24)
	title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	header.add_child(title)
	var hide_btn := _flat_button("‹")
	hide_btn.custom_minimum_size = Vector2(30, 0)
	hide_btn.pressed.connect(_collapse)
	header.add_child(hide_btn)

	var sub := Label.new()
	sub.text = "Storyboards"
	sub.add_theme_color_override("font_color", Color(0.6, 0.68, 0.8))
	col.add_child(sub)

	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	col.add_child(scroll)

	_list = VBoxContainer.new()
	_list.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_list.add_theme_constant_override("separation", 4)
	scroll.add_child(_list)
	_populate_storyboards()

	var note := Label.new()
	note.text = "scaffolding — editing tools coming"
	note.add_theme_color_override("font_color", Color(0.4, 0.46, 0.56))
	note.add_theme_font_size_override("font_size", 12)
	col.add_child(note)

	# Collapsed-state reopen tab.
	_reopen = _flat_button("Storyboards ›")
	_reopen.set_anchors_preset(Control.PRESET_TOP_LEFT)
	_reopen.position = Vector2(10, 10)
	_reopen.visible = false
	add_child(_reopen)
	_reopen.pressed.connect(_expand)


func _populate_storyboards() -> void:
	var dir := DirAccess.open("res://storyboards")
	if dir == null:
		return
	var names := []
	for fn in dir.get_files():
		if fn.ends_with(".json"):
			names.append(fn.get_basename())
	names.sort()
	for n in names:
		var b := _flat_button(n)
		b.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		b.alignment = HORIZONTAL_ALIGNMENT_LEFT
		b.pressed.connect(_select.bind(n))
		_list.add_child(b)
		_buttons[n] = b


# Switch the live session to a storyboard and jump to its first scene.
func _select(name: String) -> void:
	if Director.load_storyboard(name):
		_active = name
		Director.next()
		_restyle()


# Mark the active storyboard in the list.
func _restyle() -> void:
	for n in _buttons:
		var b: Button = _buttons[n]
		b.text = ("▶ " + n) if n == _active else ("   " + n)


func _collapse() -> void:
	_panel.visible = false
	_reopen.visible = true


func _expand() -> void:
	_panel.visible = true
	_reopen.visible = false


# A button that doesn't steal keyboard focus (so Space/Esc keep their global roles).
func _flat_button(text: String) -> Button:
	var b := Button.new()
	b.text = text
	b.focus_mode = Control.FOCUS_NONE
	return b
