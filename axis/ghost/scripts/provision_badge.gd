extends CanvasLayer

## ProvisionBadge - a small notice at the top of the frame while ghost is installing or updating
## its own dependencies, in every mode. Chrome furniture, because an install can start from
## anywhere: the voice's environment from the Generative panel, FFmpeg from a clip opened on a
## first run, an update at launch. Hidden the rest of the time.
##
## It shows the first job running - what it is doing, how far along, for how long - and how many
## more are queued behind it; the home screen's Environment panel has the full list. It never
## takes a click: everything under it stays usable.

const COL_TEXT := Color(0.78, 0.84, 0.94)
const COL_BAR := Color(0.45, 0.66, 1.0)

var _panel: PanelContainer
var _label: Label
var _bar: ProgressBar
var _tick := 0.0


func _ready() -> void:
	layer = 252
	_panel = PanelContainer.new()
	_panel.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_panel.set_anchors_preset(Control.PRESET_CENTER_TOP)
	_panel.grow_horizontal = Control.GROW_DIRECTION_BOTH
	_panel.offset_top = 12
	var sb := StyleBoxFlat.new()
	sb.bg_color = Color(0.06, 0.07, 0.10, 0.88)
	sb.border_color = Color(0.20, 0.26, 0.36)
	sb.set_border_width_all(1)
	sb.set_corner_radius_all(6)
	sb.content_margin_left = 12
	sb.content_margin_right = 12
	sb.content_margin_top = 6
	sb.content_margin_bottom = 7
	_panel.add_theme_stylebox_override("panel", sb)
	add_child(_panel)

	var col := VBoxContainer.new()
	col.mouse_filter = Control.MOUSE_FILTER_IGNORE
	col.add_theme_constant_override("separation", 4)
	_panel.add_child(col)

	_label = Label.new()
	_label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_label.add_theme_font_size_override("font_size", 12)
	_label.add_theme_color_override("font_color", COL_TEXT)
	col.add_child(_label)

	_bar = ProgressBar.new()
	_bar.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_bar.show_percentage = false
	_bar.custom_minimum_size = Vector2(300, 4)
	var bg := StyleBoxFlat.new()
	bg.bg_color = Color(0.16, 0.19, 0.25)
	bg.set_corner_radius_all(2)
	var fill := StyleBoxFlat.new()
	fill.bg_color = COL_BAR
	fill.set_corner_radius_all(2)
	_bar.add_theme_stylebox_override("background", bg)
	_bar.add_theme_stylebox_override("fill", fill)
	col.add_child(_bar)

	var agent := Provision.agent()
	if agent != null:
		agent.connect("changed", _refresh)
	_refresh("")


func _process(dt: float) -> void:
	# The elapsed time in the line moves even when the job reports nothing new (uv resolving).
	_tick += dt
	if _tick >= 0.5:
		_tick = 0.0
		_refresh("")


func _refresh(_key: String) -> void:
	var keys := Provision.active()
	_panel.visible = not keys.is_empty()
	set_process(_panel.visible)
	if keys.is_empty():
		return
	var key := keys[0]
	var s := Provision.state(key)
	var line := ("Updating " if String(s.get("action", "")) == "update" else "Installing ") \
		+ Provision.describe(key, s)
	if keys.size() > 1:
		line += "   (+%d more)" % (keys.size() - 1)
	_label.text = line
	var f := float(s.get("fraction", -1.0))
	_bar.indeterminate = f < 0.0
	if f >= 0.0:
		_bar.value = f * 100.0
