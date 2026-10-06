extends HFlowContainer

## TagField - tags as chips, the way YouTube Studio shows them: every tag visible, wrapping onto as
## many lines as it needs, each with an × that removes it, and a box at the end where a comma (or
## Enter, or leaving the box, or pasting "a, b, c") turns what was typed into chips. Backspace in
## the empty box removes the last chip. Loaded by path (no class_name), as youtube.gd is.
##
## Two kinds of chip: the field's own, and FIXED ones shown first - tags that go up with these but
## are kept elsewhere (a show's own tags), outlined, with no × because they are not this field's
## to remove. [method mark] dims a chip that will not go up and says why in its tooltip.

## The field's own tags changed: a chip was added or removed.
signal changed

## Longest a chip's text is drawn, in px - a longer tag is cut with an ellipsis and its tooltip
## holds it whole - so one tag can never widen the panel it sits in.
const MAX_CHIP := 250.0
const FONT_SIZE := 12

var editable := true:
	set(v):
		editable = v
		_input.editable = v
		_rebuild()

var _tags := PackedStringArray()
var _fixed := PackedStringArray()
var _fixed_tip := ""
var _marks := {}           # a tag, lowercased -> why it will not go up
var _input := LineEdit.new()


func _init() -> void:
	add_theme_constant_override("h_separation", 4)
	add_theme_constant_override("v_separation", 4)
	_input.placeholder_text = "Add tags, separated by commas"
	_input.custom_minimum_size = Vector2(150, 0)
	_input.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_input.add_theme_font_size_override("font_size", FONT_SIZE)
	_input.text_changed.connect(_on_typed)
	_input.text_submitted.connect(func(_t: String) -> void: commit())
	_input.focus_exited.connect(commit)
	_input.gui_input.connect(_on_key)
	add_child(_input)


## The field's own tags, replaced (no [signal changed]: this is the field being filled, not edited).
func set_tags(tags: PackedStringArray) -> void:
	_tags = _unique(tags)
	_rebuild()


## The field's own tags, with anything still typed in the box.
func get_tags() -> PackedStringArray:
	var out := _tags.duplicate()
	for t in _split(_input.text):
		if not _has(out, t):
			out.append(t)
	return out


## The fixed tags shown first, and the tooltip that says where they are kept.
func set_fixed(tags: PackedStringArray, tip := "") -> void:
	_fixed = _unique(tags)
	_fixed_tip = tip
	_rebuild()


## Chips that will not go up: a tag (any case) -> why. Dimmed, the reason in the tooltip.
func mark(reasons: Dictionary) -> void:
	var low := {}
	for k in reasons:
		low[String(k).to_lower()] = String(reasons[k])
	if low == _marks:
		return
	_marks = low
	_rebuild()


## Whether the box has the keyboard - a field being typed in is not refilled under the typist.
func typing() -> bool:
	return _input.has_focus()


## Turn what is typed in the box into chips.
func commit() -> void:
	var typed := _split(_input.text)
	_input.text = ""
	_add(typed)


func remove(tag: String) -> void:
	var at := -1
	for i in _tags.size():
		if _tags[i].to_lower() == tag.to_lower():
			at = i
	if at < 0:
		return
	_tags.remove_at(at)
	_rebuild()
	changed.emit()


func _add(tags: PackedStringArray) -> void:
	var grew := false
	for t in tags:
		if not _has(_tags, t):
			_tags.append(t)
			grew = true
	if grew:
		_rebuild()
		changed.emit()


## A comma ends a tag: everything before the last one becomes chips, the rest stays to be typed on.
func _on_typed(text: String) -> void:
	if not text.contains(","):
		return
	var cut := text.rfind(",")
	_input.text = text.substr(cut + 1).lstrip(" ")
	_input.caret_column = _input.text.length()
	_add(_split(text.substr(0, cut)))


func _on_key(e: InputEvent) -> void:
	var k := e as InputEventKey
	if k != null and k.pressed and not k.echo and k.keycode == KEY_BACKSPACE \
			and _input.text.is_empty() and not _tags.is_empty() and editable:
		remove(_tags[_tags.size() - 1])
		_input.accept_event()


func _rebuild() -> void:
	for c in get_children():
		if c != _input:
			remove_child(c)
			c.queue_free()
	for t in _fixed:
		add_child(_chip(t, true))
	for t in _tags:
		add_child(_chip(t, false))
	move_child(_input, get_child_count() - 1)


func _chip(tag: String, fixed: bool) -> Control:
	var chip := PanelContainer.new()
	var box := StyleBoxFlat.new()
	box.bg_color = Color(1, 1, 1, 0.03 if fixed else 0.13)
	box.border_color = Color(1, 1, 1, 0.35)
	box.set_border_width_all(1 if fixed else 0)
	box.set_corner_radius_all(11)
	box.content_margin_left = 9
	box.content_margin_right = 9 if fixed or not editable else 3
	box.content_margin_top = 2
	box.content_margin_bottom = 2
	chip.add_theme_stylebox_override("panel", box)
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 1)
	row.mouse_filter = Control.MOUSE_FILTER_PASS
	chip.add_child(row)
	var label := Label.new()
	label.text = tag
	label.add_theme_font_size_override("font_size", FONT_SIZE)
	label.clip_text = true
	label.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
	var w := label.get_theme_font("font").get_string_size(tag, HORIZONTAL_ALIGNMENT_LEFT, -1, FONT_SIZE).x
	label.custom_minimum_size = Vector2(minf(ceilf(w) + 2.0, MAX_CHIP), 0)
	label.mouse_filter = Control.MOUSE_FILTER_PASS
	row.add_child(label)
	var why := String(_marks.get(tag.to_lower(), ""))
	var tip := tag
	if fixed and not _fixed_tip.is_empty():
		tip += "\n\n" + _fixed_tip
	if not why.is_empty():
		tip += "\n\nNot going up: " + why + "."
		chip.modulate.a = 0.45
	chip.tooltip_text = tip
	if not fixed and editable:
		var x := Button.new()
		x.text = "×"
		x.flat = true
		x.focus_mode = Control.FOCUS_NONE
		x.add_theme_font_size_override("font_size", FONT_SIZE + 2)
		x.tooltip_text = "Remove \"%s\"" % tag
		x.pressed.connect(remove.bind(tag))
		row.add_child(x)
	return chip


## [param text] split at its commas, each piece trimmed and unquoted, the empty ones dropped.
static func _split(text: String) -> PackedStringArray:
	var out := PackedStringArray()
	for part in text.split(","):
		var t := String(part).strip_edges()
		if t.length() >= 2 and (t[0] == "\"" or t[0] == "'") and t[t.length() - 1] == t[0]:
			t = t.substr(1, t.length() - 2).strip_edges()
		if not t.is_empty():
			out.append(t)
	return out


static func _has(tags: PackedStringArray, tag: String) -> bool:
	for t in tags:
		if t.to_lower() == tag.to_lower():
			return true
	return false


static func _unique(tags: PackedStringArray) -> PackedStringArray:
	var out := PackedStringArray()
	for t in tags:
		var s := t.strip_edges()
		if not s.is_empty() and not _has(out, s):
			out.append(s)
	return out
