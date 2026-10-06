extends VBoxContainer
class_name ScriptWriter

## ScriptWriter - where a voice panel's script is written: a card in the panel, and an editor
## window it opens.
##
## THE TEXT BOX OUTGREW THE PANEL. A script carries speaker cues, hesitations, picture
## markers, pronunciations, macros and emphasis, and a 380-pixel box with a tooltip was the
## only place any of that was written down - so an author who had not read the source had no
## way to find out what ghost understands. The panel now holds a CARD (where the words come
## from, how many there are, how many voices and pictures they ask for) and the writing
## happens in a centered window: the text on the left, highlighted by what the panel will make
## of it ([ScriptHighlighter]), and on the right a palette of every mark this panel honors
## ([ScriptMarks]), each one a click away from being inserted at the caret.
##
## THE FRONTMATTER IS NOT IN THE EDITOR. It is metadata the panel owns - the voice, the look -
## and every field in it is a control on the panel already, so the editor shows the body and
## [DocSource] keeps the frontmatter on disk exactly as it was, ghost's block written by the
## panel and everything else untouched.
##
## The window's [CodeEdit] IS the panel's text box: the panel reads `text_edit.text` and
## listens to its `text_changed` exactly as it did the old one, and [member doc] drives it.

## The document's own metadata - top-level frontmatter keys, not ghost's block - edited on the
## card, never in the editor. Each is read by something: the label, its tooltip, the panels
## that use it.
const FIELDS := {
	"title": {"label": "Title", "modes": ["generative", "tarot"],
		"tip": "The chapter's title. Read aloud first, by the voice that opens the chapter, and set at the head of the chapter in the Novel and Notebook. For a tarot show, the channel's name."},
	"byline": {"label": "Byline", "modes": ["tarot"],
		"tip": "A line under the channel's name on the title screen while the intro holds - \"with Pen & Ink\". Empty, the name stands alone."},
	"author": {"label": "Author", "modes": ["generative"],
		"tip": "Who wrote it. Printed on the cover in the Novel and Notebook."},
	"book": {"label": "Book", "modes": ["generative"],
		"tip": "The book this chapter belongs to. Printed on the cover in the Novel and Notebook."},
}

## WHAT THE DOCUMENT IS CALLED, where it is not a script: a tarot show's body is the BRIEF its
## agents work from - written by hand, never read aloud.
const WORDING := {
	"tarot": {"edit": "Edit brief…", "window": "Brief",
		"edit_tip": "Open the show's brief in an editor: what the show is, who reads it and by what rules. Every agent behind every episode is handed it word for word.",
		"placeholder": "What is this show? Who reads it, how, and by what rules?"},
}

## How long typing must pause before the card's summary is recounted.
const SUMMARY_MS := 0.4

## Where the words come from (a synced file, or none), built into the window's top bar.
var doc: DocSource
## The editor. The owning panel treats this as its text box.
var text_edit: CodeEdit

var _mode := "generative"
var _window: Window
var _name: Label
var _summary: Label
var _note: Label
var _help: Label
var _speakers: HFlowContainer
var _speakers_head: Label
var _timer: Timer
var _speaker_names := PackedStringArray()
var _field_edits := {}          # key -> LineEdit


func _wording(key: String, dflt: String) -> String:
	return String((WORDING.get(_mode, {}) as Dictionary).get(key, dflt))


## Build the card, the window and the [DocSource] inside it. [param section] and [param block]
## are DocSource's (the Settings section, the frontmatter sub-key); [param mode] picks which
## marks the palette offers and the highlighter colors ("generative", "synthesis").
func setup(section: String, block: String, mode: String) -> void:
	_mode = mode
	add_theme_constant_override("separation", 4)

	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 6)
	add_child(row)
	_name = Label.new()
	_name.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_name.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
	_name.add_theme_font_size_override("font_size", 14)
	row.add_child(_name)
	var open := Button.new()
	open.text = "Open…"
	open.pressed.connect(func() -> void: doc.open())
	row.add_child(open)
	var clear := Button.new()
	clear.text = "Clear"
	clear.pressed.connect(func() -> void: doc.clear())
	row.add_child(clear)
	var edit := Button.new()
	edit.text = String(_wording("edit", "Edit script…"))
	edit.tooltip_text = String(_wording("edit_tip", "Open the script in an editor, with every mark "
		+ "this panel understands listed beside it - speakers, hesitations, pictures, "
		+ "pronunciations - ready to insert at the cursor."))
	edit.pressed.connect(open_editor)
	row.add_child(edit)
	_summary = Label.new()
	_summary.add_theme_font_size_override("font_size", 12)
	_summary.modulate = Color(1, 1, 1, 0.65)
	add_child(_summary)
	_build_fields()
	_note = Label.new()
	_note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_note.add_theme_font_size_override("font_size", 11)
	_note.modulate = Color(1, 1, 1, 0.65)
	_note.visible = false
	add_child(_note)

	_timer = Timer.new()
	_timer.one_shot = true
	_timer.wait_time = SUMMARY_MS
	_timer.timeout.connect(_refresh)
	add_child(_timer)

	_build_window(section, block)
	open.tooltip_text = DocSource.OPEN_TIP
	clear.tooltip_text = DocSource.CLEAR_TIP
	doc.dialog_host = self
	doc.noted.connect(func(msg: String) -> void:
		_note.text = msg
		_note.visible = not msg.is_empty())
	doc.mode_changed.connect(func(_sync: bool) -> void: _refresh())
	doc.opened.connect(func(_path: String) -> void: _refresh())
	text_edit.text_changed.connect(_recount_soon)
	text_edit.text_set.connect(_recount_soon)
	_refresh()


## The FIELDS this panel uses, as a two-column form under the summary. An edit is committed on
## Enter or on leaving the box (not per keystroke - synced, each commit is a write to the file),
## and the box then shows what the source holds, so a refused write is visible as a revert.
func _build_fields() -> void:
	var grid := GridContainer.new()
	grid.columns = 2
	grid.add_theme_constant_override("h_separation", 8)
	grid.add_theme_constant_override("v_separation", 2)
	for k in FIELDS:
		var f: Dictionary = FIELDS[k]
		if not (f["modes"] as Array).has(_mode):
			continue
		var l := Label.new()
		l.text = String(f["label"])
		l.add_theme_font_size_override("font_size", 12)
		l.tooltip_text = String(f["tip"])
		l.mouse_filter = Control.MOUSE_FILTER_STOP
		grid.add_child(l)
		var e := LineEdit.new()
		e.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		e.add_theme_font_size_override("font_size", 12)
		e.tooltip_text = String(f["tip"])
		e.placeholder_text = "none"
		var key := String(k)
		e.text_submitted.connect(func(_t: String) -> void:
			_commit_field(key)
			e.release_focus())
		e.focus_exited.connect(_commit_field.bind(key))
		grid.add_child(e)
		_field_edits[key] = e
	if grid.get_child_count() > 0:
		add_child(grid)
	else:
		grid.free()


func _commit_field(key: String) -> void:
	var e: LineEdit = _field_edits[key]
	doc.set_field(key, e.text)
	e.text = doc.field(key)


func _recount_soon() -> void:
	if _timer.is_inside_tree():
		_timer.start()
	else:
		_refresh()


## Show the editor, centered and sized to the window it opens over.
func open_editor() -> void:
	_refresh()
	var host := get_tree().root.get_visible_rect().size if is_inside_tree() \
		else Vector2(1600, 900)
	var want := Vector2i(int(host.x * 0.82), int(host.y * 0.82))
	want = want.max(_window.min_size)
	# THE SIZE IS SET AGAIN AFTER THE POPUP: measured, popup() and popup_centered() both
	# came back 1280 wide whatever was asked for (1574 on a 1080p window), which also left
	# popup_centered's window left of center.
	_window.popup(Rect2i((Vector2i(host) - want) / 2, want))
	_window.size = want
	text_edit.grab_focus()


func _build_window(section: String, block: String) -> void:
	_window = Window.new()
	_window.title = _wording("window", "Script")
	_window.visible = false
	_window.transient = true
	_window.min_size = Vector2i(760, 440)
	_window.close_requested.connect(_window.hide)
	_window.window_input.connect(func(e: InputEvent) -> void:
		if e is InputEventKey and e.pressed and e.keycode == KEY_ESCAPE:
			_window.hide())
	add_child(_window)

	var bg := PanelContainer.new()
	bg.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_window.add_child(bg)
	var margin := MarginContainer.new()
	for side in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 10)
	bg.add_child(margin)
	var col := VBoxContainer.new()
	col.add_theme_constant_override("separation", 8)
	margin.add_child(col)

	doc = preload("res://scripts/doc_source.gd").new()
	doc.setup(section, block)
	col.add_child(doc)

	var split := HSplitContainer.new()
	split.size_flags_vertical = Control.SIZE_EXPAND_FILL
	col.add_child(split)

	text_edit = CodeEdit.new()
	text_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	text_edit.size_flags_vertical = Control.SIZE_EXPAND_FILL
	text_edit.custom_minimum_size = Vector2(420, 240)
	text_edit.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	text_edit.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	text_edit.placeholder_text = String(_wording("placeholder", "Once upon a time..."))
	text_edit.highlight_current_line = true
	text_edit.caret_blink = true
	# Prose, not code: nothing may type a character the author did not.
	text_edit.auto_brace_completion_enabled = false
	text_edit.indent_automatic = false
	text_edit.code_completion_enabled = false
	text_edit.gutters_draw_line_numbers = false
	text_edit.add_theme_font_size_override("font_size", 16)
	text_edit.add_theme_constant_override("line_spacing", 6)
	text_edit.syntax_highlighter = ScriptHighlighter.new(_mode)
	split.add_child(text_edit)
	split.add_child(_build_palette())

	var foot := HBoxContainer.new()
	col.add_child(foot)
	var hint := Label.new()
	hint.text = ("Frontmatter is not shown here: the panel's settings are written into it on "
		+ "their own. A synced file takes your edits a moment after you stop typing.")
	hint.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	hint.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	hint.add_theme_font_size_override("font_size", 11)
	hint.modulate = Color(1, 1, 1, 0.55)
	foot.add_child(hint)
	var close := Button.new()
	close.text = "Done"
	close.pressed.connect(_window.hide)
	foot.add_child(close)


## THE PALETTE: one button per mark this panel honors, grouped and colored as the
## highlighter colors them, with what each does shown below when it is pointed at.
func _build_palette() -> Control:
	var side := VBoxContainer.new()
	side.custom_minimum_size = Vector2(270, 0)
	side.add_theme_constant_override("separation", 6)
	var head := Label.new()
	head.text = "Insert"
	head.add_theme_font_size_override("font_size", 16)
	side.add_child(head)
	var intro := Label.new()
	intro.text = "Click to insert at the cursor. A selection becomes the mark's text."
	intro.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	intro.add_theme_font_size_override("font_size", 11)
	intro.modulate = Color(1, 1, 1, 0.6)
	side.add_child(intro)

	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	side.add_child(scroll)
	var box := VBoxContainer.new()
	box.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	box.add_theme_constant_override("separation", 2)
	scroll.add_child(box)

	var keys := ScriptMarks.for_mode(_mode)
	for g in ScriptMarks.GROUPS:
		var color: Color = ScriptMarks.GROUPS[g]["color"]
		var mine: Array = []
		for k in keys:
			if String(ScriptMarks.REGISTRY[k]["group"]) == g:
				mine.append(k)
		if mine.is_empty():
			continue
		var gl := Label.new()
		gl.text = String(ScriptMarks.GROUPS[g]["label"])
		gl.add_theme_font_size_override("font_size", 12)
		gl.add_theme_color_override("font_color", color)
		if box.get_child_count() > 0:
			box.add_child(_gap(6))
		box.add_child(gl)
		for k in mine:
			box.add_child(_mark_button(String(k), color))
		if g == "voices":
			_speakers_head = Label.new()
			_speakers_head.text = "Speakers in this script"
			_speakers_head.add_theme_font_size_override("font_size", 11)
			_speakers_head.modulate = Color(1, 1, 1, 0.6)
			box.add_child(_speakers_head)
			_speakers = HFlowContainer.new()
			box.add_child(_speakers)

	_help = Label.new()
	_help.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_help.custom_minimum_size = Vector2(0, 110)
	_help.add_theme_font_size_override("font_size", 12)
	_help.text = "Point at a mark to see what it does."
	_help.modulate = Color(1, 1, 1, 0.8)
	side.add_child(HSeparator.new())
	side.add_child(_help)
	return side


func _mark_button(key: String, color: Color) -> Button:
	var e: Dictionary = ScriptMarks.REGISTRY[key]
	var b := Button.new()
	b.text = String(e["label"])
	b.alignment = HORIZONTAL_ALIGNMENT_LEFT
	b.flat = true
	b.add_theme_color_override("font_color", color)
	b.tooltip_text = "%s\n\n%s" % [ScriptMarks.example(key), String(e["blurb"])]
	b.pressed.connect(func() -> void:
		ScriptMarks.insert(text_edit, key)
		text_edit.grab_focus())
	b.mouse_entered.connect(_explain.bind(key))
	b.focus_entered.connect(_explain.bind(key))
	return b


func _explain(key: String) -> void:
	var e: Dictionary = ScriptMarks.REGISTRY[key]
	_help.text = "%s\n%s\n\n%s" % [String(e["label"]), ScriptMarks.example(key),
		String(e["blurb"])]


func _gap(px: int) -> Control:
	var c := Control.new()
	c.custom_minimum_size = Vector2(0, px)
	return c


## Recount the card and refresh what depends on the text: the summary, the window title and
## the speakers already in the script (each a one-click cue).
func _refresh() -> void:
	if text_edit == null:
		return
	var text := text_edit.text
	var source := doc.doc_path().get_file() if doc.is_sync() \
		else "Unsynced %s" % _wording("window", "Script").to_lower()
	_name.text = source
	_name.tooltip_text = ("Synced to " + doc.doc_path()) if doc.is_sync() else \
		"Kept by Ghost Notes alone, in no file. Sync to… in the editor writes it into one."
	_window.title = "%s - %s" % [_wording("window", "Script"), source]
	for k in _field_edits:
		var e: LineEdit = _field_edits[k]
		if not e.has_focus():
			e.text = doc.field(k)
	var spoken := Manuscript.unspoken(text)
	var parts := PackedStringArray([_count(Manuscript._rx("\\S+").search_all(spoken).size(), "word")])
	if _mode == "generative":
		var names := Manuscript.speakers(text)
		parts.append(_count(names.size(), "voice"))
		var pics := Manuscript.images(text).size()
		if pics > 0:
			parts.append(_count(pics, "picture"))
		if names != _speaker_names:
			_speaker_names = names
			_rebuild_speakers()
	_summary.text = " · ".join(parts)


func _rebuild_speakers() -> void:
	if _speakers == null:
		return
	for c in _speakers.get_children():
		c.queue_free()
	var cued := _speaker_names.duplicate()
	if cued.size() > 0 and cued[0] == Manuscript.NARRATOR \
			and not ("speaker: " + Manuscript.NARRATOR) in text_edit.text:
		cued.remove_at(0)             # the narrator is implied, not cued - nothing to repeat
	_speakers_head.visible = not cued.is_empty()
	for n in cued:
		var b := Button.new()
		b.text = n
		b.tooltip_text = "Insert a cue handing the text after it to %s." % n
		b.add_theme_font_size_override("font_size", 12)
		b.pressed.connect(func() -> void:
			ScriptMarks.insert(text_edit, "speaker", n)
			text_edit.grab_focus())
		_speakers.add_child(b)


static func _count(n: int, noun: String) -> String:
	var digits := str(n)
	var out := ""
	for i in digits.length():
		if i > 0 and (digits.length() - i) % 3 == 0:
			out += ","
		out += digits[i]
	return "%s %s%s" % [out, noun, "" if n == 1 else "s"]
