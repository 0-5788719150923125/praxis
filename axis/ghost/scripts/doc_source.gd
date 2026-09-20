extends VBoxContainer
class_name DocSource

## DocSource - where a reading's words come from: the text box, or a file on disk.
##
## THE TEXT BOX IS A DRAFT, and a chapter is not a draft. Everything the voice panels read
## until now was pasted in, which has three costs that only show up on real material: the
## paste is a COPY, so it goes stale the moment the document is edited anywhere else; the
## panel's settings are ghost's rather than the document's, so moving to the next chapter
## silently inherits the last one's voice and coming back has lost it; and a fix to a
## sentence means finding the panel, clearing it and pasting again.
##
## So a panel gets a SOURCE, and it is a toggle rather than a mode: INPUT is the box exactly
## as it was, SYNC points at a file. The two do not share storage - the draft survives being
## away from it - and nothing downstream of the panel knows which is in use, because both
## end up as text in the same [TextEdit].
##
## SYNC IS READ AT EVERY SPEAK, not at every keystroke and not when the document was picked.
## The point of it is that the author keeps working in their own editor while ghost is open:
## save the file, press Speak, hear the change. A watcher polling for modifications would be
## the same thing done worse - it would re-read mid-reading, which is exactly when the file
## is most likely to be half-written.
##
## FRONTMATTER IS NOT SPOKEN, AND IT IS WHERE THE VOICE LIVES. [FrontMatter] cuts the YAML
## block off the top before a word reaches the synthesizer, and ghost keeps its own key in
## there: the reader, the tone, the room, the whole cast of a multi-speaker chapter. Picking
## a document restores its voice; ↑ writes the panel's current one back. Neither happens on
## its own - an import that fired at every Speak would undo a dial the author had just
## moved, and an export that fired on every change would be ghost writing to someone's
## manuscript without being asked.
##
## The panel supplies the two halves it alone can know, as Callables: [member capture]
## returns the settings to store, [member apply] takes the ones a document carried.

const FrontMatter_ := preload("res://scripts/front_matter.gd")

## A document was opened and its voice (if it had one) applied.
signal opened(path: String)
## The toggle moved. The panel re-syncs its text box off this.
signal mode_changed(sync: bool)

## The panel's settings, as they are right now -> the dictionary to store under this
## source's block. Set by the owner; a source without it simply cannot save a voice.
var capture: Callable
## The other direction: a document's stored block -> the panel. Called on open and on
## Reload, never on a Speak.
var apply: Callable

var _section := "generative"     # the [Settings] section this source's draft and path live in
var _block := "generative"       # our sub-key inside the document's `ghost:` frontmatter
var _text: TextEdit              # the panel's box, driven by this widget in sync mode
var _mode_input: Button
var _mode_sync: Button
var _row: HBoxContainer
var _name: Label
var _status: Label
var _path := ""
var _draft := ""                 # the pasted text, kept aside while a document is open
var _body := ""                  # the last body successfully read from _path
var _dialog: FileDialog = null
var _syncing := false            # a programmatic write to the box must not mark it edited


## Build the widget and restore the last source. Call before [method bind_text].
##
## [param section] is the [Settings] section the owner already uses (its draft and the
## remembered document path go there); [param block] is the key inside a document's
## frontmatter, so two panels reading the same chapter keep separate voices in it.
func setup(section: String, block: String) -> void:
	_section = section
	_block = block
	add_theme_constant_override("separation", 4)

	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 4)
	add_child(row)
	var group := ButtonGroup.new()
	_mode_input = _mode_button("Input", group,
		"Read what is typed in the box below. The draft is remembered between sessions, "
		+ "and it is kept while a document is open - switching back brings it straight back.")
	_mode_sync = _mode_button("Sync", group,
		"Read a file on disk instead, fresh at every Speak. Keep writing in your own "
		+ "editor while ghost is open: save, press Speak, and the reading is the file as it "
		+ "is now. YAML frontmatter at the top is never spoken - it is where the voice is "
		+ "kept (see ↑).")
	row.add_child(_mode_input)
	row.add_child(_mode_sync)
	var spacer := Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row.add_child(spacer)

	_row = HBoxContainer.new()
	_row.add_theme_constant_override("separation", 4)
	add_child(_row)
	_name = Label.new()
	_name.add_theme_font_size_override("font_size", 12)
	_name.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_name.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
	_row.add_child(_name)
	_row.add_child(_tool("Open…", "Pick the document to read.", _open_dialog))
	_row.add_child(_tool("⟳", "Re-read the file now, and take the voice from its "
		+ "frontmatter again. A Speak re-reads the WORDS on its own; this is for when the "
		+ "settings in the document have changed too.", func() -> void: reload()))
	_row.add_child(_tool("↑", "Write the panel's current settings into the document's "
		+ "frontmatter, under one `ghost:` key of ghost's own. Nothing else in the file is "
		+ "touched - not the body, not another key, not a comment - and the edit is refused "
		+ "outright if it would be.", func() -> void: save()))

	_status = Label.new()
	_status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_status.add_theme_font_size_override("font_size", 11)
	_status.modulate = Color(1, 1, 1, 0.65)
	_status.visible = false
	add_child(_status)

	_path = str(Settings.read(_section, "doc_path", ""))
	var want_sync: bool = bool(Settings.read(_section, "sync", false)) and not _path.is_empty()
	_syncing = true
	(_mode_sync if want_sync else _mode_input).button_pressed = true
	_syncing = false
	_refresh_row()


## Hand the widget the panel's text box. It owns what is IN it from here on: a synced body
## is written in read-only, and the draft is put back when the toggle returns to Input.
##
## Deliberately separate from [method setup] so the widget can be built above the box it
## drives - the toggle belongs over the text, and a Control is added where it is shown.
func bind_text(te: TextEdit) -> void:
	_text = te
	_draft = str(Settings.read(_section, "text", ""))
	if is_sync():
		# The body only - reading the frontmatter aloud is the first thing this prevents.
		var body := _read_body()
		_show(body if not body.is_empty() else _draft)
	else:
		_show(_draft)
	_apply_editable()


## True while THIS widget is writing the text box, so the panel's own `text_changed`
## handler can tell a document being shown from the author typing. Without it, a synced
## Speak marks its own reading stale the instant it starts.
func is_quiet() -> bool:
	return _syncing


## True when the reading comes from a file.
func is_sync() -> bool:
	return _mode_sync != null and _mode_sync.button_pressed and not _path.is_empty()


## The document's path, or "" in Input mode.
func doc_path() -> String:
	return _path if is_sync() else ""


## WHAT TO PERSIST as the panel's text. In sync mode this is the draft that was set aside,
## never the document's body - a panel that saved the body would overwrite the author's own
## draft with a copy of a file that is already on disk.
func draft() -> String:
	if is_sync():
		return _draft
	return _text.text if _text != null else _draft


## THE REAL-TIME HALF: the words to speak, read from the disk at this instant.
##
## Call it at the top of every Speak and every export. In Input mode it is just the box. A
## read that fails keeps the last body rather than falling silent - a file being saved
## under us is a moment, not a reason to stop a reading - and says so.
func pull() -> String:
	if not is_sync():
		return _text.text if _text != null else ""
	var body := _read_body()
	if body.is_empty() and not _body.is_empty():
		return _body
	_show(body)
	return body


## Re-read the document AND take its voice again. The ⟳ button, and what picking a document
## does. Returns false when the file could not be read.
func reload() -> bool:
	if not is_sync():
		_note("Nothing to re-read - the reading is coming from the box.")
		return false
	var raw: Variant = _read_raw()
	if raw == null:
		return false
	var parts := FrontMatter_.split(String(raw))
	_body = String(parts.body)
	_show(_body)
	_import(String(raw))
	return true


## Write the panel's settings into the document's frontmatter. The ↑ button.
##
## Everything careful about this is in [method FrontMatter.write_block]; what belongs here
## is that a refusal is REPORTED rather than swallowed, because the whole value of the
## button is knowing whether the document now carries the voice.
func save() -> bool:
	if not is_sync():
		_note("Open a document first - there is nowhere to save a voice to.")
		return false
	if not capture.is_valid():
		_note("This panel cannot save a voice.")
		return false
	var block: Variant = capture.call()
	if not (block is Dictionary):
		_note("This panel cannot save a voice.")
		return false
	# READ-MODIFY-WRITE OF ONE KEY. Whatever the document says about the OTHER panel's
	# voice is carried through untouched, so a chapter can hold both.
	var raw: Variant = _read_raw()
	if raw == null:
		return false
	var res := FrontMatter_.read_block(String(raw))
	var ghost: Dictionary = (res.data as Dictionary).duplicate(true)
	ghost[_block] = block
	var err := FrontMatter_.write_block(_path, ghost)
	if not err.is_empty():
		_note("⚠  " + err)
		return false
	_note("✓  Voice saved into %s" % _path.get_file())
	return true


# --- internals ----------------------------------------------------------------


func _mode_button(label: String, group: ButtonGroup, tip: String) -> Button:
	var b := Button.new()
	b.text = label
	b.toggle_mode = true
	b.button_group = group
	b.tooltip_text = tip
	b.custom_minimum_size = Vector2(60, 0)
	b.toggled.connect(func(on: bool) -> void:
		if on:
			_on_mode(label == "Sync"))
	return b


func _tool(label: String, tip: String, action: Callable) -> Button:
	var b := Button.new()
	b.text = label
	b.tooltip_text = tip
	b.pressed.connect(action)
	return b


## THE SWAP, and the reason the draft is a member rather than a read of Settings: the box
## holds the draft at the moment the toggle moves, and Settings holds whatever the panel's
## debounce last wrote, which is up to a second behind.
func _on_mode(sync: bool) -> void:
	if _syncing:
		return
	if sync and _path.is_empty():
		_refresh_row()
		_open_dialog()
		return
	if _text != null:
		if sync:
			_draft = _text.text
		else:
			_show(_draft)
	Settings.write(_section, "sync", sync)
	_refresh_row()
	_apply_editable()
	if sync:
		reload()
	else:
		_note("")
	mode_changed.emit(sync)


func _apply_editable() -> void:
	if _text == null:
		return
	# The box is a VIEW of the document in sync mode. It is still selectable and
	# scrollable - being able to read along is most of why it is shown at all - but typing
	# into it would be typing into something the next Speak overwrites.
	_text.editable = not is_sync()
	_text.placeholder_text = "Once upon a time..." if not is_sync() \
		else "The document's text appears here when it is read."


func _refresh_row() -> void:
	var sync: bool = _mode_sync != null and _mode_sync.button_pressed
	_row.visible = sync
	if _path.is_empty():
		_name.text = "no document"
		_name.tooltip_text = "Press Open… to pick one."
	else:
		_name.text = _path.get_file()
		_name.tooltip_text = _path


func _show(body: String) -> void:
	if _text == null:
		return
	if _text.text == body:
		return
	_syncing = true
	# The caret and the scroll, kept: a re-read at every Speak would otherwise throw the
	# reader back to the top of the chapter every time.
	var col := _text.get_caret_column()
	var line := _text.get_caret_line()
	var scroll := _text.scroll_vertical
	_text.text = body
	_text.set_caret_line(mini(line, maxi(0, _text.get_line_count() - 1)))
	_text.set_caret_column(col)
	_text.scroll_vertical = scroll
	_syncing = false


## The whole file, or null with the reason on screen.
func _read_raw() -> Variant:
	if _path.is_empty():
		_note("No document is open.")
		return null
	if not FileAccess.file_exists(_path):
		_note("⚠  %s is not there any more." % _path.get_file())
		return null
	var fh := FileAccess.open(_path, FileAccess.READ)
	if fh == null:
		_note("⚠  Could not read %s (error %d)." % [_path.get_file(), FileAccess.get_open_error()])
		return null
	var raw := fh.get_as_text()
	fh.close()
	return raw


## The body alone, for a Speak: no voice import, no status chatter on success.
func _read_body() -> String:
	var raw: Variant = _read_raw()
	if raw == null:
		return ""
	var parts := FrontMatter_.split(String(raw))
	_body = String(parts.body)
	return _body


## Take the voice out of a document and hand it to the panel.
func _import(raw: String) -> void:
	var res := FrontMatter_.read_block(raw)
	if not res.ok:
		_note("Read %s. Its frontmatter could not be parsed (%s), so the voice is unchanged."
			% [_path.get_file(), String(res.error)])
		return
	var ghost := res.data as Dictionary
	var mine: Variant = ghost.get(_block, null)
	if mine is Dictionary and apply.is_valid():
		apply.call(mine as Dictionary)
		_note("Read %s and restored its voice." % _path.get_file())
	else:
		_note("Read %s. It carries no voice yet - press ↑ to put this one in it."
			% _path.get_file())


func _open_dialog() -> void:
	if _dialog != null and is_instance_valid(_dialog):
		return
	_dialog = FileDialog.new()
	_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	_dialog.access = FileDialog.ACCESS_FILESYSTEM
	# In-window, never native - the portal dialog shows nothing at all on a Linux box
	# without xdg-desktop-portal, which is the "I pressed it and nothing happened" report
	# the film importer already carries this note for.
	_dialog.use_native_dialog = false
	_dialog.title = "Read from a document"
	_dialog.filters = PackedStringArray([
		"*.md, *.markdown, *.txt ; Text", "* ; Every file"])
	if not _path.is_empty():
		_dialog.current_dir = _path.get_base_dir()
	elif not OS.get_system_dir(OS.SYSTEM_DIR_DOCUMENTS).is_empty():
		_dialog.current_dir = OS.get_system_dir(OS.SYSTEM_DIR_DOCUMENTS)
	_dialog.size = Vector2i(820, 560)
	_dialog.file_selected.connect(_on_picked)
	_dialog.canceled.connect(_close_dialog)
	add_child(_dialog)
	_dialog.popup_centered()


func _on_picked(path: String) -> void:
	_close_dialog()
	if _text != null and not is_sync():
		_draft = _text.text     # the box is about to become the document's
	_path = path
	Settings.write(_section, "doc_path", _path)
	Settings.write(_section, "sync", true)
	_syncing = true
	_mode_sync.button_pressed = true
	_syncing = false
	_refresh_row()
	_apply_editable()
	if reload():
		opened.emit(_path)
	mode_changed.emit(true)


func _close_dialog() -> void:
	if _dialog != null and is_instance_valid(_dialog):
		_dialog.queue_free()
	_dialog = null


func _note(msg: String) -> void:
	_status.text = msg
	_status.visible = not msg.is_empty()
