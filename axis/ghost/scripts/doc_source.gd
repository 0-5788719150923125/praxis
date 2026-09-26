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
## there: the reader, the tone, the room, the whole cast of a multi-speaker chapter.
##
## BOTH DIRECTIONS FOLLOW THE AUTHOR, with no button for either. SAVING is automatic - a
## dial moved is a dial written, on a quiet period, exactly the way the rest of ghost saves
## (see [Settings]). LOADING happens at every Speak and export ([method pull]), because the
## file is the authoritative source - and never on a timer, because an import firing by
## itself would undo a dial the author had just moved. A Speak cannot do that: it writes any
## change still waiting first, so what it reads back already holds it.
##
## What makes the automatic direction safe is not that the write is small, it is that the
## write is CHECKED: [method FrontMatter.write_block] re-reads from disk, replaces one key
## textually, refuses outright if the body or any other key would change, and renames a
## verified temp file over the original. Doing that on a timer is the same operation as
## doing it on a button, minus the button.
##
## ...AND NOT IN AN UNATTENDED PROCESS. An export render boots this whole app against the
## author's settings, which is exactly how a render would come to edit a manuscript nobody
## is watching. Automatic saving is refused wherever [Settings] is read-only - a render, the
## offline analyzer, a test probe. Flushing on Speak is not, because pressing Speak is a person.
##
## The panel supplies the two halves it alone can know, as Callables: [member capture]
## returns the settings to store, [member apply] takes the ones a document carried.

const FrontMatter_ := preload("res://scripts/front_matter.gd")

## How long the settings must stop moving before the document is written, in ms. Longer than
## [constant Settings.DEBOUNCE_MS] on purpose: ghost's own config is a file nobody else has
## open, and the author's manuscript may well be open in their editor right now.
const AUTOSAVE_MS := 1200
## How often the snapshot is taken. Comparing it is cheap, but not free enough to do per frame.
const POLL_MS := 250

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
# AUTOSAVE. `_seen` is the last settings snapshot observed, `_saved` the last one written to
# the document, and `_settled_ms` when `_seen` stopped changing. Three values rather than one
# flag because a drag must not be written mid-drag: a quiet period is "unchanged since", which
# a dirty bit cannot express.
var _seen := ""
var _saved := ""
var _settled_ms := 0
var _polled_ms := 0
var _autosave_note := ""         # last failure said, so a retry does not repeat it
## Let a gate that is specifically testing the autosave run it in a probe. Mirrors - and is
## deliberately separate from - [method Settings.allow_writes_for_test]: flipping the global
## would also let the gate write the author's real config, and the thing under test here is a
## file of the gate's own.
var _autosave_for_test := false


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
		+ "kept, written there on its own a moment after you stop adjusting a setting.")
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
##
## IN SYNC MODE THE DOCUMENT IS THE SOURCE OF TRUTH, from the first frame, and that is a
## correction the autosave forced. This used to read only the BODY at boot and leave the
## voice as ghost's own settings had it - on the argument that the panel's last state is what
## the author left there. That is defensible while saving is a button and indefensible once it
## is automatic: the panel would come up holding the LAST document's voice and write it over
## THIS one's a second later. Whichever way the asymmetry falls it has to fall the same way in
## both directions, and for a file the author owns, the file wins.
func bind_text(te: TextEdit) -> void:
	_text = te
	_draft = str(Settings.read(_section, "text", ""))
	if is_sync():
		# The body AND the voice - and `reload` seeds the autosave's snapshot, so opening
		# ghost on a document is not followed by ghost writing that document.
		if not reload():
			_show(_draft)
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


## THE REAL-TIME READ: the document as it is on disk at this instant - its words AND its
## voice. The file is the authoritative source, so every Speak and every export starts here
## and there is no separate "re-read" to remember to press.
##
## A SETTING NOT YET WRITTEN IS WRITTEN FIRST. The autosave waits for a quiet period, so a
## dial moved a moment before Speak exists only in the panel; reading the voice back without
## flushing it would put the OLDER value from the file straight over the one just chosen.
## Flushed, the panel's latest change is in the file, and what comes back is the file - the
## author's own edits to the frontmatter included.
##
## In Input mode it is just the box. A read that fails keeps the last body rather than
## falling silent - a file being saved under us is a moment, not a reason to stop a reading.
func pull() -> String:
	if not is_sync():
		return _text.text if _text != null else ""
	_flush()
	var raw: Variant = _read_raw()
	if raw == null:
		return _body
	var parts := FrontMatter_.split(String(raw))
	_body = String(parts.body)
	_show(_body)
	_import(String(raw), true)
	return _body


## Write the panel's settings now if they differ from what was last written - the autosave's
## own write, without its quiet period. Refused where the autosave is (a render, a probe).
func _flush() -> void:
	if not capture.is_valid() or (Settings.is_read_only() and not _autosave_for_test):
		return
	var snap := _snapshot()
	if snap.is_empty() or snap == _saved:
		return
	var err := _write_reconciled()
	if not err.is_empty():
		_autosave_note = err
		_note("⚠  " + err)


## Re-read the document AND take its voice. What picking a document does, and what boot
## does in sync mode. Returns false when the file could not be read.
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


## Write the panel's settings into the document's frontmatter, now. For gates; the panel
## itself relies on the autosave and on the flush in [method pull].
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
	if not (capture.call() is Dictionary):
		_note("This panel cannot save a voice.")
		return false
	# READ-MODIFY-WRITE OF ONE KEY. Whatever the document says about the OTHER panel's
	# voice is carried through untouched, so a chapter can hold both.
	var err := FrontMatter_.write_block(_path, _merged_block())
	if not err.is_empty():
		_autosave_note = err
		_note("⚠  " + err)
		return false
	# The autosave must not now write the same thing again a moment later.
	_seen = _snapshot()
	_saved = _seen
	_autosave_note = ""
	_note("✓  Voice saved into %s" % _path.get_file())
	return true


## THE AUTOSAVE, polled rather than wired to each control.
##
## WIRED WOULD HAVE MEANT TEN CALL SITES AND A RULE TO REMEMBER, and the panel already proves
## that does not hold: of the Generative panel's controls, the text box, the tabs, the voice,
## the speaker and the tone mark it dirty and the ten VALUE DIALS - pace, pause, dynamics,
## arc, effort, echo, room, resonance, presence, ambience - do not. Those are precisely the
## ones an author adjusts while listening, so a wired autosave would have saved everything
## except the thing being asked for. Polling a snapshot cannot be forgotten by a control
## added later, which is the same argument [method Settings.bind] is built on.
##
## It is a QUIET PERIOD, not a debounce from the first change: the snapshot has to stop moving
## before anything is written, so a slider dragged for ten seconds is one write at the end and
## not twelve along the way.
func _process(_delta: float) -> void:
	if not is_sync() or not capture.is_valid():
		return
	# An export render, the offline analyzer and a test probe all boot the whole app against
	# the author's own settings - and would find their own document open. A person pressing Speak
	# is a person; a background process is not.
	if Settings.is_read_only() and not _autosave_for_test:
		return
	var now := Time.get_ticks_msec()
	if now - _polled_ms < POLL_MS:
		return
	_polled_ms = now
	var snap := _snapshot()
	if snap != _seen:
		_seen = snap
		_settled_ms = now
		return
	if snap == _saved or now - _settled_ms < AUTOSAVE_MS:
		return
	var err := _write_reconciled()
	if err.is_empty():
		_autosave_note = ""
		return
	# A FAILURE IS LOUD. The whole value of the verify is knowing when it refused.
	if err != _autosave_note:
		_autosave_note = err
		_note("⚠  " + err)


## See [member _autosave_for_test]. Nothing but a gate may call this.
func allow_autosave_for_test() -> void:
	_autosave_for_test = true


## WRITE THE PANEL INTO THE DOCUMENT WITHOUT UNDOING AN EDIT MADE TO IT OUTSIDE GHOST.
##
## The panel writes its WHOLE block whenever anything in it changes, and the document is only
## re-read at Speak - so a value the author reverted in their own editor was written straight
## back from the panel's memory by the next unrelated change (a slider, a tab, the Handwriting
## picker): "I keep reverting the frontmatter, yet Ghost keeps resetting it". So when the block
## on disk is no longer what ghost last read or wrote (`_saved`), this is a THREE-WAY MERGE:
## the file's version, plus only what changed in the panel since then. The panel is then shown
## the result, so what it displays and what the file says are the same.
##
## Marked saved BEFORE the write, not after: a write that fails for a standing reason (the file
## is read-only, the directory is gone) must not be retried four times a second for the rest
## of the session. The next actual change moves the snapshot and tries again.
func _write_reconciled() -> String:
	var raw: Variant = _read_raw()
	var ghost := {}
	if raw != null:
		var res := FrontMatter_.read_block(String(raw))
		if res.data is Dictionary:
			ghost = (res.data as Dictionary).duplicate(true)
	var mine: Variant = capture.call()
	var theirs: Variant = ghost.get(_block, null)
	var base: Variant = JSON.parse_string(_saved) if not _saved.is_empty() else null
	if mine is Dictionary and theirs is Dictionary and base is Dictionary \
			and not same_value(theirs, base):
		var merged: Variant = merge3(base, mine, theirs)
		if apply.is_valid() and not same_value(merged, mine):
			apply.call(merged as Dictionary)
			mine = capture.call()
			_note("%s was edited outside ghost - kept those edits." % _path.get_file())
	_seen = _snapshot()
	_saved = _seen
	ghost[_block] = mine
	return FrontMatter_.write_block(_path, ghost)


## `theirs` with every value `mine` changed from `base` laid over it, key by key down through
## nested blocks (a voice's dial is three levels in). A value only one side changed survives;
## where both changed the same value, the panel - the more recent hand - wins.
static func merge3(base: Variant, mine: Variant, theirs: Variant) -> Variant:
	if not (mine is Dictionary and theirs is Dictionary):
		return mine if not same_value(mine, base) else theirs
	var out := (theirs as Dictionary).duplicate(true)
	var b: Dictionary = base if base is Dictionary else {}
	for k in mine:
		if not b.has(k):
			if not out.has(k):
				out[k] = mine[k]           # new in the panel, unknown to the file
			continue
		if same_value(mine[k], b[k]):
			if not out.has(k):
				out[k] = mine[k]           # absent from the file is not an edit (a newer key)
			continue                       # the panel did not touch it: the file's stands
		out[k] = merge3(b[k], mine[k], out.get(k, null))
	return out


## Equal as settings: numbers by value whatever their type (YAML reads 3, JSON gives 3.0), and
## to the precision a dial is shown at.
static func same_value(a: Variant, b: Variant) -> bool:
	var num := [TYPE_INT, TYPE_FLOAT]
	if typeof(a) in num and typeof(b) in num:
		return absf(float(a) - float(b)) < 0.0005
	if a is Dictionary and b is Dictionary:
		if (a as Dictionary).size() != (b as Dictionary).size():
			return false
		for k in a:
			if not (b as Dictionary).has(k) or not same_value(a[k], b[k]):
				return false
		return true
	if a is Array and b is Array:
		if (a as Array).size() != (b as Array).size():
			return false
		for i in (a as Array).size():
			if not same_value(a[i], b[i]):
				return false
		return true
	return typeof(a) == typeof(b) and a == b


## The panel's settings as a stable string, for comparing one frame to the next.
func _snapshot() -> String:
	var block: Variant = capture.call()
	return JSON.stringify(block) if block is Dictionary else ""


## Our block merged onto whatever the document already says, so the OTHER panel's voice - and
## any key a later build adds - is carried through a save rather than dropped by it.
func _merged_block() -> Dictionary:
	var raw: Variant = _read_raw()
	var ghost := {}
	if raw != null:
		var res := FrontMatter_.read_block(String(raw))
		if res.data is Dictionary:
			ghost = (res.data as Dictionary).duplicate(true)
	ghost[_block] = capture.call()
	return ghost


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


## Take the voice out of a document and hand it to the panel.
func _import(raw: String, quiet := false) -> void:
	var res := FrontMatter_.read_block(raw)
	if not res.ok:
		_note("Read %s. Its frontmatter could not be parsed (%s), so the voice is unchanged."
			% [_path.get_file(), String(res.error)])
		return
	var ghost := res.data as Dictionary
	var mine: Variant = ghost.get(_block, null)
	if mine is Dictionary and apply.is_valid():
		apply.call(mine as Dictionary)
		if not quiet:
			_note("Read %s and restored its voice." % _path.get_file())
	elif not quiet:
		_note("Read %s. It carries no voice yet - adjust anything and it will be written in."
			% _path.get_file())
	# THE SNAPSHOT IS SEEDED HERE, after the panel has taken the document's voice. Without it
	# the autosave sees "the panel does not match what we last wrote" the instant a document is
	# opened and writes it straight back - touching a file the author has only just opened, and
	# doing it every time, which is exactly the behaviour a careful writer is built to avoid.
	_seen = _snapshot()
	_saved = _seen
	_settled_ms = Time.get_ticks_msec()


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
