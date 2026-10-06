extends VBoxContainer
class_name DocSource

## DocSource - where a reading's words come from: the editor alone, or a file it syncs to.
##
## THE TEXT BOX IS A DRAFT, and a chapter is not a draft. Everything the voice panels read
## until now was pasted in, which has three costs that only show up on real material: the
## paste is a COPY, so it goes stale the moment the document is edited anywhere else; the
## panel's settings are ghost's rather than the document's, so moving to the next chapter
## silently inherits the last one's voice and coming back has lost it; and a fix to a
## sentence means finding the panel, clearing it and pasting again.
##
## So a script can be SYNCED TO A FILE, and that is a standing state rather than an act: OPEN
## reads a document and syncs to it, SYNC TO… writes the script into a new file and syncs to
## that, and CLEAR detaches and empties the editor, leaving a script that is saved nowhere
## until it is synced. Nothing downstream of the panel knows which is in use, because both end up
## as text in the same [TextEdit] (the [ScriptWriter]'s editor). Unsynced text is the panel's
## own: it persists it in Settings as before.
##
## THE DOCUMENT CAN BE WRITTEN HERE TOO. The editor shows the BODY only - the frontmatter is
## the panel's business and is written by it - and an edit made in ghost is saved on the same
## quiet period as a dial, through [method FrontMatter.write_body], which refuses if the file
## changed on disk since ghost last read it. That refusal is a CONFLICT the author resolves
## (keep ghost's text, or take the file's), never a silent winner either way.
##
## SYNC IS READ AT EVERY SPEAK, not at every keystroke and not when the document was picked.
## The point of it is that the author keeps working in their own editor while ghost is open:
## save the file, press Play, hear the change. A watcher polling for modifications would be
## the same thing done worse - it would re-read mid-reading, which is exactly when the file
## is most likely to be half-written.
##
## FRONTMATTER IS NOT SPOKEN, AND IT IS WHERE THE VOICE LIVES. [FrontMatter] cuts the YAML
## block off the top before a word reaches the synthesizer, and ghost keeps its own key in
## there: the reader, the tone, the room, the whole cast of a multi-speaker chapter.
##
## BOTH DIRECTIONS FOLLOW THE AUTHOR, with no button for either. SAVING is automatic - a
## dial moved is a dial written, on a quiet period, exactly the way the rest of ghost saves
## (see [Settings]). LOADING happens at every Play and export ([method pull]), because the
## file is the authoritative source - and never on a timer, because an import firing by
## itself would undo a dial the author had just moved. A Play cannot do that: it writes any
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
## offline analyzer, a test probe. Flushing on Play is not, because pressing Play is a person.
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
## Clear's tooltip; the button is on the [ScriptWriter] card.
const CLEAR_TIP := ("Start an empty script that is saved nowhere, and forget the file it was "
	+ "synced to. The file itself is left exactly as it is. Text that was not synced anywhere "
	+ "can be brought back with Ctrl+Z in the editor.")
## Open…'s tooltip; the button is on the [ScriptWriter] card.
const OPEN_TIP := ("Open a Markdown file and sync to it: it is read fresh at every Play, "
	+ "and edits made here are saved into it. Edit it in your own editor too if you like. "
	+ "Its YAML frontmatter is never shown or spoken - the panel keeps the voice there.")

## A document was opened and its voice (if it had one) applied.
signal opened(path: String)
## A status line was shown (or cleared, with ""). The panel's script card mirrors it, so a
## failed save is visible with the editor window closed.
signal noted(msg: String)
## The script was synced to a file, or detached from one.
signal mode_changed(sync: bool)

## The panel's settings, as they are right now -> the dictionary to store under this
## source's block. Set by the owner; a source without it simply cannot save a voice.
var capture: Callable
## The other direction: a document's stored block -> the panel. Called on open and on
## Reload, never on a Play.
var apply: Callable

var _section := "generative"     # the [Settings] section this source's draft and path live in
var _block := "generative"       # our sub-key inside the document's `ghost:` frontmatter
## Where the file dialogs are added. A dialog parented inside the editor's own Window could
## not be shown with that window closed, and Open… is on the panel's card too.
var dialog_host: Node = null

var _text: TextEdit              # the panel's box, driven by this widget in sync mode
var _sync := false               # synced to _path
var _fields := {}                # top-level frontmatter fields (title...) of an UNSYNCED script
var _name: Label
var _status: Label
var _conflict_row: HBoxContainer
var _path := ""
var _body := ""                  # the body as last read from or written to _path (LF)
var _dialog: Window = null       # the one dialog open at a time (file or confirmation)
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
# THE BODY'S AUTOSAVE, the same quiet period keyed on the editor's version counter.
var _body_seen_v := -1
var _body_settled_ms := 0
var _body_failed_v := -1         # a write refused at this version is not retried until an edit
var _conflict := false           # the file moved under an unsaved edit; the author decides
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
	_name = Label.new()
	_name.add_theme_font_size_override("font_size", 13)
	_name.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_name.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
	row.add_child(_name)
	row.add_child(_tool("Sync to…", "Write this script into a Markdown file and keep it there: "
		+ "from now on every edit is saved into the file a moment after you stop typing, with "
		+ "this panel's voice in its frontmatter.", _sync_to_dialog))

	_conflict_row = HBoxContainer.new()
	_conflict_row.add_theme_constant_override("separation", 4)
	_conflict_row.visible = false
	add_child(_conflict_row)
	var cl := Label.new()
	cl.text = "Changed on disk and here:"
	cl.add_theme_font_size_override("font_size", 12)
	cl.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_conflict_row.add_child(cl)
	_conflict_row.add_child(_tool("Keep mine", "Write the text in the editor over the file.",
		func() -> void: _write_body(true)))
	_conflict_row.add_child(_tool("Take the file's", "Throw away the edits made here and show "
		+ "the file as it is on disk.", func() -> void: reload()))

	_status = Label.new()
	_status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_status.add_theme_font_size_override("font_size", 11)
	_status.modulate = Color(1, 1, 1, 0.65)
	_status.visible = false
	add_child(_status)

	_path = str(Settings.read(_section, "doc_path", ""))
	_sync = bool(Settings.read(_section, "sync", false)) and not _path.is_empty()
	var f: Variant = Settings.read(_section, "fields", {})
	_fields = (f as Dictionary).duplicate() if f is Dictionary else {}
	_refresh_row()


## Hand the widget the panel's text box. It owns what is IN it from here on: a synced body
## is shown in it, and unsynced text is the panel's persisted draft.
##
## Deliberately separate from [method setup] so the widget can be built above the box it
## drives - the source row belongs over the text, and a Control is added where it is shown.
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
	_body_seen_v = te.get_version()
	if is_sync():
		# The body AND the voice - and `reload` seeds the autosave's snapshot, so opening
		# ghost on a document is not followed by ghost writing that document. A file that has
		# gone shows nothing and says so; Clear or Open moves on from it.
		reload()
	else:
		_show(str(Settings.read(_section, "text", "")))
	_apply_editable()


## True while THIS widget is writing the text box, so the panel's own `text_changed`
## handler can tell a document being shown from the author typing. Without it, a synced
## Play marks its own reading stale the instant it starts.
func is_quiet() -> bool:
	return _syncing


## True when the reading comes from a file.
func is_sync() -> bool:
	return _sync and not _path.is_empty()


## A TOP-LEVEL FRONTMATTER FIELD - `title:`, `author:`, `book:` - which is the document's own
## metadata rather than ghost's block. Synced, it is read from the file as it is now (TEXTUALLY,
## one line, so a head MiniYaml refuses still yields it); unsynced, it is what the panel set,
## else the script's own frontmatter if it was pasted with one.
func field(key: String) -> String:
	if is_sync():
		if not FileAccess.file_exists(_path):
			return ""
		return BookLayout.field_of(FileAccess.get_file_as_string(_path), key)
	var v := str(_fields.get(key, ""))
	if v.is_empty() and _text != null:
		v = BookLayout.field_of(_text.text, key)
	return v


## Set a field. Synced, it is written into the file at once - one key, by the same checked line
## surgery as the voice, and REMOVED when emptied rather than left as an empty key; unsynced, it
## is kept with the script and goes into the file at Sync to…. Called on a person's edit, so not
## refused in a read-only process (the same reasoning as the flush on Play).
func set_field(key: String, value: String) -> bool:
	value = value.strip_edges()
	if field(key) == value:
		return true
	if not is_sync():
		if value.is_empty():
			_fields.erase(key)
		else:
			_fields[key] = value
		Settings.write(_section, "fields", _fields.duplicate())
		return true
	var err := FrontMatter_.write_block(_path, value if not value.is_empty() else null, key)
	if not err.is_empty():
		_note("⚠  " + err)
		return false
	return true


## The document's path, or "" when the script is synced to nothing.
func doc_path() -> String:
	return _path if is_sync() else ""


## WHAT TO PERSIST as the panel's text: the script when it is synced to nothing, and nothing
## when it is - the file is where a synced script lives, and a copy in Settings would only be
## a stale one.
func draft() -> String:
	if is_sync() or _text == null:
		return ""
	return _text.text


## THE REAL-TIME READ: the document as it is on disk at this instant - its words AND its
## voice. The file is the authoritative source, so every Play and every export starts here
## and there is no separate "re-read" to remember to press.
##
## A SETTING NOT YET WRITTEN IS WRITTEN FIRST. The autosave waits for a quiet period, so a
## dial moved a moment before Play exists only in the panel; reading the voice back without
## flushing it would put the OLDER value from the file straight over the one just chosen.
## Flushed, the panel's latest change is in the file, and what comes back is the file - the
## author's own edits to the frontmatter included.
##
## Synced to nothing, it is just the box. A read that fails keeps the last body rather than
## falling silent - a file being saved under us is a moment, not a reason to stop a reading.
##
## AN EDIT MADE IN GHOST IS WRITTEN FIRST, for the same reason. If it cannot be - the file
## changed outside ghost as well - the reading is the text on screen, and the file is left
## for the author to reconcile; replacing their unsaved words with the file's would be the
## one outcome nobody chose.
func pull() -> String:
	if not is_sync():
		return _text.text if _text != null else ""
	_flush()
	_flush_body()
	var raw: Variant = _read_raw()
	if raw == null:
		return _text.text if _unsaved() else _body
	if _unsaved():
		_import(String(raw), true)
		return _text.text
	_body = FrontMatter_.lf(String(FrontMatter_.split(String(raw)).body))
	_show(_body)
	_import(String(raw), true)
	return _body


## The editor holds words the file does not.
func _unsaved() -> bool:
	return is_sync() and _text != null and _text.text != _body


## Write the editor's body now if it is ahead of the file. Refused where the autosave is.
func _flush_body() -> void:
	if not _unsaved() or (Settings.is_read_only() and not _autosave_for_test):
		return
	_write_body(false)


## THE BODY WRITE. [param force] is "Keep mine": the file's current body becomes the
## expected one, so the editor's text replaces it - asked for by a person, never automatic.
func _write_body(force: bool) -> bool:
	if _text == null or _path.is_empty():
		return false
	var expected := _body
	if force:
		var raw: Variant = _read_raw()
		if raw == null:
			return false
		expected = FrontMatter_.lf(String(FrontMatter_.split(String(raw)).body))
	var mine := _text.text
	var err := FrontMatter_.write_body(_path, mine, expected)
	if err.is_empty():
		_body = mine
		_body_failed_v = -1
		if _conflict:
			_conflict = false
			_conflict_row.visible = false
			_note("✓  %s saved." % _path.get_file())
		return true
	_body_failed_v = _text.get_version()
	if err.begins_with(FrontMatter_.CONFLICT):
		_conflict = true
		_conflict_row.visible = true
		_note("⚠  %s was changed outside Ghost Notes while you were editing it here. Nothing is "
			% _path.get_file() + "saved until you choose which to keep.")
	else:
		_note("⚠  " + err)
	return false


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
	_body = FrontMatter_.lf(String(FrontMatter_.split(String(raw)).body))
	_conflict = false
	_conflict_row.visible = false
	_body_failed_v = -1
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
	if not is_sync():
		return
	# An export render, the offline analyzer and a test probe all boot the whole app against
	# the author's own settings - and would find their own document open. A person pressing Play
	# is a person; a background process is not.
	if Settings.is_read_only() and not _autosave_for_test:
		return
	var now := Time.get_ticks_msec()
	if now - _polled_ms < POLL_MS:
		return
	_polled_ms = now
	_poll_body(now)
	if not capture.is_valid():
		return
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


## The body's half of the autosave: written once the editor's version has held still for
## [constant AUTOSAVE_MS]. A conflict or a refused write waits for the next edit (or a choice).
func _poll_body(now: int) -> void:
	if _text == null:
		return
	var v := _text.get_version()
	if v != _body_seen_v:
		_body_seen_v = v
		_body_settled_ms = now
		return
	if _conflict or v == _body_failed_v or now - _body_settled_ms < AUTOSAVE_MS:
		return
	if _unsaved():
		_write_body(false)


## See [member _autosave_for_test]. Nothing but a gate may call this.
func allow_autosave_for_test() -> void:
	_autosave_for_test = true


## WRITE THE PANEL INTO THE DOCUMENT WITHOUT UNDOING AN EDIT MADE TO IT OUTSIDE GHOST.
##
## The panel writes its WHOLE block whenever anything in it changes, and the document is only
## re-read at Play - so a value the author reverted in their own editor was written straight
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
			_note("%s was edited outside Ghost Notes - kept those edits." % _path.get_file())
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


func _tool(label: String, tip: String, action: Callable) -> Button:
	var b := Button.new()
	b.text = label
	b.tooltip_text = tip
	b.pressed.connect(action)
	return b


## Before the editor stops showing the document: write what it holds, and refuse to move
## if that cannot be done - the edits would otherwise be replaced by the draft and lost.
func _leave_document() -> bool:
	if _text == null or _path.is_empty() or _text.text == _body:
		return true
	if _write_body(false):
		return true
	_note("⚠  Edits to %s could not be written - settle that first."
		% _path.get_file())
	return false


func _apply_editable() -> void:
	if _text == null:
		return
	# Editable in both: in sync mode an edit is written back into the file's body (see
	# [method _write_body]), so it no longer has to be made in another editor.
	_text.editable = true
	_text.placeholder_text = "Once upon a time..." if not is_sync() \
		else "This document has no text yet."


func _refresh_row() -> void:
	if is_sync():
		_name.text = "Synced to %s" % _path.get_file()
		_name.tooltip_text = _path
	else:
		_name.text = "Not synced to a file"
		_name.tooltip_text = "This script is kept by Ghost Notes alone. Sync to… writes it into a file."


func _show(body: String) -> void:
	if _text == null:
		return
	if _text.text == body:
		return
	_syncing = true
	# The caret and the scroll, kept: a re-read at every Play would otherwise throw the
	# reader back to the top of the chapter every time.
	body = FrontMatter_.lf(body)
	var col := _text.get_caret_column()
	var line := _text.get_caret_line()
	var scroll := _text.scroll_vertical
	_text.text = body
	_text.set_caret_line(mini(line, maxi(0, _text.get_line_count() - 1)))
	_text.set_caret_column(col)
	_text.scroll_vertical = scroll
	# A body shown is not an edit: the autosave starts counting from here.
	_body_seen_v = _text.get_version()
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
	# doing it every time, which is exactly the behavior a careful writer is built to avoid.
	_seen = _snapshot()
	_saved = _seen
	_settled_ms = Time.get_ticks_msec()


## OPEN A DOCUMENT AND SYNC TO IT. Text synced to nothing would be replaced by the file's and
## exists nowhere else, so that is asked about first; a synced script is in its file already.
func open() -> void:
	if _dialog != null and is_instance_valid(_dialog):
		return
	if not is_sync() and _text != null and not _text.text.strip_edges().is_empty():
		var ask := ConfirmationDialog.new()
		ask.title = "Open a document"
		ask.dialog_text = ("This script is not synced to a file, and opening one replaces it.\n"
			+ "Use Sync to… first to keep it.")
		ask.ok_button_text = "Open anyway"
		ask.confirmed.connect(func() -> void:
			_close_dialog()
			_open_dialog())
		ask.canceled.connect(_close_dialog)
		_popup(ask)
		return
	_open_dialog()


func _open_dialog() -> void:
	var d := _file_dialog(FileDialog.FILE_MODE_OPEN_FILE, "Open and sync to a document")
	d.filters = PackedStringArray(["*.md, *.markdown, *.txt ; Text", "* ; Every file"])
	d.file_selected.connect(_on_picked)
	_popup(d)


func _sync_to_dialog() -> void:
	if _dialog != null and is_instance_valid(_dialog):
		return
	var d := _file_dialog(FileDialog.FILE_MODE_SAVE_FILE, "Sync this script to a file")
	d.filters = PackedStringArray(["*.md ; Markdown"])
	d.current_file = _path.get_file() if not _path.is_empty() else "chapter.md"
	d.file_selected.connect(sync_to)
	_popup(d)


func _file_dialog(mode: FileDialog.FileMode, title: String) -> FileDialog:
	var d := FileDialog.new()
	d.file_mode = mode
	d.access = FileDialog.ACCESS_FILESYSTEM
	# In-window, never native - the portal dialog shows nothing at all on a Linux box
	# without xdg-desktop-portal, which is the "I pressed it and nothing happened" report
	# the film importer already carries this note for.
	d.use_native_dialog = false
	d.title = title
	if not _path.is_empty():
		d.current_dir = _path.get_base_dir()
	elif not OS.get_system_dir(OS.SYSTEM_DIR_DOCUMENTS).is_empty():
		d.current_dir = OS.get_system_dir(OS.SYSTEM_DIR_DOCUMENTS)
	d.size = Vector2i(820, 560)
	d.canceled.connect(_close_dialog)
	return d


func _popup(d: Window) -> void:
	_dialog = d
	(dialog_host if dialog_host != null else self).add_child(d)
	d.popup_centered()


func _on_picked(path: String) -> void:
	_close_dialog()
	if is_sync() and not _leave_document():
		return
	if not is_sync():
		_fields = {}             # they belonged to the unsynced script this replaces
		Settings.write(_section, "fields", {})
	_path = path
	_sync = true
	Settings.write(_section, "doc_path", _path)
	Settings.write(_section, "sync", true)
	_refresh_row()
	_apply_editable()
	if reload():
		opened.emit(_path)
	mode_changed.emit(true)


## SYNC THE SCRIPT TO A FILE: write it there now - the panel's voice in the frontmatter - and
## keep it there from then on. Synced already, the new file takes the old one's frontmatter
## too (a title, the other panel's voice), and the old file is left as it was.
func sync_to(path: String) -> bool:
	_close_dialog()
	if _text == null:
		return false
	if not path.get_extension().to_lower() in ["md", "markdown", "txt"]:
		path += ".md"
	var template := ""
	var ghost := {}
	if is_sync():
		var raw: Variant = _read_raw()
		if raw != null:
			template = String(raw)
			var res := FrontMatter_.read_block(template)
			if res.data is Dictionary:
				ghost = (res.data as Dictionary).duplicate(true)
	if capture.is_valid() and capture.call() is Dictionary:
		ghost[_block] = capture.call()
	var fields := {}
	for k in _fields:
		fields[k] = _fields[k]
	var err := FrontMatter_.create(path, ghost, _text.text, template, FrontMatter_.KEY,
		{} if is_sync() else fields)
	if not err.is_empty():
		_note("⚠  " + err)
		return false
	_fields = {}                 # in the file now
	Settings.write(_section, "fields", {})
	_path = path
	_sync = true
	Settings.write(_section, "doc_path", _path)
	Settings.write(_section, "sync", true)
	_refresh_row()
	_apply_editable()
	reload()
	_note("✓  Synced to %s - edits are saved there from now on." % _path.get_file())
	opened.emit(_path)
	mode_changed.emit(true)
	return true


## A SCRIPT SAVED NOWHERE, AND NO FILE REMEMBERED - and no title, author or book either. Synced, the file is left exactly as it is
## (edits still waiting are written first, and a clash must be settled before the link is
## dropped); not synced, the text is deleted as an EDIT, so Ctrl+Z brings it back.
func clear() -> void:
	if _text == null:
		return
	if is_sync():
		if not _leave_document():
			return
		var was := _path.get_file()
		_sync = false
		Settings.write(_section, "text", "")
		_show("")
		_note("No longer synced to %s, which is unchanged. This script is saved nowhere until "
			% was + "you Sync it to a file.")
	else:
		_text.begin_complex_operation()
		_text.select_all()
		_text.delete_selection()
		_text.end_complex_operation()
		_note("")
	_path = ""
	_fields = {}
	Settings.write(_section, "sync", false)
	Settings.write(_section, "doc_path", "")
	Settings.write(_section, "fields", {})
	_conflict = false
	_conflict_row.visible = false
	_refresh_row()
	_apply_editable()
	mode_changed.emit(false)


func _close_dialog() -> void:
	if _dialog != null and is_instance_valid(_dialog):
		_dialog.queue_free()
	_dialog = null


func _note(msg: String) -> void:
	_status.text = msg
	_status.visible = not msg.is_empty()
	noted.emit(msg)
