extends Node

## doc_sync_check - the Generative panel READING FROM A FILE, end to end.
##
## [code]doc_source_check.gd[/code] holds [FrontMatter] to account on strings; this one
## holds the PANEL to account on a real document, because every claim the feature makes is
## about the wiring rather than about the parser, and every one of them fails silently:
##
##   THE FRONTMATTER IS NEVER SPOKEN. If it leaks, the reading opens with "title colon
##   chapter one date colon two thousand nineteen" and nothing errors.
##   A SPEAK RE-READS THE DISK. If it does not, the panel quietly performs a version of the
##   chapter that was current when the document was opened - which looks exactly like the
##   author's edit not having saved.
##   CLEAR DETACHES, IT DOES NOT DELETE. A synced file is left byte-identical, a script synced
##   to nothing comes back with one undo, and a synced script is never copied into Settings
##   (that copy would only ever be a stale one).
##   SYNC TO… IS A STANDING STATE. The script lands in the new file with the voice in its
##   frontmatter, and the edits after it land there too.
##   THE VOICE IS RESTORED, AND SAVED. This is the whole reason the frontmatter is written
##   at all - a cast of two readers that comes back as one is the feature not working.
##   THE SAVE IS AUTOMATIC, ON A QUIET PERIOD. A dial moved is a dial written, with no button
##   pressed - and NOT written while it is still moving, or a ten-second drag is a dozen
##   rewrites of a file the author may have open in their own editor. Both halves are
##   asserted, because "it saved" also passes on a version that saves constantly.
##   THE WORDS CAN BE WRITTEN IN GHOST, saved into the body on the same quiet period with the
##   frontmatter untouched - and an edit made to the file elsewhere in the meantime is a
##   CONFLICT the author resolves, never overwritten and never thrown away.
##   ...AND NEVER BY AN UNATTENDED PROCESS. An export render boots this whole app against the
##   author's settings and would find their document open. Asserted by taking the gate's own
##   seam away and watching the same dial move reach nothing.
##
## Needs a real boot: the panel's sliders are Director-backed and [Settings] is an autoload.
##   tests/run_boot_probe.sh tests/doc_sync_check.gd 120

const DIR := "user://doc_sync_check"
## A section of the settings file that belongs to nobody, so the probe cannot disturb the
## author's own remembered document. (A probe is read-only anyway - this is belt and braces,
## and it also keeps the gate from picking up whatever document is really open.)
const SECTION := "doc_sync_probe"

const HEAD := """---
title: 'Chapter One'
# an authoring note
ghost:
  generative:
    turn: 2.5
    tab: Narrator
    voices:
      Narrator:
        voice: en_US-libritts-high
        speaker: 4
        pace: 0.8
      Emily White:
        voice: en_GB-alan-medium
        speaker: 0
        pace: 1.3
---
"""
const BODY_A := "\n# Chapter One\n\nThe rain had not stopped.\n\n<!-- speaker: Emily White -->\n\nNor had I.\n"
const BODY_B := "\n# Chapter One\n\nThe rain had stopped, and the door stood open.\n\n<!-- speaker: Emily White -->\n\nNor had I.\n"

var _fails: Array = []
var _ed: GenerativeEditor
var _doc: DocSource
var _path := ""


func _ready() -> void:
	DirAccess.make_dir_recursive_absolute(DIR)
	_path = DIR + "/chapter.md"
	_write(HEAD + BODY_A)

	# Built by hand rather than added to the tree: _ready would start the voice host, and
	# nothing here needs a python process.
	_ed = GenerativeEditor.new()
	_ed._build_panel()
	# THE PANEL MUST BE IN THE TREE - Range only emits value_changed for an owner inside
	# one, so a panel built outside it has every slider callback silently dead.
	_ed.remove_child(_ed._panel)
	add_child(_ed._panel)
	_ed.remove_child(_ed._repace_timer)
	add_child(_ed._repace_timer)
	# Re-point the source at a section of its own, BEFORE it is bound to the box: setup()
	# has already read the author's real remembered document and this gate must not open it.
	_doc = _ed._doc
	_doc._section = SECTION
	_doc._path = ""
	_doc._sync = false
	_doc._refresh_row()
	_doc.bind_text(_ed._text)

	_check_input_mode()
	await _check_opening_a_document()
	_check_speak_rereads_the_disk()
	_check_clear()
	_check_saving_the_voice()
	await _check_autosave()
	await _check_autosave_waits_for_quiet()
	_check_speak_takes_the_voice()
	await _check_outside_edit_survives()
	_check_the_look_travels()
	_check_the_picture_travels()
	await _check_editing_in_ghost()
	_check_fields()
	await _check_sync_to()
	await _check_unattended_processes_never_autosave()

	_ed.free()
	DirAccess.remove_absolute(_path)
	DirAccess.remove_absolute(DIR)
	if _fails.is_empty():
		print("doc_sync_check: ALL OK")
		get_tree().quit()
		return
	for f in _fails:
		print("doc_sync_check: FAIL - ", f)
	print("doc_sync_check: %d FAILED" % _fails.size())
	get_tree().quit(1)


func _ok(cond: bool, what: String) -> void:
	if not cond:
		_fails.append(what)


## THE CONTROL FOR EVERYTHING BELOW: synced to nothing, the panel is exactly what it was,
## and the file on disk is nothing to it.
func _check_input_mode() -> void:
	_ed._text.text = "A pasted draft."
	_ok(not _doc.is_sync(), "the panel started synced with no document open")
	_ok(_doc.pull() == "A pasted draft.", "an unsynced script did not read the box")
	_ok(_doc.draft() == "A pasted draft.", "an unsynced script is not persisted")
	_ok(_ed._text.editable, "the box was read-only")


func _check_opening_a_document() -> void:
	_doc._on_picked(_path)
	_ok(_doc.is_sync(), "picking a document did not switch the source to sync")
	_ok(_doc.doc_path() == _path, "the source is not pointing at the document")
	_ok(_ed._text.editable, "the document is read-only in ghost's editor")

	# THE FRONTMATTER IS NOT THE CHAPTER. Two-sided: the raw file plainly contains it, and
	# what the panel will speak plainly does not.
	var raw := FileAccess.get_file_as_string(_path)
	_ok(raw.contains("title:") and raw.contains("ghost:"),
		"the control is wrong - the fixture has no frontmatter in it")
	var body := _doc.pull()
	_ok(not body.contains("title:") and not body.contains("ghost:") and not body.contains("---"),
		"the frontmatter reached the text to be spoken: %s" % body.substr(0, 80))
	_ok(body == BODY_A, "the body read back wrong: %s" % body)
	_ok(_ed._text.text == BODY_A, "the box is not showing the document's body")

	# THE VOICE CAME WITH IT. Two readers, each with its own settings - a cast that came
	# back as one tab would be the feature not working.
	_ok(_ed._slots.size() == 2,
		"the document's cast of 2 arrived as %d tab(s)" % _ed._slots.size())
	_ok(is_equal_approx(_ed._turn.value, 2.5),
		"the document's Turn did not reach the panel (%f)" % _ed._turn.value)
	_ok(String(_ed._cfg(0).get("voice", "")) == "en_US-libritts-high"
			and int(_ed._cfg(0).get("speaker", -1)) == 4
			and is_equal_approx(float(_ed._cfg(0).get("pace", 0.0)), 0.8),
		"the narrator did not come back as the document wrote it: %s" % str(_ed._cfg(0)))
	_ok(_ed._names == PackedStringArray(["Narrator", "Emily White"]),
		"the tabs are not the script's speakers by name: %s" % [_ed._names])
	_ok(String(_ed._cfg(1).get("voice", "")) == "en_GB-alan-medium"
			and is_equal_approx(float(_ed._cfg(1).get("pace", 0.0)), 1.3),
		"Emily White did not come back as the document wrote her: %s" % str(_ed._cfg(1)))
	# A key the document did NOT mention takes its default rather than arriving missing.
	_ok(_ed._cfg(0).has("presence") and _ed._cfg(0).has("ambience"),
		"a voice written by an older build arrived with keys missing")
	# OPENING A DOCUMENT IS NOT AN EDIT TO IT. The autosave's snapshot is seeded by the import,
	# so a file the author has only just opened is not written back a second later - which
	# would happen every single time, and is what a careful writer exists to avoid.
	_doc.allow_autosave_for_test()
	var opened := FileAccess.get_file_as_string(_path)
	await _wait(int(DocSource.AUTOSAVE_MS) + 700)
	_ok(FileAccess.get_file_as_string(_path) == opened,
		"simply opening a document caused ghost to write it back")
	_doc._autosave_for_test = false


## THE REAL-TIME CLAIM, which is the whole point of sync mode: the author keeps writing
## off-screen and the next Speak is the file as it is NOW.
func _check_speak_rereads_the_disk() -> void:
	_write(HEAD + BODY_B)
	var body := _doc.pull()
	_ok(body == BODY_B, "a second Speak did not re-read the file (got %s)"
		% body.strip_edges().substr(0, 60))
	_ok(_ed._text.text == BODY_B, "the box did not follow the document's new text")

	# ...and the control: once cleared, the file on disk is nothing to the panel, however
	# much it changes. Without this, a `pull` that simply always read the last file it saw
	# would pass the check above.
	_doc.clear()
	_ed._text.insert_text_at_caret("A typed script.")
	_write(HEAD + BODY_A)
	_ok(_doc.pull() == "A typed script.",
		"a cleared script read the document anyway: %s" % _doc.pull().substr(0, 60))
	_doc._on_picked(_path)
	_ok(_doc.pull() == BODY_A, "opening the document again did not return to it")


## CLEAR DETACHES, IT DOES NOT DELETE. Synced, the file is left byte-identical, its path is
## forgotten and the editor is empty; synced to nothing, the text goes as an edit and one undo brings it back.
func _check_clear() -> void:
	_ok(_doc.is_sync(), "the gate lost the document before checking Clear")
	_ok(_doc.draft().is_empty(), "a synced script is being copied into Settings")
	var before := FileAccess.get_file_as_string(_path)
	_doc.clear()
	_ok(not _doc.is_sync() and _ed._text.text.is_empty(), "Clear did not leave an empty, unsynced script")
	_ok(_doc._path.is_empty(), "Clear kept the file reference")
	_ok(FileAccess.get_file_as_string(_path) == before, "Clear touched the synced file")
	_ed._text.insert_text_at_caret("Kept nowhere.")
	_doc.clear()
	_ok(_ed._text.text.is_empty(), "Clear left an unsynced script's text")
	_ed._text.undo()
	_ok(_ed._text.text == "Kept nowhere.", "an unsynced script could not be brought back with undo")
	_doc._on_picked(_path)
	_ok(_doc.is_sync() and _ed._text.text == BODY_A, "the document did not open again after Clear")


## AND BACK OUT AGAIN. The dial moved here is the one the document must carry next time.
func _check_saving_the_voice() -> void:
	var before := FileAccess.get_file_as_string(_path)
	_ed._turn.value = 4.0
	_ed._rate.value = 1.45
	_ok(_doc.save(), "saving the voice into the document failed")

	var after := FileAccess.get_file_as_string(_path)
	_ok(after != before, "saving the voice changed nothing in the file")
	# THE DOCUMENT IS THE AUTHOR'S. The body and their own keys are untouched.
	_ok(after.contains("title: 'Chapter One'"), "saving the voice rewrote the author's title")
	_ok(after.contains("# an authoring note"), "saving the voice deleted the author's comment")
	_ok(after.ends_with(BODY_A), "saving the voice changed the chapter")

	# Read it back the way a fresh session would: through the panel, from the file.
	_ed._turn.value = 1.0
	_ed._slots = [GenerativeEditor.SLOT_DEFAULTS.duplicate()]
	_ed._names = PackedStringArray(["Narrator"])
	_ed._stash = {}
	_ed._slot = 0
	_ed._rebuild_tabs()
	_doc.reload()
	_ok(_ed._slots.size() == 2, "the saved cast came back as %d tab(s)" % _ed._slots.size())
	_ok(is_equal_approx(_ed._turn.value, 4.0),
		"the saved Turn came back as %f" % _ed._turn.value)
	_ok(is_equal_approx(float(_ed._cfg(0).get("pace", 0.0)), 1.45),
		"the dial moved before saving did not survive the document: %s" % str(_ed._cfg(0)))


## A DIAL MOVED IS A DIAL WRITTEN, with nothing pressed.
func _check_autosave() -> void:
	# A probe is read-only precisely so gates cannot edit the author's things; this gate is the
	# exception, only for the autosave, and only against a fixture of its own.
	_doc.allow_autosave_for_test()
	_doc.reload()
	await _wait(400)
	var before := FileAccess.get_file_as_string(_path)

	_ed._turn.value = 3.75
	await _wait(int(DocSource.AUTOSAVE_MS) + 900)
	var after := FileAccess.get_file_as_string(_path)
	_ok(after != before, "moving a dial never reached the document - autosave did nothing")
	_ok(after.contains("turn: 3.75"),
		"the document does not carry the dial that was moved")
	# ...and the document is still the author's.
	_ok(after.contains("title: 'Chapter One'") and after.contains("# an authoring note"),
		"an automatic save damaged the author's own frontmatter")
	_ok(after.ends_with(BODY_A), "an automatic save changed the chapter")

	# NOTHING FURTHER IS WRITTEN while nothing changes. Without this the panel rewrites the
	# file four times a second forever, and an editor open on it says so every time.
	await _wait(int(DocSource.AUTOSAVE_MS) + 900)
	_ok(FileAccess.get_file_as_string(_path) == after,
		"the document kept being rewritten with nothing changing")


## ...BUT NOT WHILE IT IS STILL BEING ADJUSTED. A quiet period, not a debounce from the first
## change: a session of nudging lands as ONE write at the end, not one per nudge.
##
## THE ADJUSTMENTS HAVE TO BE SPACED, and getting that wrong is why this check first passed
## against a build with no quiet period in it at all. A value changed EVERY FRAME never shows
## the same snapshot to two consecutive polls, so the poll's own "has it changed" test blocks
## the write by itself and the quiet period is never the thing being exercised. The gap here
## is deliberately LONGER than [constant DocSource.POLL_MS] - so the snapshot really does sit
## still between polls, and only the quiet period is left to stop the write - and SHORTER than
## [constant DocSource.AUTOSAVE_MS], so it never elapses. That is the case an author actually
## produces: nudge, listen, nudge again.
func _check_autosave_waits_for_quiet() -> void:
	var before := FileAccess.get_file_as_string(_path)
	var gap: int = int(DocSource.POLL_MS) + 150
	var bursts: int = 5
	for i in bursts:
		_ed._turn.value = 1.0 + 0.3 * float(i)
		await _wait(gap)
	_ok(FileAccess.get_file_as_string(_path) == before,
		"the document was written during %d adjustments %d ms apart - there is no quiet "
		% [bursts, gap] + "period, only a poll")
	# ...and it lands once the nudging stops, on the LAST value rather than an interim one.
	_ed._turn.value = 2.25
	await _wait(int(DocSource.AUTOSAVE_MS) + 900)
	var settled := FileAccess.get_file_as_string(_path)
	_ok(settled != before, "the adjustments never landed once they stopped")
	_ok(settled.contains("turn: 2.25"),
		"the document carries an interim value rather than where the adjusting ended")


## EVERY SPEAK READS THE VOICE TOO, not only the words - the file is the authoritative source
## and there is no re-read button. Two-sided: an edit made to the frontmatter outside ghost
## must arrive, AND a dial moved a moment before Speak (still inside the autosave's quiet
## period) must survive it - the flush writes it first, so the file's older value cannot
## come back over it.
func _check_speak_takes_the_voice() -> void:
	_doc.allow_autosave_for_test()
	_write(HEAD.replace("pace: 0.8", "pace: 0.65") + BODY_A)
	_doc.pull()
	_ok(is_equal_approx(float(_ed._cfg(0).get("pace", 0.0)), 0.65),
		"a Speak did not take the voice edited in the file: %s" % str(_ed._cfg(0)))
	_ed._select_tab(0)
	_ed._rate.value = 1.2
	_doc.pull()
	_ok(is_equal_approx(float(_ed._cfg(0).get("pace", 0.0)), 1.2),
		"a Speak put the file's older pace over the dial just moved: %s" % str(_ed._cfg(0)))
	_ok(FileAccess.get_file_as_string(_path).contains("pace: 1.2"),
		"the dial moved before Speak was not written to the document first")
	_doc._autosave_for_test = false


## AN EDIT MADE OUTSIDE GHOST SURVIVES THE NEXT AUTOSAVE. The panel writes its whole block
## whenever anything in it changes, and it used to write it from memory - so a voice dial the
## author reverted in their own editor came straight back the next time an UNRELATED control
## moved ("I keep reverting the frontmatter, yet Ghost keeps resetting it"). Both edits must
## land: the file's dial and the panel's own change.
func _check_outside_edit_survives() -> void:
	_doc.allow_autosave_for_test()
	_doc.pull()                      # ghost and the file agree
	var raw := FileAccess.get_file_as_string(_path)
	var re := RegEx.create_from_string("pace: [0-9.]+")
	var m := re.search(raw)
	_ok(m != null, "the fixture carries no pace to edit")
	if m == null:
		return
	_write(raw.substr(0, m.get_start()) + "pace: 0.7" + raw.substr(m.get_end()))
	_ed._turn.value = 1.9            # something else, moved in the panel
	await _wait(int(DocSource.AUTOSAVE_MS) + 900)
	var after := FileAccess.get_file_as_string(_path)
	_ok(after.contains("pace: 0.7"), "the autosave put the panel's old pace back over an edit made in the file")
	_ok(after.contains("turn: 1.9"), "the panel's own change was lost in reconciling with the file")
	_ok(is_equal_approx(float(_ed._cfg(0).get("pace", 0.0)), 0.7),
		"the panel does not show the file's edit after reconciling: %s" % str(_ed._cfg(0)))
	_doc._autosave_for_test = false


## THE PICTURE TRAVELS WITH THE DOCUMENT - medium, Look filters and the Director's dials. They
## lived only in ghost.cfg, so a chapter opened on another machine came up in that machine's
## medium and look. Set here, saved, changed, and read back by the next Speak.
func _check_the_picture_travels() -> void:
	var was := {"medium": Director.medium, "filters": Director.filters.duplicate(),
		"pacing": Director.pacing, "hand": Director.hand}
	Director.set_medium("notebook")
	Director.set_hand("patrick")
	Director.set_filter("static", 0.3)
	Director.set_filter("vignette", 0.0)
	Director.set_pacing(1.4)
	_ok(_doc.save(), "saving the picture into the document failed")
	var raw := FileAccess.get_file_as_string(_path)
	_ok(raw.contains("picture:") and raw.contains("medium: notebook"), "the picture was not written into the frontmatter")
	Director.set_medium("full")
	Director.set_filter("static", 0.0)
	Director.set_pacing(1.0)
	Director.set_hand("kalam")
	_doc.allow_autosave_for_test()
	_doc._saved = _doc._snapshot()
	_doc._autosave_for_test = false
	_doc.pull()
	_ok(Director.medium == "notebook", "the medium did not come back from the document (%s)" % Director.medium)
	_ok(is_equal_approx(Director.filter_amount("static"), 0.3), "the Look did not come back from the document")
	_ok(is_equal_approx(Director.pacing, 1.4), "the scene hold did not come back from the document")
	_ok(Director.hand == "patrick", "the handwriting did not come back from the document (%s)" % Director.hand)
	_ok(_ed._hand_pick.get_item_text(_ed._hand_pick.selected) == "Patrick",
		"the panel does not show the document's handwriting")
	_ok(is_equal_approx(_ed._scene_hold.value, 1.4), "the panel does not show the document's scene hold")
	var row: Dictionary = _ed._filter_rows["static"]
	_ok((row["box"] as CheckBox).button_pressed, "the panel does not show the document's filter")
	# put the Director back as it was
	Director.set_medium(String(was["medium"]))
	for k in Filters.REGISTRY:
		Director.set_filter(k, float((was["filters"] as Dictionary).get(k, 0.0)))
	Director.set_pacing(float(was["pacing"]))
	Director.set_hand(String(was["hand"]))


## THE PICTURES' LOOK TRAVELS WITH THE DOCUMENT - painter, style and reference images, into
## the frontmatter and back at the next Speak. The style is the awkward case on purpose: a
## colon, quotes and a line break, all of which YAML would take apart if written carelessly.
func _check_the_look_travels() -> void:
	Illustrations.use_for_test({}, false)
	var img := Image.create(8, 8, false, Image.FORMAT_RGB8)
	img.fill(Color(0.8, 0.2, 0.2))
	var ref := DIR + "/ref.png"
	img.save_png(ref)
	var style := "Ink and wash: muted, \"quiet\".\nNo text anywhere."
	Illustrations.set_style(style)
	Illustrations.set_style("Ballpoint: schematic, \"plain\".", "sketch")
	Illustrations.add_references([ProjectSettings.globalize_path(ref)])
	_ok(_doc.save(), "saving the look into the document failed")
	var raw := FileAccess.get_file_as_string(_path)
	_ok(raw.contains("illustrations:") and raw.contains("painter:"),
		"the look was not written into the frontmatter")
	_ok(raw.ends_with(BODY_A), "writing the look changed the chapter")
	# Change everything in the library, then Speak: the document's look must come back.
	Illustrations.set_style("something else")
	Illustrations.set_style("something else", "sketch")
	Illustrations.set_look({"references": []})
	_ok(Illustrations.references().is_empty(), "the control is wrong - the references did not clear")
	_doc.allow_autosave_for_test()
	_doc._saved = _doc._snapshot()      # no flush: the file must win this time
	_doc._autosave_for_test = false
	_doc.pull()
	_ok(Illustrations.style() == style, "the style came back as %s" % JSON.stringify(Illustrations.style()))
	_ok(Illustrations.style("sketch") == "Ballpoint: schematic, \"plain\".",
		"the sketch style came back as %s" % JSON.stringify(Illustrations.style("sketch")))
	_ok(Illustrations.references().size() == 1, "the reference images did not come back from the document")
	_ok(String(_ed._illustrations._style.text) == style, "the panel's style box does not show the document's style")
	DirAccess.remove_absolute(ProjectSettings.globalize_path(ref))


## THE DOCUMENT IS WRITTEN IN GHOST'S OWN EDITOR. Typed text reaches the file's body after
## the quiet period and not during it, the frontmatter is byte-identical, and the three ways a
## clash can end are each held: refused (the file keeps the outside edit, the editor keeps the
## author's, a Speak reads what is on screen), "Take the file's", and "Keep mine".
func _check_editing_in_ghost() -> void:
	_doc.allow_autosave_for_test()
	_doc.pull()
	var te: TextEdit = _ed._text
	var head := String(FrontMatter.split(FileAccess.get_file_as_string(_path)).head)
	te.set_caret_line(te.get_line_count() - 1)
	te.set_caret_column(0)
	te.insert_text_at_caret("Written in ghost.\n")
	var typed := te.text
	await _wait(300)
	_ok(not FileAccess.get_file_as_string(_path).contains("Written in ghost."),
		"the body was written while the author was still typing")
	await _wait(int(DocSource.AUTOSAVE_MS) + 900)
	var raw := FileAccess.get_file_as_string(_path)
	_ok(FrontMatter.lf(String(FrontMatter.split(raw).body)) == typed,
		"the edit made in ghost did not reach the file's body")
	_ok(String(FrontMatter.split(raw).head) == head, "writing the body changed the frontmatter")

	# THE CLASH: the file edited elsewhere, and more typed here, before either was saved.
	_write(raw.replace("Written in ghost.", "Written elsewhere."))
	te.insert_text_at_caret("And more.\n")
	await _wait(int(DocSource.AUTOSAVE_MS) + 900)
	var after := FileAccess.get_file_as_string(_path)
	_ok(after.contains("Written elsewhere.") and not after.contains("And more."),
		"an edit made in ghost was written over an edit made outside it")
	_ok(_doc._conflict, "the clash was not raised for the author to resolve")
	_ok(_doc.pull() == te.text and te.text.contains("And more."),
		"a Speak during a clash threw away the words typed in ghost")
	# ...and Clear may not discard them either.
	_doc.clear()
	_ok(_doc.is_sync() and te.text.contains("And more."),
		"leaving the document dropped edits that could not be saved")
	# TAKE THE FILE'S.
	_ok(_doc.reload() and not _doc._conflict and te.text.contains("Written elsewhere.")
			and not te.text.contains("And more."), "'Take the file's' did not show the file")
	# KEEP MINE.
	te.insert_text_at_caret("Mine.\n")
	_write(FileAccess.get_file_as_string(_path).replace("Written elsewhere.", "Third hand."))
	_doc.pull()
	_ok(_doc._conflict, "the second clash was not raised")
	_ok(_doc._write_body(true), "'Keep mine' failed")
	var kept := FileAccess.get_file_as_string(_path)
	_ok(kept.contains("Mine.") and not kept.contains("Third hand.")
			and String(FrontMatter.split(kept).head) == head and not _doc._conflict,
		"'Keep mine' did not write the editor's text over the file, frontmatter intact")
	_doc._autosave_for_test = false
	_write(HEAD + BODY_A)
	_doc.reload()


## TITLE, AUTHOR, BOOK: edited on the card, written into the synced file's own frontmatter by
## the checked one-key surgery - body and every other key untouched - removed when emptied, and
## what the reading uses (the title is read first; the book medium's cover).
func _check_fields() -> void:
	var before := FileAccess.get_file_as_string(_path)
	var parts := FrontMatter.split(before)
	_ok(_doc.field("title") == "Chapter One", "the file's title was not read")
	_ok(_doc.set_field("author", "Ann Example"), "setting the author failed")
	var after := FileAccess.get_file_as_string(_path)
	_ok(BookLayout.field_of(after, "author") == "Ann Example", "the author did not reach the file")
	_ok(String(FrontMatter.split(after).body) == String(parts.body), "setting a field changed the body")
	_ok(FrontMatter.read_block(after).data == FrontMatter.read_block(before).data,
		"setting a field changed ghost's block")
	_ok(_doc.set_field("title", "Chapter Uno") and _ed.book_document()["title"] == "Chapter Uno"
			and _ed.book_document()["author"] == "Ann Example",
		"the reading does not use the fields set on the card")
	_ok(_doc.set_field("author", "") and not FileAccess.get_file_as_string(_path).contains("author:"),
		"emptying a field left the key behind")
	# Back to the original value: everything is the original file but the rewritten title line
	# (its quoting is the emitter's now - a value that was REWRITTEN, not one left alone).
	_doc.set_field("title", "Chapter One")
	var back := FileAccess.get_file_as_string(_path)
	_ok(back == before.replace("title: 'Chapter One'", "title: Chapter One"),
		"the fields did not round-trip back to the original file:\n%s" % back)


## SYNC TO… writes the script into a new file with the voice in its frontmatter and stays
## there: an edit typed afterwards lands in that file too.
func _check_sync_to() -> void:
	_doc.allow_autosave_for_test()
	_doc.clear()
	var te: TextEdit = _ed._text
	te.insert_text_at_caret("A new script.\n")
	# Fields set on an unsynced script travel into the file at Sync to….
	_doc.set_field("title", "Untitled No More")
	_ok(_doc.field("title") == "Untitled No More", "an unsynced script did not keep its title")
	var target := DIR + "/synced.md"
	_ok(_doc.sync_to(target), "Sync to… failed")
	var raw := FileAccess.get_file_as_string(target)
	_ok(_doc.is_sync() and _doc.doc_path() == target, "Sync to… did not sync to the new file")
	_ok(FrontMatter.lf(String(FrontMatter.split(raw).body)).strip_edges() == "A new script.",
		"the script did not land in the new file: %s" % raw)
	_ok(FrontMatter.read_block(raw).data.has("generative"), "the voice did not go into its frontmatter")
	_ok(BookLayout.field_of(raw, "title") == "Untitled No More" and _doc._fields.is_empty(),
		"the unsynced title did not move into the file")
	te.set_caret_line(te.get_line_count() - 1)
	te.insert_text_at_caret("And a second line.\n")
	await _wait(int(DocSource.AUTOSAVE_MS) + 900)
	_ok(FileAccess.get_file_as_string(target).contains("And a second line."),
		"an edit after Sync to… did not reach the file - it was a one-time copy")
	_doc._autosave_for_test = false
	_doc._on_picked(_path)
	DirAccess.remove_absolute(target)


## AN UNATTENDED PROCESS MUST NOT EDIT A MANUSCRIPT. The control for everything above: with
## the gate's seam taken away this probe is read-only exactly as a render is, and the same
## dial move must reach nothing.
func _check_unattended_processes_never_autosave() -> void:
	_doc._autosave_for_test = false
	_ok(Settings.is_read_only(),
		"the control is wrong - this probe is not read-only, so it says nothing about a render")
	var before := FileAccess.get_file_as_string(_path)
	_ed._turn.value = 5.5
	_ed._text.insert_text_at_caret("Typed by nobody.")
	await _wait(int(DocSource.AUTOSAVE_MS) + 900)
	_ok(FileAccess.get_file_as_string(_path) == before,
		"a read-only process (a render, the analyzer, a probe) wrote to the author's document")


func _wait(ms: int) -> void:
	var until := Time.get_ticks_msec() + ms
	while Time.get_ticks_msec() < until:
		await get_tree().process_frame


func _write(text: String) -> void:
	var fh := FileAccess.open(_path, FileAccess.WRITE)
	fh.store_string(text)
	fh.close()
