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
##   THE DRAFT SURVIVES. The pasted text and a document are separate stores; a panel that
##   persisted the document's body over the draft would delete whatever was in the box the
##   first time a file was opened, silently, in the background debounce.
##   THE VOICE IS RESTORED, AND SAVED. This is the whole reason the frontmatter is written
##   at all - a cast of two readers that comes back as one is the feature not working.
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
    tab: 0
    voices:
      - voice: en_US-libritts-high
        speaker: 4
        pace: 0.8
      - voice: en_GB-alan-medium
        speaker: 0
        pace: 1.3
---
"""
const BODY_A := "\n# Chapter One\n\nThe rain had not stopped.\n"
const BODY_B := "\n# Chapter One\n\nThe rain had stopped, and the door stood open.\n"

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
	_doc._syncing = true
	_doc._mode_input.button_pressed = true
	_doc._syncing = false
	_doc._refresh_row()
	_doc.bind_text(_ed._text)

	_check_input_mode()
	_check_opening_a_document()
	_check_speak_rereads_the_disk()
	_check_the_draft_survives()
	_check_saving_the_voice()

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


## THE CONTROL FOR EVERYTHING BELOW: with no document open the panel is exactly what it
## was, and the file on disk is nothing to it.
func _check_input_mode() -> void:
	_ed._text.text = "A pasted draft."
	_ok(not _doc.is_sync(), "the panel started in sync mode with no document open")
	_ok(_doc.pull() == "A pasted draft.", "Input mode did not read the box")
	_ok(_doc.draft() == "A pasted draft.", "Input mode did not persist the box")
	_ok(_ed._text.editable, "the box was read-only in Input mode")


func _check_opening_a_document() -> void:
	_doc._on_picked(_path)
	_ok(_doc.is_sync(), "picking a document did not switch the source to sync")
	_ok(_doc.doc_path() == _path, "the source is not pointing at the document")
	_ok(not _ed._text.editable, "the box is still editable while showing a document")

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
		"tab 1 did not come back as the document wrote it: %s" % str(_ed._cfg(0)))
	_ok(String(_ed._cfg(1).get("voice", "")) == "en_GB-alan-medium"
			and is_equal_approx(float(_ed._cfg(1).get("pace", 0.0)), 1.3),
		"tab 2 did not come back as the document wrote it: %s" % str(_ed._cfg(1)))
	# A key the document did NOT mention takes its default rather than arriving missing.
	_ok(_ed._cfg(0).has("presence") and _ed._cfg(0).has("ambience"),
		"a voice written by an older build arrived with keys missing")


## THE REAL-TIME CLAIM, which is the whole point of sync mode: the author keeps writing
## off-screen and the next Speak is the file as it is NOW.
func _check_speak_rereads_the_disk() -> void:
	_write(HEAD + BODY_B)
	var body := _doc.pull()
	_ok(body == BODY_B, "a second Speak did not re-read the file (got %s)"
		% body.strip_edges().substr(0, 60))
	_ok(_ed._text.text == BODY_B, "the box did not follow the document's new text")

	# ...and the control: in Input mode the file on disk is nothing to the panel, however
	# much it changes. Without this, a `pull` that simply always read the last file it saw
	# would pass the check above.
	_doc._mode_input.button_pressed = true
	_write(HEAD + BODY_A)
	_ok(_doc.pull() == "A pasted draft.",
		"Input mode read the document anyway: %s" % _doc.pull().substr(0, 60))
	_doc._mode_sync.button_pressed = true
	_ok(_doc.pull() == BODY_A, "switching back to sync did not return to the document")


## THE DRAFT IS NOT OVERWRITTEN. `draft()` is what the panel persists, and in sync mode it
## must be the pasted text - never a copy of a document that is already on disk.
func _check_the_draft_survives() -> void:
	_ok(_doc.is_sync(), "the gate lost sync mode before checking the draft")
	_ok(_doc.draft() == "A pasted draft.",
		"the document's body was about to be saved over the draft: %s"
		% _doc.draft().substr(0, 60))
	_doc._mode_input.button_pressed = true
	_ok(_ed._text.text == "A pasted draft.",
		"the draft did not come back when the source returned to the box")
	_doc._mode_sync.button_pressed = true


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
	_ed._slot = 0
	_ed._rebuild_tabs()
	_doc.reload()
	_ok(_ed._slots.size() == 2, "the saved cast came back as %d tab(s)" % _ed._slots.size())
	_ok(is_equal_approx(_ed._turn.value, 4.0),
		"the saved Turn came back as %f" % _ed._turn.value)
	_ok(is_equal_approx(float(_ed._cfg(0).get("pace", 0.0)), 1.45),
		"the dial moved before saving did not survive the document: %s" % str(_ed._cfg(0)))


func _write(text: String) -> void:
	var fh := FileAccess.open(_path, FileAccess.WRITE)
	fh.store_string(text)
	fh.close()
