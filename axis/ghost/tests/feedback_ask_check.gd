extends Node

## THE ASK BOX on the ` feedback console: a note written with it off is LOGGED - the screenshot
## and the record land on disk and in the feedback list - and nothing is dispatched. Asked for as
## "Sometimes, I do not want to prompt the agent - I just want to log the screenshot and feedback."
##
##   GHOST_PROBE_GPU=1 tests/run_boot_probe.sh tests/feedback_ask_check.gd 60
##
## A BOOT probe: the box is remembered through [Settings] and the backend is read through
## [Splash]. It never touches the author's feedback: the console writes into a scratch folder,
## the backend is chosen in memory only (a probe's Settings never reach the disk), and the
## Assistant it asks is held at its concurrency limit throughout, so even an entry that wrongly
## queues cannot start a run - the control case below queues one on purpose.

const SCRATCH := "user://feedback_ask_check"

var _fails := 0
var _heard: Array = []   # every `submitted` the console emitted: [index, query, stem, ask]


func _ready() -> void:
	_run.call_deferred()


func _ok(cond: bool, what: String) -> void:
	print(("  ok    " if cond else "  FAIL  ") + what)
	if not cond:
		_fails += 1


func _run() -> void:
	var backend_was: Variant = Settings.read("assistant", "backend", "")
	var ask_was: Variant = Settings.read(FeedbackConsole.ASK_SECTION, FeedbackConsole.ASK_KEY, true)
	_clear_scratch()

	print("the rule")
	_ok(Assistant.sends(true, "claude_cli"), "asked, with a backend: sent")
	_ok(not Assistant.sends(false, "claude_cli"), "not asked: only logged")
	_ok(not Assistant.sends(true, ""), "no backend: only logged")

	var fc: FeedbackConsole = preload("res://scripts/feedback.gd").new()
	fc.dir = SCRATCH
	fc.describe = func() -> Dictionary: return {"scene": "probe"}
	fc.freeze = func(_on: bool) -> void: pass
	fc.advance = func() -> void: pass
	add_child(fc)
	fc.submitted.connect(func(i: int, q: String, s: String, a: bool) -> void: _heard.append([i, q, s, a]))
	Settings.write(FeedbackConsole.ASK_SECTION, FeedbackConsole.ASK_KEY, true)

	print("the box")
	Settings.write("assistant", "backend", "")
	fc._open_console()
	_ok(fc._ask.disabled and not fc._ask.button_pressed and not fc.asks(),
		"no assistant chosen: shown off and unclickable, though the remembered choice is on")
	fc._close()
	Settings.write("assistant", "backend", "claude_cli")
	fc._open_console()
	_ok(not fc._ask.disabled and fc._ask.button_pressed and fc.asks(),
		"an assistant chosen: the remembered choice (on) is back")
	fc._ask.button_pressed = false   # the click: emits `toggled`, which Settings.bind saves
	_ok(not fc.asks(), "unticked: this note is not asked")
	_ok(Settings.read(FeedbackConsole.ASK_SECTION, FeedbackConsole.ASK_KEY, true) == false,
		"unticked: remembered")

	print("a logged note")
	_submit(fc, "logged only, please")
	var logged: Array = _heard.back() if not _heard.is_empty() else []
	_ok(not logged.is_empty() and logged[3] == false, "the submission says: not asked")
	var stem := String(logged[2]) if not logged.is_empty() else ""
	_ok(stem != "" and FileAccess.file_exists(stem + ".json") and FileAccess.file_exists(stem + ".png"),
		"the record and the screenshot are written anyway")
	var rec: Variant = JSON.parse_string(FileAccess.get_file_as_string(stem + ".json")) if stem != "" else null
	_ok(rec is Dictionary and String(rec.get("query", "")) == "logged only, please", "the note is in the record")

	print("the box is remembered, both ways")
	fc._open_console()
	_ok(not fc._ask.button_pressed, "reopened: still off")
	fc._ask.button_pressed = true
	_submit(fc, "this one goes to the assistant")
	_ok(_heard.size() == 2 and _heard[1][3] == true, "ticked: the submission says asked")

	print("the assistant")
	var a: Assistant = preload("res://scripts/assistant.gd").new()
	a._running_count = Assistant.MAX_CONCURRENT   # nothing can start, whatever is queued
	add_child(a)
	await get_tree().process_frame
	a.enqueue(int(logged[0]), String(logged[1]), stem, false)
	var e0: Dictionary = a._entries[0]
	_ok(int(e0.index) == int(logged[0]) and String(e0.status) == "orphaned",
		"a logged note is listed, not sent (status %s)" % e0.status)
	_ok(not FileAccess.file_exists(stem + ".assistant.json"), "and has no dispatch log")
	# The control: the same note asked must take the dispatch path, or the check above is
	# asserting nothing. Removed before a frame can pass.
	a.enqueue(int(logged[0]), String(logged[1]), stem, true)
	var e1: Dictionary = a._entries[0]
	_ok(String(e1.status) == "queued", "control: the same note asked queues (status %s)" % e1.status)
	a._entries.erase(e1)
	a._entries.erase(e0)
	a._running_count = 0
	a.queue_free()

	Settings.write("assistant", "backend", backend_was)
	Settings.write(FeedbackConsole.ASK_SECTION, FeedbackConsole.ASK_KEY, ask_was)
	fc.queue_free()
	_clear_scratch()
	print("feedback_ask_check: %s" % ("ALL OK" if _fails == 0 else "%d FAILED" % _fails))
	get_tree().quit(0 if _fails == 0 else 1)


## Type a note and press Enter. Under the dummy renderer the open-time capture is empty, so a
## stand-in picture is put where the screenshot would be.
func _submit(fc: FeedbackConsole, text: String) -> void:
	if fc._shot_img == null or fc._shot_img.is_empty():
		fc._shot_img = Image.create(16, 9, false, Image.FORMAT_RGBA8)
	fc._on_submit(text)


func _clear_scratch() -> void:
	var da := DirAccess.open(SCRATCH)
	if da == null:
		return
	for fn in da.get_files():
		da.remove(fn)
