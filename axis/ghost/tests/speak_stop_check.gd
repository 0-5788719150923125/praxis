extends Node

## Gate for STOP in the Generative panel:
##
##   tests/run_boot_probe.sh tests/speak_stop_check.gd 90
##
## WHY IT NEEDS ONE. There was no way to end a reading. The Speak button's tooltip said "press
## again while it is running to stop", and that was not true of any code - `_on_speak` re-plans
## and reads again from the top whatever is playing. So the panel could start a reading and
## restart it, and the only way out was to quit ghost: "it would be preferable to stop the
## real-time scene before I do an export, and today I've just been restarting the program."
##
## A Stop that resets MOST of a reading is worse than none - it leaves the audio stream open and
## the Director still attached to a stage nothing is driving, which is the state that used to
## pile up frozen scenes. So the gate is about completeness rather than about the button: every
## field a reading owns must be back where it started, and the owner's `end_stream` must have
## been called exactly once.
##
## The editor is built by hand and never added to the tree, because its `_ready` starts the
## python voice host and nothing here needs one. The panel itself DOES go in - Range only emits
## `value_changed` for an owner inside the tree, so a panel built outside it has every slider
## callback silently dead (see multi_voice_check.gd, which learned that the hard way).

var _fails: Array = []
var _ed: GenerativeEditor
var _ended := 0


func _ready() -> void:
	_ed = GenerativeEditor.new()
	_ed._build_panel()
	_ed.remove_child(_ed._panel)
	add_child(_ed._panel)
	_ed.remove_child(_ed._repace_timer)
	add_child(_ed._repace_timer)
	_ed.end_stream = func() -> void: _ended += 1
	_run.call_deferred()


func _run() -> void:
	_check_stop_ends_a_reading()
	_check_stop_is_idempotent()
	_check_the_button_follows_the_state()
	print("")
	if _fails.is_empty():
		print("speak_stop_check: ALL OK - Stop ends a reading and hands the stage back.")
	else:
		print("speak_stop_check: %d FAILURE(S)" % _fails.size())
		for f in _fails:
			print("   ", f)
	# Freed by hand: the editor was never in the tree, and the two children that WERE were
	# reparented here, so nothing else is going to take them down.
	_ed._panel.queue_free()
	_ed._repace_timer.queue_free()
	for _i in 4:
		await get_tree().process_frame
	_ed.free()
	get_tree().quit(1 if not _fails.is_empty() else 0)


## The whole of a reading, put back. Listed field by field on purpose: a Stop that clears the
## chunk list and leaves the request map behind still splices replies for a reading that no
## longer exists into the next one.
func _check_stop_ends_a_reading() -> void:
	print("")
	print("stop_ends_a_reading")
	_ed._text.text = "The machines are running. They have been here for years. Nobody minds."
	_ed._plan(_ed._text.text)
	_ok(not _ed._chunks.is_empty(), "a reading was planned (%d chunks)" % _ed._chunks.size())
	# Pretend the stream opened and a couple of chunks went out, which is the state Stop has to
	# be able to unwind - not the tidy one just after planning.
	_ed._stream_open = true
	_ed._next_to_request = 2
	_ed._in_flight = 1
	_ed._req_chunk = {7: 0, 8: 1}
	_ed._pushed = 12345
	_ed._elapsed = 3.5
	_ed._ready_takes = [{"stub": true}]
	_ed._sub_words.append({"w": "the"})
	_ended = 0
	_ed._stop_speaking()
	_ok(_ended == 1, "the owner's end_stream was called exactly once (%d)" % _ended)
	for name in ["_chunks", "_ready_takes", "_sub_words"]:
		_ok((_ed.get(name) as Array).is_empty(), "%s is empty" % name)
	_ok(_ed._req_chunk.is_empty(),
		"the pending request map is cleared, so replies in flight are discarded on arrival")
	_ok(not _ed._stream_open, "_stream_open is false")
	_ok(_ed._playback == null, "_playback is released")
	for name in ["_next_to_request", "_next_to_play", "_in_flight", "_pushed", "_read",
			"_ring_capacity"]:
		_ok(int(_ed.get(name)) == 0, "%s is zero (%s)" % [name, str(_ed.get(name))])
	_ok(is_equal_approx(_ed._elapsed, 0.0), "the reading clock is back to zero")
	_ok(is_equal_approx(_ed._lead_in, 0.0), "the intro silence is cleared")
	_ok(_ed._pending.is_empty(), "the resampler's pending buffer is empty")


## Pressing it twice, or with nothing playing, must do nothing at all - including NOT calling
## the owner's end_stream a second time, which would detach a stage the editor no longer owns.
func _check_stop_is_idempotent() -> void:
	print("")
	print("stop_is_idempotent")
	_ended = 0
	_ed._stop_speaking()
	_ed._stop_speaking()
	_ok(_ended == 0, "stopping an already-stopped reading calls end_stream %d times" % _ended)


func _check_the_button_follows_the_state() -> void:
	print("")
	print("the_button_follows_the_state")
	_ok(_ed._stop != null, "there is a Stop button")
	_ok(_ed._stop.disabled, "Stop is greyed with nothing playing")
	_ed._text.text = "One sentence is enough."
	_ed._plan(_ed._text.text)
	_ed._sync_speak_buttons()
	_ok(not _ed._stop.disabled, "and live once a reading is planned")
	_ed._stop_speaking()
	_ok(_ed._stop.disabled, "and greyed again after Stop")
	_ok(_ed._go.text == "Speak",
		"Speak has dropped its unsaved-edit dot after a stop (%s)" % _ed._go.text)


func _ok(cond: bool, what: String) -> void:
	if cond:
		print("    ok  %s" % what)
	else:
		_fails.append(what)
		print("    FAIL %s" % what)
