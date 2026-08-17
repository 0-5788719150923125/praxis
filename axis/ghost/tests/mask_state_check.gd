extends SceneTree

## Three things the Masking editor is supposed to REMEMBER, each reported broken
## and each failing silently.
##   godot --headless --path axis/ghost --script res://tests/mask_state_check.gd
## No renderer needed - none of this draws anything.
##
##   FOLLOW    the "Follow" switch beside the marker list (does playback drag the
##             selection along) did not survive a restart. It is a preference
##             about how the EDITOR behaves rather than a property of the session
##             being edited - the same reason it sits by the list header and not
##             in the options panel - so it belongs in user://ghost.cfg beside the
##             other per-user settings. ONE cfg serves the whole app, so the write
##             has to be a READ-MODIFY-WRITE of just the [mask] section; a fresh
##             ConfigFile.save() would silently wipe splash's remembered song and
##             clip, [director]'s pacing and [synth]'s state. That is the half of
##             this a human would not notice for weeks, so it is asserted here.
##
##   KEY SWATCH  the key colour "is not persisted... the UI never reads that color
##             from state, so you never actually know what color is being keyed
##             to". The keying itself was fine: the key IS a hue and hue_a carried
##             it. The picker stored only `.h` and rebuilt the swatch as
##             from_hsv(hue_a, 0.85, 0.9), so a colour picked off the footage came
##             back a DIFFERENT colour - pick a pale yellow off a wall, reopen, get
##             a saturated one. key_sat/key_val carry the rest of the pick.
##
##   ONE DRAG, ONE UNDO  "if I click on an option and hold, and drag back and forth
##             to see how the adjustments look, every single place where I stop
##             will be added to the command history". Undo coalescing was on a 0.9s
##             TIMER, so any pause longer than that opened a fresh entry - and, the
##             other way round, two SEPARATE adjustments made inside the window
##             folded into one, so Ctrl+Z "often reverts several changes at once".
##             Both are the timer; it is gone. The boundary is the mouse button:
##             while it is down, one entry, and everything else is its own step. The VALUE must still change on every step - the live preview is
##             the point of dragging - so this checks both halves, or "fixed" could
##             mean the drag stopped previewing.

const CFG := "user://ghost.cfg"

var _fails: PackedStringArray = []
var _cfg_backup := ""
var _cfg_existed := false


func _initialize() -> void:
	_backup_cfg()
	_check_follow()
	_check_key_swatch()
	await _check_drag_undo()
	_restore_cfg()

	print("")
	if _fails.is_empty():
		print("mask_state_check: PASS - Follow persists, the key swatch round-trips, "
			+ "and a drag is one undo step.")
		quit(0)
	else:
		for f in _fails:
			print("mask_state_check: FAIL - ", f)
		quit(1)


func _expect(ok: bool, msg: String) -> void:
	if not ok:
		_fails.append(msg)


func _check_follow() -> void:
	var ed = load("res://scripts/mask_editor.gd").new()
	# A neighbouring section, planted first: the read-modify-write is what has to
	# preserve it, and a fresh ConfigFile.save() is the mistake being guarded
	# against. splash's remembered clip lives exactly like this.
	var seed_cfg := ConfigFile.new()
	seed_cfg.load(CFG)
	seed_cfg.set_value("splash", "_state_check_witness", "keep me")
	seed_cfg.save(CFG)

	_expect(ed._load_follow(), "Follow does not default to ON with nothing stored - "
		+ "that is the behaviour every session had before the switch existed")
	ed._save_follow(false)
	var back: bool = bool(ed._load_follow())
	print("Follow: stored false -> reads back %s" % back)
	_expect(not back, "Follow was stored as OFF and read back as %s - it is not "
		% back + "persisting, which is the whole report")
	ed._save_follow(true)
	_expect(ed._load_follow(), "Follow stored as ON did not read back")

	var after := ConfigFile.new()
	after.load(CFG)
	var witness := String(after.get_value("splash", "_state_check_witness", ""))
	print("the neighbouring [splash] section after two writes: %s"
		% ("intact" if witness == "keep me" else "LOST"))
	_expect(witness == "keep me",
		"writing the Follow preference wiped the [splash] section - the write must "
		+ "be a read-modify-write, or saving one editor toggle throws away the "
		+ "remembered song, clip, pacing and synth state")
	ed.free()


func _check_key_swatch() -> void:
	# A colour a person would actually pick off footage: a pale, unsaturated
	# yellow wall. The old round trip returned it at s=0.85, v=0.9 whatever it was.
	var picked := Color(0.90, 0.86, 0.42)
	var m := {}
	for k in MaskSession.VECTOR_FIELDS:
		m[k] = MaskSession.DEFAULTS.get(k, 0.0)
	# What the picker's handler stores...
	m["hue_a"] = picked.h
	m["key_sat"] = picked.s
	m["key_val"] = picked.v
	# ...through a full save/load of the session, which is where "after a restart"
	# actually happens.
	var s := MaskSession.new()
	s.markers.append(m)
	var round_trip := MaskSession.from_vector(s.to_vector(s.markers[0]))
	# ...and what the panel rebuilds from it.
	var shown := Color.from_hsv(float(round_trip.get("hue_a", 0.02)),
		float(round_trip.get("key_sat", 0.85)), float(round_trip.get("key_val", 0.9)))
	print("key swatch: picked (%.3f, %.3f, %.3f) -> shown (%.3f, %.3f, %.3f)"
		% [picked.r, picked.g, picked.b, shown.r, shown.g, shown.b])
	var off := Vector3(picked.r - shown.r, picked.g - shown.g, picked.b - shown.b).length()
	_expect(off < 0.05,
		"the key swatch comes back %.3f away from the colour that was picked - the "
		% off + "hue survives (that is what keys) but the swatch is rebuilt at a "
		+ "fixed saturation and value, so it shows a colour nobody chose")
	# A session written before these fields existed must open unchanged.
	var old := {"hue_a": 0.13}
	var legacy := Color.from_hsv(float(old.get("hue_a", 0.02)),
		float(old.get("key_sat", 0.85)), float(old.get("key_val", 0.9)))
	var was := Color.from_hsv(0.13, 0.85, 0.9)
	_expect(legacy.is_equal_approx(was),
		"a session predating key_sat/key_val no longer opens looking the way it did")


func _check_drag_undo() -> void:
	var ed = load("res://scripts/mask_editor.gd").new()
	# _snapshot() reads the session, so give it one; nothing below touches the
	# video or the panel.
	ed.session = MaskSession.new()
	root.add_child(ed)
	var key := "marker:1:fx_scale"

	# WITH THE BUTTON DOWN: one boundary, however many changes and however long the
	# pauses between them.
	_hold(true)
	await process_frame
	for i in 6:
		ed._push_undo(key, "adjusted Scale")
	var during: int = ed._undo_stack.size()
	_hold(false)
	await process_frame
	print("six changes during one held drag -> %d undo step(s)" % during)
	_expect(during == 1,
		"a single drag left %d entries in the history - every place the pointer "
		% during + "paused opened another one, which is the report")

	# ...and the next drag is its own step, or undo would swallow unrelated work.
	_hold(true)
	await process_frame
	ed._push_undo(key, "adjusted Scale")
	_hold(false)
	await process_frame
	print("a second drag of the SAME control -> %d undo step(s) in total"
		% ed._undo_stack.size())
	var total: int = ed._undo_stack.size()
	_expect(total == 2,
		"a second, separate drag did not open its own history entry (%d total) - "
		% total + "one undo would now revert two adjustments")

	# THE PREVIEW MUST STILL MOVE. "Do not record the drag" is satisfied trivially
	# by not applying it either, and that would be a worse bug than the one being
	# fixed, so the value's own path is checked to still write on every step.
	var m2 := {}
	for k in MaskSession.VECTOR_FIELDS:
		m2[k] = MaskSession.DEFAULTS.get(k, 0.0)
	ed.session.markers.append(m2)
	ed._selected = m2
	# The three pieces of chrome _edit touches AFTER writing the value - the
	# timeline strip and the marker label/list it refreshes. Without them the write
	# still happens and the assertion below still holds, but every step prints a
	# null-access error, and an expected error in the output is where a real one
	# hides.
	ed._timeline = MaskTimeline.new()
	root.add_child(ed._timeline)
	ed._marker_label = Label.new()
	ed._marker_list = VBoxContainer.new()
	root.add_child(ed._marker_label)
	root.add_child(ed._marker_list)
	_hold(true)
	await process_frame
	var seen: Array = []
	for v in [0.4, 0.9, 1.6, 2.2]:
		ed._edit("fx_scale", v)
		seen.append(float(m2.get("fx_scale", -1.0)))
	_hold(false)
	await process_frame
	print("values the marker took during the drag: %s" % str(seen))
	_expect(seen == [0.4, 0.9, 1.6, 2.2],
		"the marker did not follow the drag (%s) - the live preview is the point of "
		% str(seen) + "dragging; only the HISTORY is supposed to wait for the release")
	ed.free()


## Hold or release the left mouse button, as the editor sees it. _push_undo reads
## the button rather than a per-widget drag signal so that one rule covers sliders,
## the colour wheel and the region box.
func _hold(down: bool) -> void:
	var ev := InputEventMouseButton.new()
	ev.button_index = MOUSE_BUTTON_LEFT
	ev.pressed = down
	Input.parse_input_event(ev)


func _backup_cfg() -> void:
	_cfg_existed = FileAccess.file_exists(CFG)
	if _cfg_existed:
		var f := FileAccess.open(CFG, FileAccess.READ)
		_cfg_backup = f.get_as_text()
		f.close()


## The author's own settings live in this file. Put it back exactly as it was.
func _restore_cfg() -> void:
	if _cfg_existed:
		var f := FileAccess.open(CFG, FileAccess.WRITE)
		f.store_string(_cfg_backup)
		f.close()
	else:
		DirAccess.remove_absolute(ProjectSettings.globalize_path(CFG))
