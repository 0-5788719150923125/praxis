extends CanvasLayer
class_name Assistant

## Assistant - the feedback browser, and (opt-in) tight Claude Code integration.
##
## This node exists unconditionally (see main.gd / mask_editor.gd) - it is
## ALSO the only UI for reviewing and deleting old feedback console
## submissions (see feedback.gd), which has nothing to do with AI dispatch and
## shouldn't require opting into it.
##
## Dispatch itself is gated on the splash's Assistant dropdown (see
## splash.gd): set to anything but Off, every feedback submission is
## immediately handed to a fresh `claude -p --dangerously-skip-permissions`
## subprocess - the same one-shot workflow already used interactively all
## session, just wired to fire the moment feedback lands instead of waiting
## for a human to notice and paste it in. Sonnet 5, default (auto) effort -
## the CLI's --effort flag has no "auto" value, so it's simply omitted; the
## session picks its own. "Claude Code CLI" is the only backend implemented so
## far - the dropdown exists as a menu of one so a second backend is a new
## entry there, not a redesign here. With it set to Off, submissions still
## show up (status "orphaned") so they stay reviewable/deletable - they just
## never get sent anywhere on their own.
##
## Runs are SERIAL: only one subprocess is ever live at a time, queued entries
## wait. Several fresh, permission-bypassed agents editing the same working
## tree concurrently is exactly the kind of conflict this whole session has
## spent its time finding and fixing IN THIS EDITOR - two independent fixes to
## the same shader at once is a guaranteed mess, not a speedup.
##
## Conversations persist to feedback/NNNN.assistant.json (paired with the
## feedback console's own NNNN.json/.png) so they survive an app restart, and
## each is resumable: once a run completes, a typed follow-up continues the
## SAME claude session via --resume, not a fresh one - a real back-and-forth,
## not a one-shot fire-and-forget log.
##
## UI lives bottom-right, above the export status row (see mask_editor.gd /
## exporter.gd's shared notification corner) - CLOSED by default: a chat-
## bubble toggle button is the only thing on screen until clicked, carrying an
## activity badge ("💬 2") when something's running/queued so there's a hint
## without opening it. Expanded, it's a scrollable list of entries, each
## expandable to show the full exchange, with a delete button per entry.

const DIR := "res://feedback"
const PANEL_W := 380
const LIST_H := 280

## Only ever call `claude` with a controlled argument list via bash's safe
## positional-parameter trick ("$1", passed as a real argv element, never
## string-interpolated into the script) - the prompt carries arbitrary user
## text and must never be concatenated into the shell command itself.
var _claude_bin := "claude"
var _repo_root := ""

var _entries: Array = []   # Array[Dictionary], newest first - see enqueue()
var _dispatching := false  # true while one entry is actively running
var _timer_tick := 0.0     # throttles the running-entry elapsed-timer repaint to ~1/sec

var _panel: Control
var _list_col: VBoxContainer
var _header: Label
var _toggle_btn: Button
var _expanded := false   # closed by default - see _build_ui


func _ready() -> void:
	layer = 126   # below the feedback console itself (128), above ordinary scene UI
	_resolve_claude_bin()
	_resolve_repo_root()
	_ensure_dir()
	_build_ui()
	_load_existing()
	_refresh_list()


## `which claude` once at startup - PATH when Godot is launched from a GUI
## entry (vs. a shell) doesn't always match an interactive shell's PATH, so
## resolve the real binary rather than assume bare "claude" works.
func _resolve_claude_bin() -> void:
	var out := []
	var code := OS.execute("which", ["claude"], out)
	if code == 0 and not out.is_empty():
		var path: String = str(out[0]).strip_edges()
		if not path.is_empty():
			_claude_bin = path


## ghost's project root is axis/ghost - the actual git repo (praxis) is two
## directories up. The subprocess needs to run FROM the repo root so it sees
## the same CLAUDE.md / git context an interactive session would.
func _resolve_repo_root() -> void:
	_repo_root = ProjectSettings.globalize_path("res://../..").simplify_path()


func _ensure_dir() -> void:
	if not DirAccess.dir_exists_absolute(ProjectSettings.globalize_path(DIR)):
		DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(DIR))


# --- public API ----------------------------------------------------------------

## Called by whoever owns the FeedbackConsole (main.gd / mask_editor.gd) right
## after it writes feedback/NNNN.json - see feedback.gd's `submitted` signal.
## Fires this instant ONLY if the splash's Assistant dropdown actually has a
## backend selected; with it Off, the entry is still listed (this node exists
## unconditionally now - see main.gd/mask_editor.gd - specifically so old
## feedback stays browsable/deletable without opting into AI dispatch), it
## just never gets sent anywhere on its own.
func enqueue(index: int, query: String, stem: String) -> void:
	var entry := {
		"index": index, "query": query, "stem": stem,
		"status": "queued", "session_id": "", "response": "", "cost_usd": 0.0,
		"pid": -1, "out_path": "", "err_path": "", "error_text": "", "expanded": true,
		"pending_prompt": "",
	}
	_entries.push_front(entry)
	if Splash.assistant_backend() == "":
		entry.status = "orphaned"
		_refresh_list()
		return
	_save_entry(entry)
	_refresh_list()
	_pump_queue()


# --- dispatch --------------------------------------------------------------------

func _pump_queue() -> void:
	if _dispatching:
		return
	for e in _entries:
		if e.status == "queued":
			var pending: String = e.get("pending_prompt", "")
			e.pending_prompt = ""
			_dispatch(e, pending if pending != "" else _build_prompt(e))
			return


## `prompt` is either the initial dispatch prompt (built from the feedback
## record) or a typed follow-up - either way it lands on the SAME claude
## session once `entry.session_id` is set, via --resume.
func _dispatch(entry: Dictionary, prompt: String) -> void:
	_dispatching = true
	entry.status = "running"
	entry.started_at = Time.get_ticks_msec()
	entry.read_offset = 0
	entry.progress = "starting…"
	var base := ProjectSettings.globalize_path(DIR)
	entry.out_path = "%s/%04d.out.json" % [base, int(entry.index)]
	entry.err_path = "%s/%04d.err.log" % [base, int(entry.index)]
	var resume_part := ""
	if entry.session_id != "":
		resume_part = " --resume %s" % entry.session_id
	# stream-json (NDJSON: one event object per line, as they happen), not the
	# single-blob json mode - --output-format json gives NOTHING until the
	# whole run finishes, which is exactly what prompted this: a real,
	# correctly-running session with zero visible sign of life for however
	# long the fix takes. --verbose is required alongside --print for
	# stream-json. _poll_progress tails this file each frame for a live
	# "what's it doing right now" readout; _finish still just wants the LAST
	# line (type "result") once the process exits.
	#
	# cd/exec/redirects are all Godot-controlled strings (paths we built, never
	# user text) - safe to interpolate directly. The prompt is NOT interpolated;
	# it arrives as bash's $1, a real argv element.
	var script := "cd \"%s\" && exec \"%s\" -p --model sonnet --dangerously-skip-permissions --output-format stream-json --verbose%s \"$1\" > \"%s\" 2> \"%s\"" % [
		_repo_root, _claude_bin, resume_part, entry.out_path, entry.err_path]
	entry.pid = OS.create_process("/bin/bash", ["-c", script, "bash", prompt])
	if int(entry.pid) < 0:
		# create_process failed outright (bad binary, spawn error) - without
		# this, the entry would sit at "running" forever: _process()'s poll
		# only ever calls _finish() when is_process_running() sees an ACTUAL
		# pid, and a negative one never satisfies that, so nothing would ever
		# notice or unblock the queue.
		entry.status = "error"
		entry.error_text = "failed to start the claude subprocess (OS.create_process returned %d)" % int(entry.pid)
		_dispatching = false
		_save_entry(entry)
		_refresh_list()
		_pump_queue()
		return
	_save_entry(entry)
	_refresh_list()


## Mirrors exactly how this whole session actually worked: point the fresh
## session at the feedback record + screenshot and let it read them itself,
## rather than trying to pre-digest the state into the prompt (the record
## already carries the full resolved layer stack, markers, and playhead - it
## IS the context).
func _build_prompt(entry: Dictionary) -> String:
	var rel: String = String(entry.stem).trim_prefix("res://")   # "feedback/0051"
	return ("New feedback was left in ghost's feedback console: axis/ghost/%s.json " +
		"(screenshot at axis/ghost/%s.png). Read both to understand the exact scene/state " +
		"being reacted to, then fix the issue. Their note: \"%s\"\n\n" +
		"This runs unattended - scope the fix to what's actually reported rather than " +
		"opportunistically refactoring nearby code.") % [rel, rel, String(entry.query)]


func _process(_dt: float) -> void:
	var any_running := false
	for e in _entries:
		if e.status == "running" and int(e.pid) >= 0:
			any_running = true
			_poll_progress(e)
			if not OS.is_process_running(int(e.pid)):
				_finish(e)
	# The elapsed timer + progress line only need to repaint about once a
	# second, not every frame - rebuilding the whole list is real UI work,
	# and nobody's watching closely enough for sub-second granularity anyway.
	if any_running:
		_timer_tick += _dt
		if _timer_tick >= 1.0:
			_timer_tick = 0.0
			_refresh_list()


## Tails entry.out_path for newly-written, COMPLETE stream-json lines (a
## partial trailing line - still being written - is left for the next poll)
## and keeps entry.progress pointed at the most recent meaningful event, so
## the UI has something honest to show while a run is still in flight instead
## of a bare "running..." with no sign of life (see _dispatch's stream-json
## switch - this is the whole reason for it).
func _poll_progress(entry: Dictionary) -> void:
	if not FileAccess.file_exists(entry.out_path):
		return
	var fa := FileAccess.open(entry.out_path, FileAccess.READ)
	if fa == null:
		return
	var offset := int(entry.get("read_offset", 0))
	var total := fa.get_length()
	if offset >= total:
		fa.close()
		return
	fa.seek(offset)
	var chunk := fa.get_as_text()
	fa.close()
	var last_nl := chunk.rfind("\n")
	if last_nl < 0:
		return   # nothing complete yet - the line currently being written doesn't count
	entry.read_offset = offset + last_nl + 1
	for line in chunk.substr(0, last_nl).split("\n"):
		line = line.strip_edges()
		if line == "":
			continue
		var evt = JSON.parse_string(line)
		if evt is Dictionary:
			var desc := _describe_event(evt)
			if desc != "":
				entry.progress = desc


## One stream-json event -> a short human-readable "what's happening now"
## line. Only assistant turns carry anything worth showing: a tool call (Read/
## Edit/Bash/...) names itself and its main argument, plain text is the
## model's own words, everything else (system init, tool-result echoes, rate-
## limit pings) is silently skipped rather than shown raw.
func _describe_event(evt: Dictionary) -> String:
	if String(evt.get("type", "")) != "assistant":
		return ""
	var blocks: Array = evt.get("message", {}).get("content", [])
	for b in blocks:
		if String(b.get("type", "")) == "tool_use":
			var input: Dictionary = b.get("input", {})
			var hint := ""
			for key in ["file_path", "command", "pattern", "path", "prompt"]:
				if input.has(key):
					hint = String(input[key])
					break
			if hint.length() > 55:
				hint = hint.substr(0, 55) + "…"
			return "%s  %s" % [String(b.get("name", "tool")), hint] if hint != "" else String(b.get("name", "tool"))
	for b in blocks:
		if String(b.get("type", "")) == "text":
			var t := String(b.get("text", "")).strip_edges()
			if t != "":
				return t.substr(0, 70) + ("…" if t.length() > 70 else "")
	return "thinking…"


func _finish(entry: Dictionary) -> void:
	var out_text := _read_file(entry.out_path)
	var parsed = null
	if out_text != "":
		# stream-json is NDJSON, not one blob - the line we want (type
		# "result", same shape --output-format json would have given whole)
		# is the last one written.
		for line in out_text.split("\n"):
			line = line.strip_edges()
			if line == "":
				continue
			var evt = JSON.parse_string(line)
			if evt is Dictionary and String(evt.get("type", "")) == "result":
				parsed = evt
	if parsed is Dictionary and parsed.get("is_error", true) == false and parsed.has("result"):
		entry.status = "done"
		entry.response = str(parsed.get("result", ""))
		entry.session_id = str(parsed.get("session_id", entry.session_id))
		entry.cost_usd = float(parsed.get("total_cost_usd", entry.cost_usd))
		entry.error_text = ""
	else:
		entry.status = "error"
		var err_text := _read_file(entry.err_path)
		entry.error_text = err_text if err_text != "" else \
			(out_text if out_text != "" else "(no output - the process may have failed to start)")
	_delete_file(entry.out_path)
	_delete_file(entry.err_path)
	entry.pid = -1
	entry.progress = ""
	_save_entry(entry)
	_refresh_list()
	_dispatching = false
	_pump_queue()


func _read_file(path: String) -> String:
	if path == "" or not FileAccess.file_exists(path):
		return ""
	var fa := FileAccess.open(path, FileAccess.READ)
	if fa == null:
		return ""
	var s := fa.get_as_text()
	fa.close()
	return s


func _delete_file(path: String) -> void:
	if path != "" and FileAccess.file_exists(path):
		DirAccess.remove_absolute(path)


# --- persistence -------------------------------------------------------------

func _save_entry(entry: Dictionary) -> void:
	var path := "%s.assistant.json" % ProjectSettings.globalize_path(String(entry.stem))
	var data := {
		"index": entry.index, "query": entry.query, "status": entry.status,
		"session_id": entry.session_id, "response": entry.response,
		"cost_usd": entry.cost_usd, "error_text": entry.get("error_text", ""),
	}
	var fa := FileAccess.open(path, FileAccess.WRITE)
	if fa != null:
		fa.store_string(JSON.stringify(data, "\t"))
		fa.close()


## Conversations from a prior run of the app - rebuilt from the persisted
## summaries (see _save_entry). A run that was still queued/running when the
## app last closed never got to finish; the subprocess died with the editor,
## so that's surfaced honestly as an error rather than silently vanishing or
## claiming to still be "running" forever.
func _load_existing() -> void:
	var dir := DirAccess.open(DIR)
	if dir == null:
		return
	var loaded := []
	var has_record := {}   # feedback index -> true, for every entry ALREADY reconciled below
	for fn in dir.get_files():
		if not fn.ends_with(".assistant.json"):
			continue
		var fa := FileAccess.open("%s/%s" % [DIR, fn], FileAccess.READ)
		if fa == null:
			continue
		var data = JSON.parse_string(fa.get_as_text())
		fa.close()
		if not (data is Dictionary):
			continue
		var idx := int(data.get("index", 0))
		var status := str(data.get("status", "error"))
		var error_text := str(data.get("error_text", ""))
		if status == "running" or status == "queued":
			status = "error"
			error_text = "interrupted - the editor closed before this run finished"
		loaded.append({
			"index": idx, "query": str(data.get("query", "")), "stem": "%s/%04d" % [DIR, idx],
			"status": status, "session_id": str(data.get("session_id", "")),
			"response": str(data.get("response", "")), "cost_usd": float(data.get("cost_usd", 0.0)),
			"pid": -1, "out_path": "", "err_path": "", "error_text": error_text, "expanded": false,
			"pending_prompt": "",
		})
		has_record[idx] = true
	# Orphaned feedback: a NNNN.json with no matching NNNN.assistant.json - left
	# behind by a submission made before the Assistant dropdown was ever set to
	# anything but Off. LISTED, never auto-dispatched: a feedback console left
	# running for a while can easily have DOZENS of these built up (this is not
	# hypothetical - it happened, and enqueuing all of them at once meant a
	# real subprocess firing off against ancient feedback the instant the
	# assistant was ever turned on, with nothing to review or stop it - the
	# same mistake --dangerously-skip-permissions makes if it fires without
	# anyone asking it to). Each one needs its own deliberate click (see the
	# "orphaned" status in _build_body) before anything runs.
	for fn in dir.get_files():
		if not fn.ends_with(".json") or fn.ends_with(".assistant.json"):
			continue
		var stem_name := fn.get_basename()
		if not stem_name.is_valid_int():
			continue
		var idx := int(stem_name)
		if has_record.has(idx):
			continue
		var fa := FileAccess.open("%s/%s" % [DIR, fn], FileAccess.READ)
		if fa == null:
			continue
		var data = JSON.parse_string(fa.get_as_text())
		fa.close()
		var query := String(data.get("query", "")) if data is Dictionary else ""
		if query == "":
			continue
		loaded.append({
			"index": idx, "query": query, "stem": "%s/%04d" % [DIR, idx],
			"status": "orphaned", "session_id": "", "response": "", "cost_usd": 0.0,
			"pid": -1, "out_path": "", "err_path": "", "error_text": "", "expanded": false,
			"pending_prompt": "",
		})
	loaded.sort_custom(func(a, b): return int(a.index) > int(b.index))
	_entries = loaded
	_refresh_list()


func _delete_entry(entry: Dictionary) -> void:
	if entry.status == "running" and int(entry.pid) >= 0:
		OS.kill(int(entry.pid))
	_delete_file(entry.out_path)
	_delete_file(entry.err_path)
	var g := ProjectSettings.globalize_path(String(entry.stem))
	_delete_file(g + ".assistant.json")
	_delete_file(g + ".json")
	_delete_file(g + ".png")
	_entries.erase(entry)
	if entry.status == "running":
		_dispatching = false
		_pump_queue()
	_refresh_list()


# --- UI ------------------------------------------------------------------------

const _TOGGLE_SIZE := 40.0
const _TOGGLE_GAP := 4.0    # between the toggle button and the panel above it

func _build_ui() -> void:
	# Closed by default - a chat-bubble toggle sitting above the export status
	# row (see mask_editor.gd / exporter.gd's shared notification corner) is
	# the whole UI until clicked; the panel itself only exists on screen while
	# expanded.
	_toggle_btn = Button.new()
	_toggle_btn.text = "💬"
	_toggle_btn.tooltip_text = "Assistant"
	_toggle_btn.focus_mode = Control.FOCUS_NONE
	_toggle_btn.custom_minimum_size = Vector2(_TOGGLE_SIZE, _TOGGLE_SIZE)
	_toggle_btn.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	_toggle_btn.offset_right = -28
	_toggle_btn.offset_left = -28 - _TOGGLE_SIZE
	_toggle_btn.offset_bottom = -84   # sits just above the export status row
	_toggle_btn.offset_top = -84 - _TOGGLE_SIZE
	_toggle_btn.pressed.connect(func():
		_expanded = not _expanded
		_refresh_list())
	add_child(_toggle_btn)

	_panel = PanelContainer.new()
	_panel.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	_panel.offset_right = -28
	_panel.offset_left = -28 - PANEL_W
	_panel.offset_bottom = -84 - _TOGGLE_SIZE - _TOGGLE_GAP   # above the toggle, not overlapping it
	_panel.offset_top = _panel.offset_bottom - LIST_H - 28
	_panel.visible = false
	add_child(_panel)

	var outer := VBoxContainer.new()
	outer.add_theme_constant_override("separation", 6)
	_panel.add_child(outer)

	_header = Label.new()
	_header.text = "Assistant"
	_header.add_theme_color_override("font_color", Color(0.7, 0.85, 1.0, 0.9))
	outer.add_child(_header)

	var scroll := ScrollContainer.new()
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	scroll.custom_minimum_size = Vector2(0, LIST_H)
	outer.add_child(scroll)

	_list_col = VBoxContainer.new()
	_list_col.add_theme_constant_override("separation", 4)
	_list_col.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(_list_col)


func _refresh_list() -> void:
	var running := 0
	var queued := 0
	for e in _entries:
		if e.status == "running":
			running += 1
		elif e.status == "queued":
			queued += 1
	if Splash.assistant_backend() == "":
		_header.text = "Feedback (assistant off - browse/delete only)"
	elif running == 0 and queued == 0:
		_header.text = "Assistant"
	else:
		_header.text = "Assistant - %d running, %d queued" % [running, queued]
	# The toggle carries an activity badge even while collapsed, so something
	# running/queued is still visible without opening the panel.
	_toggle_btn.text = "💬" if running + queued == 0 else "💬 %d" % (running + queued)
	_panel.visible = _expanded
	for c in _list_col.get_children():
		c.queue_free()
	for e in _entries:
		_list_col.add_child(_build_row(e))


func _status_glyph(status: String) -> String:
	match status:
		"orphaned": return "○"
		"queued": return "⏳"
		"running": return "⚙"
		"done": return "✓"
		_: return "✕"


func _build_row(entry: Dictionary) -> Control:
	var row := PanelContainer.new()
	var col := VBoxContainer.new()
	col.add_theme_constant_override("separation", 4)
	row.add_child(col)

	var head := HBoxContainer.new()
	col.add_child(head)

	var toggle := Button.new()
	toggle.flat = true
	toggle.focus_mode = Control.FOCUS_NONE
	var preview: String = String(entry.query)
	if preview.length() > 40:
		preview = preview.substr(0, 40) + "…"
	toggle.text = "%s #%04d  %s" % [_status_glyph(entry.status), int(entry.index), preview]
	toggle.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	toggle.alignment = HORIZONTAL_ALIGNMENT_LEFT
	# Without this, a Button's minimum width is sized to fit its FULL text -
	# a 40-character preview easily exceeds the whole panel's width, and the
	# HBoxContainer has no way to shrink it back down. That doesn't just crowd
	# the delete button, it pushes it out of the row's allocated width
	# entirely - invisible even though nothing is actually "off-screen" in the
	# window sense. clip_text lets the button truncate instead of demanding
	# the space, which is the whole reason SIZE_EXPAND_FILL is on it - it's
	# supposed to take whatever's left AFTER `del` claims its own width, not
	# force everything else out of the way to fit uncropped.
	toggle.clip_text = true
	toggle.pressed.connect(func():
		entry.expanded = not entry.expanded
		_refresh_list())
	head.add_child(toggle)

	var del := Button.new()
	del.text = "✕"
	del.focus_mode = Control.FOCUS_NONE
	del.custom_minimum_size = Vector2(28, 0)   # guaranteed footprint regardless of the toggle's width
	del.tooltip_text = "Delete this feedback + conversation"
	del.pressed.connect(func(): _delete_entry(entry))
	head.add_child(del)

	if entry.expanded:
		col.add_child(_build_body(entry))
	return row


func _build_body(entry: Dictionary) -> Control:
	var body := VBoxContainer.new()
	body.add_theme_constant_override("separation", 4)

	var query_lbl := Label.new()
	query_lbl.text = String(entry.query)
	query_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	query_lbl.add_theme_color_override("font_color", Color(0.85, 0.9, 1.0, 0.9))
	body.add_child(query_lbl)

	match String(entry.status):
		"orphaned":
			if Splash.assistant_backend() != "":
				body.add_child(_dim_label("not sent automatically"))
				body.add_child(_build_send_row(entry))
			else:
				body.add_child(_dim_label("no assistant selected - pick one on the home screen to send this"))
		"queued":
			body.add_child(_dim_label("queued - waiting for the current run to finish"))
		"running":
			var elapsed_s: int = (Time.get_ticks_msec() - int(entry.get("started_at", Time.get_ticks_msec()))) / 1000
			var time_str := "%ds" % elapsed_s if elapsed_s < 60 else "%d:%02d" % [elapsed_s / 60, elapsed_s % 60]
			body.add_child(_dim_label("running (%s) - output is captured here, never printed to a terminal" % time_str))
			var prog := String(entry.get("progress", ""))
			if prog != "":
				var prog_lbl := Label.new()
				prog_lbl.text = "▸ " + prog
				prog_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
				prog_lbl.add_theme_color_override("font_color", Color(0.6, 0.85, 0.7, 0.9))
				body.add_child(prog_lbl)
		"done":
			var resp := Label.new()
			resp.text = String(entry.response)
			resp.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
			body.add_child(resp)
			var cost := _dim_label("$%.3f" % float(entry.cost_usd))
			body.add_child(cost)
			body.add_child(_build_followup_row(entry))
		_:
			var err := Label.new()
			err.text = String(entry.error_text)
			err.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
			err.add_theme_color_override("font_color", Color(1.0, 0.6, 0.55, 0.9))
			body.add_child(err)
	return body


func _dim_label(text: String) -> Label:
	var l := Label.new()
	l.text = text
	l.add_theme_color_override("font_color", Color(0.6, 0.68, 0.8, 0.8))
	return l


## An orphaned entry only ever moves once THIS button is pressed - see
## _load_existing's class doc for why nothing here fires on its own.
func _build_send_row(entry: Dictionary) -> Control:
	var send := Button.new()
	send.text = "Send to Claude Code CLI"
	send.focus_mode = Control.FOCUS_NONE
	send.pressed.connect(func():
		entry.status = "queued"
		_save_entry(entry)
		_refresh_list()
		_pump_queue())
	return send


## A completed conversation stays resumable - the follow-up is sent via
## --resume against entry.session_id (see _dispatch), a genuine continuation
## of the same claude session, not a fresh one that's forgotten everything.
func _build_followup_row(entry: Dictionary) -> Control:
	var row := HBoxContainer.new()
	var edit := LineEdit.new()
	edit.placeholder_text = "follow up..."
	edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	# _refresh_list() rebuilds EVERY row whenever any entry's status changes
	# (a different entry finishing running, say) - without stashing the draft
	# on the entry itself, that rebuild would wipe out whatever someone was
	# mid-typing here for a completely unrelated conversation.
	edit.text = String(entry.get("draft_followup", ""))
	edit.text_changed.connect(func(t): entry.draft_followup = t)
	row.add_child(edit)
	var send := Button.new()
	send.text = "Send"
	send.focus_mode = Control.FOCUS_NONE
	# Goes through the SAME serial queue as a fresh submission (see
	# enqueue/_pump_queue) - a follow-up on an idle, already-done entry must
	# still wait if some OTHER entry is currently running, never fire
	# concurrently against the working tree.
	send.pressed.connect(func():
		var text := edit.text.strip_edges()
		if text.is_empty():
			return
		entry.status = "queued"
		entry.response = ""
		entry.pending_prompt = text
		entry.draft_followup = ""
		_entries.erase(entry)
		_entries.push_front(entry)
		_save_entry(entry)
		_refresh_list()
		_pump_queue())
	edit.text_submitted.connect(func(_t): send.pressed.emit())
	row.add_child(send)
	return row
