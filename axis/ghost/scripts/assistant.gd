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
## Up to MAX_CONCURRENT subprocesses run at once; anything past that queues.
## Each is a fresh, permission-bypassed agent editing the SAME working tree,
## so two runs CAN still land conflicting edits on the same file - kept
## deliberately small, and each dispatch prompt tells the agent to scope its
## edit narrowly rather than roam, precisely so concurrent runs mostly land on
## different files.
##
## Conversations persist to feedback/NNNN.assistant.json (paired with the
## feedback console's own NNNN.json/.png) so they survive an app restart, and
## each is resumable: once a run completes, a typed follow-up continues the
## SAME claude session via --resume, not a fresh one - a real back-and-forth,
## not a one-shot fire-and-forget log.
##
## UI lives bottom-right, in the SAME row as the export button (see
## mask_editor.gd / exporter.gd's shared "Export video" button), right of it,
## right in the corner - CLOSED by default: a chat-bubble toggle button is the
## only thing on screen until clicked, carrying an activity badge ("💬 2")
## when something's running/queued so there's a hint without opening it.
## Expanded, it's a scrollable list of entries, each expandable to show the
## full exchange, with a delete button per entry. Both export buttons leave
## room for it (see their own offset_right) - the two live side by side by
## construction, not by coincidence, so don't move one without the other.

const DIR := "res://feedback"
const PANEL_W := 380
## How many dispatched subprocesses may be live at once - past this, entries
## queue same as before. Kept small: these are permission-bypassed agents
## editing the same live working tree, not isolated sandboxes.
const MAX_CONCURRENT := 3

## Only ever call `claude` with a controlled argument list via bash's safe
## positional-parameter trick ("$1", passed as a real argv element, never
## string-interpolated into the script) - the prompt carries arbitrary user
## text and must never be concatenated into the shell command itself.
var _claude_bin := "claude"
var _repo_root := ""

var _entries: Array = [] # Array[Dictionary], newest first - see enqueue()
var _running_count := 0 # how many entries are actively running right now
var _timer_tick := 0.0 # throttles the running-entry elapsed-timer repaint to ~1/sec

var _panel: Control
var _list_col: VBoxContainer
var _header: Label
var _toggle_btn: Button
var _expanded := false # closed by default - see _build_ui


## A restart requested (by the editor's F5, or auto after a run edits code) but held
## until every agent has returned - restarting mid-run would corrupt another agent's
## in-progress edits. Fired by _maybe_reload the moment we go idle.
var _pending_reload: Callable = Callable()


func _ready() -> void:
	add_to_group("assistant") # the mask editor asks us to defer its reload until idle
	layer = 126 # below the feedback console itself (128), above ordinary scene UI
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
		"pid": - 1, "out_path": "", "err_path": "", "error_text": "", "expanded": true,
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

## Anything still running OR waiting to run - the window during which a restart would
## clobber an agent's edits.
func is_busy() -> bool:
	if _running_count > 0:
		return true
	for e in _entries:
		if e.status == "queued":
			return true
	return false


## Defer a restart until every agent has returned. Fires immediately if already idle,
## otherwise held and fired by _maybe_reload once the last run finishes.
func reload_when_idle(cb: Callable) -> void:
	_pending_reload = cb
	if is_busy():
		var me := get_tree().get_first_node_in_group("mask_editor")
		if me != null and me.has_method("_set_status"):
			me.call("_set_status", "⟳  Reload pending - restarting once assistant runs finish (%d active)" % _running_count)
	_maybe_reload()


func _maybe_reload() -> void:
	if not _pending_reload.is_valid() or is_busy():
		return
	# Don't yank the reload out from under someone mid-critique - hold it until the
	# feedback console closes (submit or cancel), same deference main.gd's
	# _end_session gives it for session teardown.
	var fc := get_tree().get_first_node_in_group("feedback_console")
	if fc != null and is_instance_valid(fc) and fc.is_open():
		if not fc.closed.is_connected(_maybe_reload):
			fc.closed.connect(_maybe_reload, CONNECT_ONE_SHOT)
		return
	var cb := _pending_reload
	_pending_reload = Callable()
	cb.call()


## Newest modification time across ghost's own source (scripts + shaders). A dispatch
## records this at start (see _dispatch); if it's higher at _finish, that run edited
## code and the app should reload to pick it up.
func _max_ghost_source_mtime() -> float:
	var best := 0.0
	for d in ["res://scripts", "res://shaders"]:
		var da := DirAccess.open(d)
		if da == null:
			continue
		for fn in da.get_files():
			if fn.ends_with(".gd") or fn.ends_with(".gdshader"):
				var m := float(FileAccess.get_modified_time(ProjectSettings.globalize_path("%s/%s" % [d, fn])))
				if m > best:
					best = m
	return best


func _pump_queue() -> void:
	while _running_count < MAX_CONCURRENT:
		var started := false
		for e in _entries:
			if e.status == "queued":
				var pending: String = e.get("pending_prompt", "")
				e.pending_prompt = ""
				_dispatch(e, pending if pending != "" else _build_prompt(e))
				started = true
				break
		if not started:
			return


## `prompt` is either the initial dispatch prompt (built from the feedback
## record) or a typed follow-up - either way it lands on the SAME claude
## session once `entry.session_id` is set, via --resume.
func _dispatch(entry: Dictionary, prompt: String) -> void:
	entry.code_mtime = _max_ghost_source_mtime() # baseline: detect if THIS run edits code
	_running_count += 1
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
		_running_count -= 1
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
##
## axis/ghost/CLAUDE.md is named explicitly (not left to auto-discovery)
## because a real dispatched run (feedback 0055) burned a huge chunk of a
## 10-minute session re-deriving ghost's file layout, the effect-registry
## pattern, and the "shader edits need a real GPU compile, not just
## --editor --quit" rule from scratch via grep, all of which that file now
## states up front - forcing an explicit read is one line and removes any
## doubt about whether a -p session picks up a subdirectory CLAUDE.md on
## its own before it starts reading files.
func _build_prompt(entry: Dictionary) -> String:
	var rel: String = String(entry.stem).trim_prefix("res://") # "feedback/0051"
	return ("Read axis/ghost/CLAUDE.md first - it's the map of where things live in " +
		"this project plus hard-won gotchas (shader validation, etc.) that'll save you " +
		"from re-deriving them via grep.\n\n" +
		"New feedback was left in ghost's feedback console: axis/ghost/%s.json " +
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
		return # nothing complete yet - the line currently being written doesn't count
	entry.read_offset = offset + last_nl + 1
	for line in chunk.substr(0, last_nl).split("\n"):
		line = line.strip_edges()
		if line == "":
			continue
		var evt = _json_object(line)
		if evt is Dictionary:
			# Persist the session id the INSTANT it first appears (the "system"/init
			# event carries it), not only at _finish - so a run interrupted by a crash
			# still has a saved id and can be resumed later (see _load_existing).
			if String(entry.session_id) == "" and evt.has("session_id"):
				entry.session_id = str(evt.get("session_id", ""))
				if String(entry.session_id) != "":
					_save_entry(entry)
			var desc := _describe_event(evt)
			if desc != "" and desc != String(entry.get("progress", "")):
				# Marks the moment the underlying agent actually did something new -
				# _build_row uses this to flicker the collapsed title briefly, a
				# small "sign of life" visible even in a long collapsed list.
				entry._flicker_pulse_at = Time.get_ticks_msec()
			if desc != "":
				entry.progress = desc


## Dig a session id out of a captured output/error blob when the field itself was
## never saved (an interrupted run): every stream-json event carries
## "session_id":"<uuid>", so the first one in the text is the run's id.
func _recover_session_id(text: String) -> String:
	var key := "\"session_id\":\""
	var i := text.find(key)
	if i < 0:
		return ""
	i += key.length()
	var j := text.find("\"", i)
	return text.substr(i, j - i) if j > i else ""


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


## Parse one stream-json line into its object, or null - QUIETLY. claude's stdout is
## NDJSON (one JSON object per line), but a stray non-JSON line (a subprocess's own
## stdout that slipped through, a partial write) must be skipped, not logged. JSON.parse_string
## prints a console ERROR on every failure; a JSON instance's parse() just returns a code, so
## no spam. Anything not starting with '{' is skipped without even trying.
func _json_object(line: String) -> Variant:
	if not line.begins_with("{"):
		return null
	var json := JSON.new()
	if json.parse(line) != OK:
		return null
	return json.data


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
			var evt = _json_object(line)
			if evt is Dictionary and String(evt.get("type", "")) == "result":
				parsed = evt
	if parsed is Dictionary and parsed.get("is_error", true) == false and parsed.has("result"):
		entry.status = "done"
		entry.response = str(parsed.get("result", ""))
		entry.session_id = str(parsed.get("session_id", entry.session_id))
		entry.cost_usd = float(parsed.get("total_cost_usd", entry.cost_usd))
		entry.error_text = ""
	else:
		var err_text := _read_file(entry.err_path)
		# The run produced no clean result, but it may still have a session id (it got
		# far enough to emit the init event). If so it's RESUMABLE, not a dead end.
		if String(entry.session_id) == "":
			entry.session_id = _recover_session_id(out_text + "\n" + err_text)
		if String(entry.session_id) != "":
			entry.status = "interrupted"
			entry.error_text = "did not finish - Resume to continue this session"
		else:
			entry.status = "error"
			entry.error_text = err_text if err_text != "" else \
				(out_text if out_text != "" else "(no output - the process may have failed to start)")
	_delete_file(entry.out_path)
	_delete_file(entry.err_path)
	entry.pid = -1
	entry.progress = ""
	_save_entry(entry)
	_refresh_list()
	_running_count -= 1
	_pump_queue()
	# If this run actually edited ghost's own code, arrange a restart to pick it up -
	# held by reload_when_idle until any OTHER still-running agents also return, so no
	# restart ever lands mid-edit.
	if entry.status == "done" and _max_ghost_source_mtime() > float(entry.get("code_mtime", 0.0)):
		var me := get_tree().get_first_node_in_group("mask_editor")
		if me != null and me.has_method("_do_restart"):
			reload_when_idle(Callable(me, "_do_restart"))
	_maybe_reload()


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
	var has_record := {} # feedback index -> true, for every entry ALREADY reconciled below
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
		var sid := str(data.get("session_id", ""))
		if sid == "":
			sid = _recover_session_id(error_text) # older logs stashed the init event in error_text
		# A run that was mid-flight when the editor closed (or errored without a clean
		# result) is resumable IF we know its session id - continue it via --resume
		# rather than starting over. Only a run we can't identify is a dead "error".
		if status == "running" or status == "queued" or (status == "error" and sid != ""):
			if sid != "":
				status = "interrupted"
				error_text = "did not finish before the editor closed - Resume to continue this session"
			else:
				status = "error"
				error_text = "interrupted - the editor closed before this run could be identified"
		loaded.append({
			"index": idx, "query": str(data.get("query", "")), "stem": "%s/%04d" % [DIR, idx],
			"status": status, "session_id": sid,
			"response": str(data.get("response", "")), "cost_usd": float(data.get("cost_usd", 0.0)),
			"pid": - 1, "out_path": "", "err_path": "", "error_text": error_text, "expanded": false,
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
			"pid": - 1, "out_path": "", "err_path": "", "error_text": "", "expanded": false,
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
		_running_count -= 1
		_pump_queue()
	_refresh_list()


# --- UI ------------------------------------------------------------------------

const _TOGGLE_SIZE := 40.0
const _TOGGLE_GAP := 4.0 # between the toggle button and the panel above it
const _PANEL_TOP_MARGIN := 28.0 # never crowd closer to the viewport top than this, even when full of entries
## The toggle's own bottom edge - the same -28 margin mask_editor.gd's/exporter.gd's
## "Export video" button sits on, so the two align into one visual row. The export
## buttons' own offset_right is pulled in by _TOGGLE_SIZE + _TOGGLE_GAP to leave the
## toggle room at the true corner, right of them - see this file's class doc.
const _TOGGLE_ROW_BOTTOM := -28.0

func _build_ui() -> void:
	# Closed by default - a chat-bubble toggle sitting in the viewport's very
	# bottom-right corner, right of the export button in the same row (see
	# _TOGGLE_ROW_BOTTOM's doc), is the whole UI until clicked; the panel
	# itself only exists on screen while expanded.
	_toggle_btn = Button.new()
	_toggle_btn.text = "💬"
	_toggle_btn.tooltip_text = "Assistant"
	_toggle_btn.focus_mode = Control.FOCUS_NONE
	_toggle_btn.custom_minimum_size = Vector2(_TOGGLE_SIZE, _TOGGLE_SIZE)
	_toggle_btn.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	_toggle_btn.offset_right = -28
	_toggle_btn.offset_left = -28 - _TOGGLE_SIZE
	_toggle_btn.offset_bottom = _TOGGLE_ROW_BOTTOM
	_toggle_btn.offset_top = _TOGGLE_ROW_BOTTOM - _TOGGLE_SIZE
	_toggle_btn.pressed.connect(func():
		_expanded = not _expanded
		_refresh_list())
	add_child(_toggle_btn)

	_panel = PanelContainer.new()
	# BOTTOM_RIGHT (not RIGHT_WIDE) so the panel sits as a small window above
	# the toggle by default and only grows upward as its content needs more
	# room - see _resize_panel(), called after every _refresh_list(). Both
	# offset_top and offset_bottom end up measured from the viewport's
	# bottom edge this way, which is what lets offset_top move up (more
	# negative) as content grows instead of being pinned near the top.
	_panel.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	_panel.offset_right = -28
	_panel.offset_left = -28 - PANEL_W
	_panel.offset_bottom = _TOGGLE_ROW_BOTTOM - _TOGGLE_SIZE - _TOGGLE_GAP # above the toggle, not overlapping it
	_panel.offset_top = _panel.offset_bottom # sized properly on the first _resize_panel() call
	_panel.visible = false
	add_child(_panel)

	var outer := VBoxContainer.new()
	outer.add_theme_constant_override("separation", 6)
	_panel.add_child(outer)

	_header = Label.new()
	_header.text = "Assistant"
	# Wrap, or a long header ("Feedback (assistant off - browse/delete only)")
	# reports its full text width as a minimum and drags the right-anchored panel
	# off-screen to the right (the panel grows to its content's minimum size).
	_header.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_header.add_theme_color_override("font_color", Color(0.7, 0.85, 1.0, 0.9))
	outer.add_child(_header)
	# The panel starts (and often sits) invisible/collapsed, and Godot doesn't run
	# layout on invisible controls - so the deferred _resize_panel() call below can
	# land while an autowrapped Label is still mid-reflow (a transient near-zero
	# wrap width briefly reports a wildly inflated minimum height), pinning the
	# panel's height to that bogus value with nothing left to ever correct it.
	# Recomputing whenever the real minimum size actually changes (e.g. once the
	# panel becomes visible and reflows for real) keeps it converging on the true
	# content height instead of getting stuck on a stale bad reading.
	_header.minimum_size_changed.connect(_resize_panel)

	var scroll := ScrollContainer.new()
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	outer.add_child(scroll)

	_list_col = VBoxContainer.new()
	_list_col.add_theme_constant_override("separation", 4)
	_list_col.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(_list_col)
	_list_col.minimum_size_changed.connect(_resize_panel)


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
	# Deferred: freshly added rows haven't had a layout pass yet, so their
	# autowrapped labels don't report an accurate minimum height until next
	# idle frame.
	call_deferred("_resize_panel")


## Keeps the panel sized to its actual content (small by default, above the
## toggle) instead of permanently stretched to the viewport height - it only
## grows upward, capped so it never crowds past _PANEL_TOP_MARGIN from the
## top of the screen, beyond which the panel's own ScrollContainer takes
## over instead of the panel itself continuing to grow.
func _resize_panel() -> void:
	if not is_instance_valid(_panel):
		return
	var offset_bottom := _TOGGLE_ROW_BOTTOM - _TOGGLE_SIZE - _TOGGLE_GAP
	_panel.offset_bottom = offset_bottom
	var viewport_h := get_viewport().get_visible_rect().size.y
	var max_h: float = max(viewport_h + offset_bottom - _PANEL_TOP_MARGIN, 0.0)
	var chrome_h := 0.0
	var style := _panel.get_theme_stylebox("panel")
	if style:
		chrome_h = style.get_minimum_size().y
	var content_h: float = _header.get_combined_minimum_size().y + 6.0 + _list_col.get_combined_minimum_size().y + chrome_h
	_panel.offset_top = offset_bottom - clamp(content_h, 0.0, max_h)


func _status_glyph(status: String) -> String:
	match status:
		"orphaned": return "○"
		"queued": return "⏳"
		"running": return "⚙"
		"done": return "✓"
		"interrupted": return "↻"
		_: return "✕"


## Neon accent per status - distinct from both the desaturated near-white body
## text (see _build_body/_dim_label) and each other, so a title reads at a
## glance even collapsed. Saturated on purpose ("neon sign"), not just tinted.
func _title_color(status: String) -> Color:
	match status:
		"running": return Color(0.3, 1.0, 0.85, 0.95) # cyan - alive right now
		"done": return Color(0.55, 1.0, 0.45, 0.95) # green - finished clean
		"interrupted": return Color(1.0, 0.75, 0.25, 0.95) # amber - paused, resumable
		"queued": return Color(0.7, 0.55, 1.0, 0.9) # violet - waiting
		"orphaned": return Color(0.55, 0.6, 0.7, 0.85) # dim - nothing happening yet
		_: return Color(1.0, 0.35, 0.55, 0.95) # pink - error


## How long after a genuine progress change (see _poll_progress's
## _flicker_pulse_at) a title still shows the flicker - long enough to land on
## the very next ~1/sec repaint (_process), short enough to read as a flicker,
## not a steady state.
const _FLICKER_WINDOW_MS := 1800.0
const _FLICKER_GLYPHS := ["▚", "▞", "░", "▓", "◇", "∴", "≈"]

## Glitches ONE character of a title - half the time it just goes dark (a
## neon sign losing a letter), half the time it flickers to a brighter glyph.
## Never touches the status glyph/index prefix (see _build_row) or whitespace/
## ellipsis, so the title stays legible, just alive.
func _flicker_text(text: String) -> String:
	var candidates: Array = []
	for i in range(text.length()):
		var ch := text[i]
		if ch != " " and ch != "…":
			candidates.append(i)
	if candidates.is_empty():
		return text
	var idx: int = candidates[randi() % candidates.size()]
	var replacement: String = " " if randi() % 2 == 0 else _FLICKER_GLYPHS[randi() % _FLICKER_GLYPHS.size()]
	return text.substr(0, idx) + replacement + text.substr(idx + 1)


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
	# Only flicker while something is actually happening (a real, new progress
	# event landed recently) - see _poll_progress's _flicker_pulse_at. This is
	# what makes it read as "life", not a distracting constant twitch.
	var pulsing: bool = String(entry.status) == "running" and \
		(Time.get_ticks_msec() - float(entry.get("_flicker_pulse_at", -_FLICKER_WINDOW_MS * 10.0))) < _FLICKER_WINDOW_MS
	if pulsing:
		preview = _flicker_text(preview)
	toggle.text = "%s #%04d  %s" % [_status_glyph(entry.status), int(entry.index), preview]
	toggle.add_theme_color_override("font_color", _title_color(entry.status))
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
	del.custom_minimum_size = Vector2(28, 0) # guaranteed footprint regardless of the toggle's width
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
		"interrupted":
			body.add_child(_dim_label(String(entry.error_text)))
			body.add_child(_build_resume_row(entry))
		"running":
			var elapsed_s: int = (Time.get_ticks_msec() - int(entry.get("started_at", Time.get_ticks_msec()))) / 1000
			var time_str := "%ds" % elapsed_s if elapsed_s < 60 else "%d:%02d" % [elapsed_s / 60, elapsed_s % 60]
			body.add_child(_dim_label("running (%s)" % time_str))
			var prog := String(entry.get("progress", ""))
			if prog != "":
				var prog_lbl := Label.new()
				prog_lbl.text = "▸ " + prog
				prog_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
				prog_lbl.add_theme_color_override("font_color", Color(0.6, 0.85, 0.7, 0.9))
				body.add_child(prog_lbl)
			body.add_child(_build_interrupt_row(entry))
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
	# These carry the long status strings ("running (12s) - output is captured
	# here, never printed to a terminal", "no assistant selected - pick one on the
	# home screen to send this"). Without wrapping, each reports its full one-line
	# width as a minimum and pushes the right-anchored panel off the screen edge -
	# WORD_SMART also breaks over-long tokens (paths) so nothing overflows.
	l.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
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
	# Goes through the SAME queue as a fresh submission (see
	# enqueue/_pump_queue) - a follow-up on an idle, already-done entry only
	# waits if MAX_CONCURRENT other entries are already running.
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


## Halts a live run without discarding it (unlike _delete_entry, which also wipes
## the feedback record). Kills the subprocess and lets the normal _process()/
## _finish() polling notice it exited next frame - _finish already turns a run
## with a known session_id into "interrupted" (the same state a crash or app-close
## leaves behind), which is exactly the state _build_resume_row below knows how to
## continue via --resume. session_id is usually already captured by then (
## _poll_progress persists it the instant the run's first stream-json event
## arrives), so an interrupt reliably lands on "interrupted", not a dead "error".
func _interrupt_entry(entry: Dictionary) -> void:
	if entry.status != "running" or int(entry.pid) < 0:
		return
	OS.kill(int(entry.pid))


func _build_interrupt_row(entry: Dictionary) -> Control:
	var row := HBoxContainer.new()
	var stop := Button.new()
	stop.text = "⏸  Interrupt"
	stop.focus_mode = Control.FOCUS_NONE
	stop.tooltip_text = "Halt this run now so you can add context and resume it"
	stop.pressed.connect(func(): _interrupt_entry(entry))
	row.add_child(stop)
	return row


## Resume an interrupted run: re-queue it against its saved session_id so --resume
## picks up the SAME claude session where it left off (see _dispatch). An optional
## typed note - e.g. context that changed while the run was halted - is folded into
## the resume prompt; left blank, it falls back to the plain "pick up where you left
## off" prompt. Same queue as any other dispatch.
func _build_resume_row(entry: Dictionary) -> Control:
	var col := VBoxContainer.new()
	col.add_theme_constant_override("separation", 4)

	var row := HBoxContainer.new()
	var edit := LineEdit.new()
	edit.placeholder_text = "extra context for the resume (optional)..."
	edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	# Same reasoning as _build_followup_row's draft stash - _refresh_list() rebuilds
	# every row whenever any entry's status changes, so an in-progress note here
	# would otherwise vanish the moment an unrelated entry finishes.
	edit.text = String(entry.get("draft_resume", ""))
	edit.text_changed.connect(func(t): entry.draft_resume = t)
	row.add_child(edit)

	var resume := Button.new()
	resume.text = "↻  Resume"
	resume.focus_mode = Control.FOCUS_NONE
	resume.tooltip_text = "Continue the same Claude session (--resume %s) where it left off" % String(entry.session_id)
	var do_resume := func():
		var extra := edit.text.strip_edges()
		var prompt := "The previous run was interrupted before finishing."
		if extra != "":
			prompt += " The user has additional context: \"%s\"" % extra
		prompt += " Please pick up exactly where you left off and complete the work."
		entry.status = "queued"
		entry.response = ""
		entry.error_text = ""
		entry.pending_prompt = prompt
		entry.draft_resume = ""
		_entries.erase(entry)
		_entries.push_front(entry)
		_save_entry(entry)
		_refresh_list()
		_pump_queue()
	resume.pressed.connect(do_resume)
	edit.text_submitted.connect(func(_t): do_resume.call())
	row.add_child(resume)

	col.add_child(row)
	return col
