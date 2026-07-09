extends CanvasLayer
class_name Assistant

## Assistant - tight Claude Code integration for the feedback loop.
##
## When ghost is launched with --assistant, every feedback console submission
## (see feedback.gd) is immediately handed to a fresh `claude -p
## --dangerously-skip-permissions` subprocess - the same one-shot workflow
## already used interactively all session, just wired to fire the moment
## feedback lands instead of waiting for a human to notice and paste it in.
## Sonnet 5, default (auto) effort - the CLI's --effort flag has no "auto"
## value, so it's simply omitted; the session picks its own.
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
## UI lives bottom-right, stacked above the export status row (see
## mask_editor.gd / exporter.gd's shared notification corner) - a scrollable
## list of entries, each expandable to show the full exchange, with a delete
## button per entry.

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

var _panel: Control
var _list_col: VBoxContainer
var _header: Label


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
func enqueue(index: int, query: String, stem: String) -> void:
	var entry := {
		"index": index, "query": query, "stem": stem,
		"status": "queued", "session_id": "", "response": "", "cost_usd": 0.0,
		"pid": -1, "out_path": "", "err_path": "", "error_text": "", "expanded": true,
		"pending_prompt": "",
	}
	_entries.push_front(entry)
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
	var base := ProjectSettings.globalize_path(DIR)
	entry.out_path = "%s/%04d.out.json" % [base, int(entry.index)]
	entry.err_path = "%s/%04d.err.log" % [base, int(entry.index)]
	var resume_part := ""
	if entry.session_id != "":
		resume_part = " --resume %s" % entry.session_id
	# cd/exec/redirects are all Godot-controlled strings (paths we built, never
	# user text) - safe to interpolate directly. The prompt is NOT interpolated;
	# it arrives as bash's $1, a real argv element.
	var script := "cd \"%s\" && exec \"%s\" -p --model sonnet --dangerously-skip-permissions --output-format json%s \"$1\" > \"%s\" 2> \"%s\"" % [
		_repo_root, _claude_bin, resume_part, entry.out_path, entry.err_path]
	entry.pid = OS.create_process("/bin/bash", ["-c", script, "bash", prompt])
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
	for e in _entries:
		if e.status == "running" and int(e.pid) >= 0 and not OS.is_process_running(int(e.pid)):
			_finish(e)


func _finish(entry: Dictionary) -> void:
	var out_text := _read_file(entry.out_path)
	var parsed = JSON.parse_string(out_text) if out_text != "" else null
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
	loaded.sort_custom(func(a, b): return int(a.index) > int(b.index))
	_entries = loaded


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

func _build_ui() -> void:
	_panel = PanelContainer.new()
	_panel.set_anchors_preset(Control.PRESET_BOTTOM_RIGHT)
	_panel.offset_right = -28
	_panel.offset_left = -28 - PANEL_W
	_panel.offset_bottom = -84   # sits just above the export status row
	_panel.offset_top = -84 - LIST_H - 28
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
	_header.text = "Assistant" if running == 0 and queued == 0 else \
		"Assistant - %d running, %d queued" % [running, queued]
	for c in _list_col.get_children():
		c.queue_free()
	for e in _entries:
		_list_col.add_child(_build_row(e))


func _status_glyph(status: String) -> String:
	match status:
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
	toggle.pressed.connect(func():
		entry.expanded = not entry.expanded
		_refresh_list())
	head.add_child(toggle)

	var del := Button.new()
	del.text = "✕"
	del.focus_mode = Control.FOCUS_NONE
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
		"queued":
			body.add_child(_dim_label("queued - waiting for the current run to finish"))
		"running":
			body.add_child(_dim_label("running..."))
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
