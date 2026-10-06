extends RefCounted
class_name AgentJobs

## AgentJobs - every piece of writing and painting ghost asks an AI for, in one queue.
##
## A job is ONE run of a [TextGen] writer or an [ImageGen] painter. Whoever wants the work
## submits it and keeps the id; this file starts it when there is room, notices when it ends,
## and holds the outcome until it is asked for. It knows nothing about what the words or the
## pictures are FOR - that is the caller's - so anything in ghost that wants an agent's output
## asks here rather than growing its own copy of a subprocess pump.
##
## LANES. A job may name a lane, and jobs in one lane run ONE AT A TIME, in the order they were
## submitted - which is how a chain of pictures that reference each other (each one is sent the
## ones before it) and a writer that continues its own earlier replies are expressed, without
## the queue knowing either thing. Jobs in different lanes, or in none, run side by side up to
## [constant LIMITS] per kind.
##
## THE JOB'S DIRECTORY IS THE CALLER'S, not this file's: it holds the prompt exactly as sent, the
## reply and the log, and it is KEPT. So the record of what a writer was shown lives beside what
## it wrote, wherever the caller keeps its own data - the evidence a reading never cheated is on
## disk, not in a promise.
##
## A BACKEND MAY WORK IN STEPS: when a step's process ends, its `advance(job)` may start the next
## one and return that pid, and the job runs on - a writer that has to look its route up before it
## can write does both without ever blocking a frame. A one-step backend's `advance` returns 0 and
## the job lands, as it always did. Its timeout counts from the first step.
##
## Polled from `main._process`, the [Films] / [Illustrations] rule: a job is a subprocess, and
## something with a frame has to see it end. NOTHING STARTS ON ITS OWN - a job runs because a
## person pressed something - and a read-only process (a render, a probe) refuses every
## submission, since a render booted against the author's settings must not spend their quota.

## How many runs of each kind at once. Each mostly waits on a network round trip, so more than
## one is cheap; more than a couple is a burst against the author's quota.
const LIMITS := {"text": 2, "image": 2}
## A run that takes longer than this is abandoned rather than watched forever.
const TIMEOUT_S := {"text": 420, "image": 600}

## id -> the job as submitted, while it waits: [{id, kind, ...}], in submission order.
static var _queue: Array = []
## id -> the running job (with its backend and pid).
static var _running := {}
## id -> {ok, text, path, error, kind, label}, once ended, until forgotten.
static var _ended := {}
static var _seq := 0
## The test/tool seam: -1 = ask Settings, 0 = writable, 1 = read-only.
static var _forced_read_only := -1


## For a gate or a command-line tool booted with `--script`, where no [Settings] autoload exists
## (which [method read_only] would otherwise take as read-only).
static func allow_for_tool(writable := true) -> void:
	_forced_read_only = 0 if writable else 1


static func read_only() -> bool:
	if _forced_read_only >= 0:
		return _forced_read_only == 1
	var tree := Engine.get_main_loop() as SceneTree
	var st: Node = tree.root.get_node_or_null("Settings") if tree != null else null
	return st == null or bool(st.is_read_only())


## Where a painter is told to save its picture: inside the job's own directory, never at the
## caller's target - a file appearing at the target is a finished picture, and a painter writing
## there directly made a half-written one look finished (the next card was sent it as a
## reference).
static func paint_target(dir: String) -> String:
	return dir.path_join("image.png")


## A JOB DIRECTORY IS REUSED - a redo runs in the folder its first run left - so whatever that run
## wrote must be gone before this one starts, or a run that ends without writing (a painter that
## skips the copy, a quota refusal) hands back the OLD output as its own and the step shows as
## made. The prompt, the system prompt and the log are rewritten by the run itself. "" when
## clear, else why not.
static func clear_outputs(job: Dictionary) -> String:
	var dir := String(job.get("dir", ""))
	var outs: Array = []
	if String(job.get("kind", "")) == "image":
		outs.append(paint_target(dir))
	else:
		var p := TextGen.Backend.paths(job)
		outs.append_array([String(p["reply"]), String(p["last"])])
	for f in outs:
		if FileAccess.file_exists(String(f)):
			DirAccess.remove_absolute(String(f))
			if FileAccess.file_exists(String(f)):
				return "could not clear the last run's output (%s)" % String(f).get_file()
	return ""


## Queue a job; its id, or "" when it was refused (read-only, unknown kind, no directory).
##
## [param spec] keys:
##   kind     - "text" or "image"
##   backend  - a [TextGen] / [ImageGen] registry key
##   dir      - the job's own directory, absolute; created if missing
##   prompt   - the prompt, verbatim
##   system   - text only: the system prompt
##   tier     - text only: one of [constant TextGen.TIERS]
##   refs     - image only: reference images, absolute paths, in the order the prompt names them
##   target   - image only: where the finished PNG goes. The painter is NOT told this path - it
##              paints into its job directory ([method paint_target], which the prompt names) and
##              the picture is moved here whole, so nothing ever sees a half-written file there
##   lane     - optional: jobs sharing a lane run one at a time, in order
##   label    - optional: a few words for logs and status lines
static func submit(spec: Dictionary) -> String:
	if read_only():
		return ""
	var kind := String(spec.get("kind", ""))
	if not LIMITS.has(kind) or String(spec.get("dir", "")).is_empty():
		push_warning("ghost: AgentJobs refused a job with kind '%s' and no directory" % kind)
		return ""
	_seq += 1
	var id := "%s-%d-%d" % [kind, Time.get_ticks_msec(), _seq]
	var job := spec.duplicate(true)
	job["id"] = id
	if kind == "image":
		job["dest"] = String(spec.get("target", ""))
		job["target"] = paint_target(String(spec["dir"]))
	_queue.append(job)
	pump()
	return id


## "queued", "running", "done", "failed", or "" for an id this queue does not hold.
static func state(id: String) -> String:
	if _running.has(id):
		return "running"
	if _ended.has(id):
		return "done" if bool((_ended[id] as Dictionary)["ok"]) else "failed"
	for q in _queue:
		if String((q as Dictionary)["id"]) == id:
			return "queued"
	return ""


## The outcome of an ended job: {ok, text (text jobs), path (image jobs), error}. Empty while it
## has not ended.
static func result(id: String) -> Dictionary:
	return (_ended.get(id, {}) as Dictionary).duplicate()


## Drop an ended job's outcome once the caller has taken it.
static func forget(id: String) -> void:
	_ended.erase(id)


## Withdraw a queued job or stop a running one. Its outcome is a failure reading "cancelled".
static func cancel(id: String) -> void:
	for i in _queue.size():
		if String((_queue[i] as Dictionary)["id"]) == id:
			_queue.remove_at(i)
			_ended[id] = {"ok": false, "error": "cancelled", "text": "", "path": ""}
			return
	if _running.has(id):
		Subprocess.stop(int((_running[id] as Dictionary)["pid"]))
		_running.erase(id)
		_ended[id] = {"ok": false, "error": "cancelled", "text": "", "path": ""}


## Jobs waiting or running.
static func busy() -> int:
	return _queue.size() + _running.size()


## Notice what ended, start what fits.
static func pump() -> void:
	for id in _running.keys():
		var job: Dictionary = _running[id]
		var pid := int(job["pid"])
		if Subprocess.alive(pid):
			if int(Time.get_unix_time_from_system()) - int(job["started"]) \
					> int(TIMEOUT_S[String(job["kind"])]):
				Subprocess.stop(pid)
				_running.erase(id)
				_ended[id] = {"ok": false, "text": "", "path": "", "label": job.get("label", ""),
					"error": "timed out after %d s" % int(TIMEOUT_S[String(job["kind"])])}
			continue
		Subprocess.forget(pid)
		var next := int(job["gen"].advance(job))
		if next > 0:
			job["pid"] = next
			continue
		_running.erase(id)
		_ended[id] = _land(job)
	if read_only():
		return
	while true:
		var i := _next_startable()
		if i < 0:
			break
		_start(_queue.pop_at(i))


## The first queued job that may start now: its kind has a free slot, and nothing ahead of it
## in its lane is waiting or running.
static func _next_startable() -> int:
	var counts := {}
	var busy_lanes := {}
	for j in _running.values():
		var k := String((j as Dictionary)["kind"])
		counts[k] = int(counts.get(k, 0)) + 1
		var lane := String((j as Dictionary).get("lane", ""))
		if not lane.is_empty():
			busy_lanes[lane] = true
	for i in _queue.size():
		var q: Dictionary = _queue[i]
		var lane := String(q.get("lane", ""))
		var blocked := not lane.is_empty() and busy_lanes.has(lane)
		if not lane.is_empty():
			busy_lanes[lane] = true          # later jobs in this lane wait behind this one
		if blocked:
			continue
		if int(counts.get(String(q["kind"]), 0)) >= int(LIMITS[String(q["kind"])]):
			continue
		return i
	return -1


static func _start(job: Dictionary) -> void:
	var id := String(job["id"])
	var dir := String(job["dir"])
	DirAccess.make_dir_recursive_absolute(dir)
	job["started"] = int(Time.get_unix_time_from_system()) - 1
	var stale := clear_outputs(job)
	if not stale.is_empty():
		_ended[id] = {"ok": false, "text": "", "path": "", "label": job.get("label", ""), "error": stale}
		return
	var gen: Variant = null
	if String(job["kind"]) == "text":
		gen = TextGen.make(String(job.get("backend", "")))
	else:
		gen = ImageGen.make(String(job.get("backend", "")))
	job["gen"] = gen
	var pid: int = gen.start(job)
	if pid <= 0:
		_ended[id] = {"ok": false, "text": "", "path": "", "label": job.get("label", ""),
			"error": String(gen.failure(job))}
		return
	job["pid"] = pid
	_running[id] = job
	print("ghost: agent job %s started (%s)" % [String(job.get("label", id)), String(job["kind"])])


## A job ended: take its output, or record why there is none.
static func _land(job: Dictionary) -> Dictionary:
	var gen: Variant = job["gen"]
	var out := {"ok": false, "text": "", "path": "", "error": "", "label": job.get("label", "")}
	if String(job["kind"]) == "text":
		var text: String = gen.resolve(job)
		if text.is_empty():
			out["error"] = String(gen.failure(job))
		else:
			out["ok"] = true
			out["text"] = text
	else:
		var made: String = gen.resolve(job)
		var img := Image.new()
		if made.is_empty() or img.load(made) != OK or img.is_empty():
			out["error"] = String(gen.failure(job))
		else:
			# RE-ENCODED, never copied: whatever the painter wrote (a JPEG named .png has been
			# seen from image tools), the caller is handed a real PNG at the place it asked for -
			# by rename, so it appears there whole or not at all.
			var target := String(job.get("dest", ""))
			if target.is_empty():
				target = String(job["dir"]).path_join("final.png")
			DirAccess.make_dir_recursive_absolute(target.get_base_dir())
			var tmp := target + ".part.png"
			if img.save_png(tmp) != OK or DirAccess.rename_absolute(tmp, target) != OK:
				out["error"] = "could not write " + target
			else:
				out["ok"] = true
				out["path"] = target
	if bool(out["ok"]):
		print("ghost: agent job %s done" % String(job.get("label", job["id"])))
	else:
		push_warning("ghost: agent job %s failed - %s" % [String(job.get("label", job["id"])),
			String(out["error"])])
	return out
