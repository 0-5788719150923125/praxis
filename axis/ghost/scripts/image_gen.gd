extends RefCounted
class_name ImageGen

## ImageGen - who paints the pictures. The backend axis of [Illustrations].
##
## A backend turns ONE finished prompt plus a list of reference images into ONE PNG on
## disk. Everything above that - the style, the prompt, the cache, the versions - belongs to
## [Illustrations], so a second painter is a registry entry and a class here, never a
## branch in the library or the panel.
##
## THE CONTRACT is three calls, all on the main thread (spawns go through [Subprocess],
## whose death pact is per-thread):
##
##   start(job)    - launch the work; return a pid (<= 0 is a failure, with job.error set)
##   resolve(job)  - after the pid exits: the absolute path of the image it made, or ""
##   failure(job)  - after an empty resolve: one line saying why, for the panel
##
## `job` is a Dictionary the library owns: {prompt, refs, dir, target, started}. A backend
## may add its own keys to it (the event log, a thread id) and read them back later.

## Keys are what `[illustrations] backend` in `user://ghost.cfg` stores.
const REGISTRY := {
	"codex": Codex,
}

const LABELS := {
	"codex": "OpenAI (Codex CLI)",
}

## One string literal per entry, for the picker's tooltip - same rule as Medium.BLURBS.
const BLURBS := {
	"codex": "OpenAI's image model, driven through the Codex CLI's built-in image tool. Uses the Codex login, not an API key. About a minute per picture.",
}


## Build the backend for [param key], falling back to the first entry for an unknown one: a
## stale config must never leave the panel without a painter.
static func make(key: String) -> Backend:
	var cls: Variant = REGISTRY.get(key, REGISTRY[REGISTRY.keys()[0]])
	return cls.new()


## The base every backend extends. It does nothing; a backend that forgets a method fails
## visibly (no pid, no image) rather than silently.
class Backend:
	extends RefCounted

	func available() -> bool:
		return false

	func start(_job: Dictionary) -> int:
		return -1

	func resolve(_job: Dictionary) -> String:
		return ""

	func failure(job: Dictionary) -> String:
		return String(job.get("error", "the painter produced nothing"))


## OPENAI THROUGH THE CODEX CLI.
##
## `codex exec` is an agent, not an image API: it is asked, in words, to call its built-in
## image tool and save the result under an exact name in its working directory. Measured on
## codex-cli 0.150.0 - the tool writes the original to
## `$CODEX_HOME/generated_images/<thread_id>/*.png` and the agent then copies it where asked.
## The copy is the step an agent can skip, so [method resolve] falls back to that directory,
## keyed by the thread id from the first JSONL event and filtered to files made after the job
## began - never "the newest png anywhere", which would hand one chapter another's picture.
##
## Reasoning effort is pinned LOW for these runs: the author's default is the maximum, which
## is right for code and pure latency for "call one tool and copy one file".
class Codex:
	extends Backend

	func binary() -> String:
		var p := Deps.resolve("codex")
		return p if not p.is_empty() else "codex"

	func available() -> bool:
		return Deps.has("codex")

	func start(job: Dictionary) -> int:
		var dir := String(job["dir"])
		DirAccess.make_dir_recursive_absolute(dir)
		job["events"] = dir.path_join("events.jsonl")
		job["log"] = dir.path_join("stderr.log")
		# The prompt carries the author's own text, so it goes on stdin (`-`), never in argv:
		# see AssistantBackends for why.
		var prompt_path := dir.path_join("prompt.txt")
		var pf := FileAccess.open(prompt_path, FileAccess.WRITE)
		if pf == null:
			job["error"] = "could not write the prompt file in " + dir
			return -1
		pf.store_string(String(job["prompt"]))
		pf.close()
		var args := PackedStringArray([
			"exec", "--skip-git-repo-check", "--json",
			"-s", "workspace-write", "-C", dir,
			"-c", "model_reasoning_effort=low"])
		# `--image=` per file, then `--`: the flag takes MANY values, and a bare `-i a.png`
		# followed by the prompt swallowed the prompt as a second image ("No prompt provided").
		for r in job.get("refs", []):
			args.append("--image=" + String(r))
		args.append("--")
		args.append("-")
		var pid := Subprocess.start_redirected(binary(), args, {"stdin": prompt_path,
			"out": String(job["events"]), "err": String(job["log"])}, "codex image")
		if pid <= 0:
			job["error"] = "could not start codex (is the Codex CLI installed and logged in?)"
		return pid

	func resolve(job: Dictionary) -> String:
		var target := String(job.get("target", ""))
		if not target.is_empty() and FileAccess.file_exists(target) \
				and FileAccess.get_file_as_bytes(target).size() > 0:
			return target
		var tid := thread_id(String(job.get("events", "")))
		if tid.is_empty():
			return ""
		return newest_png(generated_dir().path_join(tid), int(job.get("started", 0)))

	func failure(job: Dictionary) -> String:
		var last := last_message(String(job.get("events", "")))
		if not last.is_empty():
			return last
		var err := FileAccess.get_file_as_string(String(job.get("log", ""))).strip_edges()
		if not err.is_empty():
			var lines := err.split("\n")
			return String(lines[lines.size() - 1]).substr(0, 240)
		return String(job.get("error", "codex finished without an image"))

	## `$CODEX_HOME/generated_images`, defaulting to ~/.codex like the CLI does.
	static func generated_dir() -> String:
		var home := OS.get_environment("CODEX_HOME")
		if home.is_empty():
			home = Deps.home().path_join(".codex")
		return home.path_join("generated_images")

	## The thread id from a `codex exec --json` event log, or "".
	static func thread_id(events_path: String) -> String:
		for ev in _events(events_path):
			if String(ev.get("type", "")) == "thread.started":
				return String(ev.get("thread_id", ""))
		return ""

	## The agent's last words, which is where it says why it made nothing.
	static func last_message(events_path: String) -> String:
		var msg := ""
		for ev in _events(events_path):
			var item: Variant = ev.get("item", null)
			if item is Dictionary and String((item as Dictionary).get("type", "")) == "agent_message":
				msg = String((item as Dictionary).get("text", ""))
			elif String(ev.get("type", "")) in ["error", "turn.failed"]:
				msg = JSON.stringify(ev.get("error", ev.get("message", ev)))
		return msg.strip_edges().substr(0, 240)

	## The most recently written png in [param dir] no older than [param since] (unix seconds).
	static func newest_png(dir: String, since: int) -> String:
		var da := DirAccess.open(dir)
		if da == null:
			return ""
		var best := ""
		var best_t := -1
		for f in da.get_files():
			if not f.to_lower().ends_with(".png"):
				continue
			var p := dir.path_join(f)
			var t := int(FileAccess.get_modified_time(p))
			if t >= since and t > best_t:
				best = p
				best_t = t
		return best

	static func _events(path: String) -> Array:
		var out: Array = []
		if path.is_empty() or not FileAccess.file_exists(path):
			return out
		for line in FileAccess.get_file_as_string(path).split("\n"):
			var s := String(line).strip_edges()
			if not s.begins_with("{"):
				continue
			var j := JSON.new()
			if j.parse(s) == OK and j.data is Dictionary:
				out.append(j.data)
		return out
