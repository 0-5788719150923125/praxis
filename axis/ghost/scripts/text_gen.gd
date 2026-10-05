extends RefCounted
class_name TextGen

## TextGen - who writes the words. The text counterpart of [ImageGen].
##
## A backend turns ONE system prompt plus ONE prompt into ONE reply on disk. Everything above
## that - what is asked, what the reply is for, where it is kept - belongs to the caller (see
## [AgentJobs], which queues the work, and [TarotProducer], which asks for it), so a second
## writer is a registry entry and a class here, never a branch at a call site.
##
## THE CONTRACT is [ImageGen]'s, all on the main thread (spawns go through [Subprocess], whose
## death pact is per-thread):
##
##   start(job)    - launch the work; return a pid (<= 0 is a failure, with job.error set)
##   resolve(job)  - after the pid exits: the reply text, or ""
##   failure(job)  - after an empty resolve: one line saying why
##
## `job` is a Dictionary the queue owns: {prompt, system, dir, tier, images}. `dir` is the job's
## own directory - the prompt, the system prompt, the reply and the log are all written there and
## KEPT, so what a writer was shown can always be read back afterwards.
##
## PICTURES go with the words when a caller has them: `images` is `[{path, label, flip}]`, each
## sent ahead of the prompt under its label (turned over when `flip` - a card that came up
## reversed is seen upside down), and listed at the top of `prompt.txt`, so the record says
## which pictures a writer was shown as plainly as which words.
##
## A WRITER, NOT AN AGENT. A run works from its prompt alone: Claude is run with no tools at
## all; Codex, whose `exec` keeps a read-only shell, is TOLD to work only from the message and
## never to open, list or read a file ([constant Codex.ONLY_THIS]) - the user's call (2026-10-04):
## asking is enough, agents follow it. Each run is also STATELESS (no session is kept or
## resumed) - a caller that wants a conversation sends the conversation, so every run's whole
## input is the prompt file beside its reply.
##
## THE PROMPT IS NEVER IN ARGV, for the reason [AssistantBackends] gives: it reaches the CLI on
## stdin from a file, and the system prompt from a file the CLI reads itself.

## A picture is sent no larger than this on its long edge: enough to see what is painted, and a
## fraction of the tokens of the painter's full size.
const PICTURE_EDGE := 768

## Keys are what `ghost: tarot: writer` (and anything else that asks for words) stores.
const REGISTRY := {
	"claude": Claude,
	"codex": Codex,
}

const LABELS := {
	"claude": "Claude (Claude Code CLI)",
	"codex": "OpenAI (Codex CLI)",
}

## One string literal per entry, for a picker's tooltip - same rule as Medium.BLURBS.
const BLURBS := {
	"claude": "Anthropic's models through the Claude Code CLI, run with no tools and no project context. Uses the Claude Code login.",
	"codex": "OpenAI's models through the Codex CLI, run read-only in the job's own folder and told to work only from the message it is given. Uses the Codex login.",
}

## THE TIER is what a caller asks for instead of a model name: `best` for words somebody will
## hear or read, `fast` for structured bookkeeping nobody sees verbatim. Each backend maps it to
## its own models, so a caller never learns one.
const TIERS := ["best", "fast"]


## Build the backend for [param key], falling back to the first entry for an unknown one: a
## stale document must never leave a caller without a writer.
static func make(key: String) -> Backend:
	var cls: Variant = REGISTRY.get(key, REGISTRY[REGISTRY.keys()[0]])
	return cls.new()


static func has(key: String) -> bool:
	return REGISTRY.has(key)


## THE JSON IN A REPLY, leniently. A writer asked for "only a JSON object" still sometimes
## fences it, or says a word first; neither should cost a run. Takes the outermost object (or
## array) in [param text], with code fences ignored. null when there is none that parses.
static func extract_json(text: String) -> Variant:
	var t := text.strip_edges()
	if t.begins_with("```"):
		var nl := t.find("\n")
		t = t.substr(nl + 1) if nl >= 0 else ""
		var close := t.rfind("```")
		if close >= 0:
			t = t.substr(0, close)
	for pair in [["{", "}"], ["[", "]"]]:
		var a := t.find(String(pair[0]))
		var b := t.rfind(String(pair[1]))
		if a < 0 or b <= a:
			continue
		var j := JSON.new()
		if j.parse(t.substr(a, b - a + 1)) == OK:
			return j.data
	return null


## A picture as a writer is sent it: no larger than [constant PICTURE_EDGE], turned over when
## [param flip]; null when it cannot be read.
static func picture(path: String, flip := false) -> Image:
	var img := Image.load_from_file(path) if FileAccess.file_exists(path) else null
	if img == null or img.is_empty():
		return null
	if img.is_compressed():
		img.decompress()
	img.convert(Image.FORMAT_RGB8)
	var edge := maxi(img.get_width(), img.get_height())
	if edge > PICTURE_EDGE:
		var k := float(PICTURE_EDGE) / float(edge)
		img.resize(maxi(1, roundi(img.get_width() * k)), maxi(1, roundi(img.get_height() * k)),
			Image.INTERPOLATE_LANCZOS)
	if flip:
		img.rotate_180()
	return img


## [method picture] as a message content block (a JPEG, base64); empty when it cannot be read.
static func image_block(path: String, flip := false) -> Dictionary:
	var img := picture(path, flip)
	if img == null:
		return {}
	return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
		"data": Marshalls.raw_to_base64(img.save_jpg_to_buffer(0.9))}}


## The record of which pictures went with a prompt: one line each, ahead of the prompt.
static func picture_line(d: Dictionary) -> String:
	return "[picture: %s%s - %s]" % [String(d.get("label", "")).trim_suffix(":"),
		" (turned over, as it lies)" if bool(d.get("flip", false)) else "", String(d.get("path", "")).get_file()]


## Write [param text] to [param path], creating the directory. "" on success, else why not.
static func put(path: String, text: String) -> String:
	DirAccess.make_dir_recursive_absolute(path.get_base_dir())
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f == null:
		return "could not write " + path
	f.store_string(text)
	f.close()
	return ""


## The base every backend extends. It does nothing; a backend that forgets a method fails
## visibly (no pid, no reply) rather than silently.
class Backend:
	extends RefCounted

	func available() -> bool:
		return false

	func start(_job: Dictionary) -> int:
		return -1

	func resolve(_job: Dictionary) -> String:
		return ""

	func failure(job: Dictionary) -> String:
		return String(job.get("error", "the writer produced nothing"))

	## The files every backend writes into the job's directory, named once. `prompt` is the
	## record a person reads; `input` is exactly what the CLI was sent; `reply` is its output
	## stream and `last` its final message, where a backend writes one separately.
	static func paths(job: Dictionary) -> Dictionary:
		var dir := String(job["dir"])
		return {"prompt": dir.path_join("prompt.txt"), "system": dir.path_join("system.txt"),
			"input": dir.path_join("input.jsonl"), "reply": dir.path_join("reply.jsonl"),
			"last": dir.path_join("reply.txt"), "log": dir.path_join("stderr.log")}


## CLAUDE THROUGH THE CLAUDE CODE CLI, as a bare writer.
##
## `--safe-mode` leaves out everything the author has customized - CLAUDE.md, memory, skills,
## hooks, MCP servers, plugins, output styles - and `--system-prompt-file` replaces Claude
## Code's own agent prompt, so the model sees the two files and nothing else (measured: ~480
## input tokens for a one-line prompt). `--tools ""` takes every tool away, so a reply is
## words and only words. `--no-session-persistence`: nothing is resumed, see the class note.
## The message goes in as STREAM-JSON (one user message on stdin), the only input that carries
## pictures; the reply comes back as the stream's `result` event (`--verbose` is required with
## stream-json output). Measured with a picture attached: the tool list stays empty.
## `--bare` would be tighter still and is NOT used - it reads only ANTHROPIC_API_KEY, never the
## subscription login the rest of ghost's Claude runs use.
class Claude:
	extends Backend

	const MODELS := {"best": "opus", "fast": "sonnet"}

	func binary() -> String:
		var p := Deps.resolve("claude")
		return p if not p.is_empty() else "claude"

	func available() -> bool:
		return Deps.has("claude")

	func start(job: Dictionary) -> int:
		var p := Backend.paths(job)
		var m := Claude.compose(job)
		if not String(m["error"]).is_empty():
			job["error"] = m["error"]
			return -1
		for pair in [["prompt", m["shown"]], ["system", String(job.get("system", ""))], ["input", m["input"]]]:
			var err := TextGen.put(String(p[pair[0]]), String(pair[1]))
			if not err.is_empty():
				job["error"] = err
				return -1
		var args := PackedStringArray(["-p",
			"--model", String(MODELS.get(String(job.get("tier", "best")), MODELS["best"])),
			"--safe-mode", "--tools", "",
			"--system-prompt-file", String(p["system"]),
			"--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
			"--no-session-persistence"])
		var pid := Subprocess.start_redirected(binary(), args, {"cwd": String(job["dir"]),
			"stdin": String(p["input"]), "out": String(p["reply"]), "err": String(p["log"])},
			"claude writer")
		if pid <= 0:
			job["error"] = "could not start claude (is the Claude Code CLI installed and logged in?)"
		return pid

	## THE MESSAGE: each picture under its label, then the prompt - as the stream-json line the
	## CLI reads (`input`), and as the record a person reads (`shown`: the pictures listed, then
	## the prompt). `error` names a picture that could not be read: a writer told to talk about
	## a card it was never sent would make it up.
	static func compose(job: Dictionary) -> Dictionary:
		var content: Array = []
		var record := PackedStringArray()
		for im in job.get("images", []):
			var d: Dictionary = im
			var path := String(d.get("path", ""))
			var block := TextGen.image_block(path, bool(d.get("flip", false)))
			if block.is_empty():
				return {"error": "could not read the picture %s" % path.get_file(), "input": "", "shown": ""}
			var label := String(d.get("label", ""))
			if not label.is_empty():
				content.append({"type": "text", "text": label})
			content.append(block)
			record.append(TextGen.picture_line(d))
		var prompt := String(job.get("prompt", ""))
		content.append({"type": "text", "text": prompt})
		return {"error": "",
			"shown": ("\n".join(record) + "\n\n" + prompt) if not record.is_empty() else prompt,
			"input": JSON.stringify({"type": "user", "message": {"role": "user", "content": content}}) + "\n"}

	func resolve(job: Dictionary) -> String:
		var d: Variant = _reply(job)
		if not (d is Dictionary) or bool((d as Dictionary).get("is_error", false)):
			return ""
		return String((d as Dictionary).get("result", "")).strip_edges()

	func failure(job: Dictionary) -> String:
		var d: Variant = _reply(job)
		if d is Dictionary:
			var r := String((d as Dictionary).get("result", "")).strip_edges()
			if not r.is_empty():
				return r.substr(0, 240)
			return "claude ended %s" % String((d as Dictionary).get("subtype", "without a reply"))
		var err := FileAccess.get_file_as_string(String(Backend.paths(job)["log"])).strip_edges()
		if not err.is_empty():
			var lines := err.split("\n")
			return String(lines[lines.size() - 1]).substr(0, 240)
		return String(job.get("error", "claude finished without a reply"))

	## The stream's `result` event - the run's outcome - or null when it never came.
	static func _reply(job: Dictionary) -> Variant:
		var out: Variant = null
		for line in FileAccess.get_file_as_string(String(Backend.paths(job)["reply"])).split("\n"):
			var t := String(line).strip_edges()
			if not t.begins_with("{"):
				continue
			var j := JSON.new()
			if j.parse(t) == OK and j.data is Dictionary and String((j.data as Dictionary).get("type", "")) == "result":
				out = j.data
		return out


## OPENAI THROUGH THE CODEX CLI.
##
## `codex exec` has no system prompt of its own to replace, so the system prompt leads the
## message, after [constant ONLY_THIS]: an agent that keeps a shell is asked to work from the
## message alone. Read-only, in the job's own folder, `--ephemeral` so no session is kept, and
## `-o` names the file the final message is written to. Pictures go as `--image` files - each
## prepared the way Claude is sent it ([method TextGen.picture]) and saved beside the prompt -
## and the message says which is which. The tier is the reasoning effort; the model is the CLI's.
class Codex:
	extends Backend

	const EFFORT := {"best": "medium", "fast": "low"}
	## What an agent with tools is told before anything else.
	const ONLY_THIS := ("You are writing, not working: everything you need is in this message and the "
		+ "pictures attached to it. Do not run commands, and do not open, list or read any file - "
		+ "in this folder or anywhere else. Reply with the text asked for and nothing else.")

	func binary() -> String:
		var p := Deps.resolve("codex")
		return p if not p.is_empty() else "codex"

	func available() -> bool:
		return Deps.has("codex")

	func start(job: Dictionary) -> int:
		var p := Backend.paths(job)
		var m := Codex.compose(job)
		if not String(m["error"]).is_empty():
			job["error"] = m["error"]
			return -1
		for pic in m["pictures"]:
			var d: Dictionary = pic
			var img := TextGen.picture(String(d["from"]), bool(d["flip"]))
			if img == null or img.save_jpg(String(d["to"]), 0.9) != OK:
				job["error"] = "could not prepare the picture %s" % String(d["from"]).get_file()
				return -1
		var err := TextGen.put(String(p["prompt"]), String(m["prompt"]))
		if not err.is_empty():
			job["error"] = err
			return -1
		var pid := Subprocess.start_redirected(binary(), Codex.argv(job, m["pictures"]),
			{"cwd": String(job["dir"]), "stdin": String(p["prompt"]), "out": String(p["reply"]),
			"err": String(p["log"])}, "codex writer")
		if pid <= 0:
			job["error"] = "could not start codex (is the Codex CLI installed and logged in?)"
		return pid

	## THE MESSAGE: what an agent with tools is told first, the system prompt, which picture is
	## which, then the prompt - and the pictures to attach, `{from, flip, to}`, in that order.
	static func compose(job: Dictionary) -> Dictionary:
		var head := PackedStringArray([ONLY_THIS])
		var sys := String(job.get("system", "")).strip_edges()
		if not sys.is_empty():
			head.append(sys)
		var pictures: Array = []
		var lines := PackedStringArray()
		for im in job.get("images", []):
			var d: Dictionary = im
			var from := String(d.get("path", ""))
			if not FileAccess.file_exists(from):
				return {"error": "could not read the picture %s" % from.get_file(), "prompt": "", "pictures": []}
			pictures.append({"from": from, "flip": bool(d.get("flip", false)),
				"to": String(job["dir"]).path_join("picture_%d.jpg" % (pictures.size() + 1))})
			lines.append("Attached picture %d - %s" % [pictures.size(), String(d.get("label", "")).trim_suffix(":")])
		var body := "\n\n".join(head) + "\n\n---\n\n"
		if not lines.is_empty():
			body += "\n".join(lines) + "\n\n"
		return {"error": "", "prompt": body + String(job.get("prompt", "")), "pictures": pictures}

	static func argv(job: Dictionary, pictures: Array) -> PackedStringArray:
		var args := PackedStringArray(["exec", "--skip-git-repo-check", "--json", "--ephemeral",
			"-s", "read-only", "-C", String(job["dir"]),
			"-c", "model_reasoning_effort=" + String(EFFORT.get(String(job.get("tier", "best")), "medium")),
			"-o", String(Backend.paths(job)["last"])])
		# `--image=` per file, then `--`: the flag takes many values (see ImageGen.Codex)
		for pic in pictures:
			args.append("--image=" + String((pic as Dictionary)["to"]))
		args.append_array(["--", "-"])
		return args

	func resolve(job: Dictionary) -> String:
		return FileAccess.get_file_as_string(String(Backend.paths(job)["last"])).strip_edges()

	func failure(job: Dictionary) -> String:
		var last := ImageGen.Codex.last_message(String(Backend.paths(job)["reply"]))
		if not last.is_empty():
			return last
		var err := FileAccess.get_file_as_string(String(Backend.paths(job)["log"])).strip_edges()
		if not err.is_empty():
			var lines := err.split("\n")
			return String(lines[lines.size() - 1]).substr(0, 240)
		return String(job.get("error", "codex finished without a reply"))
