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
## A WRITER, NOT AN AGENT - unless ghost hands it tools. A run works from its prompt alone: Claude
## is run with none of its own tools; Codex, whose `exec` keeps a read-only shell, is TOLD to work
## only from the message and never to open, list or read a file ([constant Codex.ONLY_THIS]) - the
## user's call (2026-10-04): asking is enough, agents follow it. Each run is also STATELESS (no
## session is kept or resumed) - a caller that wants a conversation sends the conversation, so
## every run's whole input is the prompt file beside its reply.
##
## GHOST'S TOOLS. A job carrying `tools_url` (an [AgentTools] endpoint) is run as an agent whose
## ONLY tools are the ones ghost serves there - it calls them, sees what they return, and works in
## that loop until it is done; every call is kept beside its prompt (`tools.jsonl`). A backend that
## can take them says so ([method Backend.takes_tools]); one that cannot is asked the one-shot way.
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
	"bedrock": Bedrock,
}

const LABELS := {
	"claude": "Claude (Claude Code CLI)",
	"codex": "OpenAI (Codex CLI)",
	"bedrock": "Amazon Bedrock (AWS CLI)",
}

## One string literal per entry, for a picker's tooltip - same rule as Medium.BLURBS.
const BLURBS := {
	"claude": "Anthropic's models through the Claude Code CLI, run with no tools and no project context. Uses the Claude Code login.",
	"codex": "OpenAI's models through the Codex CLI, run read-only in the job's own folder and told to work only from the message it is given. Uses the Codex login.",
	"bedrock": "Models on Amazon Bedrock - Amazon's own Nova by default, and the open models Bedrock hosts - through the AWS CLI with your AWS credentials and region (aws configure). Billed per token to your AWS account; a model from another provider may subscribe the account to it through AWS Marketplace on first use.",
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

	## Whether a job's `tools_url` reaches this writer as tools it can call (see the class note).
	func takes_tools() -> bool:
		return false

	## The models a picker offers for this writer: `[{key, label}]`, the default (key "") first.
	static func models() -> Array:
		return [{"key": "", "label": "Default"}]

	func start(_job: Dictionary) -> int:
		return -1

	func resolve(_job: Dictionary) -> String:
		return ""

	func failure(job: Dictionary) -> String:
		return String(job.get("error", "the writer produced nothing"))

	## A writer that works in steps starts the next one here once a step's process has ended, and
	## returns its pid; 0 when there is no next step (see [AgentJobs]).
	func advance(_job: Dictionary) -> int:
		return 0

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

	## Claude Code's own aliases - each the newest model of its family, so the list never goes
	## stale. "Default" is the tiers above: Opus for what is heard, Sonnet for the bookkeeping.
	static func models() -> Array:
		return [{"key": "", "label": "Default (Opus)"}, {"key": "fable", "label": "Fable"},
			{"key": "opus", "label": "Opus"}, {"key": "sonnet", "label": "Sonnet"}]

	## The model a job runs on: the one chosen for it, else its tier's.
	static func model_of(job: Dictionary) -> String:
		var chosen := String(job.get("model", ""))
		return chosen if not chosen.is_empty() else String(MODELS.get(String(job.get("tier", "best")), MODELS["best"]))

	func binary() -> String:
		var p := Deps.resolve("claude")
		return p if not p.is_empty() else "claude"

	func available() -> bool:
		return Deps.has("claude")

	func takes_tools() -> bool:
		return true

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
		var tools := Claude.tool_args(job)
		if tools.is_empty() and not String(job.get("tools_url", "")).is_empty():
			job["error"] = "could not write the tools' config into " + String(job["dir"])
			return -1
		var pid := Subprocess.start_redirected(binary(), Claude.argv(job, String(p["system"]), tools), {"cwd": String(job["dir"]),
			"stdin": String(p["input"]), "out": String(p["reply"]), "err": String(p["log"])},
			"claude writer")
		if pid <= 0:
			job["error"] = "could not start claude (is the Claude Code CLI installed and logged in?)"
		return pid

	## Flags and paths only - the prompt is on stdin. [param tools]: [method tool_args]'s.
	##
	## NONE OF THE AUTHOR'S OWN SETUP: `--safe-mode`, which also drops every MCP server - ghost's
	## included (measured, CLI 2.1.291) - so a writer given ghost's tools loads no settings at all
	## instead (`--setting-sources ""`: no plugin, hook or memory; measured, it is shown no
	## CLAUDE.md), and the system prompt replaces Claude Code's either way.
	static func argv(job: Dictionary, system_path: String, tools: PackedStringArray) -> PackedStringArray:
		var args := PackedStringArray(["-p", "--model", Claude.model_of(job)])
		args.append_array(PackedStringArray(["--setting-sources", ""]) if not tools.is_empty() else PackedStringArray(["--safe-mode"]))
		args.append_array(["--tools", "",
			"--system-prompt-file", system_path,
			"--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
			"--no-session-persistence"])
		args.append_array(tools)
		return args

	## GHOST'S TOOLS for a job with a `tools_url`: the server named in a config file beside the
	## prompt (`mcp.json` - a path in argv, never the URL's token), no other server the author has
	## set up (`--strict-mcp-config`), and its tools allowed to run without asking - a print run has
	## nobody to ask. Claude's own tools stay off (`--tools ""`), so ghost's are the only ones. Empty
	## for a job without tools, or when the config cannot be written.
	static func tool_args(job: Dictionary) -> PackedStringArray:
		var url := String(job.get("tools_url", ""))
		if url.is_empty():
			return PackedStringArray()
		var cfg := String(job["dir"]).path_join("mcp.json")
		var err := TextGen.put(cfg, JSON.stringify({"mcpServers": {"ghost": {"type": "http", "url": url}}}, "\t"))
		if not err.is_empty():
			return PackedStringArray()
		return PackedStringArray(["--mcp-config", cfg, "--strict-mcp-config", "--allowedTools", "mcp__ghost"])

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

	## The models Codex offers this installation (see [method ImageGen.Codex.models]).
	static func models() -> Array:
		return ImageGen.Codex.models()

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
		if not String(job.get("model", "")).is_empty():
			args.append_array(["-m", String(job["model"])])
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


## MODELS ON AMAZON BEDROCK, THROUGH THE AWS CLI: Amazon's own Nova (the default) and the open
## models Bedrock hosts, run on the machine's AWS credentials and region (`aws configure`) and
## billed per token to that account.
##
## One `bedrock-runtime converse` call per job. Its whole request - the system prompt, the pictures
## (JPEG, base64) and the prompt - is a JSON file the CLI reads itself (`--cli-input-json
## file://...`), so nothing the author wrote is ever in argv. `--cli-binary-format base64` is
## explicit because a `cli_binary_format = raw-in-base64-out` in the user's config would send the
## pictures' base64 as the bytes themselves.
##
## WHICH MODELS, AND WHERE TO CALL THEM, is the account's own answer: [BedrockCatalog]. A model is
## kept by its base id and called on its route, in the region that serves it; a job finding no
## fresh catalog looks it up first, as steps of its own ([method advance]).
##
## ONE API, MANY MODELS: Converse evens out most differences between providers, and the two that
## remain are retried as steps: a model that takes no system prompt is sent it at the head of the
## message, and one whose output limit is under [constant MAX_TOKENS] is asked again inside it.
class Bedrock:
	extends Backend

	const Catalog := preload("res://scripts/bedrock_catalog.gd")
	## The model per tier: Amazon's newest for what is heard, its cheaper one for bookkeeping. Both
	## see pictures - a tarot card's passage is sent its card.
	const MODELS := {"best": "amazon.nova-2-lite-v1:0", "fast": "amazon.nova-lite-v1:0"}
	## Nova 1 models stop at 10K output tokens (Nova 2 Lite at 64K); a tarot plan runs to ~6K.
	const MAX_TOKENS := 10000
	## What the model picker offers before the account has been asked; the catalog replaces it.
	const SEED := [
		{"id": "amazon.nova-2-lite-v1:0", "name": "Nova 2 Lite", "provider": "Amazon", "images": true},
		{"id": "amazon.nova-pro-v1:0", "name": "Nova Pro", "provider": "Amazon", "images": true},
		{"id": "amazon.nova-lite-v1:0", "name": "Nova Lite", "provider": "Amazon", "images": true},
		{"id": "amazon.nova-micro-v1:0", "name": "Nova Micro", "provider": "Amazon", "images": false},
	]

	func binary() -> String:
		var p := Deps.resolve("aws")
		return p if not p.is_empty() else "aws"

	func available() -> bool:
		return Deps.has("aws")

	## The account's text models when it has been asked, else [constant SEED].
	static func listed() -> Array:
		var list := Catalog.models("TEXT")
		return list if not list.is_empty() else SEED

	static func models() -> Array:
		var out: Array = [{"key": "", "label": "Default (%s)" % Bedrock.name_of(MODELS["best"])}]
		for m in Bedrock.listed():
			var d: Dictionary = m
			out.append({"key": String(d["id"]),
				"label": Catalog.label(d) + ("" if bool(d.get("images", true)) else " (text only)")})
		return out

	static func name_of(id: String) -> String:
		for m in Bedrock.listed() + SEED:
			if String((m as Dictionary)["id"]) == id:
				return Catalog.label(m)
		return id

	## The base model a job runs on: the one chosen for it, else its tier's.
	static func model_of(job: Dictionary) -> String:
		var chosen := String(job.get("model", ""))
		return chosen if not chosen.is_empty() else String(MODELS.get(String(job.get("tier", "best")), MODELS["best"]))

	## False only for a model known to read text alone.
	static func sees_pictures(id: String) -> bool:
		for m in Bedrock.listed() + SEED:
			if String((m as Dictionary)["id"]) == id:
				return bool((m as Dictionary).get("images", true))
		return true

	func start(job: Dictionary) -> int:
		var model := Bedrock.model_of(job)
		if not (job.get("images", []) as Array).is_empty() and not Bedrock.sees_pictures(model):
			job["error"] = "%s reads text only, and this job sends pictures - choose another model" % Bedrock.name_of(model)
			return -1
		var m := Bedrock.compose(job)
		if not String(m["error"]).is_empty():
			job["error"] = m["error"]
			return -1
		var p := Backend.paths(job)
		for pair in [["prompt", m["shown"]], ["system", String(job.get("system", ""))]]:
			var err := TextGen.put(String(p[pair[0]]), String(pair[1]))
			if not err.is_empty():
				job["error"] = err
				return -1
		job["content"] = m["content"]
		job["max_tokens"] = MAX_TOKENS
		job["catalog"] = Catalog.fresh()
		if (job["catalog"] as Dictionary).is_empty():
			return Catalog.start_lookup(job, binary())
		return _converse(job)

	## After a step: the next lookup, the call itself, or the call again where a retry applies.
	func advance(job: Dictionary) -> int:
		match String(job.get("step", "")):
			"lookup":
				var next := Catalog.after_lookup(job, binary())
				return _converse(job) if next == 0 else maxi(next, 0)
			"converse":
				if not Provision.read_json(String(Backend.paths(job)["reply"])).is_empty():
					return 0
				match Bedrock.retry_of(job, Catalog.cli_error(String(Backend.paths(job)["log"]))):
					"fold":
						job["fold"] = true
						return _converse(job)
					"cap":
						job["max_tokens"] = Bedrock.output_cap(Catalog.cli_error(String(Backend.paths(job)["log"])),
							int(job["max_tokens"]))
						return _converse(job)
		return 0

	## What to do about a failed call, from the CLI's complaint: "fold" (the model takes no system
	## prompt - send it in the message), "cap" (ask inside the model's own output limit) or "".
	## Each is tried once.
	static func retry_of(job: Dictionary, why: String) -> String:
		var t := why.to_lower()
		if t.contains("system message") and not bool(job.get("fold", false)) \
				and not String(job.get("system", "")).strip_edges().is_empty():
			return "fold"
		if Bedrock.output_cap(why, int(job.get("max_tokens", MAX_TOKENS))) > 0:
			return "cap"
		return ""

	## The output limit a refusal names ("... is not less or equal to 4096"): the largest number in
	## it under [param asked], when the complaint is about the token limit at all; else 0.
	static func output_cap(why: String, asked: int) -> int:
		var t := why.to_lower()
		if not (t.contains("max_tokens") or t.contains("maxtokens") or t.contains("max tokens") or t.contains("maximum tokens")):
			return 0
		var best := 0
		for m in RegEx.create_from_string("\\d+").search_all(t):
			var n := int(m.get_string())
			if n >= 256 and n < asked and n > best:
				best = n
		return best

	func _converse(job: Dictionary) -> int:
		job["step"] = "converse"
		var p := Backend.paths(job)
		var model := Bedrock.model_of(job)
		var e := Catalog.entry(job.get("catalog", {}), model)
		if e.is_empty():
			job["error"] = "%s is not offered to this AWS account in any region it reaches" % Bedrock.name_of(model)
			return -1
		job["route"] = "%s in %s" % [String(e["route"]), String(e["region"])]
		var req := Bedrock.request(String(e["route"]), String(job.get("system", "")), job.get("content", []),
			int(job.get("max_tokens", MAX_TOKENS)), bool(job.get("fold", false)))
		var err := TextGen.put(String(p["input"]), JSON.stringify(req))
		if not err.is_empty():
			job["error"] = err
			return -1
		return Catalog.run(job, binary(), Bedrock.argv(String(p["input"]), String(e["region"])),
			String(p["reply"]), String(p["log"]), "bedrock writer")

	## THE MESSAGE: each picture under its label, then the prompt, as Converse content blocks
	## (`content`), and the record a person reads (`shown`). `error` names a picture that could not
	## be read: a writer told to talk about a card it was never sent would make it up.
	static func compose(job: Dictionary) -> Dictionary:
		var content: Array = []
		var record := PackedStringArray()
		for im in job.get("images", []):
			var d: Dictionary = im
			var block := Bedrock.picture_block(String(d.get("path", "")), bool(d.get("flip", false)))
			if block.is_empty():
				return {"error": "could not read the picture %s" % String(d.get("path", "")).get_file(),
					"content": [], "shown": ""}
			var label := String(d.get("label", ""))
			if not label.is_empty():
				content.append({"text": label})
			content.append(block)
			record.append(TextGen.picture_line(d))
		var prompt := String(job.get("prompt", ""))
		content.append({"text": prompt})
		return {"error": "", "content": content,
			"shown": ("\n".join(record) + "\n\n" + prompt) if not record.is_empty() else prompt}

	## A picture as a Converse image block (JPEG, base64); {} when it cannot be read.
	static func picture_block(path: String, flip := false) -> Dictionary:
		var img := TextGen.picture(path, flip)
		if img == null:
			return {}
		return {"image": {"format": "jpeg", "source": {"bytes": Marshalls.raw_to_base64(img.save_jpg_to_buffer(0.9))}}}

	## The Converse request, as the CLI reads it from its input file. [param fold] sends the system
	## prompt at the head of the message, for a model that takes none.
	static func request(model_id: String, system: String, content: Array, max_tokens := MAX_TOKENS,
			fold := false) -> Dictionary:
		var blocks := content.duplicate()
		var has_system := not system.strip_edges().is_empty()
		if fold and has_system:
			blocks.push_front({"text": system})
		var req := {"modelId": model_id,
			"messages": [{"role": "user", "content": blocks}],
			"inferenceConfig": {"maxTokens": max_tokens}}
		if has_system and not fold:
			req["system"] = [{"text": system}]
		return req

	## Flags and a path only - the request is in the file.
	static func argv(request_path: String, region := "") -> PackedStringArray:
		var args := PackedStringArray(["bedrock-runtime", "converse",
			"--cli-input-json", "file://" + request_path,
			"--cli-binary-format", "base64", "--cli-read-timeout", "300"])
		if not region.is_empty():
			args.append_array(["--region", region])
		args.append_array(["--output", "json", "--no-cli-pager"])
		return args

	func resolve(job: Dictionary) -> String:
		return Bedrock.reply_text(job, String(Backend.paths(job)["reply"]))

	## A Converse reply's words: its text blocks, joined (a reasoning model's thinking is left out);
	## "" for a reply cut off at the token limit, which is a failure and not a short answer.
	static func reply_text(job: Dictionary, path: String) -> String:
		if String(job.get("step", "")) != "converse":
			return ""
		var d := Provision.read_json(path)
		if d.is_empty() or String(d.get("stopReason", "")) == "max_tokens":
			return ""
		var parts := PackedStringArray()
		for c in ((d.get("output", {}) as Dictionary).get("message", {}) as Dictionary).get("content", []):
			if c is Dictionary and (c as Dictionary).has("text"):
				parts.append(String((c as Dictionary)["text"]))
		var u: Dictionary = d.get("usage", {})
		print("ghost: %s wrote %d tokens from %d (%s)" % [Bedrock.name_of(Bedrock.model_of(job)),
			int(u.get("outputTokens", 0)), int(u.get("inputTokens", 0)), String(job.get("route", ""))])
		return "".join(parts).strip_edges()

	func failure(job: Dictionary) -> String:
		if job.has("error"):
			return String(job["error"])
		var d := Provision.read_json(String(Backend.paths(job)["reply"]))
		if String(d.get("stopReason", "")) == "max_tokens":
			return "%s ran past %d tokens and was cut off" % [Bedrock.name_of(Bedrock.model_of(job)),
				int(job.get("max_tokens", MAX_TOKENS))]
		var err := Catalog.cli_error(String(Backend.paths(job)["log"]))
		return err if not err.is_empty() else "aws finished without a reply"
