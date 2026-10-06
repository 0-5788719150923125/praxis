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
	"bedrock": Bedrock,
}

const LABELS := {
	"codex": "OpenAI (Codex CLI)",
	"bedrock": "Amazon Bedrock (AWS CLI)",
}

## One string literal per entry, for the picker's tooltip - same rule as Medium.BLURBS.
const BLURBS := {
	"codex": "OpenAI's image model, driven through the Codex CLI's built-in image tool. Uses the Codex login, not an API key. About a minute per picture.",
	"bedrock": "Image models on Amazon Bedrock - Stability AI's (Amazon retired its own in 2026) - through the AWS CLI with your AWS credentials, billed per picture to your AWS account. An Amazon Nova model reads each request and its reference pictures and writes the image model's prompt. A Stability model subscribes the account to it through AWS Marketplace on first use.",
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

	## A painter that works in steps starts the next one here once a step's process has ended,
	## and returns its pid; 0 when there is no next step (see [AgentJobs]).
	func advance(_job: Dictionary) -> int:
		return 0


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
##
## `--ephemeral`: ghost keeps the picture, so Codex keeps no session. A session log holds every
## reference and the result as base64 (5-20 MB a picture) and Codex never prunes them. Measured on
## codex-cli 0.157.0: the thread id and `generated_images` are unaffected.
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
			"exec", "--skip-git-repo-check", "--json", "--ephemeral",
			"-s", "workspace-write", "-C", dir,
			"-c", "model_reasoning_effort=low"])
		# the model of the AGENT that calls the image tool - the picture is the tool's either way
		if not String(job.get("model", "")).is_empty():
			args.append_array(["-m", String(job["model"])])
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
		return home().path_join("generated_images")

	## `$CODEX_HOME`, defaulting to ~/.codex like the CLI does.
	static func home() -> String:
		var h := OS.get_environment("CODEX_HOME")
		return h if not h.is_empty() else Deps.home().path_join(".codex")

	## THE MODELS CODEX OFFERS, as a picker shows them: `[{key, label}]`, the CLI's own default first
	## (key "", labelled with the model the author's config names). Read from the catalog the CLI
	## keeps for itself (`models_cache.json` - what `codex debug models` prints), so it is the list
	## this installation actually has, and reading it costs nothing. Hidden entries stay hidden.
	static func models() -> Array:
		var dflt := ""
		for line in FileAccess.get_file_as_string(home().path_join("config.toml")).split("\n"):
			var t := String(line).strip_edges()
			if t.begins_with("model ") or t.begins_with("model="):
				dflt = t.get_slice("=", 1).strip_edges().trim_prefix("\"").trim_suffix("\"")
				break
		var out: Array = [{"key": "", "label": "Default" + ((" (%s)" % dflt) if not dflt.is_empty() else "")}]
		var j := JSON.new()
		if j.parse(FileAccess.get_file_as_string(home().path_join("models_cache.json"))) == OK and j.data is Dictionary:
			for m in (j.data as Dictionary).get("models", []):
				if m is Dictionary and String((m as Dictionary).get("visibility", "list")) == "list":
					out.append({"key": String(m["slug"]), "label": String((m as Dictionary).get("display_name", m["slug"]))})
		return out

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



## IMAGE MODELS ON AMAZON BEDROCK, THROUGH THE AWS CLI - Stability AI's, today (Amazon retired
## Titan Image Generator and Nova Canvas in 2026), on the machine's AWS credentials, billed per
## picture.
##
## AN IMAGE MODEL IS NOT AN AGENT. Every painter prompt in ghost is written for one - "use your
## image tool, match the attached pictures, save it here" - and a diffusion model can neither read
## instructions nor see an attachment. So a job is a DIRECTOR and a painter: Amazon's Nova
## ([constant DIRECTOR], through [TextGen]'s Bedrock writer) reads the request and looks at its
## references, and writes the one paragraph the image model sees, the picture's faults to avoid and
## its aspect ratio; then the image model paints that. The deck's hand reaches the painter only as
## words - except through a "style" model ([constant SCHEMAS]), which is also handed the job's
## first reference and paints in its style.
##
## WHAT AN IMAGE MODEL IS ASKED is per family ([constant SCHEMAS]): the catalog lists every image
## model the account can reach, and a painter offers the ones it knows how to ask. A new family is
## an entry there and a body in [method body]. Steps, each its own process: the catalog lookup if
## it is stale (see [BedrockCatalog]), the director, the painting.
class Bedrock:
	extends Backend

	const Catalog := preload("res://scripts/bedrock_catalog.gd")
	## By model id prefix: "text" paints from a prompt; "style" also takes a style image.
	const SCHEMAS := {
		"stability.stable-image-core": "text",
		"stability.sd3-5-large": "text",
		"stability.stable-image-ultra": "text",
		"stability.stable-image-style-guide": "style",
	}
	## Fast and cheap; the one to try a deck with.
	const DEFAULT := "stability.stable-image-core-v1:1"
	const DIRECTOR := "amazon.nova-2-lite-v1:0"
	const RATIOS := ["16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"]
	const PROMPT_MAX := 1500
	const NEGATIVE_MAX := 400
	## How strongly a style model follows its style image (0..1; Stability's default is 0.5).
	const FIDELITY := 0.5
	## What the model picker offers before the account has been asked; the catalog replaces it.
	const SEED := [
		{"id": "stability.stable-image-core-v1:1", "name": "Stable Image Core", "provider": "Stability AI"},
		{"id": "stability.sd3-5-large-v1:0", "name": "Stable Diffusion 3.5 Large", "provider": "Stability AI"},
		{"id": "stability.stable-image-ultra-v1:1", "name": "Stable Image Ultra", "provider": "Stability AI"},
		{"id": "stability.stable-image-style-guide-v1:0", "name": "Stable Image Style Guide", "provider": "Stability AI"},
	]
	const DIRECTOR_SYSTEM := """You write the prompt for an image model. You are given a request written for a painting agent that has an image tool, and the pictures attached to it. The image model sees only what you write - not the request, and not the pictures - so everything it must show, and every quality of the attached pictures it must match, has to be in your words.

Reply with ONE JSON object and nothing else:
{"prompt": "...", "negative_prompt": "...", "aspect_ratio": "..."}

prompt: one paragraph, at most 1200 characters. The subject and composition the request asks for, then the look: medium, palette, light, texture, level of detail. Where the request says to match attached pictures, describe what they share - the hand, the palette, the materials - instead of referring to them.
negative_prompt: at most 300 characters. What must not appear: text, letters, numbers, borders, frames, watermarks, signatures, and anything else the request forbids.
aspect_ratio: the request's format, one of 16:9, 1:1, 21:9, 2:3, 3:2, 4:5, 5:4, 9:16, 9:21.

Ignore anything in the request about tools, files, paths or saving."""

	func binary() -> String:
		var p := Deps.resolve("aws")
		return p if not p.is_empty() else "aws"

	func available() -> bool:
		return Deps.has("aws")

	## "text", "style", or "" for a model this painter does not know how to ask.
	static func schema_of(id: String) -> String:
		for prefix in SCHEMAS:
			if id.begins_with(String(prefix)):
				return String(SCHEMAS[prefix])
		return ""

	## The account's image models this painter can ask, when it has been asked; else [constant SEED].
	static func listed() -> Array:
		var list := Catalog.models("IMAGE").filter(
			func(m: Dictionary) -> bool: return not Bedrock.schema_of(String(m["id"])).is_empty())
		return list if not list.is_empty() else SEED

	static func models() -> Array:
		var out: Array = [{"key": "", "label": "Default (%s)" % Bedrock.name_of(DEFAULT)}]
		for m in Bedrock.listed():
			var d: Dictionary = m
			out.append({"key": String(d["id"]), "label": Catalog.label(d)
				+ (" - in the style of the first reference" if Bedrock.schema_of(String(d["id"])) == "style" else "")})
		return out

	static func name_of(id: String) -> String:
		for m in Bedrock.listed() + SEED:
			if String((m as Dictionary)["id"]) == id:
				return Catalog.label(m)
		return id

	static func model_of(job: Dictionary) -> String:
		var chosen := String(job.get("model", ""))
		return chosen if not chosen.is_empty() else DEFAULT

	## The files a job writes, named once: the director's request/reply/log, the painter's
	## body/response/stdout/log. The picture itself goes to `job.target`.
	static func paths(job: Dictionary) -> Dictionary:
		var dir := String(job["dir"])
		return {"prompt": dir.path_join("prompt.txt"), "brief_in": dir.path_join("brief_request.json"),
			"brief": dir.path_join("brief_reply.json"), "brief_log": dir.path_join("brief.log"),
			"body": dir.path_join("paint_body.json"), "painted": dir.path_join("paint_response.json"),
			"paint_out": dir.path_join("paint_stdout.json"), "paint_log": dir.path_join("paint.log")}

	func start(job: Dictionary) -> int:
		var model := Bedrock.model_of(job)
		if Bedrock.schema_of(model).is_empty():
			job["error"] = "ghost does not know how to ask %s for a picture" % model
			return -1
		DirAccess.make_dir_recursive_absolute(String(job["dir"]))
		var m := Bedrock.brief_content(job)
		if not String(m["error"]).is_empty():
			job["error"] = m["error"]
			return -1
		var err := TextGen.put(String(Bedrock.paths(job)["prompt"]), String(m["shown"]))
		if not err.is_empty():
			job["error"] = err
			return -1
		job["content"] = m["content"]
		job["catalog"] = Catalog.fresh()
		if (job["catalog"] as Dictionary).is_empty():
			return Catalog.start_lookup(job, binary())
		return _direct(job)

	func advance(job: Dictionary) -> int:
		match String(job.get("step", "")):
			"lookup":
				var next := Catalog.after_lookup(job, binary())
				return _direct(job) if next == 0 else maxi(next, 0)
			"direct":
				var brief := Bedrock.read_brief(job)
				if brief.is_empty():
					var why := Catalog.cli_error(String(Bedrock.paths(job)["brief_log"]))
					job["error"] = "the director (%s) wrote no usable prompt%s" % [TextGen.Bedrock.name_of(DIRECTOR),
						(": " + why) if not why.is_empty() else " - see brief_reply.json in the job's folder"]
					return 0
				job["brief"] = brief
				return _paint(job)
		return 0

	## THE DIRECTOR'S MESSAGE: each reference under the name the request gives it ("Attached image
	## 1:" - the requests say "the first attached image"), then the request; and the record a
	## person reads. `error` names a reference that could not be read.
	static func brief_content(job: Dictionary) -> Dictionary:
		var content: Array = []
		var record := PackedStringArray()
		var refs: Array = job.get("refs", [])
		for i in refs.size():
			var block := TextGen.Bedrock.picture_block(String(refs[i]))
			if block.is_empty():
				return {"error": "could not read the reference %s" % String(refs[i]).get_file(), "content": [], "shown": ""}
			content.append({"text": "Attached image %d:" % (i + 1)})
			content.append(block)
			record.append("[attached image %d - %s]" % [i + 1, String(refs[i]).get_file()])
		var prompt := String(job.get("prompt", ""))
		content.append({"text": "THE REQUEST:\n" + prompt})
		return {"error": "", "content": content,
			"shown": ("\n".join(record) + "\n\n" + prompt) if not record.is_empty() else prompt}

	func _direct(job: Dictionary) -> int:
		job["step"] = "direct"
		var p := Bedrock.paths(job)
		var e := Catalog.entry(job.get("catalog", {}), DIRECTOR)
		if e.is_empty():
			job["error"] = "the director, %s, is not offered to this AWS account" % TextGen.Bedrock.name_of(DIRECTOR)
			return -1
		DirAccess.remove_absolute(String(p["brief"]))
		var req := TextGen.Bedrock.request(String(e["route"]), DIRECTOR_SYSTEM, job.get("content", []), 2000)
		var err := TextGen.put(String(p["brief_in"]), JSON.stringify(req))
		if not err.is_empty():
			job["error"] = err
			return -1
		return Catalog.run(job, binary(), TextGen.Bedrock.argv(String(p["brief_in"]), String(e["region"])),
			String(p["brief"]), String(p["brief_log"]), "bedrock director")

	## The director's reply as `{prompt, negative_prompt, aspect_ratio}`, reshaped to what an
	## image model takes; {} when it wrote nothing usable. A missing or unknown ratio is read off
	## the request's own FORMAT line, else square.
	static func read_brief(job: Dictionary) -> Dictionary:
		var text := TextGen.Bedrock.reply_text({"step": "converse", "model": DIRECTOR, "route": "director"},
			String(Bedrock.paths(job)["brief"]))
		var d: Variant = TextGen.extract_json(text)
		if not (d is Dictionary) or String((d as Dictionary).get("prompt", "")).strip_edges().is_empty():
			return {}
		var ratio := String((d as Dictionary).get("aspect_ratio", "")).strip_edges()
		if not RATIOS.has(ratio):
			ratio = Bedrock.ratio_in(String(job.get("prompt", "")))
		return {"prompt": String(d["prompt"]).strip_edges().substr(0, PROMPT_MAX),
			"negative_prompt": String((d as Dictionary).get("negative_prompt", "")).strip_edges().substr(0, NEGATIVE_MAX),
			"aspect_ratio": ratio}

	## The first aspect ratio a request names ("FORMAT: PORTRAIT 2:3"), else "1:1".
	static func ratio_in(request: String) -> String:
		for m in RegEx.create_from_string("\\b\\d{1,2}:\\d{1,2}\\b").search_all(request):
			if RATIOS.has(m.get_string()):
				return m.get_string()
		return "1:1"

	func _paint(job: Dictionary) -> int:
		job["step"] = "paint"
		var p := Bedrock.paths(job)
		var model := Bedrock.model_of(job)
		var refs: Array = job.get("refs", [])
		# a style model paints after a picture; with none to follow, the default paints it
		if Bedrock.schema_of(model) == "style" and refs.is_empty():
			print("ghost: %s needs a reference to follow, and this picture has none - %s paints it"
				% [Bedrock.name_of(model), Bedrock.name_of(DEFAULT)])
			model = DEFAULT
		job["painter_model"] = model
		var e := Catalog.entry(job.get("catalog", {}), model)
		if e.is_empty():
			job["error"] = "%s is not offered to this AWS account in any region it reaches" % Bedrock.name_of(model)
			return -1
		var body := Bedrock.body(Bedrock.schema_of(model), job["brief"], String(refs[0]) if not refs.is_empty() else "")
		if body.is_empty():
			job["error"] = "could not read the style reference %s" % String(refs[0]).get_file()
			return -1
		var err := TextGen.put(String(p["body"]), JSON.stringify(body))
		if not err.is_empty():
			job["error"] = err
			return -1
		# the response file is written only by a call that succeeds: an earlier run's must go first
		DirAccess.remove_absolute(String(p["painted"]))
		return Catalog.run(job, binary(), Bedrock.argv(String(e["route"]), String(e["region"]),
			String(p["body"]), String(p["painted"])), String(p["paint_out"]), String(p["paint_log"]), "bedrock painter")

	## What [param schema] is asked: the director's prompt, faults and ratio, a PNG back - and for a
	## style model, the style image (base64). {} when that image cannot be read.
	static func body(schema: String, brief: Dictionary, style_ref := "") -> Dictionary:
		var b := {"prompt": String(brief.get("prompt", "")), "aspect_ratio": String(brief.get("aspect_ratio", "1:1")),
			"output_format": "png"}
		if not String(brief.get("negative_prompt", "")).is_empty():
			b["negative_prompt"] = String(brief["negative_prompt"])
		if schema == "style":
			var img := TextGen.picture(style_ref)
			if img == null:
				return {}
			b["image"] = Marshalls.raw_to_base64(img.save_png_to_buffer())
			b["fidelity"] = FIDELITY
		return b

	## Flags and paths only: the body is read from its file, the picture written to [param out].
	static func argv(route: String, region: String, body_path: String, out: String) -> PackedStringArray:
		var args := PackedStringArray(["bedrock-runtime", "invoke-model", "--model-id", route,
			"--body", "fileb://" + body_path, "--content-type", "application/json",
			"--accept", "application/json", "--cli-read-timeout", "300"])
		if not region.is_empty():
			args.append_array(["--region", region])
		args.append_array(["--output", "json", "--no-cli-pager", out])
		return args

	## The picture, decoded from the response into `job.target`; "" when there is none.
	func resolve(job: Dictionary) -> String:
		if String(job.get("step", "")) != "paint":
			return ""
		var d := Provision.read_json(String(Bedrock.paths(job)["painted"]))
		var images: Array = d.get("images", [])
		var reasons: Array = d.get("finish_reasons", [])
		if images.is_empty() or (not reasons.is_empty() and reasons[0] != null):
			return ""
		var img := Image.new()
		if img.load_png_from_buffer(Marshalls.base64_to_raw(String(images[0]))) != OK:
			return ""
		var target := String(job.get("target", ""))
		if target.is_empty():
			target = String(job["dir"]).path_join("image.png")
		if img.save_png(target) != OK:
			return ""
		print("ghost: %s painted %dx%d (%s)" % [Bedrock.name_of(String(job.get("painter_model", ""))),
			img.get_width(), img.get_height(), String((job.get("brief", {}) as Dictionary).get("aspect_ratio", ""))])
		return target

	func failure(job: Dictionary) -> String:
		if job.has("error"):
			return String(job["error"])
		var d := Provision.read_json(String(Bedrock.paths(job)["painted"]))
		var reasons: Array = d.get("finish_reasons", [])
		if not reasons.is_empty() and reasons[0] != null:
			return "%s declined: %s" % [Bedrock.name_of(String(job.get("painter_model", ""))), String(reasons[0])]
		var err := Catalog.cli_error(String(Bedrock.paths(job)["paint_log"]))
		return err if not err.is_empty() else "aws finished without a picture"
