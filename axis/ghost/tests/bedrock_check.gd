extends SceneTree

## bedrock_check - Amazon Bedrock as a writer and a painter ([BedrockCatalog], [TextGen]'s and
## [ImageGen]'s `bedrock`): everything that can be held to a rule WITHOUT calling a model, which
## bills per token or per picture, and the queue's STEPS the backends run on.
##
##   godot --headless --path . --script res://tests/bedrock_check.gd
##
## - THE CATALOG, from canned answers shaped like a real account's (us-east-2, 2026-10-05): chat
##   models and text-to-image models only - not embeddings, the reranker, speech, an upscaler that
##   takes no prompt, provisioned-only variants, a legacy model or a retired one whose profile
##   lingers. The route to each: the geographic profile over the global one, the base id where the
##   home region serves it on demand, else ANOTHER REGION that does (Stability's painters live in
##   us-west-2). No account number kept; a week per region asked.
## - THE REGION as the CLI reads it.
## - THE WRITER: pictures as base64 JPEG (turned over when reversed), the request in a file and
##   nothing authored in argv, the region named; a model that takes no system prompt is sent it in
##   the message and one with a lower output limit is asked inside it - each once; the reply's text
##   blocks, a cut-off reply a failure, the CLI's complaint reported; a text-only model refuses
##   pictures before anything starts.
## - THE PAINTER: the director is shown each reference by the name the request gives it, and its
##   reply is reshaped (clipped, the ratio checked against what the image model takes, else read
##   off the request); the body per model family - a style model is handed the first reference,
##   and with none the default paints; the picture decoded from the response into the target; a
##   declined picture says why; a family it cannot ask is refused.
## - THE QUEUE'S STEPS: `advance` keeps a job running on a new pid until it returns 0 - in
##   [AgentJobs] and in [Illustrations]' own pump.

## By path, not by class name: a new class is unknown to `--script` runs until the editor rescans.
const Catalog := preload("res://scripts/bedrock_catalog.gd")
const ACCOUNT := "123456789012"

var _fails := 0


func _init() -> void:
	for check in [_catalog, _region, _writer, _writer_retries, _painter, _steps]:
		if not bool(check.call()):
			_ok(false, "a check stopped part way (script error?)")
	print("bedrock_check: %s" % ("ALL OK" if _fails == 0 else "%d FAILED" % _fails))
	quit(0 if _fails == 0 else 1)


func _ok(cond: bool, what: String) -> void:
	print(("  ok    " if cond else "  FAIL  ") + what)
	if not cond:
		_fails += 1


func _scratch() -> String:
	var dir := ProjectSettings.globalize_path("user://bedrock_check")
	DirAccess.make_dir_recursive_absolute(dir)
	return dir


func _picture(name: String) -> String:
	var img := Image.create(1200, 1800, false, Image.FORMAT_RGB8)
	img.fill(Color(0.9, 0.1, 0.1))
	img.fill_rect(Rect2i(0, 900, 1200, 900), Color(0.1, 0.1, 0.9))
	var path := _scratch().path_join(name)
	img.save_png(path)
	return path


# --- canned answers ------------------------------------------------------------------------------

func _model(id: String, name: String, provider: String, ins: Array, outs: Array, stream: bool,
		infer: Array, life := "ACTIVE") -> Dictionary:
	return {"modelId": id, "modelName": name, "providerName": provider, "inputModalities": ins,
		"outputModalities": outs, "responseStreamingSupported": stream,
		"inferenceTypesSupported": infer, "modelLifecycle": {"status": life}}


func _profile(id: String, status := "ACTIVE") -> Dictionary:
	return {"inferenceProfileId": id, "status": status, "type": "SYSTEM_DEFINED",
		"inferenceProfileArn": "arn:aws:bedrock:us-east-2:%s:inference-profile/%s" % [ACCOUNT, id]}


## us-east-2's models and profiles, us-west-2's models: [home_models, home_profiles, others].
func _docs() -> Array:
	var home := {"modelSummaries": [
		_model("amazon.nova-pro-v1:0", "Nova Pro", "Amazon", ["TEXT", "IMAGE", "VIDEO"], ["TEXT"], true, ["INFERENCE_PROFILE"]),
		_model("amazon.nova-2-lite-v1:0", "Nova 2 Lite", "Amazon", ["TEXT", "IMAGE", "VIDEO"], ["TEXT"], true, ["INFERENCE_PROFILE"]),
		_model("amazon.titan-embed-text-v2:0", "Titan Text Embeddings V2", "Amazon", ["TEXT"], ["EMBEDDING"], false, ["ON_DEMAND"]),
		_model("amazon.nova-lite-v1:0", "Nova Lite", "Amazon", ["TEXT", "IMAGE", "VIDEO"], ["TEXT"], true, ["ON_DEMAND", "INFERENCE_PROFILE"]),
		_model("amazon.nova-micro-v1:0", "Nova Micro", "Amazon", ["TEXT"], ["TEXT"], true, ["INFERENCE_PROFILE"]),
		_model("amazon.nova-lite-v1:0:300k", "Nova Lite", "Amazon", ["TEXT", "IMAGE"], ["TEXT"], true, ["PROVISIONED"]),
		_model("amazon.rerank-v1:0", "Rerank 1.0", "Amazon", ["TEXT"], ["TEXT"], false, ["ON_DEMAND"]),
		_model("amazon.nova-2-sonic-v1:0", "Nova 2 Sonic", "Amazon", ["SPEECH"], ["SPEECH", "TEXT"], true, ["ON_DEMAND"]),
		_model("amazon.nova-old-v1:0", "Nova Old", "Amazon", ["TEXT"], ["TEXT"], true, ["ON_DEMAND"], "LEGACY"),
		_model("meta.llama3-3-70b-instruct-v1:0", "Llama 3.3 70B Instruct", "Meta", ["TEXT"], ["TEXT"], true, ["INFERENCE_PROFILE"]),
		_model("stability.stable-image-style-guide-v1:0", "Stable Image Style Guide", "Stability AI", ["TEXT", "IMAGE"], ["IMAGE"], false, ["INFERENCE_PROFILE"]),
		_model("stability.stable-fast-upscale-v1:0", "Stable Fast Upscale", "Stability AI", ["IMAGE"], ["IMAGE"], false, ["INFERENCE_PROFILE"]),
	]}
	var profiles := {"inferenceProfileSummaries": [
		_profile("us.amazon.nova-premier-v1:0"),
		_profile("global.amazon.nova-2-lite-v1:0"),
		_profile("us.amazon.nova-2-lite-v1:0"),
		_profile("us.amazon.nova-pro-v1:0"),
		_profile("us.amazon.nova-micro-v1:0"),
		_profile("eu.amazon.nova-lite-v1:0", "INACTIVE"),
		_profile("us.meta.llama3-3-70b-instruct-v1:0"),
		_profile("us.stability.stable-image-style-guide-v1:0"),
		_profile("us.stability.stable-fast-upscale-v1:0"),
	]}
	var west := {"modelSummaries": [
		_model("amazon.nova-pro-v1:0", "Nova Pro", "Amazon", ["TEXT", "IMAGE"], ["TEXT"], true, ["INFERENCE_PROFILE"]),
		_model("stability.sd3-5-large-v1:0", "Stable Diffusion 3.5 Large", "Stability AI", ["TEXT", "IMAGE"], ["IMAGE"], false, ["ON_DEMAND"]),
		_model("stability.stable-image-core-v1:1", "Stable Image Core", "Stability AI", ["TEXT"], ["IMAGE"], false, ["ON_DEMAND"]),
		_model("stability.stable-image-ultra-v1:1", "Stable Image Ultra", "Stability AI", ["TEXT"], ["IMAGE"], false, ["ON_DEMAND"]),
		_model("mistral.mistral-large-2407-v1:0", "Mistral Large (24.07)", "Mistral AI", ["TEXT"], ["TEXT"], true, ["ON_DEMAND"]),
		_model("luma.ray-v2:0", "Ray v2", "Luma AI", ["TEXT", "IMAGE"], ["VIDEO"], false, ["ON_DEMAND"]),
	]}
	return [home, profiles, {"us-west-2": west}]


## The saved catalog swapped for [param cat] while [param body] runs, then put back.
func _with_catalog(cat: Dictionary, body: Callable) -> void:
	var path := ProjectSettings.globalize_path(Catalog.PATH)
	var had := FileAccess.file_exists(path)
	var kept := FileAccess.get_file_as_string(path) if had else ""
	TextGen.put(path, JSON.stringify(cat))
	body.call()
	if had:
		TextGen.put(path, kept)
	else:
		DirAccess.remove_absolute(path)


func _catalog() -> bool:
	print("the catalog")
	var d := _docs()
	var cat := Catalog.build(d[0], d[1], d[2], "us-east-2", 1000)
	var ids: Array = (cat["models"] as Array).map(func(m: Dictionary) -> String: return String(m["id"]))
	var want := ["amazon.nova-pro-v1:0", "amazon.nova-2-lite-v1:0", "amazon.nova-lite-v1:0", "amazon.nova-micro-v1:0",
		"meta.llama3-3-70b-instruct-v1:0", "stability.stable-image-style-guide-v1:0",
		"stability.sd3-5-large-v1:0", "stability.stable-image-core-v1:1", "stability.stable-image-ultra-v1:1",
		"mistral.mistral-large-2407-v1:0"]
	_ok(ids == want, "chat and text-to-image models, and nothing else: %s" % [ids])
	var e := func(id: String) -> Dictionary: return Catalog.entry(cat, id)
	_ok(String(e.call("amazon.nova-2-lite-v1:0")["route"]) == "us.amazon.nova-2-lite-v1:0", "the geographic profile beats the global one")
	_ok(String(e.call("amazon.nova-lite-v1:0")["route"]) == "amazon.nova-lite-v1:0"
		and String(e.call("amazon.nova-lite-v1:0")["region"]) == "us-east-2",
		"an inactive profile is no route: the home region's on-demand base id is")
	_ok(String(e.call("stability.stable-image-style-guide-v1:0")["route"]) == "us.stability.stable-image-style-guide-v1:0",
		"a painter reached through a profile from home")
	var sd: Dictionary = e.call("stability.sd3-5-large-v1:0")
	_ok(String(sd["region"]) == "us-west-2" and String(sd["route"]) == "stability.sd3-5-large-v1:0",
		"a painter home cannot reach is called in the region that serves it")
	_ok(String(e.call("amazon.nova-pro-v1:0")["region"]) == "us-east-2", "a model home reaches is not sent elsewhere")
	_ok(not bool(e.call("amazon.nova-micro-v1:0")["images"]) and bool(e.call("amazon.nova-pro-v1:0")["images"])
		and String(e.call("stability.stable-image-core-v1:1")["out"]) == "IMAGE", "what each reads and makes")
	_ok(String(cat["region"]) == "us-east-2" and not JSON.stringify(cat).contains(ACCOUNT),
		"the region comes from the profiles, and the account number is not kept")
	_ok(Catalog.entry(cat, "amazon.nova-premier-v1:0").is_empty(), "a retired model's lingering profile is no model")
	var only_global := Catalog.build(d[0], {"inferenceProfileSummaries": [_profile("global.amazon.nova-2-lite-v1:0")]},
		{}, "us-east-2", 1000)
	_ok(String(Catalog.entry(only_global, "amazon.nova-2-lite-v1:0")["route"]) == "global.amazon.nova-2-lite-v1:0",
		"the global profile when it is the only one")
	_ok(Catalog.lookups("us-west-2").size() == 3 and Catalog.lookups("eu-west-1").size() == 4,
		"the other regions are read, never the home region twice")
	var argv := Catalog.lookup_argv({"region": "us-west-2", "what": "models"})
	_ok(argv.slice(0, 2) == PackedStringArray(["bedrock", "list-foundation-models"]) and argv[argv.find("--region") + 1] == "us-west-2",
		"a lookup names its region")
	_ok(Catalog.lookup_argv({"region": "", "what": "profiles"}).find("--region") < 0,
		"an unknown home region is left to the CLI")
	# kept a week, for the region asked; listed Amazon first
	var now := int(Time.get_unix_time_from_system())
	var region := Catalog.region()
	for case in [[region, now, true, "this region, today"], ["xx-nowhere-1", now, false, "another region"],
			[region, now - 8 * 86400, false, "eight days old"]]:
		_with_catalog(Catalog.build(d[0], d[1], d[2], String(case[0]), int(case[1])), func() -> void:
			_ok(Catalog.fresh().is_empty() != bool(case[2]), "the saved catalog is %s: %s"
				% ["used" if bool(case[2]) else "not used", String(case[3])]))
	_with_catalog(cat, func() -> void:
		var text: Array = Catalog.models("TEXT").map(func(m: Dictionary) -> String: return String(m["provider"]))
		_ok(text.slice(0, 4) == ["Amazon", "Amazon", "Amazon", "Amazon"] and text.slice(4) == ["Meta", "Mistral AI"],
			"the writers on offer: Amazon's own first, then by provider: %s" % [text])
		var wl: Array = TextGen.Bedrock.models().map(func(m: Dictionary) -> String: return String(m["label"]))
		_ok(String(wl[0]).begins_with("Default (Amazon · Nova 2 Lite") and wl.has("Amazon · Nova Micro (text only)")
			and wl.has("Meta · Llama 3.3 70B Instruct (text only)"), "the writer picker: %s" % [wl])
		var pl: Array = ImageGen.Bedrock.models().map(func(m: Dictionary) -> String: return String(m["key"]))
		_ok(pl == ["", "stability.sd3-5-large-v1:0", "stability.stable-image-core-v1:1",
			"stability.stable-image-style-guide-v1:0", "stability.stable-image-ultra-v1:1"],
			"the painter picker: the default, then every family it can ask: %s" % [pl]))
	_ok(String((TextGen.Bedrock.models()[0] as Dictionary)["key"]) == "" and String((ImageGen.Bedrock.models()[0] as Dictionary)["key"]) == "",
		"both pickers open on the default")
	return true


func _region() -> bool:
	print("the region")
	var names := ["AWS_REGION", "AWS_DEFAULT_REGION", "AWS_CONFIG_FILE", "AWS_PROFILE", "AWS_DEFAULT_PROFILE"]
	var was := {}
	for n in names:
		was[n] = OS.get_environment(n)
		OS.unset_environment(n)
	var cfg := _scratch().path_join("config")
	TextGen.put(cfg, "[default]\nregion = eu-west-3\n\n[profile art]\noutput = json\nregion=ap-south-1\n")
	OS.set_environment("AWS_CONFIG_FILE", cfg)
	_ok(Catalog.region() == "eu-west-3", "the default profile's region (%s)" % Catalog.region())
	OS.set_environment("AWS_PROFILE", "art")
	_ok(Catalog.region() == "ap-south-1", "the active profile's region (%s)" % Catalog.region())
	OS.set_environment("AWS_DEFAULT_REGION", "us-west-2")
	_ok(Catalog.region() == "us-west-2", "AWS_DEFAULT_REGION over the config file")
	OS.set_environment("AWS_REGION", "ca-central-1")
	_ok(Catalog.region() == "ca-central-1", "AWS_REGION over everything")
	for n in names:
		if String(was[n]).is_empty():
			OS.unset_environment(n)
		else:
			OS.set_environment(n, String(was[n]))
	return true


func _writer() -> bool:
	print("the writer")
	var path := _picture("card.png")
	var m := TextGen.Bedrock.compose({"prompt": "Read it.", "images": [{"path": path, "label": "Card 1:", "flip": true}]})
	var content: Array = m["content"]
	_ok(content.size() == 3 and String((content[0] as Dictionary).get("text", "")) == "Card 1:"
		and (content[1] as Dictionary).has("image") and String((content[2] as Dictionary).get("text", "")) == "Read it.",
		"the label, the picture, then the prompt")
	var pic: Dictionary = (content[1] as Dictionary)["image"]
	var back := Image.new()
	_ok(String(pic["format"]) == "jpeg" and back.load_jpg_from_buffer(Marshalls.base64_to_raw(String((pic["source"] as Dictionary)["bytes"]))) == OK
		and maxi(back.get_width(), back.get_height()) == TextGen.PICTURE_EDGE,
		"the picture is a base64 JPEG at %d px on its long edge" % TextGen.PICTURE_EDGE)
	var top := back.get_pixel(back.get_width() / 2, 10)
	_ok(top.b > top.r, "a reversed card arrives upside down")
	_ok(String(m["shown"]).begins_with("[picture: Card 1 (turned over, as it lies) - card.png]"), "the record lists the picture")
	_ok(not String(TextGen.Bedrock.compose({"prompt": "x", "images": [{"path": path + ".missing"}]})["error"]).is_empty(),
		"an unreadable picture stops the run")
	var req := TextGen.Bedrock.request("us.amazon.nova-pro-v1:0", "SYSTEM-SECRET", [{"text": "PROMPT-SECRET"}], 4096)
	_ok(String(req["modelId"]) == "us.amazon.nova-pro-v1:0" and String(((req["system"] as Array)[0] as Dictionary)["text"]) == "SYSTEM-SECRET"
		and int((req["inferenceConfig"] as Dictionary)["maxTokens"]) == 4096, "the request: route, system block, output cap")
	_ok(not TextGen.Bedrock.request("m", "  ", []).has("system"), "an empty system prompt sends no system block")
	var argv := TextGen.Bedrock.argv("/tmp/job/input.jsonl", "us-west-2")
	_ok(argv.slice(0, 2) == PackedStringArray(["bedrock-runtime", "converse"])
		and argv[argv.find("--cli-input-json") + 1] == "file:///tmp/job/input.jsonl"
		and argv[argv.find("--cli-binary-format") + 1] == "base64" and argv[argv.find("--region") + 1] == "us-west-2"
		and argv.has("--no-cli-pager"), "the call: Converse, the request from its file, base64 named, the region")
	_ok(not " ".join(argv).contains("SECRET"), "nothing the author wrote is in argv")
	# the reply
	var gen := TextGen.Bedrock.new()
	var job := {"dir": _scratch(), "step": "converse", "tier": "best"}
	var p := TextGen.Backend.paths(job)
	TextGen.put(String(p["reply"]), JSON.stringify({"output": {"message": {"role": "assistant", "content": [
		{"reasoningContent": {"reasoningText": {"text": "thinking aloud"}}}, {"text": "The Tower. "}, {"text": "It falls."}]}},
		"stopReason": "end_turn", "usage": {"inputTokens": 12, "outputTokens": 5}}))
	_ok(gen.resolve(job) == "The Tower. It falls.", "the reply is its text blocks, joined, the thinking left out")
	TextGen.put(String(p["reply"]), JSON.stringify({"output": {"message": {"content": [{"text": "The Tow"}]}}, "stopReason": "max_tokens"}))
	_ok(gen.resolve(job).is_empty() and gen.failure(job).contains("cut off"), "a reply cut off at the limit is a failure: %s" % gen.failure(job))
	DirAccess.remove_absolute(String(p["reply"]))
	TextGen.put(String(p["log"]), "\nAn error occurred (AccessDeniedException) when calling the Converse operation: no access\n")
	_ok(gen.failure(job).begins_with("An error occurred (AccessDeniedException)"), "the CLI's own complaint is reported")
	_ok(gen.resolve({"dir": _scratch(), "step": "lookup"}).is_empty(), "a job still looking its route up has no reply")
	var tjob := {"dir": _scratch(), "prompt": "x", "model": "amazon.nova-micro-v1:0", "images": [{"path": path, "label": "Card 1:"}]}
	_ok(gen.start(tjob) <= 0 and String(tjob.get("error", "")).contains("text only") and not tjob.has("step"),
		"a text-only model refuses a job with pictures, and starts nothing: %s" % tjob.get("error", ""))
	return true


func _writer_retries() -> bool:
	print("the writer's retries")
	var job := {"system": "Be brief.", "max_tokens": 10000}
	_ok(TextGen.Bedrock.retry_of(job, "An error occurred (ValidationException) when calling the Converse operation: This model doesn't support system messages.") == "fold",
		"a model that takes no system prompt is sent it in the message")
	_ok(TextGen.Bedrock.retry_of(dict_with(job, "fold", true), "This model doesn't support system messages.") == "",
		"...once")
	var folded := TextGen.Bedrock.request("m", "Be brief.", [{"text": "Hi"}], 10000, true)
	var blocks: Array = ((folded["messages"] as Array)[0] as Dictionary)["content"]
	_ok(not folded.has("system") and String((blocks[0] as Dictionary)["text"]) == "Be brief." and String((blocks[1] as Dictionary)["text"]) == "Hi",
		"folded: the system prompt leads the message")
	var refusal := "An error occurred (ValidationException) when calling the Converse operation: Malformed input request: #/max_tokens: 10000 is not less or equal to 4096, please reformat your input and try again."
	_ok(TextGen.Bedrock.retry_of(job, refusal) == "cap" and TextGen.Bedrock.output_cap(refusal, 10000) == 4096,
		"a lower output limit is read off the refusal (%d)" % TextGen.Bedrock.output_cap(refusal, 10000))
	_ok(TextGen.Bedrock.output_cap("The maximum tokens you requested exceeds the model limit of 8192 for a context of 128000", 10000) == 8192,
		"the limit, not the context size")
	_ok(TextGen.Bedrock.output_cap(refusal, 4096) == 0, "...and asked inside it, never again")
	_ok(TextGen.Bedrock.retry_of(job, "An error occurred (ThrottlingException): Too many requests, wait 60 seconds") == "",
		"any other failure is not retried")
	return true


func _painter() -> bool:
	print("the painter")
	var back := _picture("back.png")
	var card := _picture("card1.png")
	var m := ImageGen.Bedrock.brief_content({"prompt": "THE FIRST ATTACHED IMAGE is the BACK.", "refs": [back, card]})
	var content: Array = m["content"]
	_ok(content.size() == 5 and String((content[0] as Dictionary)["text"]) == "Attached image 1:" and (content[1] as Dictionary).has("image")
		and String((content[2] as Dictionary)["text"]) == "Attached image 2:"
		and String((content[4] as Dictionary)["text"]).begins_with("THE REQUEST:\nTHE FIRST ATTACHED IMAGE"),
		"the director sees each reference by the name the request gives it, then the request")
	_ok(not String(ImageGen.Bedrock.brief_content({"prompt": "x", "refs": [back + ".missing"]})["error"]).is_empty(),
		"an unreadable reference stops the run")
	# the director's reply, reshaped
	var job := {"dir": _scratch(), "prompt": "FORMAT: PORTRAIT 2:3 (1024x1536). Paint it."}
	var paths := ImageGen.Bedrock.paths(job)
	var reply := func(text: String) -> void:
		TextGen.put(String(paths["brief"]), JSON.stringify({"output": {"message": {"content": [{"text": text}]}}, "stopReason": "end_turn"}))
	reply.call("```json\n{\"prompt\": \"%s\", \"negative_prompt\": \"text, borders\", \"aspect_ratio\": \"2:3\"}\n```" % "x".repeat(3000))
	var brief := ImageGen.Bedrock.read_brief(job)
	_ok(String(brief["aspect_ratio"]) == "2:3" and String(brief["prompt"]).length() == ImageGen.Bedrock.PROMPT_MAX
		and String(brief["negative_prompt"]) == "text, borders", "a fenced reply is read, the prompt clipped to %d" % ImageGen.Bedrock.PROMPT_MAX)
	reply.call("{\"prompt\": \"a tower\", \"aspect_ratio\": \"1024x1536\"}")
	_ok(String(ImageGen.Bedrock.read_brief(job)["aspect_ratio"]) == "2:3", "a ratio the model does not take is read off the request instead")
	reply.call("I cannot help with that.")
	_ok(ImageGen.Bedrock.read_brief(job).is_empty(), "a reply with no prompt is no brief")
	_ok(ImageGen.Bedrock.ratio_in("FORMAT: LANDSCAPE 3:2 (1536x1024)") == "3:2" and ImageGen.Bedrock.ratio_in("no format") == "1:1",
		"the request's ratio, else square")
	# the body per family
	var b := {"prompt": "a tower", "negative_prompt": "text", "aspect_ratio": "2:3"}
	var tb := ImageGen.Bedrock.body("text", b)
	_ok(String(tb["prompt"]) == "a tower" and String(tb["aspect_ratio"]) == "2:3" and String(tb["output_format"]) == "png"
		and String(tb["negative_prompt"]) == "text" and not tb.has("image"), "a text model is asked the prompt, the ratio, a PNG")
	var sb := ImageGen.Bedrock.body("style", b, back)
	var style := Image.new()
	_ok(sb.has("image") and style.load_png_from_buffer(Marshalls.base64_to_raw(String(sb["image"]))) == OK
		and float(sb["fidelity"]) == ImageGen.Bedrock.FIDELITY, "a style model is also handed the first reference")
	_ok(ImageGen.Bedrock.body("style", b, back + ".missing").is_empty(), "an unreadable style reference stops the run")
	_ok(ImageGen.Bedrock.schema_of("stability.stable-image-core-v1:1") == "text" and ImageGen.Bedrock.schema_of("stability.stable-image-style-guide-v1:0") == "style"
		and ImageGen.Bedrock.schema_of("stability.stable-fast-upscale-v1:0") == "", "the families it can ask, and one it cannot")
	var gen := ImageGen.Bedrock.new()
	var bad := {"dir": _scratch(), "prompt": "x", "model": "stability.stable-fast-upscale-v1:0"}
	_ok(gen.start(bad) <= 0 and String(bad.get("error", "")).contains("does not know how to ask") and not bad.has("step"),
		"a family it cannot ask is refused before anything starts")
	var argv := ImageGen.Bedrock.argv("stability.sd3-5-large-v1:0", "us-west-2", "/tmp/job/body.json", "/tmp/job/out.json")
	_ok(argv.slice(0, 2) == PackedStringArray(["bedrock-runtime", "invoke-model"]) and argv[argv.find("--model-id") + 1] == "stability.sd3-5-large-v1:0"
		and argv[argv.find("--body") + 1] == "fileb:///tmp/job/body.json" and argv[argv.find("--region") + 1] == "us-west-2"
		and argv[argv.size() - 1] == "/tmp/job/out.json", "the call: the body from its file, the region, the picture to its file")
	# the response
	var target := _scratch().path_join("painted.png")
	DirAccess.remove_absolute(target)
	var pj := {"dir": _scratch(), "step": "paint", "target": target, "painter_model": "stability.stable-image-core-v1:1", "brief": b}
	var small := Image.create(64, 96, false, Image.FORMAT_RGB8)
	small.fill(Color(0.2, 0.6, 0.3))
	TextGen.put(String(ImageGen.Bedrock.paths(pj)["painted"]), JSON.stringify({"seeds": [7], "finish_reasons": [null],
		"images": [Marshalls.raw_to_base64(small.save_png_to_buffer())]}))
	var made := gen.resolve(pj)
	var landed := Image.new()
	_ok(made == target and landed.load(target) == OK and landed.get_width() == 64, "the picture is decoded into the target")
	TextGen.put(String(ImageGen.Bedrock.paths(pj)["painted"]), JSON.stringify({"finish_reasons": ["Filter reason: prompt"]}))
	_ok(gen.resolve(pj).is_empty() and gen.failure(pj).contains("declined: Filter reason: prompt"),
		"a declined picture says why: %s" % gen.failure(pj))
	return true


## A backend in steps, as a pump sees it: the pids are not this process's children, so each reads
## as ended at once and the next pump advances it.
class Stepper:
	extends ImageGen.Backend
	var steps := 0

	func advance(job: Dictionary) -> int:
		steps += 1
		job["step"] = steps
		return 900000 + steps if steps < 3 else 0

	func resolve(job: Dictionary) -> String:
		return "after %d steps" % int(job.get("step", 0)) if String(job.get("kind", "")) == "text" else ""

	func failure(_job: Dictionary) -> String:
		return "no picture after %d steps" % steps


func _steps() -> bool:
	print("the queue's steps")
	AgentJobs.allow_for_tool(false)        # read-only: the pump lands jobs and starts none
	var s := Stepper.new()
	AgentJobs._running = {"j": {"id": "j", "kind": "text", "pid": 900000, "gen": s,
		"started": int(Time.get_unix_time_from_system()), "label": "stepper"}}
	AgentJobs.pump()
	_ok(AgentJobs.state("j") == "running" and int(AgentJobs._running["j"]["pid"]) == 900001,
		"a step that starts another keeps the job running, on the new pid")
	AgentJobs.pump()
	AgentJobs.pump()
	_ok(AgentJobs.state("j") == "done" and String(AgentJobs.result("j")["text"]) == "after 3 steps",
		"it lands once advance returns 0 (%s)" % AgentJobs.state("j"))
	AgentJobs.forget("j")
	# the book's illustrations run their own pump, and steps there too
	var t := Stepper.new()
	Illustrations._jobs = {"k": {"key": "k", "pid": 900000, "gen": t, "started": int(Time.get_unix_time_from_system())}}
	for i in 3:
		Illustrations._advance_jobs()
	_ok(t.steps == 3 and not Illustrations._jobs.has("k") and String(Illustrations._errors.get("k", "")) == "no picture after 3 steps",
		"the illustrations' pump runs a painter's steps, then lands it (%d steps)" % t.steps)
	Illustrations._errors.erase("k")
	return true


func dict_with(d: Dictionary, k: String, v: Variant) -> Dictionary:
	var out := d.duplicate()
	out[k] = v
	return out
