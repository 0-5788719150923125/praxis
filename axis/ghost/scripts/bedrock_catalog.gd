extends RefCounted
class_name BedrockCatalog

## BedrockCatalog - which Amazon Bedrock models this AWS account can call, and how to reach each.
## Shared by the Bedrock writer ([TextGen]'s `bedrock`) and painter ([ImageGen]'s `bedrock`); it
## only ever makes the free listing calls, never a model call.
##
## THE ACCOUNT'S OWN ANSWER, never a hand-kept list: which models exist changes under us (Amazon
## retired Nova Premier, Nova Canvas and the last Titan generators in 2026), so the list comes from
## the AWS CLI - `list-foundation-models` and `list-inference-profiles` - and is kept
## [constant DAYS] days in [constant PATH], for the region it was asked in.
##
## ACROSS REGIONS. A model is served in some regions and not others - Stability AI's text-to-image
## models only in us-west-2 (2026-10), Amazon's Nova through inference profiles nearly everywhere -
## so the models are read in the configured region AND in [constant ALSO]. A model is called where
## the configured region reaches it, through its geographic inference profile (data stays in the
## geography), else the global one, else directly; failing that, directly in the first other region
## that serves it on demand.
##
## A MODEL IS KEPT BY ITS BASE ID (`amazon.nova-pro-v1:0`), so a show's choice means the same thing
## on any machine; [method entry] turns it into this account's `{route, region}`.
##
## LOOKING UP IS A JOB'S FIRST STEPS. A job that finds no fresh catalog runs the listing calls one
## after another as steps of its own ([method start_lookup] / [method after_lookup], driven by the
## backend's `advance`), so no frame ever waits on the network.

const PATH := "user://bedrock/catalog.json"
const DAYS := 7
## Where Bedrock serves the most models, read besides the configured region.
const ALSO := ["us-west-2", "us-east-1"]


## The configured region's two calls, then the other regions' model lists (their profiles are
## not needed: a model reached from elsewhere is called on demand). `required` marks the calls
## without which there is no catalog; another region failing (not enabled for the account, say)
## just contributes nothing.
static func lookups(region: String) -> Array:
	var out: Array = [{"region": region, "what": "models", "required": true},
		{"region": region, "what": "profiles", "required": true}]
	for r in ALSO:
		if String(r) != region:
			out.append({"region": String(r), "what": "models", "required": false})
	return out


static func lookup_argv(step: Dictionary) -> PackedStringArray:
	var args := PackedStringArray(["bedrock",
		"list-foundation-models" if String(step["what"]) == "models" else "list-inference-profiles"])
	if not String(step["region"]).is_empty():
		args.append_array(["--region", String(step["region"])])
	args.append_array(["--output", "json", "--no-cli-pager"])
	return args


static func lookup_path(job: Dictionary, i: int) -> String:
	return String(job["dir"]).path_join("bedrock_lookup_%d.json" % i)


## Begin a job's lookup: its first call's pid (<= 0 with job.error set).
static func start_lookup(job: Dictionary, binary: String) -> int:
	job["step"] = "lookup"
	job["lookup"] = 0
	job["lookups"] = lookups(region())
	return _run_lookup(job, binary)


## A lookup call has ended: the next call's pid; 0 when the catalog is built, saved and handed to
## the job as `job.catalog`; -1 when it cannot be (job.error says why).
static func after_lookup(job: Dictionary, binary: String) -> int:
	var steps: Array = job["lookups"]
	var i := int(job["lookup"])
	var step: Dictionary = steps[i]
	if bool(step["required"]) and Provision.read_json(lookup_path(job, i)).is_empty():
		job["error"] = "could not list the Bedrock %s: %s" % [String(step["what"]),
			cli_error(lookup_path(job, i) + ".log")]
		return -1
	i += 1
	job["lookup"] = i
	if i < steps.size():
		return _run_lookup(job, binary)
	var docs: Array = []
	for k in steps.size():
		docs.append(Provision.read_json(lookup_path(job, k)))
	var others := {}
	for k in range(2, steps.size()):
		others[String((steps[k] as Dictionary)["region"])] = docs[k]
	var cat := build(docs[0], docs[1], others, region(), int(Time.get_unix_time_from_system()))
	if (cat["models"] as Array).is_empty():
		job["error"] = "this AWS account lists no Bedrock models it can call"
		return -1
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(PATH).get_base_dir())
	var f := FileAccess.open(ProjectSettings.globalize_path(PATH), FileAccess.WRITE)
	if f != null:
		f.store_string(JSON.stringify(cat, "\t"))
		f.close()
	job["catalog"] = cat
	return 0


static func _run_lookup(job: Dictionary, binary: String) -> int:
	var i := int(job["lookup"])
	var out := lookup_path(job, i)
	return run(job, binary, lookup_argv((job["lookups"] as Array)[i]), out, out + ".log", "bedrock lookup")


## THE CATALOG from the CLI's answers - pure, so the gate can hand it canned documents.
## [param home_models] / [param home_profiles] are the configured region's; [param others] maps
## another region to its model list. An entry: `{id, name, provider, out ("TEXT"|"IMAGE"),
## images (reads pictures), region, route}`. Kept: ACTIVE models answering in text alone that
## stream (chat models do; embeddings and the reranker do not), or in images alone from a text
## prompt; callable on demand or through a profile - never a provisioned-only variant, and never
## a retired model whose profile has outlived it.
static func build(home_models: Dictionary, home_profiles: Dictionary, others: Dictionary,
		asked: String, now: int) -> Dictionary:
	var routes := {}
	var home := ""
	for v in home_profiles.get("inferenceProfileSummaries", []):
		var p: Dictionary = v
		var id := String(p.get("inferenceProfileId", ""))
		if String(p.get("status", "")) != "ACTIVE" or id.find(".") < 0:
			continue
		if home.is_empty():
			home = String(p.get("inferenceProfileArn", "")).get_slice(":", 3)
		var base := id.substr(id.find(".") + 1)
		# a geographic profile keeps the data in its geography, so it wins over the global one
		if not routes.has(base) or (String(routes[base]).begins_with("global.") and not id.begins_with("global.")):
			routes[base] = id
	if home.is_empty():
		home = asked
	var list: Array = []
	var seen := {}
	for m in _usable(home_models):
		var d: Dictionary = m
		var id := String(d["id"])
		if routes.has(id):
			d["region"] = home
			d["route"] = routes[id]
		elif bool(d["on_demand"]):
			d["region"] = home
			d["route"] = id
		else:
			continue
		d.erase("on_demand")
		list.append(d)
		seen[id] = true
	for r in others:
		for m in _usable(others[r]):
			var d: Dictionary = m
			if seen.has(String(d["id"])) or not bool(d["on_demand"]):
				continue
			d.erase("on_demand")
			d["region"] = String(r)
			d["route"] = String(d["id"])
			list.append(d)
			seen[String(d["id"])] = true
	return {"asked": asked, "region": home, "fetched": now, "models": list}


static func _usable(doc: Dictionary) -> Array:
	var out: Array = []
	for v in doc.get("modelSummaries", []):
		var m: Dictionary = v
		var ins: Array = m.get("inputModalities", [])
		var outs: Array = m.get("outputModalities", [])
		var infer: Array = m.get("inferenceTypesSupported", [])
		if String((m.get("modelLifecycle", {}) as Dictionary).get("status", "ACTIVE")) != "ACTIVE" \
				or not ins.has("TEXT") or not (infer.has("ON_DEMAND") or infer.has("INFERENCE_PROFILE")):
			continue
		var kind := ""
		if outs == ["TEXT"] and bool(m.get("responseStreamingSupported", false)):
			kind = "TEXT"
		elif outs == ["IMAGE"]:
			kind = "IMAGE"
		if kind.is_empty():
			continue
		out.append({"id": String(m["modelId"]), "name": String(m.get("modelName", m["modelId"])),
			"provider": String(m.get("providerName", "")), "out": kind, "images": ins.has("IMAGE"),
			"on_demand": infer.has("ON_DEMAND")})
	return out


## The saved catalog when it is this region's and under [constant DAYS] old, else {}.
static func fresh() -> Dictionary:
	var cat := saved()
	if cat.is_empty() or String(cat.get("asked", "")) != region() \
			or int(Time.get_unix_time_from_system()) - int(cat.get("fetched", 0)) > DAYS * 86400:
		return {}
	return cat


## The saved catalog at any age - for showing what there is, not for calling it.
static func saved() -> Dictionary:
	return Provision.read_json(ProjectSettings.globalize_path(PATH))


## The account's models answering in [param out] ("TEXT" or "IMAGE"), as saved; [] before the
## account has been asked. Amazon's own first, then by provider and name.
static func models(out: String) -> Array:
	var list: Array = (saved().get("models", []) as Array).filter(
		func(m: Dictionary) -> bool: return String(m.get("out", "")) == out)
	list.sort_custom(func(a: Dictionary, b: Dictionary) -> bool:
		var pa := String(a.get("provider", ""))
		var pb := String(b.get("provider", ""))
		if (pa == "Amazon") != (pb == "Amazon"):
			return pa == "Amazon"
		if pa != pb:
			return pa.naturalnocasecmp_to(pb) < 0
		return String(a.get("name", "")).naturalnocasecmp_to(String(b.get("name", ""))) < 0)
	return list


## [param id]'s entry in [param cat]: `{route, region, ...}`; {} when the account cannot reach it.
static func entry(cat: Dictionary, id: String) -> Dictionary:
	for m in cat.get("models", []):
		if String((m as Dictionary).get("id", "")) == id:
			return m
	return {}


## "Amazon · Nova 2 Lite": how a picker names a catalog entry.
static func label(m: Dictionary) -> String:
	var p := String(m.get("provider", ""))
	return ("%s · %s" % [p, String(m.get("name", ""))]) if not p.is_empty() else String(m.get("name", m.get("id", "")))


## The region the CLI will use, as far as ghost can tell without running it: AWS_REGION, then
## AWS_DEFAULT_REGION, then the active profile's `region` in the config file. "" when none is
## named (the CLI then decides; the catalog is keyed by what was asked, so it still holds).
static func region() -> String:
	for v in ["AWS_REGION", "AWS_DEFAULT_REGION"]:
		if not OS.get_environment(v).is_empty():
			return OS.get_environment(v)
	var path := OS.get_environment("AWS_CONFIG_FILE")
	if path.is_empty():
		path = Deps.home().path_join(".aws").path_join("config")
	var profile := OS.get_environment("AWS_PROFILE")
	if profile.is_empty():
		profile = OS.get_environment("AWS_DEFAULT_PROFILE")
	var want := "default" if profile.is_empty() or profile == "default" else "profile " + profile
	var inside := false
	for line in FileAccess.get_file_as_string(path).split("\n"):
		var t := String(line).strip_edges()
		if t.begins_with("["):
			inside = t.trim_prefix("[").trim_suffix("]").strip_edges() == want
		elif inside and t.get_slice("=", 0).strip_edges() == "region":
			return t.get_slice("=", 1).strip_edges()
	return ""


## The AWS CLI's own complaint, from a stderr log: its last non-empty line.
static func cli_error(log_path: String) -> String:
	var lines := FileAccess.get_file_as_string(log_path).strip_edges().split("\n")
	return String(lines[lines.size() - 1]).strip_edges().substr(0, 300) if not lines.is_empty() else ""


## Run [param args] on the AWS CLI for a job step: its pid (<= 0 with job.error set).
static func run(job: Dictionary, binary: String, args: PackedStringArray, out: String, err: String,
		tag: String) -> int:
	var pid := Subprocess.start_redirected(binary, args, {"cwd": String(job["dir"]), "out": out, "err": err}, tag)
	if pid <= 0:
		job["error"] = "could not start aws (is the AWS CLI installed?)"
	return pid
