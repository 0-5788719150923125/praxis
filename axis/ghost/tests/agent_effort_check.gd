extends SceneTree

## agent_effort_check - the REASONING EFFORT a tarot show sets for its writer and its painter
## (2026-10-06, the user: "if they do - I think we should make that an option that we would be able
## to set"), each CLI's own setting, with no agent run.
##
##   godot --headless --path . --script res://tests/agent_effort_check.gd
##
## - CLAUDE: `--effort <level>` when one is chosen, nothing when not (the author's own Claude Code
##   default for the model, read from `modelSettings` to label Default); the CLI's five levels.
## - CODEX: `-c model_reasoning_effort=<level>`, the chosen one or ghost's own (the writer's tier:
##   medium, low for designs; the painter: low); the levels each model's catalog entry offers.
## - AMAZON NOVA: Nova 2's `reasoningConfig` when a level is chosen, nothing otherwise and nothing
##   for a model that does not think; at "high" no output cap, as Nova 2 requires.
## - THE PANEL: an Effort picker beside each model picker, its choice in the show's knobs and the
##   producer's spec, reset when the agent changes, kept (marked) when a model does not offer it.

const CODEX_FIXTURE := {"models": [
	{"slug": "m-small", "visibility": "list", "supported_reasoning_levels": [{"effort": "low"}, {"effort": "medium"}]},
	{"slug": "m-big", "visibility": "list", "supported_reasoning_levels": [{"effort": "low"}, {"effort": "high"}, {"effort": "max"}]}]}

var _fails := 0


func _init() -> void:
	_run.call_deferred()


func _ok(cond: bool, what: String) -> void:
	print(("  ok    " if cond else "  FAIL  ") + what)
	if not cond:
		_fails += 1


func _run() -> void:
	var settings := root.get_node_or_null("Settings")
	if settings != null:
		settings.set("_read_only", true)
	var keep_home := OS.get_environment("HOME")
	var keep_codex := OS.get_environment("CODEX_HOME")
	var scratch := ProjectSettings.globalize_path("user://agent_effort_check")
	DirAccess.make_dir_recursive_absolute(scratch.path_join(".claude"))
	DirAccess.make_dir_recursive_absolute(scratch.path_join("codex"))
	_put(scratch.path_join(".claude/settings.json"), JSON.stringify({"modelSettings": {"claude-opus-9": {"effortLevel": "high"}}}))
	_put(scratch.path_join("codex/models_cache.json"), JSON.stringify(CODEX_FIXTURE))
	_put(scratch.path_join("codex/config.toml"), "model = \"m-big\"\n")
	OS.set_environment("HOME", scratch)
	OS.set_environment("CODEX_HOME", scratch.path_join("codex"))
	for check in [_claude, _codex, _nova, _panel]:
		if not bool(check.call()):
			_ok(false, "a check stopped part way (script error?)")
	OS.set_environment("HOME", keep_home)
	if keep_codex.is_empty():
		OS.unset_environment("CODEX_HOME")
	else:
		OS.set_environment("CODEX_HOME", keep_codex)
	for f in [".claude/settings.json", "codex/models_cache.json", "codex/config.toml"]:
		DirAccess.remove_absolute(scratch.path_join(f))
	for d in [".claude", "codex", ""]:
		DirAccess.remove_absolute(scratch.path_join(d))
	print("agent_effort_check: %s" % ("ALL OK" if _fails == 0 else "%d FAILED" % _fails))
	quit(0 if _fails == 0 else 1)


func _put(path: String, text: String) -> void:
	var f := FileAccess.open(path, FileAccess.WRITE)
	f.store_string(text)
	f.close()


static func _keys(list: Array) -> Array:
	return list.map(func(e: Dictionary) -> String: return String(e["key"]))


func _claude() -> bool:
	var job := {"dir": "/tmp/x", "tier": "best"}
	var plain: PackedStringArray = TextGen.Claude.argv(job, "/tmp/x/system.txt", PackedStringArray())
	_ok(not plain.has("--effort"), "Claude with no level chosen: no --effort, its own default")
	job["effort"] = "xhigh"
	var set: PackedStringArray = TextGen.Claude.argv(job, "/tmp/x/system.txt", PackedStringArray())
	var at := set.find("--effort")
	_ok(at >= 0 and set[at + 1] == "xhigh", "Claude with a level chosen: --effort xhigh")
	var list: Array = TextGen.Claude.efforts("")
	_ok(_keys(list) == ["", "low", "medium", "high", "xhigh", "max"], "Claude offers its CLI's five levels")
	_ok(String(list[0]["label"]) == "Default (High)", "its Default names the author's own setting for the model (%s)" % list[0]["label"])
	_ok(String(TextGen.Claude.efforts("sonnet")[0]["label"]) == "Default", "a model with no setting of its own: plain Default")
	return true


func _codex() -> bool:
	var job := {"dir": "/tmp/x", "tier": "fast"}
	var w: PackedStringArray = TextGen.Codex.argv(job, [])
	_ok(w.has("model_reasoning_effort=low"), "the Codex writer, unasked: its tier's effort (low for a design)")
	job["tier"] = "best"
	_ok(TextGen.Codex.argv(job, []).has("model_reasoning_effort=medium"), "...medium for the plan and the reading")
	job["effort"] = "max"
	_ok(TextGen.Codex.argv(job, []).has("model_reasoning_effort=max"), "a chosen level overrides the tier")
	var pjob := {"dir": "/tmp/x", "refs": []}
	_ok(ImageGen.Codex.argv(pjob).has("model_reasoning_effort=low"), "the Codex painter, unasked: low")
	pjob["effort"] = "high"
	_ok(ImageGen.Codex.argv(pjob).has("model_reasoning_effort=high"), "the painter with a level chosen")
	_ok(_keys(ImageGen.Codex.efforts("m-small")) == ["", "low", "medium"], "a model offers its own catalog's levels")
	_ok(_keys(TextGen.Codex.efforts("")) == ["", "low", "high", "max"], "Default model: the config's model's levels")
	_ok(_keys(TextGen.Codex.efforts("not-listed")) == ["", "low", "medium", "high", "max"],
		"a model the catalog does not list: every level any listed model takes")
	return true


func _nova() -> bool:
	var plain: Dictionary = TextGen.Bedrock.request("us.amazon.nova-2-lite-v1:0", "sys", [{"text": "hi"}])
	_ok(not plain.has("additionalModelRequestFields") and plain.has("inferenceConfig"), "Nova 2 unasked: thinking off")
	var low: Dictionary = TextGen.Bedrock.request("us.amazon.nova-2-lite-v1:0", "sys", [{"text": "hi"}], 10000, false, "low")
	_ok(low.get("additionalModelRequestFields") == {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "low"}}
		and low.has("inferenceConfig"), "Nova 2 with a level: its reasoningConfig")
	var high: Dictionary = TextGen.Bedrock.request("us.amazon.nova-2-lite-v1:0", "sys", [{"text": "hi"}], 10000, false, "high")
	_ok(not high.has("inferenceConfig") and high["additionalModelRequestFields"]["reasoningConfig"]["maxReasoningEffort"] == "high",
		"at high, no output cap (Nova 2 refuses one)")
	var lite: Dictionary = TextGen.Bedrock.request("us.amazon.nova-lite-v1:0", "sys", [{"text": "hi"}], 10000, false, "medium")
	_ok(not lite.has("additionalModelRequestFields"), "a model that does not think is asked nothing")
	_ok(_keys(TextGen.Bedrock.efforts("")) == ["", "low", "medium", "high"] and _keys(TextGen.Bedrock.efforts("amazon.nova-micro-v1:0")) == [""],
		"Nova 2 offers three levels; the others none")
	_ok(_keys(ImageGen.Bedrock.efforts("")) == [""], "Bedrock's painter takes no setting")
	return true


func _panel() -> bool:
	var ed = load("res://scripts/tarot_editor.gd").new()
	ed._build_panel()
	ed._knobs["writer"] = "claude"
	ed._show_knobs()
	var pick: OptionButton = ed._writer_effort
	_ok(pick != null and pick.item_count == 6 and pick.get_item_text(0).begins_with("Default") and not pick.disabled,
		"the writer row has an Effort picker with the agent's levels")
	pick.select(4)
	pick.item_selected.emit(4)
	_ok(ed._knobs["writer_effort"] == "xhigh" and ed._spec()["writer_effort"] == "xhigh", "a level chosen is the show's knob and the producer's")
	ed._knobs["writer_model"] = "sonnet"
	ed._fill_efforts(pick)
	_ok(String(pick.get_item_metadata(pick.selected)) == "xhigh", "another model keeps the chosen level")
	var agents: OptionButton = ed._writer_pick
	var codex_at: int = TextGen.REGISTRY.keys().find("codex")
	agents.select(codex_at)
	agents.item_selected.emit(codex_at)
	_ok(ed._knobs["writer_effort"] == "" and pick.get_item_text(0) == "Default (medium, low for designs)",
		"another agent: back to its own default, with its own levels")
	ed._knobs["painter"] = "bedrock"
	ed._knobs["painter_effort"] = ""
	ed._fill_efforts(ed._painter_effort)
	_ok(ed._painter_effort.disabled, "a painter with no effort setting grays the picker out")
	ed._knobs["writer_effort"] = "ultra"
	ed._knobs["writer_model"] = "m-small"
	ed._fill_efforts(pick)
	_ok(pick.get_item_text(pick.selected) == "Ultra  (not offered)", "a level this model does not offer stays, marked")
	ed.free()
	return true
