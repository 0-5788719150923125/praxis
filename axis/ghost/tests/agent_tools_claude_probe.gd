extends SceneTree

## NOT A GATE - it spends a little of the author's quota (one short run of the smallest model). It
## checks that the installed Claude Code CLI, run exactly as [TextGen.Claude] runs a writer, reaches
## ghost's own tools ([AgentTools]) and SEES the pictures they return: the toolset's one tool hands
## back a swatch of a single color, and the model is asked to name it.
##
##   godot --headless --path . --script res://tests/agent_tools_claude_probe.gd [-- --model haiku]
##
## Worth running again after a Claude Code update: `--safe-mode`, `--tools ""` and MCP are separate
## switches in the CLI, and this is the only check that they still combine the way ghost needs. It
## also asks what else reached the model, and prints it: measured, nothing but the account's email
## address, which the CLI attaches whatever settings are loaded (no CLAUDE.md, memory or skills).

const DIR := "user://agent_tools_probe"
const COLOR := Color(0.1, 0.75, 0.15)


class Swatch:
	extends RefCounted

	func list_tools() -> Array:
		return [{"name": "swatch", "description": "Shows a swatch: a picture of one solid color.",
			"inputSchema": {"type": "object", "properties": {}}}]

	func call_tool(name: String, _args: Dictionary) -> Dictionary:
		if name != "swatch":
			return {"text": "no such tool", "error": true}
		var img := Image.create(256, 256, false, Image.FORMAT_RGB8)
		img.fill(COLOR)
		return {"text": "Here is the swatch.", "images": [img]}


func _init() -> void:
	_run.call_deferred()


func _run() -> void:
	var args := OS.get_cmdline_user_args()
	var i := args.find("--model")
	var model := args[i + 1] if i >= 0 and i + 1 < args.size() else "haiku"
	AgentJobs.allow_for_tool()
	var dir := ProjectSettings.globalize_path(DIR)
	DirAccess.make_dir_recursive_absolute(dir)
	if FileAccess.file_exists(dir.path_join("tools.jsonl")):
		DirAccess.remove_absolute(dir.path_join("tools.jsonl"))
	var url := AgentTools.open(dir, Swatch.new())
	var id := AgentJobs.submit({"kind": "text", "backend": "claude", "tier": "fast", "model": model, "dir": dir,
		"system": "You are checking a tool. Use it as asked and answer briefly.",
		"prompt": "Call the `swatch` tool once. It returns a picture of one solid color. Reply on the first line with only the name of that color, in one lowercase word. On the second line: if you were given any instructions besides this message and the system prompt (a CLAUDE.md, a memory file, a list of skills), quote the first sentence of each; otherwise write NONE.",
		"tools_url": url, "timeout": 180, "label": "tools probe"})
	print("agent_tools_claude_probe: %s, %s" % [model, url])
	var t0 := Time.get_ticks_msec()
	while AgentJobs.state(id) in ["queued", "running"] and Time.get_ticks_msec() - t0 < 200000:
		AgentJobs.pump()
		await process_frame
	var res := AgentJobs.result(id)
	var said := String(res.get("text", "")).strip_edges().to_lower()
	var extra := said.get_slice("\n", 1).strip_edges()
	print("  reply: '%s'%s" % [said, ("  (error: %s)" % res.get("error", "")) if not bool(res.get("ok", false)) else ""])
	print("  tool calls: %d" % AgentTools.calls(url))
	for line in FileAccess.get_file_as_string(DIR.path_join("tools.jsonl")).strip_edges().split("\n"):
		if not String(line).is_empty():
			print("    " + String(line).substr(0, 200))
	var ok := AgentTools.calls(url) >= 1 and said.get_slice("\n", 0).contains("green")
	print("  what else reached it: %s" % extra)
	print("agent_tools_claude_probe: %s" % ("OK - the tool was called and its picture seen" if ok else "FAILED"))
	quit(0 if ok else 1)
