extends SceneTree

## The agent tool server's gate: [AgentTools] spoken to over a real socket, the way an agent CLI
## speaks to it - with no agent and no model.
##
##   godot --headless --path . --script res://tests/agent_tools_check.gd
##
## - A JOB'S URL is its own: an unknown token is not found, a closed one stops answering.
## - THE HANDSHAKE: initialize echoes the client's protocol version and names the toolset's
##   instructions; a notification is accepted with nothing to say; an id comes back as it was sent.
## - A CALL reaches the toolset and comes back as content: words, then pictures as JPEGs that decode
##   to the picture made. A tool that takes frames is waited for. Every call is logged in the job's
##   folder, and every picture kept there.
## - THE WIRE: a request split across writes is answered only once whole (two-sided: the first half
##   alone is incomplete); a chunked body reads as the same request; a batch gets a batch; GET is not
##   served; a browser's Origin is refused and a local one is not; junk is a parse error.
## - A RUNAWAY LOOP ENDS: past the most calls a job may make, a call is refused.
## - A READ-ONLY process serves nothing.

const DIR := "user://agent_tools_check"

var _fails := 0


class Toolset:
	extends RefCounted

	var tree: SceneTree
	var seen: Array = []

	func instructions() -> String:
		return "Tools for the gate."

	func list_tools() -> Array:
		return [{"name": "echo", "description": "Say it back.", "inputSchema": {"type": "object",
				"properties": {"say": {"type": "string"}}}},
			{"name": "picture", "description": "A picture.", "inputSchema": {"type": "object", "properties": {}}},
			{"name": "slow", "description": "Takes a few frames.", "inputSchema": {"type": "object", "properties": {}}}]

	func call_tool(name: String, args: Dictionary) -> Dictionary:
		seen.append(name)
		match name:
			"echo":
				return {"text": "you said: %s" % String(args.get("say", ""))}
			"picture":
				var img := Image.create(64, 36, false, Image.FORMAT_RGB8)
				img.fill(Color(0.1, 0.8, 0.2))
				return {"text": "a green picture", "images": [img]}
			"slow":
				for i in 3:
					await tree.process_frame
				return {"text": "done after three frames"}
		return {"text": "no tool called %s" % name, "error": true}


func _init() -> void:
	_run.call_deferred()


func _ok(cond: bool, what: String) -> void:
	if not cond:
		_fails += 1
		print("  FAIL: " + what)


func _run() -> void:
	var dir := ProjectSettings.globalize_path(DIR)
	if DirAccess.dir_exists_absolute(dir):
		for f in DirAccess.get_files_at(dir):
			DirAccess.remove_absolute(dir.path_join(f))
	AgentJobs.allow_for_tool(false)
	var ts := Toolset.new()
	ts.tree = self
	_ok(AgentTools.open(dir, ts).is_empty(), "a read-only process opened a tool server")
	AgentJobs.allow_for_tool()
	var url := AgentTools.open(dir, ts)
	var m := RegEx.create_from_string("^http://127\\.0\\.0\\.1:(\\d+)/mcp/([0-9a-f]{32})$").search(url)
	_ok(m != null and int(m.get_string(1)) == AgentTools.port() and AgentTools.port() > 0,
		"the job's URL is not a local one with its own token: %s" % url)
	var path := "/mcp/" + url.get_file()

	# THE HANDSHAKE
	var r := await _post(path, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
		"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "gate", "version": "1"}}})
	var res: Dictionary = (r["json"] as Dictionary).get("result", {}) if r["json"] is Dictionary else {}
	_ok(int(r["status"]) == 200 and String(r["headers"].get("content-type", "")) == "application/json",
		"initialize was answered %d (%s)" % [r["status"], r["headers"].get("content-type", "")])
	_ok(String(res.get("protocolVersion", "")) == "2025-06-18", "initialize did not echo the client's version: %s" % str(res))
	_ok(res.get("capabilities", {}).has("tools") and String(res.get("serverInfo", {}).get("name", "")) == "ghost",
		"initialize did not offer tools as ghost: %s" % str(res))
	_ok(String(res.get("instructions", "")) == "Tools for the gate.", "the toolset's instructions were not passed on")
	_ok(String(r["text"]).contains("\"id\":1,") or String(r["text"]).contains("\"id\":1}"),
		"the id did not come back as the integer sent: %s" % r["text"])
	r = await _post(path, {"jsonrpc": "2.0", "method": "notifications/initialized"})
	_ok(int(r["status"]) == 202 and String(r["text"]).is_empty(), "a notification was answered %d '%s'" % [r["status"], r["text"]])
	r = await _post(path, {"jsonrpc": "2.0", "id": "a", "method": "tools/list"})
	var names: Array = []
	for t in (r["json"] as Dictionary).get("result", {}).get("tools", []):
		names.append(String((t as Dictionary)["name"]))
	_ok(names == ["echo", "picture", "slow"] and String((r["json"] as Dictionary)["id"]) == "a",
		"tools/list gave %s" % str(names))

	# A CALL, its content, its log
	r = await _post(path, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "echo", "arguments": {"say": "hi"}}})
	var content: Array = (r["json"] as Dictionary).get("result", {}).get("content", [])
	_ok(content.size() == 1 and String(content[0]["text"]) == "you said: hi"
		and not bool((r["json"] as Dictionary)["result"].get("isError", true)), "echo came back as %s" % str(r["json"]))
	r = await _post(path, {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "picture", "arguments": {}}})
	content = (r["json"] as Dictionary).get("result", {}).get("content", [])
	var pic := Image.new()
	var decoded := content.size() == 2 and String(content[1].get("type", "")) == "image" \
		and String(content[1].get("mimeType", "")) == "image/jpeg" \
		and pic.load_jpg_from_buffer(Marshalls.base64_to_raw(String(content[1]["data"]))) == OK
	_ok(decoded and pic.get_size() == Vector2i(64, 36) and pic.get_pixel(32, 18).g > 0.6 and pic.get_pixel(32, 18).r < 0.3,
		"the picture did not come back as the JPEG made: %s" % str(content.slice(0, 1)))
	_ok(FileAccess.file_exists(dir.path_join("look_01.jpg")), "the picture was not kept in the job's folder")
	r = await _post(path, {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "slow", "arguments": {}}})
	content = (r["json"] as Dictionary).get("result", {}).get("content", [])
	_ok(content.size() == 1 and String(content[0]["text"]) == "done after three frames", "a tool taking frames was not waited for: %s" % str(r))
	var log := FileAccess.get_file_as_string(dir.path_join("tools.jsonl")).strip_edges().split("\n")
	var tools: Array = []
	for line in log:
		var j := JSON.new()
		if j.parse(String(line)) == OK and j.data is Dictionary:
			tools.append(String((j.data as Dictionary)["tool"]))
	_ok(tools == ["echo", "picture", "slow"], "the calls were not logged in order: %s" % str(tools))
	_ok(log.size() >= 2 and String(log[1]).contains("look_01.jpg"), "the log does not name the picture kept")
	_ok(AgentTools.calls(url) == 3, "the job's calls were counted as %d" % AgentTools.calls(url))

	# THE WIRE
	var body := JSON.stringify({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "echo", "arguments": {"say": "in two"}}})
	var raw := _raw("POST", path, body)
	var half := raw.slice(0, raw.size() / 2 + 7)
	_ok(AgentTools.parse_request(half).is_empty(), "control: half a request read as a whole one")
	r = await _send(raw, 2)
	_ok(not bool(r["early"]), "a request split in two was answered before its second half came")
	_ok(int(r["status"]) == 200 and String(r["text"]).contains("you said: in two"), "a request sent in two parts was answered %s" % str(r["text"]))
	r = await _send(_chunked(path, body))
	_ok(int(r["status"]) == 200 and String(r["text"]).contains("you said: in two"), "a chunked request was answered %d %s" % [r["status"], r["text"]])
	r = await _send(_raw("POST", path, JSON.stringify([{"jsonrpc": "2.0", "id": 7, "method": "ping"},
		{"jsonrpc": "2.0", "method": "notifications/cancelled"}, {"jsonrpc": "2.0", "id": 8, "method": "ping"}])))
	_ok(r["json"] is Array and (r["json"] as Array).size() == 2 and int(r["json"][1]["id"]) == 8,
		"a batch was answered %s" % str(r["text"]))
	r = await _post(path, {"jsonrpc": "2.0", "id": 9, "method": "resources/list"})
	_ok(int((r["json"] as Dictionary).get("error", {}).get("code", 0)) == -32601, "an unknown method was answered %s" % str(r["text"]))
	r = await _send(_raw("POST", path, "{not json"))
	_ok(int(r["status"]) == 400 and int((r["json"] as Dictionary).get("error", {}).get("code", 0)) == -32700,
		"junk was answered %d %s" % [r["status"], r["text"]])
	r = await _send(_raw("GET", path, ""))
	_ok(int(r["status"]) == 405, "GET was answered %d" % r["status"])
	r = await _send(_raw("POST", path, body, {"Origin": "http://evil.example"}))
	_ok(int(r["status"]) == 403, "a browser page's request was answered %d" % r["status"])
	r = await _send(_raw("POST", path, body, {"Origin": "http://localhost:6274"}))
	_ok(int(r["status"]) == 200, "a local origin was refused (%d)" % r["status"])
	r = await _send(_raw("POST", "/mcp/" + "0".repeat(32), body))
	_ok(int(r["status"]) == 404, "an unknown token was answered %d" % r["status"])

	# A RUNAWAY LOOP ENDS
	AgentTools._the._jobs[url.get_file()]["calls"] = AgentTools.MAX_CALLS
	var echoes := ts.seen.count("echo")
	r = await _post(path, {"jsonrpc": "2.0", "id": 10, "method": "tools/call", "params": {"name": "echo", "arguments": {"say": "more"}}})
	var over: Dictionary = (r["json"] as Dictionary).get("result", {})
	_ok(bool(over.get("isError", false)) and String(over.get("content", [{}])[0].get("text", "")).begins_with("No calls left"),
		"a call past the most a job may make was answered %s" % str(over))
	_ok(ts.seen.count("echo") == echoes, "a call past the most a job may make reached its tool")

	# A CLOSED JOB stops answering
	AgentTools.close(url)
	r = await _post(path, {"jsonrpc": "2.0", "id": 11, "method": "ping"})
	_ok(int(r["status"]) == 404, "a closed job was answered %d" % r["status"])

	print("agent_tools_check: %s (%d failure%s)" % ["ALL OK" if _fails == 0 else "FAILED", _fails, "" if _fails == 1 else "s"])
	quit(1 if _fails > 0 else 0)


func _raw(method: String, path: String, body: String, extra := {}) -> PackedByteArray:
	var lines := PackedStringArray(["%s %s HTTP/1.1" % [method, path], "Host: 127.0.0.1:%d" % AgentTools.port(),
		"Accept: application/json, text/event-stream", "Content-Type: application/json"])
	for k in extra:
		lines.append("%s: %s" % [k, extra[k]])
	var b := body.to_utf8_buffer()
	lines.append("Content-Length: %d" % b.size())
	var out := ("\r\n".join(lines) + "\r\n\r\n").to_utf8_buffer()
	out.append_array(b)
	return out


func _chunked(path: String, body: String) -> PackedByteArray:
	var lines := PackedStringArray(["POST %s HTTP/1.1" % path, "Host: 127.0.0.1", "Content-Type: application/json",
		"Transfer-Encoding: chunked"])
	var out := ("\r\n".join(lines) + "\r\n\r\n").to_utf8_buffer()
	var b := body.to_utf8_buffer()
	var at := 0
	while at < b.size():
		var piece := b.slice(at, mini(at + 17, b.size()))
		out.append_array(("%x\r\n" % piece.size()).to_utf8_buffer())
		out.append_array(piece)
		out.append_array("\r\n".to_utf8_buffer())
		at += piece.size()
	out.append_array("0\r\n\r\n".to_utf8_buffer())
	return out


func _post(path: String, msg: Variant) -> Dictionary:
	return await _send(_raw("POST", path, JSON.stringify(msg)))


## Send [param raw] in [param parts] writes, the server polled between them, and read the reply to
## the end. `early` is whether any reply came before the last part was sent.
func _send(raw: PackedByteArray, parts := 1) -> Dictionary:
	var c := StreamPeerTCP.new()
	c.connect_to_host("127.0.0.1", AgentTools.port())
	var t0 := Time.get_ticks_msec()
	while c.get_status() == StreamPeerTCP.STATUS_CONNECTING and Time.get_ticks_msec() - t0 < 3000:
		c.poll()
		AgentTools.poll()
		await process_frame
	var got := PackedByteArray()
	var early := false
	var step := ceili(raw.size() / float(parts))
	for i in parts:
		c.put_data(raw.slice(i * step, mini((i + 1) * step, raw.size())))
		for k in 4:
			AgentTools.poll()
			await process_frame
			c.poll()
			if i < parts - 1 and c.get_available_bytes() > 0:
				early = true
	while Time.get_ticks_msec() - t0 < 5000:
		AgentTools.poll()
		c.poll()
		if c.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			break
		var n := c.get_available_bytes()
		if n > 0:
			got.append_array(c.get_partial_data(n)[1] as PackedByteArray)
		await process_frame
	var text := got.get_string_from_utf8()
	var split := text.find("\r\n\r\n")
	var head := text.substr(0, split).split("\r\n") if split >= 0 else PackedStringArray([text])
	var headers := {}
	for i in range(1, head.size()):
		var colon := String(head[i]).find(":")
		if colon > 0:
			headers[String(head[i]).substr(0, colon).to_lower()] = String(head[i]).substr(colon + 1).strip_edges()
	var body := text.substr(split + 4) if split >= 0 else ""
	var j := JSON.new()
	var parsed: Variant = j.data if not body.is_empty() and j.parse(body) == OK else null
	return {"status": int(String(head[0]).get_slice(" ", 1)), "headers": headers, "text": body, "json": parsed, "early": early}
