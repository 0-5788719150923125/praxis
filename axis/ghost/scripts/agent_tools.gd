extends RefCounted
class_name AgentTools

## AgentTools - the tools an agent can call while it works, served by ghost itself.
##
## A writer run by [TextGen] used to be ONE prompt in and ONE reply out: it could not look at what
## it made, check it, or change it. A job given a TOOLSET instead works in a loop - it calls a tool,
## reads (or looks at) what comes back, and calls again - and ghost answers every call from inside
## the running app, so a tool can do anything ghost can: build a thing, stand it on the table,
## photograph it.
##
## THE WIRE IS MCP over HTTP (the "streamable HTTP" transport, answered with plain JSON): a server on
## 127.0.0.1 at a port the system picks, started the first time a job asks for tools, polled from
## [method AgentJobs.pump] like everything else a job needs. Every agent CLI that speaks MCP can be
## pointed at it with a URL, so the toolset is written once for all of them.
##
## EACH JOB GETS ITS OWN URL, with a random token in its path ([method open]): the token says which
## toolset answers, so a job can only ever reach its own tools, and what those tools let it see is
## the toolset's decision (a reader's could stop at the cards drawn so far). Requests carrying a
## browser's Origin are refused, as the protocol asks of a local server.
##
## EVERY CALL IS KEPT in the job's folder beside its prompt and reply: `tools.jsonl`, one line per
## call (the tool, what it was given, what it said), and each picture it returned as `look_NN.jpg`.
## What an agent was shown while it worked is as plain on disk as what it was told.
##
## A TOOLSET is any object with `list_tools() -> Array` (MCP tool descriptors: name, description,
## inputSchema), `call_tool(name, args) -> Dictionary` (`{text, images: [Image], error}`, and it may
## await - a picture can take frames) and, optionally, `instructions() -> String`.

## Where the endpoint lives; a job's token follows it.
const PATH := "/mcp/"
## The protocol version answered when a client names none (otherwise its own is echoed: a toolset
## server uses nothing that differs between versions).
const PROTOCOL := "2025-06-18"
## The largest request read, and how long a half-sent one is waited for (ms).
const MAX_REQUEST := 8 * 1024 * 1024
const STALL_MS := 30000
## The most calls one job may make: a runaway loop ends here, not in the author's quota.
const MAX_CALLS := 200
## A picture goes over the wire as a JPEG of this quality: a fraction of a PNG's size, and the
## model sees it the same.
const JPEG := 0.88

static var _the: AgentTools = null

var _tcp: TCPServer = null
var _port := 0
var _conns: Array = []          # [{peer, buf, busy, t}]
var _jobs := {}                 # token -> {toolset, dir, calls, images}


## A URL for [param toolset], whose calls are logged into [param dir] (the job's folder); "" when
## no server can run here (a read-only process, or no port).
static func open(dir: String, toolset: Object) -> String:
	if AgentJobs.read_only() or toolset == null:
		return ""
	var s := _server()
	if s == null:
		return ""
	var token := Crypto.new().generate_random_bytes(16).hex_encode()
	s._jobs[token] = {"toolset": toolset, "dir": dir, "calls": 0, "images": 0}
	DirAccess.make_dir_recursive_absolute(dir)
	return "http://127.0.0.1:%d%s%s" % [s._port, PATH, token]


## Stop answering for the job behind [param url] (its token alone works too).
static func close(url: String) -> void:
	if _the != null:
		_the._jobs.erase(url.get_file())


## Calls made so far through [param url], or -1 for one this server does not know.
static func calls(url: String) -> int:
	if _the == null or not _the._jobs.has(url.get_file()):
		return -1
	return int((_the._jobs[url.get_file()] as Dictionary)["calls"])


## Take what has arrived and answer what is complete. Called every frame.
static func poll() -> void:
	if _the != null:
		_the._poll()


## The port the server listens on, 0 before it has started.
static func port() -> int:
	return _the._port if _the != null else 0


static func _server() -> AgentTools:
	if _the != null:
		return _the
	var s := AgentTools.new()
	s._tcp = TCPServer.new()
	if s._tcp.listen(0, "127.0.0.1") != OK:
		push_warning("ghost: the agent tool server could not listen on 127.0.0.1")
		return null
	s._port = s._tcp.get_local_port()
	_the = s
	print("ghost: agent tools served on 127.0.0.1:%d" % s._port)
	return s


func _poll() -> void:
	while _tcp.is_connection_available():
		var peer := _tcp.take_connection()
		if peer != null:
			_conns.append({"peer": peer, "buf": PackedByteArray(), "busy": false, "t": Time.get_ticks_msec()})
	for c in _conns.duplicate():
		var conn: Dictionary = c
		var peer: StreamPeerTCP = conn["peer"]
		peer.poll()
		if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			_conns.erase(conn)
			continue
		if bool(conn["busy"]):
			continue
		var n := peer.get_available_bytes()
		if n > 0:
			var got: Array = peer.get_partial_data(n)
			if int(got[0]) == OK:
				var buf: PackedByteArray = conn["buf"]
				buf.append_array(got[1] as PackedByteArray)
				conn["buf"] = buf
				conn["t"] = Time.get_ticks_msec()
		var req := AgentTools.parse_request(conn["buf"])
		if req.is_empty():
			if Time.get_ticks_msec() - int(conn["t"]) > STALL_MS:
				peer.disconnect_from_host()
				_conns.erase(conn)
			continue
		conn["busy"] = true
		_handle(conn, req)


## One request, answered when its tool is done - which may be frames later: the connection waits.
func _handle(conn: Dictionary, req: Dictionary) -> void:
	var out: Dictionary = await _respond(req)
	var peer: StreamPeerTCP = conn["peer"]
	if peer.get_status() == StreamPeerTCP.STATUS_CONNECTED:
		peer.put_data(AgentTools.http_response(int(out["status"]), out.get("body", ""), out.get("headers", {})))
		peer.disconnect_from_host()
	_conns.erase(conn)


func _respond(req: Dictionary) -> Dictionary:
	if req.has("error"):
		return {"status": int(req["error"]), "body": ""}
	var method := String(req["method"])
	if method != "POST":
		# no stream of the server's own messages, and no sessions to end: POST is the protocol here
		return {"status": 405, "body": "", "headers": {"Allow": "POST"}}
	var origin := String((req["headers"] as Dictionary).get("origin", ""))
	if not origin.is_empty() and origin != "null" and not AgentTools.local_origin(origin):
		return {"status": 403, "body": ""}
	var path := String(req["path"]).get_slice("?", 0)
	var token := path.trim_prefix(PATH) if path.begins_with(PATH) else ""
	if token.is_empty() or not _jobs.has(token):
		return {"status": 404, "body": ""}
	var job: Dictionary = _jobs[token]
	var j := JSON.new()
	if j.parse((req["body"] as PackedByteArray).get_string_from_utf8()) != OK:
		return {"status": 400, "body": JSON.stringify(AgentTools.rpc_error(null, -32700, "Parse error"))}
	var batch: bool = j.data is Array
	var messages: Array = j.data if batch else [j.data]
	var replies: Array = []
	for m in messages:
		if not (m is Dictionary):
			replies.append(AgentTools.rpc_error(null, -32600, "Invalid Request"))
			continue
		var msg: Dictionary = m
		if not msg.has("method") or not msg.has("id"):
			continue                  # a notification, or the client answering: nothing to say
		replies.append(await _rpc(job, msg))
	if replies.is_empty():
		return {"status": 202, "body": ""}
	return {"status": 200, "body": JSON.stringify(replies if batch else replies[0]),
		"headers": {"Content-Type": "application/json"}}


func _rpc(job: Dictionary, msg: Dictionary) -> Dictionary:
	var id: Variant = msg["id"]
	var params: Dictionary = msg.get("params", {}) if msg.get("params") is Dictionary else {}
	var toolset: Object = job["toolset"]
	match String(msg["method"]):
		"initialize":
			var asked := String(params.get("protocolVersion", "")) if params.get("protocolVersion") is String else ""
			var result := {"protocolVersion": asked if not asked.is_empty() else PROTOCOL,
				"capabilities": {"tools": {"listChanged": false}},
				"serverInfo": {"name": "ghost", "version": "1.0"}}
			if toolset.has_method("instructions"):
				result["instructions"] = String(toolset.call("instructions"))
			return AgentTools.rpc_result(id, result)
		"ping":
			return AgentTools.rpc_result(id, {})
		"tools/list":
			return AgentTools.rpc_result(id, {"tools": toolset.call("list_tools")})
		"tools/call":
			var name := String(params.get("name", ""))
			var args: Dictionary = params.get("arguments", {}) if params.get("arguments") is Dictionary else {}
			job["calls"] = int(job["calls"]) + 1
			var res: Dictionary
			if int(job["calls"]) > MAX_CALLS:
				res = {"text": "No calls left: this job has made %d. Stop here." % MAX_CALLS, "error": true}
			else:
				var got: Variant = await toolset.call("call_tool", name, args)
				res = got if got is Dictionary else {"text": "the tool returned nothing", "error": true}
			return AgentTools.rpc_result(id, _content(job, name, args, res))
	return AgentTools.rpc_error(id, -32601, "Method not found: %s" % String(msg["method"]))


## A tool's outcome as MCP content - its words, then its pictures (as JPEGs) - logged as it goes.
func _content(job: Dictionary, name: String, args: Dictionary, res: Dictionary) -> Dictionary:
	var text := String(res.get("text", ""))
	var content: Array = [{"type": "text", "text": text}]
	var saved := PackedStringArray()
	for im in res.get("images", []):
		if not (im is Image) or (im as Image).is_empty():
			continue
		var img := (im as Image).duplicate() as Image
		if img.is_compressed():
			img.decompress()
		img.convert(Image.FORMAT_RGB8)
		var jpg := img.save_jpg_to_buffer(JPEG)
		job["images"] = int(job["images"]) + 1
		var file := "look_%02d.jpg" % int(job["images"])
		var f := FileAccess.open(String(job["dir"]).path_join(file), FileAccess.WRITE)
		if f != null:
			f.store_buffer(jpg)
			f.close()
			saved.append(file)
		content.append({"type": "image", "data": Marshalls.raw_to_base64(jpg), "mimeType": "image/jpeg"})
	var error := bool(res.get("error", false))
	AgentTools.log_call(String(job["dir"]), {"at": int(Time.get_unix_time_from_system()), "tool": name,
		"args": args, "text": text, "images": Array(saved), "error": error})
	return {"content": content, "isError": error}


## One line in the job's `tools.jsonl`.
static func log_call(dir: String, entry: Dictionary) -> void:
	var path := dir.path_join("tools.jsonl")
	var f := FileAccess.open(path, FileAccess.READ_WRITE if FileAccess.file_exists(path) else FileAccess.WRITE)
	if f == null:
		return
	f.seek_end()
	f.store_line(JSON.stringify(entry))
	f.close()


# --- the wire, pure ---------------------------------------------------------------------------------

## A complete HTTP request out of [param buf]: `{method, path, headers (lowercase names), body}`;
## `{error: status}` for one that cannot be served; empty while more is still to come.
static func parse_request(buf: PackedByteArray) -> Dictionary:
	var end := -1
	for i in range(0, buf.size() - 3):
		if buf[i] == 13 and buf[i + 1] == 10 and buf[i + 2] == 13 and buf[i + 3] == 10:
			end = i
			break
	if end < 0:
		return {"error": 431} if buf.size() > 65536 else {}
	var head := buf.slice(0, end).get_string_from_utf8().split("\r\n")
	var first := String(head[0]).split(" ")
	if first.size() < 2:
		return {"error": 400}
	var headers := {}
	for i in range(1, head.size()):
		var line := String(head[i])
		var colon := line.find(":")
		if colon > 0:
			headers[line.substr(0, colon).strip_edges().to_lower()] = line.substr(colon + 1).strip_edges()
	var rest := buf.slice(end + 4)
	var body := PackedByteArray()
	if String(headers.get("transfer-encoding", "")).to_lower().contains("chunked"):
		var at := 0
		while true:
			var nl := -1
			for i in range(at, rest.size() - 1):
				if rest[i] == 13 and rest[i + 1] == 10:
					nl = i
					break
			if nl < 0:
				return {}
			var size_text := rest.slice(at, nl).get_string_from_ascii().get_slice(";", 0).strip_edges()
			if not size_text.is_valid_hex_number():
				return {"error": 400}
			var size := size_text.hex_to_int()
			if size == 0:
				break
			if rest.size() < nl + 2 + size + 2:
				return {}
			body.append_array(rest.slice(nl + 2, nl + 2 + size))
			if body.size() > MAX_REQUEST:
				return {"error": 413}
			at = nl + 2 + size + 2
	else:
		var length := int(String(headers.get("content-length", "0")))
		if length > MAX_REQUEST:
			return {"error": 413}
		if rest.size() < length:
			return {}
		body = rest.slice(0, length)
	return {"method": String(first[0]).to_upper(), "path": String(first[1]), "headers": headers, "body": body}


## An HTTP response, every connection closed after it.
static func http_response(status: int, body: String, headers: Dictionary = {}) -> PackedByteArray:
	var reason := {200: "OK", 202: "Accepted", 400: "Bad Request", 403: "Forbidden", 404: "Not Found",
		405: "Method Not Allowed", 413: "Content Too Large", 431: "Request Header Fields Too Large"}
	var bytes := body.to_utf8_buffer()
	var lines := PackedStringArray(["HTTP/1.1 %d %s" % [status, String(reason.get(status, "Error"))]])
	for k in headers:
		lines.append("%s: %s" % [String(k), String(headers[k])])
	lines.append("Content-Length: %d" % bytes.size())
	lines.append("Connection: close")
	var out := ("\r\n".join(lines) + "\r\n\r\n").to_utf8_buffer()
	out.append_array(bytes)
	return out


## Whether [param origin] is this machine (a page anywhere else may not call a local server).
static func local_origin(origin: String) -> bool:
	var host := origin.get_slice("://", 1).get_slice("/", 0)
	if host.begins_with("["):
		host = host.get_slice("]", 0) + "]"
	else:
		host = host.get_slice(":", 0)
	return host in ["127.0.0.1", "localhost", "[::1]"]


static func rpc_result(id: Variant, result: Variant) -> Dictionary:
	return {"jsonrpc": "2.0", "id": _id(id), "result": result}


static func rpc_error(id: Variant, code: int, message: String) -> Dictionary:
	return {"jsonrpc": "2.0", "id": _id(id), "error": {"code": code, "message": message}}


## A request's id as it came: JSON reads every number as a float, and a client keyed on the integer
## it sent must get that integer back.
static func _id(id: Variant) -> Variant:
	if id is float and is_equal_approx(id, roundf(id)) and absf(id) < 9.0e15:
		return int(id)
	return id
