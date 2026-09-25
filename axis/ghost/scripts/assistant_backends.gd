extends RefCounted
class_name AssistantBackends

## AssistantBackends - the CLIs [Assistant] can dispatch a feedback note to, and everything
## that differs between them: how a run is launched, how it is resumed, and how its event
## stream reads.
##
## A REGISTRY, NOT A BRANCH per call site. The first backend was written inline, so its
## command line, its event names and its cost field were spread through assistant.gd as
## literals. Adding a second as `if backend == ...` at each of those places is how one of
## them gets missed and a Codex run is parsed as Claude's stream - which fails silently,
## as a run that never reports progress and never finishes cleanly.
##
## Everything here is PURE (strings and dictionaries in, strings and dictionaries out), so
## tests/assistant_backend_check.gd can drive it with recorded streams and no subprocess.
##
## THE PROMPT IS NEVER IN ARGV. [method argv] returns flags, ids and paths only; the prompt
## reaches the CLI on stdin (`claude -p` reads it there when no prompt argument is given,
## `codex exec -` by request), from a file [method Subprocess.start_redirected] feeds in. No
## shell ever parses it, and Windows' command-line quoting - which does not escape quotes
## inside an argument - never sees it.

## Key -> display label + the [Deps] row that resolves its binary. Keys are what
## `[assistant] backend` in ghost.cfg stores (see splash.gd); "claude_cli" predates this
## file and must never change, or every saved choice and entry would stop resolving.
const REGISTRY := {
	"claude_cli": {"label": "Claude Code CLI", "dep": "claude"},
	"codex_cli": {"label": "Codex CLI (OpenAI)", "dep": "codex"},
}

## An entry saved before backends were recorded was dispatched to Claude, the only
## backend there was.
const LEGACY := "claude_cli"


static func has(key: String) -> bool:
	return REGISTRY.has(key)


static func label(key: String) -> String:
	return String((REGISTRY.get(key, REGISTRY[LEGACY]) as Dictionary)["label"])


static func dep(key: String) -> String:
	return String((REGISTRY.get(key, REGISTRY[LEGACY]) as Dictionary)["dep"])


## The CLI's own argument vector for one run, the prompt excluded (it goes on stdin).
## [param session] is empty for a fresh run and the saved session/thread id for a resume or
## follow-up.
static func argv(key: String, bin: String, session: String, repo_root: String) -> PackedStringArray:
	var a := PackedStringArray([bin])
	match key:
		"codex_cli":
			# `resume` is its own subcommand with a narrower flag set - no -C, which is why
			# the launcher cd's into the repo for every run rather than relying on the flag.
			a.append("exec")
			if not session.is_empty():
				a.append_array(["resume", session])
			a.append_array(["--json", "--skip-git-repo-check",
				"--dangerously-bypass-approvals-and-sandbox"])
			if session.is_empty():
				a.append_array(["-C", repo_root])
			a.append("-")         # the prompt: read it from stdin
		_:
			# Sonnet 5, default (auto) effort - the CLI's --effort has no "auto" value, so it
			# is simply omitted. stream-json needs --verbose alongside -p.
			a.append_array(["-p", "--model", "sonnet", "--dangerously-skip-permissions",
				"--output-format", "stream-json", "--verbose"])
			if not session.is_empty():
				a.append_array(["--resume", session])
	return a


## The session/thread id an event carries, or "".
static func session_of(key: String, evt: Dictionary) -> String:
	match key:
		"codex_cli":
			if String(evt.get("type", "")) == "thread.started":
				return str(evt.get("thread_id", ""))
			return ""
		_:
			return str(evt.get("session_id", "")) if evt.has("session_id") else ""


## Dig an id out of raw captured text, for a run whose id was never saved.
static func recover_session(key: String, text: String) -> String:
	var k := "\"thread_id\":\"" if key == "codex_cli" else "\"session_id\":\""
	var i := text.find(k)
	if i < 0:
		return ""
	i += k.length()
	var j := text.find("\"", i)
	return text.substr(i, j - i) if j > i else ""


## One event -> a short "what is it doing now" line, or "" for events not worth showing.
static func describe(key: String, evt: Dictionary) -> String:
	match key:
		"codex_cli":
			var t := String(evt.get("type", ""))
			if t != "item.started" and t != "item.completed":
				return ""
			var item: Dictionary = evt.get("item", {})
			match String(item.get("type", "")):
				"command_execution":
					return "shell  " + _clip(String(item.get("command", "")), 55)
				"agent_message":
					var s := String(item.get("text", "")).strip_edges()
					return _clip(s, 70) if not s.is_empty() else ""
				"file_change":
					return "edit  " + _clip(_file_change_paths(item), 55)
				"reasoning":
					return "thinking…"
				"error":
					return "⚠ " + _clip(String(item.get("message", "")), 65)
				"":
					return ""
				var other:
					return String(other)
			return ""
		_:
			if String(evt.get("type", "")) != "assistant":
				return ""
			var blocks: Array = evt.get("message", {}).get("content", [])
			for b in blocks:
				if String(b.get("type", "")) == "tool_use":
					var input: Dictionary = b.get("input", {})
					var hint := ""
					for k in ["file_path", "command", "pattern", "path", "prompt"]:
						if input.has(k):
							hint = String(input[k])
							break
					hint = _clip(hint, 55)
					return "%s  %s" % [String(b.get("name", "tool")), hint] if hint != "" else String(b.get("name", "tool"))
			for b in blocks:
				if String(b.get("type", "")) == "text":
					var t := String(b.get("text", "")).strip_edges()
					if t != "":
						return _clip(t, 70)
			return "thinking…"


## The outcome of a finished run, from every event it wrote:
## `{ok, response, session, cost_usd, usage}` - `usage` is the line shown under a done entry
## (dollars for Claude, tokens for Codex, which reports no price).
static func result(key: String, events: Array) -> Dictionary:
	var out := {"ok": false, "response": "", "session": "", "cost_usd": 0.0, "usage": "", "error": ""}
	match key:
		"codex_cli":
			var last_msg := ""
			var completed := false
			var tokens := {}
			for e in events:
				var evt: Dictionary = e
				match String(evt.get("type", "")):
					"thread.started":
						out.session = str(evt.get("thread_id", ""))
					"item.completed":
						var item: Dictionary = evt.get("item", {})
						if String(item.get("type", "")) == "agent_message":
							last_msg = String(item.get("text", ""))
					"turn.completed":
						completed = true
						var u: Dictionary = evt.get("usage", {})
						for k in u:
							tokens[k] = int(tokens.get(k, 0)) + int(u[k])
					"turn.failed":
						completed = false
						out.error = str((evt.get("error", {}) as Dictionary).get("message", "turn failed")) \
							if evt.get("error") is Dictionary else "turn failed"
					"error":
						out.error = str(evt.get("message", "error"))
			out.response = last_msg
			# A top-level `error` is not by itself a failure - the CLI also reports stream
			# retries that way and then carries on. `turn.failed` is, and it clears
			# `completed` because it is the last word on the turn.
			out.ok = completed and last_msg != ""
			if out.ok:
				out.error = ""
			if not tokens.is_empty():
				out.usage = "%s in (%s cached) · %s out" % [_k(int(tokens.get("input_tokens", 0))),
					_k(int(tokens.get("cached_input_tokens", 0))), _k(int(tokens.get("output_tokens", 0)))]
		_:
			var parsed: Dictionary = {}
			for e in events:
				var evt: Dictionary = e
				if out.session == "" and evt.has("session_id"):
					out.session = str(evt.get("session_id", ""))
				if String(evt.get("type", "")) == "result":
					parsed = evt
			if not parsed.is_empty() and parsed.get("is_error", true) == false and parsed.has("result"):
				out.ok = true
				out.response = str(parsed.get("result", ""))
				out.session = str(parsed.get("session_id", out.session))
				out.cost_usd = float(parsed.get("total_cost_usd", 0.0))
				out.usage = "$%.3f" % out.cost_usd
	return out


## Every complete JSON object line in [param text], quietly - see Assistant._json_object.
static func events_in(text: String) -> Array:
	var out: Array = []
	for line in text.split("\n"):
		var e = json_object(line.strip_edges())
		if e is Dictionary:
			out.append(e)
	return out


static func json_object(line: String) -> Variant:
	if not line.begins_with("{"):
		return null
	var json := JSON.new()
	if json.parse(line) != OK:
		return null
	return json.data


static func _clip(s: String, n: int) -> String:
	return s.substr(0, n) + "…" if s.length() > n else s


static func _k(n: int) -> String:
	return ("%.1fk" % (n / 1000.0)) if n >= 1000 else str(n)


static func _file_change_paths(item: Dictionary) -> String:
	var paths := PackedStringArray()
	for c in item.get("changes", []):
		if c is Dictionary:
			paths.append(String((c as Dictionary).get("path", "")).get_file())
	return ", ".join(paths)
